import argparse
import numpy as np
import time
import torch
import json
import torch.nn as nn
import cv2
import random
import torch.nn.functional as F

from solver_learnToAdd import Solver
from models.synthesizer import gumbel_softmax_sample
from os.path import basename, exists, join, splitext
from os import makedirs
from torch.autograd import Variable
from utils.data_loader_stargan import get_dataset
from torch.backends import cudnn
import operator
from collections import OrderedDict
from utils.utils import show

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


class ParamObject(object):

    def __init__(self, adict):
        """Convert a dictionary to a class

        @param :adict Dictionary
        """

        self.__dict__.update(adict)

        for k, v in adict.items():
            if isinstance(v, dict):
                self.__dict__[k] = ParamObject(v)

    def __getitem__(self,key):
        return self.__dict__[key]

    def values(self):
        return self.__dict__.values()

    def itemsAsDict(self):
        return dict(self.__dict__.items())


def make_image_with_text(img_size, text):
    fVFrm = 255*np.ones(img_size,dtype=np.uint8)
    font = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(fVFrm, text,(4,img_size[0]-10), font, 0.8,(0,0,0), 1,cv2.LINE_AA)
    return fVFrm

def make_coco_labels(real_c):
    """Generate domain labels for CelebA for debugging/testing.

    if dataset == 'CelebA':
        return single and multiple attribute changes
    elif dataset == 'Both':
        return single attribute changes
    """
    y = np.eye(real_c.size(1))

    fixed_c_list = []

    # single object addition and removal
    for i in range(2*real_c.size(1)):
        fixed_c = real_c.clone()
        for c in fixed_c:
            if i%2:
                c[i//2] = 0.
            else:
                c[i//2] = 1.
        fixed_c_list.append(Variable(fixed_c, volatile=True).cuda())

    # multi-attribute transfer (H+G, H+A, G+A, H+G+A)
    #if self.dataset == 'CelebA':
    #    for i in range(4):
    #        fixed_c = real_c.clone()
    #        for c in fixed_c:
    #            if i in [0, 1, 3]:   # Hair color to brown
    #                c[:3] = y[2]
    #            if i in [0, 2, 3]:   # Gender
    #                c[3] = 0 if c[3] == 1 else 1
    #            if i in [1, 2, 3]:   # Aged
    #                c[4] = 0 if c[4] == 1 else 1
    #        fixed_c_list.append(self.to_var(fixed_c, volatile=True))
    return fixed_c_list

def make_celeb_labels(real_c, c_dim=5, dataset='CelebA'):
    """Generate domain labels for CelebA for debugging/testing.

    if dataset == 'CelebA':
        return single and multiple attribute changes
    elif dataset == 'Both':
        return single attribute changes
    """
    y = [torch.FloatTensor([1, 0, 0]),  # black hair
         torch.FloatTensor([0, 1, 0]),  # blond hair
         torch.FloatTensor([0, 0, 1])]  # brown hair

    fixed_c_list = []

    # single attribute transfer
    for i in range(c_dim):
        fixed_c = real_c.clone()
        for c in fixed_c:
            if i < 3:
                c[:3] = y[i]
            else:
                c[i] = 0 if c[i] == 1 else 1   # opposite value
        fixed_c_list.append(Variable(fixed_c, volatile=True).cuda())

    # multi-attribute transfer (H+G, H+A, G+A, H+G+A)
    if dataset == 'CelebA':
        for i in range(4):
            fixed_c = real_c.clone()
            for c in fixed_c:
                if i in [0, 1, 3]:   # Hair color to brown
                    c[:3] = y[2]
                if i in [0, 2, 3]:   # Gender
                    c[3] = 0 if c[3] == 1 else 1
                if i in [1, 2, 3]:   # Aged
                    c[4] = 0 if c[4] == 1 else 1
            fixed_c_list.append(Variable(fixed_c, volatile=True).cuda())
    return fixed_c_list

def make_image(img_list, padimg=None, equalize=False):
    edit_images = []

    for img in img_list:
        img = img[:,[0,0,0], ::] if img.shape[1] == 1 else img
        img = np.clip(img.data.cpu().numpy().transpose(0, 2, 3, 1),-1,1)
        img  = 255*((img[0,::] + 1) / 2)
        if equalize:
            img = cv2.equalizeHist(img[:,:,0].astype(np.uint8))[:,:,None][:,:,[0,0,0]]
        edit_images.append(img)
        if padimg is not None:
            edit_images.append(padimg)
        #img_out = 255 * ((x_hat[i] + 1) / 2)
        #img_out_flip = 255 * ((x_hat_flip[i] + 1) / 2)
        #img_diff = np.clip(5*np.abs(img_out - img_out_flip), 0, 255)
        #img = Image.fromarray(stacked.astype(np.uint8))
        #img = Image.fromarray(stacked.astype(np.uint8))
    #stacked = stacked.transpose(2,1,0)
    stacked = np.hstack((edit_images))

    stacked = cv2.cvtColor(stacked.astype(np.uint8), cv2.COLOR_BGR2RGB)
    return stacked

def simple_make_image(img):
    img = img[:,[0,0,0], ::] if img.shape[1] == 1 else img
    img = np.clip(img.data.cpu().numpy().transpose(0, 2, 3, 1),-1,1)
    img = 255*((img[0,::] + 1) / 2)
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    return img

def saveIndividImages(image_list, mask_image_list, nameList,sample_dir, fp, cls):
    #sample_dir = join(params['sample_dump_dir'], basename(params['model'][0]).split('.')[0])
    fdir = join(sample_dir, splitext(basename(fp[0]))[0]+'_'+cls)
    if not exists(fdir):
        makedirs(fdir)

    for i, img in enumerate(image_list):
        fname = join(fdir, nameList[i]+'.png')
        img = simple_make_image(img)
        cv2.imwrite(fname, img)
        print('Saving into file: ' + fname)

    if mask_image_list is not None:
        for i, img in enumerate(mask_image_list):
            # Skip the first one. It is just empty image
            if i > 0:
                fname = join(fdir, 'mask_'+nameList[i]+'.png')
                img = simple_make_image(img)
                cv2.imwrite(fname, img)
                print('Saving into file: ' + fname)

def draw_arrows(img, pt1 , pt2):
    imgSz = img.shape[0]
    pt1 = ((pt1.data.cpu().numpy()+1.)/2.) * imgSz
    pt2 = ((pt2.data.cpu().numpy()[0,::]+1.)/2.) * imgSz
    for i in xrange(0,pt1.shape[1],2):
        for j in xrange(0,pt1.shape[2],2):
            if np.abs(pt1[0,i,j]-pt2[0,i,j]) > 2. or  np.abs(pt1[1,i,j]-pt2[1,i,j]) > 2. :
                img = cv2.arrowedLine(img.astype(np.uint8), tuple(pt1[:,i,j]), tuple(pt2[:,i,j]), color=(0,0,255), line_type=cv2.LINE_AA, thickness=1, tipLength = 0.4)
    return img

def make_image_with_deform(img_list, deformList, padimg=None):
    edit_images = []

    for i, img in enumerate(img_list):
        img = img[:,[0,0,0], ::] if img.shape[1] == 1 else img
        img = np.clip(img.data.cpu().numpy().transpose(0, 2, 3, 1),-1,1)
        img = 255*((img[0,::] + 1) / 2)
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        cur_deform=[]
        if len(deformList[i])>0 and len(deformList[i][0])>0:
            for d in deformList[i]:
                cur_deform.append(draw_arrows(img, d[1], d[0]))
        else:
            cur_deform=[img*0, img*0, img*0]

        edit_images.append(np.vstack(cur_deform))
        if padimg is not None:
            edit_images.append(padimg)
    stacked = np.hstack((edit_images))

    return stacked

def compute_deform_statistics(pt1 , pt2):
    imgSz = 128
    pt1 = ((pt1.data.cpu().numpy()+1.)/2.) * imgSz
    pt2 = ((pt2.data.cpu().numpy()[0,::]+1.)/2.) * imgSz
    lengths = np.linalg.norm(pt1-pt2,axis=0).flatten()
    mean = lengths.mean()
    maxl = lengths.max()

    return lengths, mean, maxl

def forward_kpextract(solver, boxImg, boxlabel, encode_kp):
    with torch.no_grad():
        pEncRand, _, kpRand  = solver.forward_patchenc(boxImg, boxlabel) # kpRandImg, kpRandStr
        kp_origmean = None
        kp_origvar = None
        kp_origstr = None
        mEncRand = None
        if encode_kp:
            mEncRand = solver.G.forward_inpEnc(kpRand['img'],  boxlabel)
        else:
            kp_origmean = kpRand['code']['mean'].clone()
            kp_origvar = kpRand['code']['var'].clone()
            kp_origstr = kpRand['str'].clone()

    return kpRand,  mEncRand, (kp_origmean, kp_origvar, kp_origstr), pEncRand


def gen_samples(params):
    # For fast training
    #cudnn.benchmark = True
    gpu_id = 0
    use_cuda = params['cuda']
    b_sz  = params['batch_size']

    if params['use_same_g']:
        if len(params['use_same_g']) == 1:
           gCV = torch.load(params['use_same_g'][0])
    solvers = []
    configs = []
    for i,mfile in enumerate(params['model']):
        model = torch.load(mfile)
        configs.append(model['arch'])
        configs[-1]['pretrained_model'] = mfile
        configs[-1]['load_encoder'] = 1
        configs[-1]['load_discriminator'] = 0
        configs[-1]['image_size'] = params['image_size']
        #configs[-1]['m_upsample_type'] = 'bilinear'
        #if 'g_downsamp_layers' not in configs[-1]:
        #    configs[-1]['g_downsamp_layers'] = 2
        #if 'g_dil_start' not in configs[-1]:
        #    configs[-1]['g_dil_start'] = 0
        #    configs[-1]['e_norm_type'] = 'drop'
        #    configs[-1]['e_ksize'] = 4
        #if len(params['withExtMask']) and params['mask_size']!= 32:
        #    if params['withExtMask'][i]:
        #        configs[-1]['lowres_mask'] = 0
        #        configs[-1]['load_encoder'] = 0

        solvers.append(Solver(None, None, ParamObject(configs[-1]), mode='test', pretrainedcv=model))
        solvers[-1].G.eval()
        solvers[-1].Patch_E.eval()
        solvers[-1].StyleMLP.eval()
        #solvers[-1].D.eval()
    bad_classes = [] if not params['show_only'] else [cls for cls in configs[0]['selected_attrs'] if cls not in params['show_only']]

    dataset = get_dataset('', '', params['image_size'], params['image_size'], params['dataset'], params['split'],
                          select_attrs=configs[0]['selected_attrs'], datafile=params['datafile'], bboxLoader=1,
                          bbox_size = params['box_size'], randomrotate = 0, randomscale = params['randomscale'],
                          max_object_size=params['max_object_size'], min_box_size = params['min_box_size'],
                          use_gt_mask = configs[0]['use_gtmask_inp'], n_fg_boxes = params['n_boxes'],
                          onlyrandBoxes= (params['extmask_type'] == 'randbox'), patch_mode = True,
                          use_highres_patch = params['use_highres_patch'], remove_invalid= params['remove_invalid'],
                          filter_class_instances=bad_classes if params['show_only'] else [])
    #data_iter = DataLoader(targ_split, batch_size=b_sz, shuffle=True, num_workers=8)
    targ_split =  dataset #train if params['split'] == 'train' else valid if params['split'] == 'val' else test
    data_iter = np.random.permutation(len(targ_split))

    if len(params['show_ids'])> 0:
        cocoIdToindex = {v:i for i,v in enumerate(dataset.valid_ids)}
        data_iter = [cocoIdToindex[k] for k in params['show_ids']]

    print(len(data_iter))

    print('-----------------------------------------')
    print('%s'%(' | '.join(targ_split.selected_attrs)))
    print('-----------------------------------------')

    flatten = lambda l: [item for sublist in l for item in sublist]

    if params['showreconst'] and len(params['names'])>0:
        params['names'] = flatten([[nm,nm+'-R'] for nm in params['names']])

    #discriminator.load_state_dict(cv['discriminator_state_dict'])
    c_idx = 0
    totalItems = len(data_iter)
    ridx = np.random.randint(0,totalItems)
    np.set_printoptions(precision=2)
    padimg = np.zeros((params['image_size'],5,3),dtype=np.uint8)
    padimg[:,:,:] = 128
    cimg_cnt = 0
    mean_hist = [[],[],[]]
    max_hist = [[],[],[]]
    lengths_hist = [[],[],[]]
    if len(params['n_iter']) == 0:
        params['n_iter'] = [0]*len(params['model'])

    curRandMaskIndex = None
    prevRandMaskIndex = None
    kp_colors = [solvers[i].get_kpcolors() for i in range(len(solvers))]
    #solvers[0].use_kpstrength = False
    #solvers[1].use_kpstrength = False
    if params['int_type'] == 'feat':
        interp_kp = False
        interp_feat = True
    elif params['int_type'] == 'kp':
        interp_kp = True
        interp_feat = False
    elif params['int_type'] == 'kpandfeat':
        interp_kp = True
        interp_feat = True

    while True:
        cimg_cnt+=1
        #import ipdb; ipdb.set_trace()
        idx = data_iter[c_idx]
        x, real_label, boxImg, boxlabel, mask, bbox, curCls, boxmask, affineParams = targ_split[data_iter[c_idx]]
        fp = [targ_split.getfilename(data_iter[c_idx])]
        boxImgRand, boxmaskRand, randMaskIndex, _, _ = dataset.getRandMaskByClass(configs[0]['selected_attrs'][curCls[0]], curRandMaskIndex) #targ_split[data_iter[(c_idx + ridx)%totalItems]]

        #if configs[0]['use_gtmask_inp']:
        #    mask = mask[1:,::]
        boxImgOrig  = solvers[0].to_var(boxImg[None,::][:,:3,::])
        boxImgTrans = solvers[0].to_var(boxImg[None,::][:,3:,::])
        boxMaskOrig  = solvers[0].to_var(boxmask[None,::][:,:1,::])
        boxMaskTrans = solvers[0].to_var(boxmask[None,::][:,1:,::])
        boxlabel= solvers[0].to_var(boxlabel[None,::])

        boxImgRand       = solvers[0].to_var(boxImgRand[None,::][:,:3,::])
        boxMaskRandOrig  = solvers[0].to_var(boxmaskRand[None,::][:,:1,::])

        mask_image_list = [[None]*(2+params['n_steps']) for i in range(len(solvers))]
        fake_image_list = [[None]*(2+params['n_steps']) for i in range(len(solvers))]
        kp_image_list = [[None]*(2+params['n_steps']) for i in range(len(solvers))]

        for i in range(len(solvers)):
            fake_image_list[i][0] = boxImgOrig
            fake_image_list[i][-1] = boxImgRand
            mask_image_list[i][0] = boxMaskOrig
            mask_image_list[i][-1] = boxMaskRandOrig
            kpOrig, mEncOrig, _, pEncOrig = forward_kpextract(solvers[i], boxImgOrig, boxlabel, not (interp_kp))
            pProjOrig = solvers[i].StyleMLP.get_projected_feat(pEncOrig, kpOrig)

            kp_type = solvers[i].encode_keypoints
            if interp_kp:
                if kp_type==1:
                    kpTempMean = kpOrig['code']['mean'].clone()
                    kpTempVar = kpOrig['code']['var'].clone()
                else:
                    kpTemp = kpOrig['heatmap'].clone()
                kpTempStr = kpOrig['str'].clone()
            if interp_feat:
                featTemp = pProjOrig.clone()

            kpRand, _ , _, pEncRand = forward_kpextract(solvers[i], boxImgRand, boxlabel, False)
            pProjRand = solvers[i].StyleMLP.get_projected_feat(pEncRand, kpRand)

            kp_image_list[i][0] = solvers[i].visualize_keypoints(kpOrig['img'], kp_colors[i])
            kp_image_list[i][-1] = solvers[i].visualize_keypoints(kpRand['img'], kp_colors[i])

            for n in range(params['n_steps']):
                rand_idx = curCls[0]
                clsname = configs[0]['selected_attrs'][rand_idx]
                print('chosing class %s'%clsname)
                #if i > 0:
                #    import ipdb; ipdb.set_trace()
                with torch.no_grad():
                    if interp_kp:
                        kp_alpha = (float(n)/(params['n_steps']-1))
                        if kp_type == 1:
                            kpOrig['code']['mean'] =  (1.-kp_alpha) * kpTempMean + kp_alpha * kpRand['code']['mean']
                            kpOrig['code']['var']  =  (1.-kp_alpha) * kpTempVar  + kp_alpha * kpRand['code']['var']
                            kpOrig['img'] = kp2gaussian(kpOrig['code'], (kpOrig['img'].shape[2], kpOrig['img'].shape[3]), kp_variance=solvers[i].kp_variance_type, dist_type=solvers[i].kp_dist_type)
                        else:
                            kpOrig['heatmap']      =  (1.-kp_alpha) * kpTemp + (kp_alpha) * kpRand['heatmap']
                            kpOrig['img'] = F.interpolate(gumbel_softmax_sample(kpOrig['heatmap'], hard=True), size=kpOrig['img'].shape[-1], mode='nearest')
                        kpOrig['str'] =  (1.-kp_alpha) * kpTempStr + (kp_alpha) * kpRand['str']
                        if solvers[0].use_kpstrength:
                            kpRand['img'] = kpRand['str'].unsqueeze(-1).unsqueeze(-1) * kpRand['img']
                        mEncOrig = solvers[i].G.forward_inpEnc(kpOrig['img'],  boxlabel)

                    if interp_feat:
                        feat_alpha = (float(n)/(params['n_steps']-1))
                        pProjOrig = (1.-feat_alpha) * featTemp + (feat_alpha) * pProjRand

                    predRand,_, kpRand, predMaskRand = solvers[i].forward_patchtransform(boxImgOrig, boxMaskRandOrig, boxlabel, targpatch=boxImgRand)

                    styleOrig2Rand = solvers[i].StyleMLP.map_feat_to_target(pProjOrig, kpOrig, pEncRand.shape[2:])
                    #import ipdb; ipdb.set_trace()
                    predRand, predMaskRand, _ = solvers[i].forward_featToImg(mEncOrig, styleOrig2Rand, boxlabel, targmask = None, targKp = kpOrig, mode='test')
                    predMaskRand = (predMaskRand > params['mask_threshold']).float()

                fake_image_list[i][1+n] = predRand
                mask_image_list[i][1+n] = predMaskRand

                if params['showkp'] and solvers[i].encode_keypoints:
                    kp_image_list[i][1+n] = solvers[i].visualize_keypoints(kpOrig['img'], kp_colors[i])
                #import ipdb; ipdb.set_trace()

        for i in range(len(solvers)):
            imgOne = make_image(fake_image_list[i], padimg)
            img =  imgOne if i==0 else np.vstack([img,imgOne])
            if params['showkp'] and solvers[i].encode_keypoints:
                imgKp = make_image(kp_image_list[i], padimg)
                img = np.vstack([img, imgKp])
            if params['showmask']==2:
                imgmask = make_image(mask_image_list[i], padimg)
                img = np.vstack([img, imgmask])
        if params['showmask']==1:
            imgmask = make_image(mask_image_list[-1], padimg)
            img = np.vstack([img, imgmask])

        #imgOut = make_image(out_image_list, padimg, equalize=True)
        #img = np.vstack([img, imgOut])
        #if params['compmodel']:
        #    imgcomp = make_image(fake_image_list_comp)
        #    img = np.vstack([img, imgcomp])
        #    if params['showdiff']:
        #        imgdiffcomp = make_image([fimg - fake_image_list_comp[0] for fimg in fake_image_list_comp])
        #        img = np.vstack([img, imgdiffcomp])
        cv2.imshow('frame',img if params['scaleDisp']==0 else cv2.resize(img,None, fx = params['scaleDisp'], fy=params['scaleDisp']))#, interpolation=cv2.INTER_CUBIC))
        keyInp = cv2.waitKey(0)

        if keyInp & 0xFF == ord('q'):
            break
        elif keyInp & 0xFF == ord('b'):
            #print keyInp & 0xFF
            c_idx = c_idx-1
            curRandMaskIndex = None
            prevRandMaskIndex = None
        elif (keyInp & 0xFF == ord('s')):
            #sample_dir = join(params['sample_dump_dir'], basename(params['model'][0]).split('.')[0])
            sample_dir = join(params['sample_dump_dir'],'_'.join([params['split']]+params['names']))
            if not exists(sample_dir):
                makedirs(sample_dir)
            fnames = [''.join(clsname.split()) +'_'+'%s.png' % splitext(basename(f))[0] for f in fp]
            fpaths = [join(sample_dir, f) for f in fnames]
            imgSaveName = fpaths[0]
            if params['savesepimages']:
                saveIndividImages(fake_image_list, mask_image_list, nameList, sample_dir, fp, configs[0]['selected_attrs'][rand_idx])
            else:
                print('Saving into file: ' + imgSaveName)
                cv2.imwrite(imgSaveName, img)
            c_idx += 1
            prevRandMaskIndex = randMaskIndex
            curRandMaskIndex = None
        else:
            c_idx += 1
            prevRandMaskIndex = randMaskIndex
            curRandMaskIndex = None

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--showdiff', type=int, default=0)
  parser.add_argument('--showperceptionloss', type=int, default=0)
  parser.add_argument('--showdeform', type=int, default=0)
  parser.add_argument('--showmask', type=int, default=2)
  parser.add_argument('--showkp', type=int, default=1)
  #parser.add_argument('--showclassifier', type=int, default=0)
  parser.add_argument('--showreconst', type=int, default=0)
  parser.add_argument('-d', '--dataset', dest='dataset',  type=str, default='coco', help='dataset: celeb')
  parser.add_argument('-m', '--model', type=str, default=[], nargs='+', help='checkpoint to resume training from')
  parser.add_argument('-n', '--names', type=str, default=[], nargs='+', help='checkpoint to resume training from')
  parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=1, help='max batch size')
  parser.add_argument('--sample_dump_dir', type=str, default='gen_samples', help='print every x iters')
  parser.add_argument('--swap_attr', type=str, default='rand', help='which attribute to swap')
  parser.add_argument('--split', type=str, default='val', help='which attribute to swap')
  parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')

  parser.add_argument('--sort_by', type=str, default=[], nargs='+', help='Evaluation scores to visualize')
  parser.add_argument('--sort_score', type=str, default='iou', help='Evaluation scores to visualize')
  parser.add_argument('--show_also', type=str, nargs = '+', default=[], help='Evaluation scores to visualize')
  parser.add_argument('--use_same_g', type=str, default=[], nargs='+', help='Evaluation scores to visualize')

  # Deformations applied to mnist images;
  parser.add_argument('--no_maskgen', type=int, default=0)
  parser.add_argument('--randomrotate', type=int, default=90)
  parser.add_argument('--randomscale', type=float, nargs='+', default=[0.5,0.5])
  parser.add_argument('--image_size', type=int, default=128)
  parser.add_argument('--scaleDisp', type=int, default=0)
  parser.add_argument('--box_size', type=int, default=64)
  parser.add_argument('--mask_threshold', type=float, default=0.5)
  parser.add_argument('--withExtMask', type=int, nargs ='+', default=[])
  parser.add_argument('--extmask_type', type=str, default='mask')
  parser.add_argument('--n_iter', type=int, nargs ='+', default=[])
  parser.add_argument('--mask_size', type=int, default=128)
  parser.add_argument('--dilateMask', type=int, default=[], nargs='+')
  parser.add_argument('--datafile', type=str, default='datasetCompleteYoloSplit.json')
  parser.add_argument('--extMask_source', type=str, default='gt')
  parser.add_argument('--n_boxes', type=int, default=4)
  parser.add_argument('--show_ids', type=int, default=[], nargs='+')
  parser.add_argument('--nosoftmax', type=int, default=0)
  parser.add_argument('--n_steps', type=int, default=5)
  parser.add_argument('--int_type', type=str, default='feat')

  parser.add_argument('--savesepimages', type=int, default=0)
  parser.add_argument('--filter_by_mincooccur', type=float, default=-1.)
  parser.add_argument('--only_indiv_occur', type=float, default=0)
  parser.add_argument('--use_highres_patch', type=int, default=1)

  parser.add_argument('--max_object_size', type=float, default=1.0)
  parser.add_argument('--min_box_size', type=float, default=80)
  parser.add_argument('--remove_invalid', type=float, default=1)
  parser.add_argument('--show_only', type=str, default=None, nargs='+')


  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  params['cuda'] = not args.no_cuda
  print(json.dumps(params, indent = 2))
  gen_samples(params)


