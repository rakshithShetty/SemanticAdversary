import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as FN
from torchvision.datasets import ImageFolder, MNIST
from PIL import Image
import json
import numpy as np
import random
from .utils import computeIOU, computeContainment, computeUnionArea
from pycocotools.coco import COCO as COCOTool
from collections import defaultdict
from random import shuffle
from copy import copy
import matplotlib
from matplotlib import cm
from albumentations import RGBShift, RandomBrightnessContrast, HueSaturationValue, RandomGamma, Compose
import math
import cv2

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

class CocoDatasetBBoxSample(Dataset):
    def __init__(self, transform, mode, select_attrs=[], datafile='datasetBoxAnn.json', out_img_size=128, bbox_out_size=64,
                 balance_classes=0, onlyrandBoxes=False, max_object_size=0., min_box_size = 16, max_with_union=True, use_gt_mask=False,
                 boxrotate=0, n_fg_boxes = 1, n_extra_boxes=0, square_resize=0, filter_by_mincooccur = -1., only_indiv_occur = 0., patch_mode=0,
                 use_highres_patch=0, get_all_inst=False, remove_invalid = 0, color_or_affine='affine', get_kp_class=0, get_rand_mask=0,
                 detectorMode=0, canvas_image_size=416, filter_class_instances = [], patch_with_bg=False, only_selIds=[]):
        self.image_path = os.path.join('data','coco','images')
        self.transform = transform
        self.mode = mode
        self.n_fg_boxes = n_fg_boxes
        self.n_extra_boxes = n_extra_boxes
        self.iouThresh = 1.
        self.dataset = json.load(open(os.path.join('data','coco',datafile),'r'))
        if len(only_selIds):
            only_selIds = set(only_selIds)
            self.dataset['images'] = [img for img in self.dataset['images'] if img['cocoid'] in only_selIds]
        self.num_data = len(self.dataset['images'])
        self.attr2idx = {}
        self.idx2attr = {}
        self.catid2attr = {}
        self.out_img_size = out_img_size
        self.square_resize = square_resize
        self.bbox_out_size = bbox_out_size
        self.filter_by_mincooccur = filter_by_mincooccur
        self.only_indiv_occur = only_indiv_occur
        #self.selected_attrs = ['person', 'book', 'car', 'bird', 'chair'] if select_attrs== [] else select_attrs
        self.selected_attrs = select_attrs
        self.get_kp_class = get_kp_class
        self.get_rand_mask = get_rand_mask
        self.patch_with_bg = patch_with_bg
        self.detectorMode = detectorMode
        self.canvas_image_size = canvas_image_size

        self.apply_random_affine = True if color_or_affine=='affine' or color_or_affine=='both' else False
        self.apply_randcolor_transform = True if color_or_affine=='color' or color_or_affine=='both' else False
        if self.apply_randcolor_transform:
            #self.color_augment = Compose([RGBShift(r_shift_limit=30, g_shift_limit=20, b_shift_limit=30), RandomBrightnessContrast(),HueSaturationValue(hue_shift_limit=0, sat_shift_limit=10, val_shift_limit=40), RandomGamma()],p=0.99, additional_targets={'patch':'image'})
            self.color_augment = Compose([RGBShift(r_shift_limit=150, g_shift_limit=150, b_shift_limit=150), RandomBrightnessContrast(),HueSaturationValue(hue_shift_limit=50, sat_shift_limit=50, val_shift_limit=40), RandomGamma()],p=1.0)
        if len(select_attrs) == 0 or (select_attrs[0] == 'all'):
            self.selected_attrs = [attr['name'] for attr in self.dataset['categories']]
        self.filter_class_instances = set(filter_class_instances)
        print('Removing instances of classes ', filter_class_instances)
        self.balance_classes = balance_classes
        self.onlyrandBoxes = onlyrandBoxes
        self.max_object_size = max_object_size
        self.max_with_union= max_with_union
        self.min_box_size = min_box_size
        #self.min_minbox_size = 10
        self.use_gt_mask = use_gt_mask
        self.boxrotate = boxrotate
        self.patch_mode = patch_mode
        self.use_highres_patch = use_highres_patch
        self.remove_invalid= remove_invalid
        if self.boxrotate:
            self.rotateTrans = transforms.Compose([transforms.RandomRotation(boxrotate,resample=Image.NEAREST)])
        if use_gt_mask == 1:
            self.mask_trans =transforms.Compose([transforms.Resize(out_img_size if not square_resize else [out_img_size, out_img_size] , interpolation=Image.NEAREST), transforms.CenterCrop(out_img_size)])
            self.mask_provider = CocoMaskDataset(self.mask_trans if not self.use_highres_patch else None, mode, select_attrs=self.selected_attrs, balance_classes=balance_classes)

        self.getBoxMap = 1

        if self.getBoxMap:
            self.classTocolor = cm.get_cmap('Spectral')
            self.cNorm = matplotlib.colors.Normalize(vmin=0., vmax=len(self.selected_attrs))
        self.randHFlip = 'Flip' in transform

        print('Start preprocessing dataset..!')
        self.preprocess()
        print('Finished preprocessing dataset..!')
        self.imgId2idx = {imid:i for i,imid in enumerate(self.valid_ids)}

        self.get_all_inst = get_all_inst
        self.num_data = len(self.dataset['images']) if not get_all_inst else len(self.all_imgIdandAnn)

    #COCO Dataset
    def preprocess(self):
        for i, attr in enumerate(self.dataset['categories']):
            self.attr2idx[attr['name']] = i
            self.idx2attr[i] = attr['name']
            self.catid2attr[attr['id']] = attr['name']

        self.sattr_to_idx = {att:i for i, att in enumerate(self.selected_attrs)}

        # First remove unwanted splits:
        if self.mode == 'train':
            self.dataset['images'] = [img for img in self.dataset['images'] if img['split'] == self.mode]
        else:
            self.dataset['images'] = [img for img in self.dataset['images'] if img['split'] == self.mode]

        #self.dataset['images'] = [img for img in self.dataset['images'] if img['split'] == self.mode]
        if self.max_object_size > 0.:
            validImgs = []
            for img in self.dataset['images']:
                if not self.max_with_union:
                    maxSize = max([bb['bbox'][2]*bb['bbox'][3] for bb in img['bboxAnn']])
                else:
                    boxByCls = defaultdict(list)
                    for bb in img['bboxAnn']:
                        boxByCls[bb['cid']].append(bb['bbox'])
                    unionAreas = [computeUnionArea(boxes) for cid,boxes in boxByCls.items()]
                    maxSize = max(unionAreas) if len(unionAreas) else 0.
                if maxSize < self.max_object_size:
                    validImgs.append(img)
            print(' %d of %d images left after size filtering'%(len(validImgs), len(self.dataset['images'])))
            self.dataset['images'] = validImgs

        self.valid_ids = [img['cocoid'] for img in self.dataset['images']]
        self.catsInImg = {}

        selset = set(self.selected_attrs) - self.filter_class_instances
        for i, img in enumerate(self.dataset['images']):
            self.dataset['images'][i]['label'] = np.zeros(max(len(self.selected_attrs),1))
            self.dataset['images'][i]['bboxAnn'] = [bb for bb in img['bboxAnn'] if self.catid2attr[bb['cid']] in selset]

            # Correct BBox for Resize(of smaller edge) and CenterCrop
            fixedbbox = []
            imgSize = self.dataset['images'][i]['imgSize']
            maxSide = np.argmax(imgSize)
            for j in xrange(len(self.dataset['images'][i]['bboxAnn'])):
                cbbox = self.dataset['images'][i]['bboxAnn'][j]
                cbbox['cls'] = self.catid2attr[cbbox['cid']]
                if not self.use_highres_patch:
                    # If sampling high resolution patches, no need to adjust for resize and crop
                    # The box and the patch will be taken from the the original image
                    maxSideLen = int(float(self.out_img_size * imgSize[maxSide]) / (imgSize[1-maxSide])) if not self.square_resize else self.out_img_size
                    assert(maxSideLen >= self.out_img_size)
                    newStartCord = round((maxSideLen - self.out_img_size)/2.)
                    boxStart = min( max(cbbox['bbox'][maxSide]*maxSideLen - newStartCord, 0),  self.out_img_size)
                    boxEnd =  min(max((cbbox['bbox'][maxSide]+cbbox['bbox'][maxSide+2])*maxSideLen - newStartCord, 0), self.out_img_size)
                    length = boxEnd - boxStart
                    if length >= self.min_box_size and (not cbbox['iscrowd']) and ((not self.remove_invalid) or (('valid' in cbbox) and cbbox['valid'])):
                        cbbox['bbox'][maxSide] = float(boxStart)/self.out_img_size
                        cbbox['bbox'][maxSide+2] = float(length)/self.out_img_size
                        #if cbbox['bbox'][1-maxSide+2] >= 1./self.out_img_size:
                        if cbbox['bbox'][1-maxSide+2] >= (float(self.min_box_size)/self.out_img_size):
                            fixedbbox.append(cbbox)
                            if cbbox['bbox'][0]<0. or cbbox['bbox'][1] < 0. or cbbox['bbox'][0]>1.0 or cbbox['bbox'][1]> 1.0:
                                import ipdb; ipdb.set_trace()

                else:
                    # Largest side of the box will be resized to image_size
                    box_size = (cbbox['bbox'][2] * imgSize[0], cbbox['bbox'][3]*imgSize[1])
                    sc_ratio = min(1.0, self.out_img_size/(float(box_size[0]+1e-8)), self.out_img_size / float(box_size[1]+1e-8))
                    new_box_size = (int(box_size[0]*sc_ratio) ,  int(box_size[1]*sc_ratio))
                    if (max(new_box_size) >= self.min_box_size) and (min(new_box_size) >= 3) and (not cbbox['iscrowd']) and ((not self.remove_invalid) or (('valid' in cbbox) and cbbox['valid'])):
                        fixedbbox.append(cbbox)

            self.dataset['images'][i]['bboxAnn'] = fixedbbox
            self.dataset['images'][i]['label'][[self.sattr_to_idx[self.catid2attr[bb['cid']]] for bb in img['bboxAnn']]] = 1.

            # Convert bbox data to numpy arrays
            #for j, bb in enumerate(self.dataset['images'][i]['bboxAnn']):
            #    self.dataset['images'][i]['bboxAnn'][j]['bbox'] = np.array(bb['bbox'])
            # Create bbox labels.
            #if self.patch_mode:
            #    lab_in_img = img['label'].nonzero()[0]
            #    self.dataset['images'][i]['label_seq'] = lab_in_img
            #    n_lab_in_img = len(lab_in_img)
            #    #self.dataset['images'][i]['cls_affect']  = np.zeros((n_lab_in_img,n_lab_in_img))
            #    idx2aidx = {l:li for li,l in enumerate(lab_in_img)}
            #    boxByCls = defaultdict(list)
            #    for bb in img['bboxAnn']:
            #        boxByCls[idx2aidx[self.sattr_to_idx[self.catid2attr[bb['cid']]]]].append(bb['bbox'])
            #    self.dataset['images'][i]['cls_affect'] = np.array([[min([max([computeContainment(bb1, bb2)[0] for bb1 in boxByCls[li1]]) for bb2 in boxByCls[li2]]) for li2 in xrange(n_lab_in_img)] for li1 in xrange(n_lab_in_img)])

            for j, bb in enumerate(self.dataset['images'][i]['bboxAnn']):
                #Check for IOU > 0.5 with other bbox
                #iouAr = [computeContainment(bb['bbox'], bother['bbox'])[0] for bother in self.dataset['images'][i]['bboxAnn']]
                self.dataset['images'][i]['bboxAnn'][j]['box_label'] = np.zeros(len(self.selected_attrs))
                #self.dataset['images'][i]['bboxAnn'][j]['box_label'][[self.sattr_to_idx[self.catid2attr[self.dataset['images'][i]['bboxAnn'][ii]['cid']]] for ii,iv in enumerate(iouAr) if iv>self.iouThresh]] = 1.
                self.dataset['images'][i]['bboxAnn'][j]['box_label'][self.sattr_to_idx[self.catid2attr[bb['cid']]]] = 1.

        if self.filter_by_mincooccur >= 0. or self.only_indiv_occur:
            clsToSingleOccur = defaultdict(list)
            clsCounts = np.zeros(len(self.selected_attrs))
            clsIndivCounts = np.zeros(len(self.selected_attrs))
            for i, img in enumerate(self.dataset['images']):
                imgCls = set()
                for bb in img['bboxAnn']:
                    imgCls.add(self.catid2attr[bb['cid']])
                imgCls = list(imgCls)
                if len(imgCls)==1:
                    clsIndivCounts[self.sattr_to_idx[imgCls[0]]] += 1.
                    clsToSingleOccur[imgCls[0]].append(i)
                else:
                    clsCounts[[self.sattr_to_idx[cls] for cls in imgCls]] += 1.

            if self.filter_by_mincooccur >= 0.:
                n_rem_counts = clsIndivCounts - self.filter_by_mincooccur/(1-self.filter_by_mincooccur) * clsCounts
                allRemIds = set()
                for cls in self.selected_attrs:
                    if n_rem_counts[self.sattr_to_idx[cls]] > 0:
                        n_rem_idx = np.arange(len(clsToSingleOccur[cls]))
                        np.random.shuffle(n_rem_idx)
                        n_rem_idx = n_rem_idx[:int(n_rem_counts[self.sattr_to_idx[cls]])]
                        allRemIds.update([clsToSingleOccur[cls][ri] for ri in n_rem_idx])

                self.dataset['images'] = [img for i,img in enumerate(self.dataset['images']) if i not in allRemIds]
            elif self.only_indiv_occur:
                allKeepIds = set()
                for cls in self.selected_attrs:
                    allKeepIds.update(clsToSingleOccur[cls])

                self.dataset['images'] = [img for i,img in enumerate(self.dataset['images']) if i in allKeepIds]

            self.valid_ids = [img['cocoid'] for img in self.dataset['images']]
            print(' %d images left after co_occurence filtering'%(len(self.valid_ids)))

        if (self.mode == 'train') or self.patch_mode:
            self.dataset['images'] = [img for i,img in enumerate(self.dataset['images']) if len(img['bboxAnn'])>0]
            self.valid_ids = [img['cocoid'] for img in self.dataset['images']]
            print(' %d images left after empty img filtering'%(len(self.valid_ids)))

        self.attToImgId = defaultdict(set)
        self.all_imgIdandAnn = []
        self.imid2kpClusts = {}
        self.imidToAnnList= {}
        for i, img in enumerate(self.dataset['images']):
            classesInImg = [self.catid2attr[bb['cid']] for bb in img['bboxAnn'] if self.catid2attr[bb['cid']] in selset]
            annsInImg = [(self.catid2attr[bb['cid']], bb['id']) for bb in img['bboxAnn'] if self.catid2attr[bb['cid']] in selset]
            self.imidToAnnList[img['cocoid']]  = annsInImg
            self.all_imgIdandAnn.extend([(i,a[0],a[1]) for a in annsInImg])
            if len(classesInImg):
                self.catsInImg[i] = classesInImg
                for att in classesInImg:
                    self.attToImgId[att].add(i)
            else:
                self.attToImgId['bg'].add(i)
                self.catsInImg[i] = ['bg']

            if self.get_kp_class:
                self.imid2kpClusts.update({(img['cocoid'],ann['id']):[ann['cid'], ann['kp_class'] if ann['kp_class'] else -1, ann['feat_kp_class'] if ann['feat_kp_class'] else -1] for ann in img['bboxAnn']  if self.catid2attr[bb['cid']] in selset})
        self.attToImgId = {k:list(v) for k,v in self.attToImgId.items()}

    #COCO Dataset
    def randomBBoxSample(self, index, max_area = -1, no_random_box=0., prevBoxList = []):
        # With 50% chance sample from background or foreground
        # Minimum size
        minLen = 0.1
        maxLen = 0.5
        maxIou = 0.3
        cbboxList = self.dataset['images'][index]['bboxAnn'] if not self.onlyrandBoxes else []
        n_t = 0
        while 1:
            if len(cbboxList) and (random.random()<(0.5+no_random_box)):
                cbid = random.randrange(len(cbboxList))
                sbox = self.dataset['images'][index]['bboxAnn'][cbid]
                return copy(sbox['bbox']),sbox['box_label'], cbid
            else:
                # sample a random background box
                cbid = None
                tL_x, tL_y = random.uniform(0,1.-minLen-0.01), random.uniform(0,1.-minLen-0.01)
                l_x = random.uniform(minLen, min(1.-tL_x,maxLen))
                l_y = random.uniform(minLen, min(1.-tL_y,maxLen))
                sbox = [tL_x, tL_y, l_x, l_y]
                # Prepare label for this box
                bboxLabel = np.zeros(max(len(self.selected_attrs),1))
                # Test for overlap with foreground objects
                noOverlap = True
                #if len(cbboxList):
                for bb in cbboxList+prevBoxList:
                    iou, aInb, bIna = computeIOU(sbox, bb['bbox'])
                    if iou > maxIou or aInb >0.8:
                        noOverlap = False
                    if bIna > 0.8 and bb['cid'] > 0:
                        bboxLabel[self.sattr_to_idx[self.catid2attr[bb['cid']]]] = 1
                if (noOverlap and ((max_area < 0) or ((sbox[2]*sbox[3])< max_area))) or (n_t>10):
                    return sbox, bboxLabel, cbid
            n_t += 1

    #COCO Dataset
    def getBboxForClass(self, index, cid, annId=None):
        if annId is None:
            cbboxList = [(bb,i) for i,bb in enumerate(self.dataset['images'][index]['bboxAnn']) if bb['cls'] ==cid]
        else:
            cbboxList = [(bb,i) for i,bb in enumerate(self.dataset['images'][index]['bboxAnn']) if ((bb['cls'] ==cid) and (bb['id'] == annId))]
        if len(cbboxList):
            cbid = random.randrange(len(cbboxList))
            sbox = cbboxList[cbid][0]
            return copy(sbox['bbox']),sbox['box_label'], sbox, cbboxList[cbid][1]
        else:
            print(index, self.dataset['images'][index]['bboxAnn'], cid)
            assert(0)


    #COCO Dataset
    def __getitem__(self, index):
        # In this situation ignore index and sample classes uniformly
        if self.detectorMode:
            returnvals = self.getCanvasImg(index=index, image_size=self.canvas_image_size)
        else:
            if self.get_all_inst:
                index, currCls, annId = self.all_imgIdandAnn[index]
            else:
                if self.balance_classes==1:
                    currCls = random.choice(list(self.attToImgId.keys()))
                    index = random.choice(self.attToImgId[currCls])
                elif self.balance_classes==2:
                    currCls = random.choice(self.catsInImg[index])
                else:
                    currCls = random.choice(self.catsInImg[index])

                annId= None
            cid = [self.sattr_to_idx[currCls]] if currCls != 'bg' else [0]

            if self.patch_mode:
                returnvals = self.getPatchAndCanvasbyIndexAndclass(index,cid, annId=annId)
            else:
                returnvals = self.getbyIndexAndclass(index, cid)

        return tuple(returnvals)

    #COCO Dataset
    def getbyIdAndclass(self, imgid, cls, hflip=0):
        index = self.imgId2idx[imgid]
        cid = [self.sattr_to_idx[cls]] if cls != 'bg' else [0]
        returnvals = self.getbyIndexAndclass(index, cid)
        return tuple(returnvals)

    #COCO Dataset
    def getbyClass(self, cls):
        cid = [self.sattr_to_idx[cls]] if cls != 'bg' else [0]
        index = random.choice(self.attToImgId[cls])
        returnvals = self.getPatchAndCanvasbyIndexAndclass(index, cid)
        return tuple(returnvals)

    #COCO Dataset
    def getbyIdAndclassAndAnn(self, imgid, cls, annId):
        index = self.imgId2idx[imgid]
        cid = [self.sattr_to_idx[cls]] if cls != 'bg' else [0]
        returnvals = self.getPatchAndCanvasbyIndexAndclass(index, cid, annId)
        return tuple(returnvals)

    #COCO Dataset
    def getbyIdAndclassAndAnnLightBatch(self, imgAndAnn, cls):
        boxAll = []; labAll = []; boxMaskAll = []
        for imA,c in zip(imgAndAnn,cls):
            boxImg, boxLabel, boxMask = self.getPatchAndCanvasbyIndexAndclassLight(self.imgId2idx[imA[0]], [self.sattr_to_idx[c]] if c !='bg' else [0], imA[1])
            boxAll.append(boxImg.unsqueeze(0))
            labAll.append(boxLabel.unsqueeze(0))
            boxMaskAll.append(boxMask.unsqueeze(0))
        return torch.cat(boxAll,dim=0), torch.cat(labAll,dim=0), torch.cat(boxMaskAll, dim=0)

    #COCO Dataset
    def getPatchAndCanvasbyIndexAndclassLight(self, index, cid, annId=None):
        image = Image.open(os.path.join(self.image_path,self.dataset['images'][index]['filepath'], self.dataset['images'][index]['filename']))
        currCls = self.selected_attrs[cid[0]]
        if image.mode != 'RGB':
            # print image.mode
            image = image.convert('RGB')
        c_imgsize = image.size

        sampbbox, bboxLabel, sbox, cbid = self.getBboxForClass(index, currCls,annId=annId)
        annId = sbox['id']
        cocoid = self.dataset['images'][index]['cocoid']
        bboxId = cid

        extra_boxes = []
        if self.n_extra_boxes:
            # Sample random number of boxes between 1 and n_boxes
            c_nbox = np.random.randint(0,self.n_extra_boxes+1)
            c_area = sampbbox[2]*sampbbox[3]
            for i in xrange(c_nbox):
                # Also stop at total area > 50%
                if c_area < 0.5:
                    bsamp, _, _ = self.randomBBoxSample(index, 0.6-c_area) # Extra 10% to make the sampling easier
                    extra_boxes.append(bsamp)
                    c_area += bsamp[2]*bsamp[3]
                else:
                    break

        label = self.dataset['images'][index]['label']

        # Apply transforms to the image.
        if not self.use_highres_patch:
            image = self.transform[0](image)
            src_img_size = (self.out_img_size, self.out_img_size)
            out_img_size = (self.out_img_size, self.out_img_size)
        else:
            src_img_size = self.dataset['images'][index]['imgSize']
            out_img_size = (self.out_img_size, self.out_img_size)

        # Now do the flipping
        hflip = 0
        if self.randHFlip and random.random()>0.5:
            hflip = 1
            image = FN.hflip(image)
            sampbbox[0] = 1.0-(sampbbox[0]+sampbbox[2])

        if self.use_gt_mask==1:
            # Use GT masks as input
            gtMask = self.mask_provider.getbyImgAnnId(cocoid, sbox['id'] , hflip=hflip)

        #Convert BBox to actual co-ordinates
        sampbbox = [int(bc*src_img_size[i%2]) for i,bc in enumerate(sampbbox)]
        # Prepare the patch, no Resizing now
        boxCrop = np.zeros((out_img_size[1], out_img_size[0],3))
        boxMask = torch.zeros(1, out_img_size[1], out_img_size[0])
        if self.patch_with_bg:
            if self.patch_with_bg ==1:
                maxlen = max(max(sampbbox[2:]),self.out_img_size)
                extract_box = [max(sampbbox[0]-(maxlen-sampbbox[2])//2,0),
                               max(sampbbox[1]-(maxlen-sampbbox[3])//2,0),
                               maxlen, maxlen]
            else:
                extract_box = sampbbox

            src_patch = np.array(image)[extract_box[1]:extract_box[1]+extract_box[3], extract_box[0]:extract_box[0]+extract_box[2],:]
        else:
            extract_box = sampbbox
            src_patch = (image*(gtMask.byte().numpy().transpose(1,2,0)))[extract_box[1]:extract_box[1]+extract_box[3], extract_box[0]:extract_box[0]+extract_box[2],:]
        src_mask = gtMask[0,extract_box[1]:extract_box[1]+extract_box[3], extract_box[0]:extract_box[0]+extract_box[2]]

        if self.use_highres_patch:
            scale_ratio = min(1.0, float(out_img_size[0])/float(extract_box[2]), float(out_img_size[1])/extract_box[3])
            onlyObjtarg_boxsize = [int(sampbbox[2]*scale_ratio), int(sampbbox[3]*scale_ratio)]
            targ_boxsize = [int(extract_box[2]*scale_ratio), int(extract_box[3]*scale_ratio)]
            targ_boxloc = [out_img_size[0]//2 - targ_boxsize[0]//2, # Center the patch in the destination
                           out_img_size[1]//2 - targ_boxsize[1]//2]
            src_patch = np.array(FN.resize(Image.fromarray(src_patch), (targ_boxsize[1],targ_boxsize[0])))
            src_mask = torch.tensor(np.array(FN.resize(Image.fromarray(src_mask.numpy()), (targ_boxsize[1],targ_boxsize[0]), interpolation=Image.NEAREST)))
        else:
            # co-ordinates of the patch (x_lc, y_lc)
            scale_ratio = 1
            onlyObjtarg_boxsize = [int(sampbbox[2]*scale_ratio), int(sampbbox[3]*scale_ratio)]
            targ_boxsize = [extract_box[2]*scale_ratio, extract_box[3]*scale_ratio]
            targ_boxloc = [out_img_size[0]//2 - targ_boxsize[0]//2, # Center the patch in the destination
                           out_img_size[1]//2 - targ_boxsize[1]//2]

        boxCrop[targ_boxloc[1]: targ_boxloc[1]+targ_boxsize[1], targ_boxloc[0]: targ_boxloc[0]+targ_boxsize[0],:] = src_patch
        boxMask[0,targ_boxloc[1]: targ_boxloc[1]+targ_boxsize[1], targ_boxloc[0]: targ_boxloc[0]+targ_boxsize[0]] = src_mask
        boxCrop = Image.fromarray(boxCrop.astype(np.uint8))


        #patch = FN.resized_crop(image, sampbbox[1], sampbbox[0], sampbbox[3],sampbbox[2], (self.bbox_out_size, self.bbox_out_size))
        # Now resize to bboxImage and mask to the right sizes

        # Create Mask
        #patch = FN.resized_crop(image, sampbbox[1], sampbbox[0], sampbbox[3],sampbbox[2], (self.bbox_out_size, self.bbox_out_size))

        if self.patch_with_bg:
            final_boxImg = self.transform[-1](boxCrop)
        else:
            final_boxImg = boxMask*self.transform[-1](boxCrop)
        final_boxMask = boxMask

        return final_boxImg, \
               torch.FloatTensor(bboxLabel), \
               boxMask

    #COCO Dataset
    def getCanvasAndRandBoxByClass(self, cls, index=None, get_img=False, image_size=128, nbox =1):
        cid = [self.sattr_to_idx[cls]] if cls != 'bg' else [0]
        index = random.choice(self.attToImgId[cls]) if index == None else index

        image = Image.open(os.path.join(self.image_path,self.dataset['images'][index]['filepath'], self.dataset['images'][index]['filename']))
        if image.mode != 'RGB':
            #print image.mode
            image = image.convert('RGB')

        src_img_size = self.dataset['images'][index]['imgSize']
        out_image_size = (int(src_img_size[0]*image_size / max(src_img_size)), int(src_img_size[1]*image_size / max(src_img_size)))
        image = FN.resize(image, (out_image_size[1], out_image_size[0]))

        allSampBox = []
        allSampBoxDict = []
        for i in range(nbox):
            sampbbox, bboxLabel, cbid = self.randomBBoxSample(index, 0.5, no_random_box=-1., prevBoxList=allSampBoxDict)
            allSampBox.append(sampbbox)
            allSampBoxDict.append({'bbox':sampbbox, 'cid': -1})
        return self.transform[-1](image), torch.FloatTensor(allSampBox)

    #COCO Dataset
    def getCanvasImg(self, cls=None, index=None, image_size=128):
        if cls is None and index is None:
            assert(0)
        elif (cls is None) and (index is not None):
            cls = random.choice(self.catsInImg[index])

        index = random.choice(self.attToImgId[cls]) if index == None else index

        image = Image.open(os.path.join(self.image_path,self.dataset['images'][index]['filepath'], self.dataset['images'][index]['filename']))
        if image.mode != 'RGB':
            #print image.mode
            image = image.convert('RGB')

        src_img_size = self.dataset['images'][index]['imgSize']
        out_img_size = (self.out_img_size, self.out_img_size)
        cocoid = self.dataset['images'][index]['cocoid']

        sampbboxOrig, bboxLabel, sbox, _ = self.getBboxForClass(index, cls)
        gtMask = self.mask_provider.getbyImgAnnId(cocoid, sbox['id'] , hflip=0)
        annId = sbox['id']

        sampbbox = [int(bc*src_img_size[i%2]) for i,bc in enumerate(sampbboxOrig)]
        all_boxes = [sampbboxOrig]
        if self.get_kp_class:
            all_boxes[-1] += (self.imid2kpClusts[cocoid, annId])
        all_boxes[-1] += [annId]
        all_boxesPix = [sampbbox]
        all_bboxlabel = [bboxLabel]
        all_bboxData =[sbox]
        all_boxGtMask = [gtMask]
        n_boxes = self.n_extra_boxes + self.n_fg_boxes
        if n_boxes > 1:
            sampAnnList = set([sbox['id']])
            e_boxesPix = []
            e_boxes = []
            for i in range(n_boxes -1):
                remAnnList = [i for i in range(len(self.imidToAnnList[cocoid])) if self.imidToAnnList[cocoid][i][1] not in sampAnnList]
                if len(remAnnList):
                    ei = random.choice(remAnnList)
                    clsExtra = self.imidToAnnList[cocoid][ei][0]
                    aidExtra = self.imidToAnnList[cocoid][ei][1]
                    ebox, ebLabel, eboxData, _ = self.getBboxForClass(index, clsExtra, annId=aidExtra)
                    sampAnnList.add(eboxData['id'])
                    all_bboxData.append(eboxData)
                else:
                    ebox, ebLabel, eboxData = self.randomBBoxSample(index, max_area=0.25, no_random_box=-1., prevBoxList=all_bboxData) # Extra 10% to make the sampling easier
                    all_bboxData.append({'bbox':ebox, 'cid':-1})

                if i < self.n_fg_boxes -1:
                    egtMask = self.mask_provider.getbyImgAnnId(cocoid, eboxData['id'] , hflip=0) if eboxData is not None else torch.zeros_like(gtMask)
                    all_boxGtMask.append(egtMask)

                all_boxes.append(ebox)
                if self.get_kp_class:
                    all_boxes[-1] += (self.imid2kpClusts[cocoid, eboxData['id']] if eboxData is not None else [-1,-1,-1])
                all_boxes[-1] += ([eboxData['id']] if eboxData is not None else [-5])
                all_bboxlabel.append(ebLabel)
                all_boxesPix.append([int(bc*src_img_size[i%2]) for i,bc in enumerate(ebox)])

        e_boxes = all_boxes[self.n_fg_boxes:]
        fg_boxes = all_boxes[:self.n_fg_boxes]

        allBoxImgs = []
        allBoxMask = []
        for i,fgb in enumerate(fg_boxes):
            sampbbox = all_boxesPix[i]
            gtMask = all_boxGtMask[i]
            boxCrop = np.zeros((out_img_size[1], out_img_size[0],3))
            boxMask = torch.zeros(1, out_img_size[1], out_img_size[0])
            src_patch = (image*(gtMask.byte().numpy().transpose(1,2,0)))[sampbbox[1]:sampbbox[1]+sampbbox[3], sampbbox[0]:sampbbox[0]+sampbbox[2],:]
            src_mask = gtMask[0,sampbbox[1]:sampbbox[1]+sampbbox[3], sampbbox[0]:sampbbox[0]+sampbbox[2]]

            scale_ratio = min(1.0, float(out_img_size[0])/float(sampbbox[2]), float(out_img_size[1])/sampbbox[3])
            targ_boxsize = [int(sampbbox[2]*scale_ratio), int(sampbbox[3]*scale_ratio)]
            targ_boxloc = [out_img_size[0]//2 - targ_boxsize[0]//2, # Center the patch in the destination
                           out_img_size[1]//2 - targ_boxsize[1]//2]
            src_patch = np.array(FN.resize(Image.fromarray(src_patch), (targ_boxsize[1],targ_boxsize[0])))
            src_mask = torch.tensor(np.array(FN.resize(Image.fromarray(src_mask.numpy()), (targ_boxsize[1],targ_boxsize[0]), interpolation=Image.NEAREST)))

            boxCrop[targ_boxloc[1]: targ_boxloc[1]+targ_boxsize[1], targ_boxloc[0]: targ_boxloc[0]+targ_boxsize[0],:] = src_patch
            boxMask[0,targ_boxloc[1]: targ_boxloc[1]+targ_boxsize[1], targ_boxloc[0]: targ_boxloc[0]+targ_boxsize[0]] = src_mask
            boxCrop = Image.fromarray(boxCrop.astype(np.uint8))
            allBoxImgs.append(boxCrop)
            allBoxMask.append(boxMask)

        boxImgOut = torch.cat([(allBoxMask[i]*self.transform[-1](allBoxImgs[i])).unsqueeze(0) for i in range(len(allBoxImgs))], dim=0)
        boxMaskOut = torch.stack(allBoxMask)
        bboxLabelOut = torch.FloatTensor(np.stack(all_bboxlabel[:self.n_fg_boxes]))

        mask = torch.zeros(1, src_img_size[1], src_img_size[0])
        for ebox in all_boxesPix:
            mask[:,int(ebox[1]):int(ebox[1]+ebox[3]),int(ebox[0]):int(ebox[0]+ebox[2])] = 1.
        if self.use_highres_patch:
            gain = float(max(image_size)) / float(max(src_img_size))
            canvas_image_size = (int(src_img_size[0]*gain), int(src_img_size[1]*gain))
            #image = FN.resize(image, (canvas_image_size[1], canvas_image_size[0]))
            image = Image.fromarray(cv2.resize(np.array(image), (canvas_image_size[0], canvas_image_size[1]), interpolation=cv2.INTER_AREA))
            mask = torch.tensor(np.array(FN.resize(Image.fromarray(mask[0].numpy().astype(np.uint8)), (canvas_image_size[1], canvas_image_size[0]), interpolation=Image.NEAREST )).astype(np.float),dtype=torch.float32)[None,::]
            if self.use_gt_mask:
                all_boxGtMask = [torch.tensor(np.array(FN.resize(Image.fromarray(gtMask[0].numpy().astype(np.uint8)), (canvas_image_size[1],canvas_image_size[0]), interpolation=Image.NEAREST )).astype(np.float),dtype=torch.float32)[None,::] for gtMask in all_boxGtMask]
                mask = torch.cat([mask]+all_boxGtMask, dim=0)

        image = self.transform[-1](image)[None,::]
        cs = image.shape[2:]
        ns = image_size
        pad = [(ns[1]-cs[1])//2, math.ceil((ns[1]-cs[1])/2),(ns[0]-cs[0])//2, math.ceil((ns[0]-cs[0])/2)]
        image = torch.nn.functional.pad(image, pad, value=0.)[0]
        mask = torch.nn.functional.pad(mask[None,::], pad, value=0.)[0]

        return image, mask, torch.FloatTensor(pad), torch.IntTensor(list(cs)), boxImgOut, boxMaskOut, torch.FloatTensor(all_boxes), bboxLabelOut, torch.IntTensor([cocoid])

    #COCO Dataset
    def getRandMaskByClassBatch(self, cls=None, index=None, get_img=False, image_size=128):
        boxImgTargAll = []; boxmaskRandAll = []; sBoxAll = []; labAll = []
        for c in cls:
            boxImgTarg, boxmaskRand, _, sBox, lab= self.getRandMaskByClass(cls=c, index=None, get_img=get_img, image_size=image_size)
            boxImgTargAll.append(boxImgTarg.unsqueeze(0))
            boxmaskRandAll.append(boxmaskRand.unsqueeze(0))
            sBoxAll.append(sBox.unsqueeze(0))
            labAll.append(lab.unsqueeze(0))

        return torch.cat(boxImgTargAll, dim=0), torch.cat(boxmaskRandAll, dim=0), torch.cat(sBoxAll, dim=0), torch.cat(labAll, dim=0)

    #COCO Dataset
    def getRandMaskByClass(self, cls=None, index=None, get_img=False, image_size=128, n_boxes=1):
        if cls is None and index is None:
            assert(0)
        elif (cls is None) and (index is not None):
            cls = random.choice(self.catsInImg[index])

        index = random.choice(self.attToImgId[cls]) if index == None else index

        image = Image.open(os.path.join(self.image_path,self.dataset['images'][index]['filepath'], self.dataset['images'][index]['filename']))
        if image.mode != 'RGB':
            #print image.mode
            image = image.convert('RGB')

        if not self.use_highres_patch:
            image = self.transform[0](image)
            src_img_size = (self.out_img_size, self.out_img_size)
            out_img_size = (self.out_img_size, self.out_img_size)
        else:
            src_img_size = self.dataset['images'][index]['imgSize']
            out_img_size = (self.out_img_size, self.out_img_size)

        sampbboxOrig, bboxLabel, sbox, cbid = self.getBboxForClass(index, cls)
        annId = sbox['id']
        cocoid = self.dataset['images'][index]['cocoid']
        gtMask = self.mask_provider.getbyImgAnnId(cocoid, sbox['id'] , hflip=0)
        sampbbox = [int(bc*src_img_size[i%2]) for i,bc in enumerate(sampbboxOrig)]
        if n_boxes > 1:
            e_boxesPix = []
            e_boxes = []
            for i in range(n_boxes -1):
                clsExtra = random.choice(self.catsInImg[index])
                ebox, _, _,_ = self.getBboxForClass(index, clsExtra)
                e_boxes.append(ebox)
                e_boxesPix.append([int(bc*src_img_size[i%2]) for i,bc in enumerate(ebox)])


        boxCrop = np.zeros((out_img_size[1], out_img_size[0],3))
        boxMask = torch.zeros(1, out_img_size[1], out_img_size[0])

        if self.patch_with_bg:
            if self.patch_with_bg ==1:
                maxlen = max(max(sampbbox[2:]),self.out_img_size)
                extract_box = [max(sampbbox[0]-(maxlen-sampbbox[2])//2,0),
                               max(sampbbox[1]-(maxlen-sampbbox[3])//2,0),
                               maxlen, maxlen]
            else:
                extract_box = sampbbox
            src_patch = np.array(image)[extract_box[1]:extract_box[1]+extract_box[3], extract_box[0]:extract_box[0]+extract_box[2],:]
        else:
            extract_box = sampbbox
            src_patch = (image*(gtMask.byte().numpy().transpose(1,2,0)))[extract_box[1]:extract_box[1]+extract_box[3], extract_box[0]:extract_box[0]+extract_box[2],:]

        src_mask = gtMask[0,extract_box[1]:extract_box[1]+extract_box[3], extract_box[0]:extract_box[0]+extract_box[2]]

        if self.use_highres_patch:
            scale_ratio = min(1.0, float(out_img_size[0])/float(sampbbox[2]), float(out_img_size[1])/sampbbox[3])
            targ_boxsize = [int(sampbbox[2]*scale_ratio), int(sampbbox[3]*scale_ratio)]
            targ_boxloc = [out_img_size[0]//2 - targ_boxsize[0]//2, # Center the patch in the destination
                           out_img_size[1]//2 - targ_boxsize[1]//2]
            src_patch = np.array(FN.resize(Image.fromarray(src_patch), (targ_boxsize[1],targ_boxsize[0])))
            src_mask = torch.tensor(np.array(FN.resize(Image.fromarray(src_mask.numpy()), (targ_boxsize[1],targ_boxsize[0]), interpolation=Image.NEAREST)))
        else:
            # co-ordinates of the patch (x_lc, y_lc)
            scale_ratio = 1
            targ_boxsize = [sampbbox[2]*scale_ratio, sampbbox[3]*scale_ratio]
            targ_boxloc = [out_img_size[0]//2 - targ_boxsize[0]//2, # Center the patch in the destination
                           out_img_size[1]//2 - targ_boxsize[1]//2]

        boxCrop[targ_boxloc[1]: targ_boxloc[1]+targ_boxsize[1], targ_boxloc[0]: targ_boxloc[0]+targ_boxsize[0],:] = src_patch
        boxMask[0,targ_boxloc[1]: targ_boxloc[1]+targ_boxsize[1], targ_boxloc[0]: targ_boxloc[0]+targ_boxsize[0]] = src_mask
        boxCrop = Image.fromarray(boxCrop.astype(np.uint8))

        if self.get_kp_class:
            sampbboxOrig = sampbboxOrig + self.imid2kpClusts[cocoid, annId]
        if get_img:
            mask = torch.zeros(1, src_img_size[1], src_img_size[0])
            mask[:,sampbbox[1]:sampbbox[1]+sampbbox[3],sampbbox[0]:sampbbox[0]+sampbbox[2]] = 1.
            if n_boxes > 1:
                for ebox in e_boxesPix:
                    mask[:,int(ebox[1]):int(ebox[1]+ebox[3]),int(ebox[0]):int(ebox[0]+ebox[2])] = 1.
            if self.use_highres_patch:
                gain = float(max(image_size)) / float(max(src_img_size))
                out_image_size = (int(src_img_size[0]*gain), int(src_img_size[1]*gain))
                #out_image_size = (int(src_img_size[0]*image_size[1] / src_img_size[0]), int(src_img_size[1]*image_size[0] / src_img_size[1]))
                #image = FN.resize(image, (out_image_size[1], out_image_size[0]), interpolation=Image.NEAREST)
                image = Image.fromarray(cv2.resize(np.array(image), (out_image_size[0], out_image_size[1]), interpolation=cv2.INTER_AREA))
                mask = torch.tensor(np.array(FN.resize(Image.fromarray(mask[0].numpy().astype(np.uint8)), (out_image_size[1],out_image_size[0]), interpolation=Image.NEAREST )).astype(np.float),dtype=torch.float32)[None,::]
                if self.use_gt_mask:
                    gtMask = torch.tensor(np.array(FN.resize(Image.fromarray(gtMask[0].numpy().astype(np.uint8)), (out_image_size[1],out_image_size[0]), interpolation=Image.NEAREST )).astype(np.float),dtype=torch.float32)[None,::]
                    mask = torch.cat([mask, gtMask], dim=0)
            if n_boxes > 1:
                for ebox in e_boxes:
                    sampbboxOrig += ebox
            boxImgOut = self.transform[-1](boxCrop) if self.patch_with_bg else boxMask*self.transform[-1](boxCrop)
            return boxImgOut, boxMask, index, self.transform[-1](image), mask, torch.FloatTensor(sampbboxOrig), torch.FloatTensor(bboxLabel)
        else:
            # Create Mask
            #patch = FN.resized_crop(image, sampbbox[1], sampbbox[0], sampbbox[3],sampbbox[2], (self.bbox_out_size, self.bbox_out_size))
            boxImgOut = self.transform[-1](boxCrop) if self.patch_with_bg else boxMask*self.transform[-1](boxCrop)
            return boxImgOut, boxMask, index, torch.FloatTensor(sampbboxOrig),torch.FloatTensor(bboxLabel)

    #COCO Dataset
    def getPatchAndCanvasbyIndexAndclass(self, index, cid, annId=None):
        image = Image.open(os.path.join(self.image_path,self.dataset['images'][index]['filepath'], self.dataset['images'][index]['filename']))
        currCls = self.selected_attrs[cid[0]]
        if image.mode != 'RGB':
            #print image.mode
            image = image.convert('RGB')
        c_imgsize = image.size

        sampbbox, bboxLabel, sbox, cbid = self.getBboxForClass(index, currCls,annId=annId)
        annId = sbox['id']
        cocoid = self.dataset['images'][index]['cocoid']
        bboxId = cid

        extra_boxes = []
        if self.n_extra_boxes> 1:
            # Sample random number of boxes between 1 and n_boxes
            c_nbox = np.random.randint(0,self.n_extra_boxes)
            c_area = sampbbox[2]*sampbbox[3]
            for i in xrange(c_nbox):
                # Also stop at total area > 50%
                if c_area < 0.5:
                    bsamp, _, _ = self.randomBBoxSample(index, 0.6-c_area) # Extra 10% to make the sampling easier
                    extra_boxes.append(bsamp)
                    c_area += bsamp[2]*bsamp[3]
                else:
                    break

        label = self.dataset['images'][index]['label']

        # Apply transforms to the image.
        if not self.use_highres_patch:
            image = self.transform[0](image)
            src_img_size = (self.out_img_size, self.out_img_size)
            out_img_size = (self.out_img_size, self.out_img_size)
        else:
            src_img_size = self.dataset['images'][index]['imgSize']
            out_img_size = (self.out_img_size, self.out_img_size)

        # Now do the flipping
        hflip = 0
        if self.randHFlip and random.random()>0.5:
            hflip = 1
            image = FN.hflip(image)
            sampbbox[0] = 1.0-(sampbbox[0]+sampbbox[2])

        if self.use_gt_mask==1:
            # Use GT masks as input
            gtMask = self.mask_provider.getbyImgAnnId(cocoid, sbox['id'] , hflip=hflip)

        #Convert BBox to actual co-ordinates
        sampbbox = [int(bc*src_img_size[i%2]) for i,bc in enumerate(sampbbox)]
        # Prepare the patch, no Resizing now
        boxCrop = np.zeros((out_img_size[1], out_img_size[0],3))
        boxMask = torch.zeros(1, out_img_size[1], out_img_size[0])
        if self.patch_with_bg:
            if self.patch_with_bg ==1:
                maxlen = max(max(sampbbox[2:]),self.out_img_size)
                extract_box = [max(sampbbox[0]-(maxlen-sampbbox[2])//2,0),
                               max(sampbbox[1]-(maxlen-sampbbox[3])//2,0),
                               maxlen, maxlen]
            else:
                extract_box = sampbbox

            src_patch = np.array(image)[extract_box[1]:extract_box[1]+extract_box[3], extract_box[0]:extract_box[0]+extract_box[2],:]
        else:
            extract_box = sampbbox
            src_patch = (image*(gtMask.byte().numpy().transpose(1,2,0)))[extract_box[1]:extract_box[1]+extract_box[3], extract_box[0]:extract_box[0]+extract_box[2],:]
        src_mask = gtMask[0,extract_box[1]:extract_box[1]+extract_box[3], extract_box[0]:extract_box[0]+extract_box[2]]

        if self.use_highres_patch:
            scale_ratio = min(1.0, float(out_img_size[0])/float(extract_box[2]), float(out_img_size[1])/extract_box[3])
            onlyObjtarg_boxsize = [int(sampbbox[2]*scale_ratio), int(sampbbox[3]*scale_ratio)]
            targ_boxsize = [int(extract_box[2]*scale_ratio), int(extract_box[3]*scale_ratio)]
            targ_boxloc = [out_img_size[0]//2 - targ_boxsize[0]//2, # Center the patch in the destination
                           out_img_size[1]//2 - targ_boxsize[1]//2]
            src_patch = np.array(FN.resize(Image.fromarray(src_patch), (targ_boxsize[1],targ_boxsize[0])))
            src_mask = torch.tensor(np.array(FN.resize(Image.fromarray(src_mask.numpy()), (targ_boxsize[1],targ_boxsize[0]), interpolation=Image.NEAREST)))
        else:
            # co-ordinates of the patch (x_lc, y_lc)
            scale_ratio = 1
            onlyObjtarg_boxsize = [int(sampbbox[2]*scale_ratio), int(sampbbox[3]*scale_ratio)]
            targ_boxsize = [extract_box[2]*scale_ratio, extract_box[3]*scale_ratio]
            targ_boxloc = [out_img_size[0]//2 - targ_boxsize[0]//2, # Center the patch in the destination
                           out_img_size[1]//2 - targ_boxsize[1]//2]

        boxCrop[targ_boxloc[1]: targ_boxloc[1]+targ_boxsize[1], targ_boxloc[0]: targ_boxloc[0]+targ_boxsize[0],:] = src_patch
        boxMask[0,targ_boxloc[1]: targ_boxloc[1]+targ_boxsize[1], targ_boxloc[0]: targ_boxloc[0]+targ_boxsize[0]] = src_mask
        boxCrop = Image.fromarray(boxCrop.astype(np.uint8))


        #patch = FN.resized_crop(image, sampbbox[1], sampbbox[0], sampbbox[3],sampbbox[2], (self.bbox_out_size, self.bbox_out_size))
        # Now resize to bboxImage and mask to the right sizes
        if self.apply_randcolor_transform and self.apply_random_affine:
            aorc = random.random()>=0.3
        else:
            aorc = self.apply_random_affine

        if aorc:
            # max_scale, min_scale
            min_box_size = 64#self.min_box_size
            sc_range = [ np.maximum(float(min_box_size)   /float(targ_boxsize[1]), float(min_box_size)   /float(targ_boxsize[0])),
                         np.minimum(float(out_img_size[1])/float(targ_boxsize[1]), float(out_img_size[0])/float(targ_boxsize[0]))]
            #sc = 0.5
            #dg = np.random.randint(-180,180)
            dg = np.random.randint(-90,90)
            sh = np.random.randint(-30,30)
            #sh = np.random.randint(-45,45)
            # -----------------------------------------------------------------------
            # check if the object goes out of bounds when rotated and scale appropriately.
            # Do all computations in normalized co-ordinates.
            # top right = (1,1), bottom left = (-1,-1), center = (0,0)
            # -----------------------------------------------------------------------
            # compute co-ordinates of top-right diagonal of the box
            # For this use the actual object box, not the full box with bg
            od_x, od_y = onlyObjtarg_boxsize[0]/out_img_size[0], onlyObjtarg_boxsize[1]/out_img_size[1]
            # new co-ordinates of the diag is given as:
            rad_ang = np.radians(dg)
            # for top-right diag
            nd_x_1 = od_x * np.cos(rad_ang) + od_y * np.sin(rad_ang) + 1e-8
            nd_y_1 = od_y * np.cos(rad_ang) - od_x * np.sin(rad_ang) + 1e-8
            # for top-left diag
            nd_x_2 = -od_x * np.cos(rad_ang) + od_y * np.sin(rad_ang) + 1e-8
            nd_y_2 = od_y * np.cos(rad_ang) + od_x * np.sin(rad_ang) + 1e-8
            # Compute maximum allowed scaling such that all the four co-ordintes are between -1 and 1.
            max_scale = min(1./abs(nd_x_1), 1./abs(nd_y_1), 1./abs(nd_x_2), 1./abs(nd_y_2))

            sc_range[0] =  min(sc_range[0],max_scale)
            sc_range[1] =  min(sc_range[1],max_scale)
            sc = np.random.rand()*(sc_range[1] - sc_range[0]) + sc_range[0]
            # New co-ordinates after clockwise roatations
            affineParams = torch.FloatTensor([dg, sh, sc, 1.])
            boxAffine = FN.affine(boxCrop, dg, (0,0), sc,sh, resample=Image.BILINEAR)
            boxMaskAffine = torch.FloatTensor(np.asarray(FN.affine(Image.fromarray(boxMask.numpy()[0]), dg, (0,0), sc,sh, resample=Image.NEAREST)))[None,::]
        elif self.apply_randcolor_transform:
            transformed = self.color_augment(image=np.array(boxCrop))
            boxAffine = Image.fromarray(transformed['image'])
            boxMaskAffine = boxMask
            affineParams = torch.FloatTensor([0., 0., 1., 0.])
        else:
            affineParams = torch.FloatTensor([0., 0., 1., -1.])


        #if self.bbox_out_size != self.out_image_size:
        #   boxMask = FN.resize(boxMask, (self.bbox_out_size, self.bbox_out_size))

        # Create Mask
        #patch = FN.resized_crop(image, sampbbox[1], sampbbox[0], sampbbox[3],sampbbox[2], (self.bbox_out_size, self.bbox_out_size))

        mask = torch.zeros(1+3*self.getBoxMap, src_img_size[1], src_img_size[0])
        if self.getBoxMap:
            cbboxList = self.dataset['images'][index]['bboxAnn']
            for i in xrange(len(cbboxList)):
                if i != cbid:
                    box = copy(cbboxList[i]['bbox'])
                    if hflip:
                        box[0] = 1.0-(box[0]+box[2])
                    box = [int(bc*src_img_size[i%2]) for i,bc in enumerate(box)]
                    boxcid = self.sattr_to_idx[self.catid2attr[cbboxList[i]['cid']]]
                    boxCol = self.classTocolor(self.cNorm(boxcid))[:3]
                    mask[0,box[1]:box[1]+box[3],box[0]:box[0]+box[2]] = boxCol[0]
                    mask[1,box[1]:box[1]+box[3],box[0]:box[0]+box[2]] = boxCol[1]
                    mask[2,box[1]:box[1]+box[3],box[0]:box[0]+box[2]] = boxCol[2]

            mask[3,sampbbox[1]:sampbbox[1]+sampbbox[3],sampbbox[0]:sampbbox[0]+sampbbox[2]] = 1.
            if self.n_extra_boxes > 1 and len(extra_boxes):
                for box in extra_boxes:
                    box = [int(bc*self.out_img_size) for bc in box]
                    mask[3,box[1]:box[1]+box[3],box[0]:box[0]+box[2]] = 1.
        else:
            mask[:,sampbbox[1]:sampbbox[1]+sampbbox[3],sampbbox[0]:sampbbox[0]+sampbbox[2]] = 1.
            if self.n_extra_boxes> 1 and len(extra_boxes):
                for box in extra_boxes:
                    box = [int(bc*self.out_img_size) for bc in box]
                    mask[0,box[1]:box[1]+box[3],box[0]:box[0]+box[2]] = 1.

        if self.boxrotate:
            mask = torch.FloatTensor(np.asarray(self.rotateTrans(Image.fromarray(mask.numpy()[0]))))[None,::]
        if self.use_gt_mask:
            mask = torch.cat([mask, gtMask], dim=0)

        if self.use_highres_patch:
            image = self.transform[0](image)
            mask = torch.cat([torch.tensor(np.array(self.mask_trans(Image.fromarray(mask[i].numpy().astype(np.uint8)))).astype(np.float),dtype=torch.float32)[None,::] for i in range(mask.shape[0])] ,dim=0)

        classVec = torch.LongTensor([bboxId]) if not self.get_all_inst else torch.LongTensor(np.array([self.valid_ids[index], bboxId[0], annId]))

        if self.get_kp_class:
            sampbbox = sampbbox + self.imid2kpClusts[cocoid, annId]

        if self.patch_with_bg:
            final_boxImg = torch.cat([self.transform[-1](boxCrop), self.transform[-1](boxAffine)],dim=0)
        else:
            final_boxImg = torch.cat([boxMask*self.transform[-1](boxCrop), boxMaskAffine*self.transform[-1](boxAffine)],dim=0)
        final_boxMask = torch.cat([boxMask, boxMaskAffine],dim=0)
        if self.get_rand_mask:
            boxImgRand, boxMaskRand, _, _, _ = self.getRandMaskByClass(currCls, image_size=self.out_img_size) #targ_split[data_iter[(c_idx + ridx)%totalItems]]
            final_boxImg = torch.cat([final_boxImg, boxImgRand],dim=0)
            final_boxMask = torch.cat([final_boxMask, boxMaskRand],dim=0)

        return self.transform[-1](image), \
               torch.FloatTensor(label), \
               final_boxImg, \
               torch.FloatTensor(bboxLabel), \
               mask, \
               torch.IntTensor(sampbbox), \
               classVec, \
               final_boxMask, \
               affineParams

    #COCO Dataset
    def getbyIndexAndclass(self, index, cid):

        image = Image.open(os.path.join(self.image_path,self.dataset['images'][index]['filepath'], self.dataset['images'][index]['filename']))
        currCls = self.selected_attrs[cid[0]]
        if image.mode != 'RGB':
            #print image.mode
            image = image.convert('RGB')

        sampbbox, bboxLabel, cbid = self.randomBBoxSample(index, 0.5, no_random_box=1.)
        if cbid == None:
            bboxId = -1
        else:
            bboxId = self.sattr_to_idx[self.catid2attr[self.dataset['images'][index]['bboxAnn'][cbid]['cid']]]
        extra_boxes = []
        if self.n_boxes > 1:
            # Sample random number of boxes between 1 and n_boxes
            c_nbox = np.random.randint(0,self.n_boxes)
            c_area = sampbbox[2]*sampbbox[3]
            for i in xrange(c_nbox):
                # Also stop at total area > 50%
                if c_area < 0.5:
                    bsamp, _, _ = self.randomBBoxSample(index, 0.6-c_area) # Extra 10% to make the sampling easier
                    extra_boxes.append(bsamp)
                    c_area += bsamp[2]*bsamp[3]
                else:
                    break

        label = self.dataset['images'][index]['label']

        # Apply transforms to the image.
        image = self.transform[0](image)
        # Now do the flipping
        hflip = 0
        if self.randHFlip and random.random()>0.5:
            hflip = 1
            image = FN.hflip(image)
            sampbbox[0] = 1.0-(sampbbox[0]+sampbbox[2])
        if self.use_gt_mask==1:
            # Use GT masks as input
            gtMask = self.mask_provider.getbyIdAndclass(self.dataset['images'][index]['cocoid'], currCls, hflip=hflip)
        elif self.use_gt_mask==2:
            # Use GT boxes as input
            gtBoxes = [bbox for bbox in self.dataset['images'][index]['bboxAnn'] if self.catid2attr[bbox['cid']]== currCls]
            gtMask = torch.zeros(1,self.out_img_size,self.out_img_size)
            for box in gtBoxes:
                bbox = copy(box['bbox'])
                if hflip:
                    bbox[0] = 1.0-(bbox[0]+bbox[2])
                bbox = [int(bc*self.out_img_size) for bc in bbox]
                gtMask[0,bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]] = 1.
        elif self.use_gt_mask==3:
            # Use GT centerpoints as input
            gtBoxes = [bbox for bbox in self.dataset['images'][index]['bboxAnn'] if self.catid2attr[bbox['cid']]== currCls]
            gtMask = torch.zeros(1,self.out_img_size,self.out_img_size)
            for box in gtBoxes:
                bbox = copy(box['bbox'])
                if hflip:
                    bbox[0] = 1.0-(bbox[0]+bbox[2])
                bbox = [int(bc*self.out_img_size) for bc in bbox]
                cent = [bbox[0] + bbox[2]//2, bbox[1]+bbox[3]//2]
                # center is marked by a 3x3 square patch
                gtMask[0,cent[1]-1:cent[1]+2,cent[0]-1:cent[0]+2] = 1.


        #Convert BBox to actual co-ordinates
        sampbbox = [int(bc*self.out_img_size) for bc in sampbbox]
        boxCrop = FN.resized_crop(image, sampbbox[1], sampbbox[0], sampbbox[3],sampbbox[2], (self.bbox_out_size, self.bbox_out_size))

        # Create Mask
        mask = torch.zeros(1+3*self.getBoxMap,self.out_img_size,self.out_img_size)
        if self.getBoxMap:
            cbboxList = self.dataset['images'][index]['bboxAnn']
            for i in xrange(len(cbboxList)):
                if i != cbid:
                    box = copy(cbboxList[i]['bbox'])
                    if hflip:
                        box[0] = 1.0-(box[0]+box[2])
                    box = [int(bc*self.out_img_size) for bc in box]
                    boxcid = self.sattr_to_idx[self.catid2attr[cbboxList[i]['cid']]]
                    boxCol = self.classTocolor(self.cNorm(boxcid))[:3]
                    mask[0,box[1]:box[1]+box[3],box[0]:box[0]+box[2]] = boxCol[0]
                    mask[1,box[1]:box[1]+box[3],box[0]:box[0]+box[2]] = boxCol[1]
                    mask[2,box[1]:box[1]+box[3],box[0]:box[0]+box[2]] = boxCol[2]

            mask[3,sampbbox[1]:sampbbox[1]+sampbbox[3],sampbbox[0]:sampbbox[0]+sampbbox[2]] = 1.
            if self.n_boxes > 1 and len(extra_boxes):
                for box in extra_boxes:
                    box = [int(bc*self.out_img_size) for bc in box]
                    mask[3,box[1]:box[1]+box[3],box[0]:box[0]+box[2]] = 1.
        else:
            mask[:,sampbbox[1]:sampbbox[1]+sampbbox[3],sampbbox[0]:sampbbox[0]+sampbbox[2]] = 1.
            if self.n_boxes > 1 and len(extra_boxes):
                for box in extra_boxes:
                    box = [int(bc*self.out_img_size) for bc in box]
                    mask[0,box[1]:box[1]+box[3],box[0]:box[0]+box[2]] = 1.

        if self.boxrotate:
            mask = torch.FloatTensor(np.asarray(self.rotateTrans(Image.fromarray(mask.numpy()[0]))))[None,::]
        if self.use_gt_mask:
            mask = torch.cat([mask, gtMask], dim=0)

        return self.transform[-1](image), torch.FloatTensor(label), self.transform[-1](boxCrop), torch.FloatTensor(bboxLabel), mask, torch.IntTensor(sampbbox), torch.LongTensor([bboxId])

    #COCO Dataset
    def getbyIndexAndclassAugmentMode(self, index):

        imgData = self.dataset['images'][index]

        image = Image.open(os.path.join(self.image_path,imgData['filepath'], imgData['filename']))
        if image.mode != 'RGB':
            #print image.mode
            image = image.convert('RGB')

        gtBoxes = np.zeros((len(self.selected_attrs),4))
        for bbox in imgData['bboxAnn']:
            gtBoxes[self.sattr_to_idx[self.catid2attr[bbox['cid']]],:] = bbox['bbox']

        label = imgData['label']

        # Apply transforms to the image.
        image = self.transform[0](image)
        # Now do the flipping
        hflip = 0
        if self.randHFlip and random.random()>0.5:
            hflip = 1
            image = FN.hflip(image)
            gtBoxes[np.array(label,dtype=np.int),0] = 1.0-(gtBoxes[np.array(label,dtype=np.int),0]+gtBoxes[np.array(label,dtype=np.int),2])

        #Get class effect;
        class_effect  = np.zeros((len(self.selected_attrs),len(self.selected_attrs)))
        class_effect[np.meshgrid(imgData['label_seq'],imgData['label_seq'])] = imgData['cls_affect']

        #Convert BBox to actual co-ordinates
        return self.transform[-1](image), torch.FloatTensor(label), torch.FloatTensor(gtBoxes), torch.LongTensor([imgData['cocoid']]), torch.LongTensor([hflip]), torch.FloatTensor(class_effect.T)

    #COCO Dataset
    def __len__(self):
        return self.num_data

    #COCO Dataset
    def getfilename(self, index):
        return self.dataset['images'][index]['filename']

    #COCO Dataset
    def getfilename_bycocoid(self, cocoid):
        return self.dataset['images'][self.imgId2idx[cocoid]]['filename']

    #COCO Dataset
    def getcocoid(self, index):
        return self.dataset['images'][index]['cocoid']

    #COCO Dataset
    def getGTMaskInp(self, index, cls, hflip=False, mask_type=None):
        what_mask = self.use_gt_mask if mask_type is None else mask_type
        if what_mask==1:
            # Use GT masks as input
            gtMask = self.mask_provider.getbyIdAndclass(self.dataset['images'][index]['cocoid'], cls, hflip=hflip)
        elif what_mask==2:
            # Use GT boxes as input
            gtBoxes = [bbox for bbox in self.dataset['images'][index]['bboxAnn'] if self.catid2attr[bbox['cid']]== cls]
            gtMask = torch.zeros(1,self.out_img_size,self.out_img_size)
            for box in gtBoxes:
                bbox = copy(box['bbox'])
                if hflip:
                    bbox[0] = 1.0-(bbox[0]+bbox[2])
                bbox = [int(bc*self.out_img_size) for bc in bbox]
                gtMask[0,bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]] = 1.
        elif what_mask==3:
            # Use GT centerpoints as input
            gtBoxes = [bbox for bbox in self.dataset['images'][index]['bboxAnn'] if self.catid2attr[bbox['cid']]== cls]
            gtMask = torch.zeros(1,self.out_img_size,self.out_img_size)
            for box in gtBoxes:
                bbox = copy(box['bbox'])
                if hflip:
                    bbox[0] = 1.0-(bbox[0]+bbox[2])
                bbox = [int(bc*self.out_img_size) for bc in bbox]
                cent = [bbox[0] + bbox[2]//2, bbox[1]+bbox[3]//2]
                # center is marked by a 3x3 square patch
                gtMask[0,cent[1]-1:cent[1]+2,cent[0]-1:cent[0]+2] = 1.
        else:
            gtMask = None

        return gtMask

class BDD100kDataset(Dataset):
    def __init__(self, transform, mode, select_attrs=[], datafile='bdd100k_labels_images_det_coco_train.json', out_img_size=128, bbox_out_size=64,
                 balance_classes=0, onlyrandBoxes=False, max_object_size=0., min_box_size = 16, max_with_union=True, use_gt_mask=False,
                 boxrotate=0, n_fg_boxes = 1, n_extra_boxes= 0, square_resize=0, filter_by_mincooccur = -1., only_indiv_occur = 0., patch_mode=0,
                 use_highres_patch=0, get_all_inst=False, remove_invalid = 0, color_or_affine='affine', get_kp_class=0, get_rand_mask=0,
                 detectorMode=0, canvas_image_size=416, filter_class_instances = []):

        self.transform = transform
        self.mode = mode
        datafile = 'bdd100k_labels_images_det_coco_train.json' if mode=='train' else 'bdd100k_labels_images_det_coco_minival.json'
        self.image_path = os.path.join('data','bdd100k','images','100k',self.mode)
        self.n_fg_boxes = n_fg_boxes
        self.n_extra_boxes= n_extra_boxes
        self.dataset = json.load(open(os.path.join('data','bdd100k',datafile),'r'))
        self.num_data = len(self.dataset['images'])
        self.attr2idx = {}
        self.idx2attr = {}
        self.catid2attr = {}
        self.out_img_size = out_img_size
        self.square_resize = square_resize
        self.bbox_out_size = bbox_out_size
        self.filter_by_mincooccur = filter_by_mincooccur
        self.only_indiv_occur = only_indiv_occur
        #self.selected_attrs = ['person', 'book', 'car', 'bird', 'chair'] if select_attrs== [] else select_attrs
        self.selected_attrs = select_attrs
        self.get_kp_class = get_kp_class
        self.get_rand_mask = get_rand_mask
        self.detectorMode = detectorMode
        self.canvas_image_size = canvas_image_size

        self.apply_random_affine = True if color_or_affine=='affine' else False
        self.apply_randcolor_transform = True if color_or_affine=='color' else False
        if self.apply_randcolor_transform:
            self.color_augment = Compose([RGBShift(r_shift_limit=30, g_shift_limit=20, b_shift_limit=30), RandomBrightnessContrast(),HueSaturationValue(hue_shift_limit=0, sat_shift_limit=10, val_shift_limit=40), RandomGamma()],p=0.99, additional_targets={'patch':'image'})

        if len(select_attrs) == 0 or (select_attrs[0] == 'all'):
            self.selected_attrs = [attr['name'] for attr in self.dataset['categories']]
        self.filter_class_instances = set(filter_class_instances)
        print('Removing instances of classes ', filter_class_instances)
        self.balance_classes = balance_classes
        self.onlyrandBoxes = onlyrandBoxes
        self.max_object_size = max_object_size
        self.max_with_union= max_with_union
        self.min_box_size = min_box_size
        #self.min_minbox_size = 10
        self.use_gt_mask = use_gt_mask
        self.remove_truncated = True
        self.remove_occluded = True
        self.boxrotate = boxrotate
        self.patch_mode = patch_mode
        self.use_highres_patch = use_highres_patch
        self.remove_invalid= remove_invalid
        if self.boxrotate:
            self.rotateTrans = transforms.Compose([transforms.RandomRotation(boxrotate,resample=Image.NEAREST)])
        if use_gt_mask == 1:
            self.mask_trans =transforms.Compose([transforms.Resize(out_img_size if not square_resize else [out_img_size, out_img_size] , interpolation=Image.NEAREST), transforms.CenterCrop(out_img_size)])

        self.getBoxMap = 0

        if self.getBoxMap:
            self.classTocolor = cm.get_cmap('Spectral')
            self.cNorm = matplotlib.colors.Normalize(vmin=0., vmax=len(self.selected_attrs))
        self.randHFlip = 'Flip' in transform

        print('Start preprocessing dataset..!')
        self.preprocess()
        print('Finished preprocessing dataset..!')
        self.imgId2idx = {imid:i for i,imid in enumerate(self.valid_ids)}

        self.get_all_inst = get_all_inst
        self.num_data = len(self.dataset['images']) if not get_all_inst else len(self.all_imgIdandAnn)

    #Bdd100k
    def preprocess(self):
        for i, attr in enumerate(self.dataset['categories']):
            self.attr2idx[attr['name']] = i
            self.idx2attr[i] = attr['name']
            self.catid2attr[attr['id']] = attr['name']

        self.sattr_to_idx = {att:i for i, att in enumerate(self.selected_attrs)}

        #--------------------------------------------------------
        # Keep only the required annotations
        # 1. required classes
        # 2. based on occluded and truncated filters
        #--------------------------------------------------------
        selset = set(self.selected_attrs) - self.filter_class_instances
        self.dataset['annotations'] = [ann
                                       for ann in self.dataset['annotations']
                                       if (self.catid2attr[ann['category_id']] in selset) and
                                       (not (self.remove_truncated and ann['attributes']['truncated'])) and
                                       (not (self.remove_occluded and ann['attributes']['occluded']))
                                       ]

        print('%d annotations left after occlusion/truncation filters'%(len(self.dataset['annotations'])))

        fixed_ann = []
        for i, ann in enumerate(self.dataset['annotations']):
            # Correct BBox for Resize(of smaller edge) and CenterCrop
            imgSize = [720, 1280]
            maxSide = np.argmax(imgSize)
            cbbox = ann['bbox']
            # Largest side of the box will be resized to image_size
            box_size = (cbbox[2], cbbox[3])
            sc_ratio = min(1.0, self.out_img_size/(float(box_size[0]+1e-8)), self.out_img_size / float(box_size[1]+1e-8))
            new_box_size = (int(box_size[0]*sc_ratio) ,  int(box_size[1]*sc_ratio))
            if (max(new_box_size) >= self.min_box_size) and (min(new_box_size) >= 3) and (not ann['iscrowd']):# and ((not self.remove_invalid) or ann['valid']):
                ann['box_label'] = np.zeros(max(len(self.selected_attrs),1))
                ann['box_label'][self.sattr_to_idx[self.catid2attr[ann['category_id']]]] = 1.
                ann['cls'] = self.catid2attr[ann['category_id']]
                fixed_ann.append(ann)

        self.dataset['annotations'] = fixed_ann
        print('%d annotations left after size filters'%(len(self.dataset['annotations'])))

        self.catsInImg = defaultdict(set)
        self.imidToAnnList = defaultdict(list)
        self.attToannId = defaultdict(list)
        self.annIdToIdx = {}
        for i, ann in enumerate(self.dataset['annotations']):
            self.imidToAnnList[ann['image_id']].append((ann['cls'], ann['id']))
            self.annIdToIdx[ann['id']] = i
            self.catsInImg[ann['image_id']].add(ann['cls'])
            self.attToannId[ann['cls']].append(ann['id'])

        self.dataset['images'] = [img for img in self.dataset['images'] if img['id'] in self.imidToAnnList and len(self.imidToAnnList[img['id']]) > 0]
        self.valid_ids= []
        self.imidToIndex = {}
        for i,img in enumerate(self.dataset['images']):
            self.dataset['images'][i]['label'] = np.zeros(max(len(self.selected_attrs),1))
            self.catsInImg[img['id']] = list(self.catsInImg[img['id']])
            self.dataset['images'][i]['label'][[self.sattr_to_idx[cat] for cat in self.catsInImg[img['id']]]] = 1
            self.valid_ids.append(img['id'])
            self.imidToIndex[img['id']] = i
        print('%d images left after empty img filtering'%(len(self.valid_ids)))

    #Bdd100k
    def __getitem__(self, index):
        # In this situation ignore index and sample classes uniformly
        if self.detectorMode:
            returnvals = self.getCanvasImg(index=index, image_size=self.canvas_image_size)
        else:
            if self.get_all_inst:
                index, currCls, annId = self.all_imgIdandAnn[index]
            else:
                if self.balance_classes==1:
                    currCls = random.choice(list(self.attToannId.keys()))
                    annId = random.choice(self.attToannId[currCls])
                    index = self.imidToIndex[self.dataset['annotations'][self.annIdToIdx[annId]]['image_id']]
                else:#if self.balance_classes==2:
                    imid = self.dataset['images'][index]['id']
                    currCls = random.choice(self.catsInImg[imid])

                annId= None

            cid = [self.sattr_to_idx[currCls]] if currCls != 'bg' else [0]
            returnvals = self.getPatchAndCanvasbyIndexAndclass(index, cid, annId=annId)

        return tuple(returnvals)

    #Bdd100k
    def __len__(self):
        return self.num_data

    #Bdd100k
    def getfilename(self, index):
        return self.dataset['images'][index]['file_name']

    #Bdd100k
    def randomBBoxSample(self, imid, max_area = -1, prevBoxList=[], imgSize=[]):
        # With 50% chance sample from background or foreground
        # Minimum size
        minLen = 0.05
        maxLen = 0.3
        maxIou = 0.3
        cbboxList = [self.dataset['annotations'][self.annIdToIdx[aid]] for acls,aid in self.imidToAnnList[imid]]
        n_t = 0
        while 1:
            # sample a random background box
            cbid = None
            tL_x, tL_y = random.uniform(0,1.-minLen-0.01), random.uniform(0,1.-minLen-0.01)
            l_x = random.uniform(minLen, min(1.-tL_x,maxLen))
            l_y = random.uniform(minLen, min(1.-tL_y,maxLen))
            sbox = [tL_x, tL_y, l_x, l_y]
            # Prepare label for this box
            bboxLabel = np.zeros(max(len(self.selected_attrs),1))
            #if len(cbboxList):
            noOverlap = True
            for bb in cbboxList+prevBoxList:
                ibox = [float(sbox[0]*imgSize[0]), float(sbox[1]*imgSize[1]), float(sbox[2]*imgSize[0]), float(sbox[3]*imgSize[1])]
                iou, aInb, bIna = computeIOU(ibox, bb['bbox'])
                if iou > maxIou or aInb >0.8:
                    noOverlap = False
                #if bIna > 0.8 and bb['cid'] > 0:
                #    bboxLabel[self.sattr_to_idx[self.catid2attr[bb['cid']]]] = 1
            if (noOverlap and ((max_area < 0) or ((sbox[2]*sbox[3])< max_area))) or (n_t>10):
                return ibox, bboxLabel, cbid
            n_t += 1

    #Bdd100k
    def getBboxForClass(self, index, cls, annId=None):
        imid = self.dataset['images'][index]['id']
        if annId is None:
            cbboxList = [self.dataset['annotations'][self.annIdToIdx[aid]] for acls,aid in self.imidToAnnList[imid] if acls==cls]
        else:
            cbboxList = [self.dataset['annotations'][self.annIdToIdx[annId]]]
        if len(cbboxList):
            cbid = random.randrange(len(cbboxList))
            sbox = cbboxList[cbid]
            return copy(sbox['bbox']),sbox['box_label'], sbox
        else:
            print(index, self.dataset['images'][index], cls, annId)
            assert(0)

    #Bdd100k
    def getCanvasImg(self, cls=None, index=None, image_size=128):
        if type(image_size) is not list:
            image_size = [image_size, image_size]
        if cls is None and index is None:
            assert(0)
        elif (cls is None) and (index is not None):
            imgId = self.dataset['images'][index]['id']
            cls = random.choice(self.catsInImg[imgId])
            annId = None
        elif index is None:
            annId = random.choice(self.attToannId[cls])
            index = self.imidToIndex[self.dataset['annotations'][self.annIdToIdx[annId]]['image_id']]

        #----------------------------------------------------
        # Read and load the image
        #----------------------------------------------------
        imD = self.dataset['images'][index]
        image = Image.open(os.path.join(self.image_path, imD['file_name']))
        if image.mode != 'RGB':
            #print image.mode
            image = image.convert('RGB')
        c_imgsize = image.size
        src_img_size = c_imgsize
        out_img_size = (self.out_img_size, self.out_img_size)

        #----------------------------------------------------
        # Sample the required number of boxes
        #----------------------------------------------------
        all_boxes = []
        all_boxData = []
        all_bboxlabel = []
        n_boxes = self.n_fg_boxes + self.n_extra_boxes
        imid = self.dataset['images'][index]['id']
        sampAnnList = set()
        j=0
        for i in range(n_boxes):
            remAnnList = [i for i in range(len(self.imidToAnnList[imid])) if self.imidToAnnList[imid][i][1] not in sampAnnList]
            if len(remAnnList):
                ei = random.choice(remAnnList)
                clsExtra = self.imidToAnnList[imid][ei][0]
                aidExtra = self.imidToAnnList[imid][ei][1]
                ebox, ebLabel, eboxData = self.getBboxForClass(index, clsExtra, annId=aidExtra)
                sampAnnList.add(eboxData['id'])
                all_boxData.append(eboxData)
            else:
                ebox, ebLabel, eboxData = self.randomBBoxSample(imid, max_area=0.3, prevBoxList=all_boxData, imgSize=src_img_size) # Extra 10% to make the sampling easier
                all_boxData.append({'bbox':ebox, 'cid':-1})
            all_boxes.append(ebox)
            all_bboxlabel.append(ebLabel)

        e_boxes = all_boxes[self.n_fg_boxes:]
        fg_boxes = all_boxes[:self.n_fg_boxes]

        allBoxImgs = []
        allBoxMask = []
        for i,fgb in enumerate(fg_boxes):
            sampbbox = [int(bc) for bc in fgb]
            boxCrop = np.zeros((out_img_size[1], out_img_size[0],3))
            boxMask = torch.zeros(1, out_img_size[1], out_img_size[0])
            src_patch = np.array(image)[sampbbox[1]:sampbbox[1]+sampbbox[3], sampbbox[0]:sampbbox[0]+sampbbox[2],:]
            src_mask = np.ones(list(src_patch.shape[:2]))

            scale_ratio = min(1.0, float(out_img_size[0])/float(sampbbox[2]), float(out_img_size[1])/sampbbox[3])
            targ_boxsize = [int(sampbbox[2]*scale_ratio), int(sampbbox[3]*scale_ratio)]
            targ_boxloc = [out_img_size[0]//2 - targ_boxsize[0]//2, # Center the patch in the destination
                           out_img_size[1]//2 - targ_boxsize[1]//2]
            src_patch = np.array(FN.resize(Image.fromarray(src_patch), (targ_boxsize[1],targ_boxsize[0])))
            src_mask = torch.tensor(np.array(FN.resize(Image.fromarray(src_mask), (targ_boxsize[1],targ_boxsize[0]), interpolation=Image.NEAREST)))

            boxCrop[targ_boxloc[1]: targ_boxloc[1]+targ_boxsize[1], targ_boxloc[0]: targ_boxloc[0]+targ_boxsize[0],:] = src_patch
            boxMask[0,targ_boxloc[1]: targ_boxloc[1]+targ_boxsize[1], targ_boxloc[0]: targ_boxloc[0]+targ_boxsize[0]] = src_mask
            boxCrop = Image.fromarray(boxCrop.astype(np.uint8))
            allBoxImgs.append(boxCrop)
            allBoxMask.append(boxMask)

        boxImgOut = torch.cat([(allBoxMask[i]*self.transform[-1](allBoxImgs[i])).unsqueeze(0) for i in range(len(allBoxImgs))], dim=0)
        boxMaskOut = torch.stack(allBoxMask)
        bboxLabelOut = torch.FloatTensor(np.stack(all_bboxlabel[:self.n_fg_boxes]))

        #if self.get_kp_class:
        #    sampbboxOrig = sampbboxOrig + self.imid2kpClusts[cocoid, annId]
        mask = torch.zeros(1, src_img_size[1], src_img_size[0])
        for ebox in all_boxes:
            mask[:,int(ebox[1]):int(ebox[1]+ebox[3]),int(ebox[0]):int(ebox[0]+ebox[2])] = 1.
        if self.use_highres_patch:
            out_image_size = (int(src_img_size[0]*image_size[1] / src_img_size[0]), int(src_img_size[1]*image_size[0] / src_img_size[1]))
            image = FN.resize(image, (out_image_size[1], out_image_size[0]))
            mask = torch.tensor(np.array(FN.resize(Image.fromarray(mask[0].numpy().astype(np.uint8)), (out_image_size[1],out_image_size[0]), interpolation=Image.NEAREST )).astype(np.float),dtype=torch.float32)[None,::]
            if self.use_gt_mask:
                mask = torch.cat([mask, mask], dim=0)

        #import ipdb; ipdb.set_trace()#src_img_size
        sampbbox = []
        for ebox in all_boxes:
            sampbbox.append([float(ebox[0])/src_img_size[0], float(ebox[1])/src_img_size[1], float(ebox[2])/src_img_size[0], float(ebox[3])/src_img_size[1]])

        image = self.transform[-1](image)
        cs = image.shape[1:]
        pad = [0, 0, 0, 0]

        return image, mask, torch.FloatTensor(pad), torch.IntTensor(list(cs)), boxImgOut, boxMaskOut, torch.FloatTensor(sampbbox), bboxLabelOut, torch.IntTensor([index])

    #Bdd100k
    def getRandMaskByClass(self, cls=None, index=None, get_img=False, image_size=128, n_fg_boxes=1, n_extra_boxes=0):
        if type(image_size) is not list:
            image_size = [image_size, image_size]
        if cls is None and index is None:
            assert(0)
        elif (cls is None) and (index is not None):
            imgId = self.dataset['images'][index]['id']
            cls = random.choice(self.catsInImg[imgId])
            annId = None
        elif index is None:
            annId = random.choice(self.attToannId[cls])
            index = self.imidToIndex[self.dataset['annotations'][self.annIdToIdx[annId]]['image_id']]

        imD = self.dataset['images'][index]
        image = Image.open(os.path.join(self.image_path, imD['file_name']))
        if image.mode != 'RGB':
            #print image.mode
            image = image.convert('RGB')
        c_imgsize = image.size

        src_img_size = c_imgsize
        out_img_size = (self.out_img_size, self.out_img_size)


        sampbboxOrig, bboxLabel, sbox = self.getBboxForClass(index, cls, annId=annId)
        all_boxes = [sampbboxOrig]
        all_boxData = [sbox]
        all_bboxlabel = [bboxLabel]
        n_boxes = n_fg_boxes + n_extra_boxes
        if n_boxes > 1:
            sampAnnList = set([sbox['id']])
            imid = self.dataset['images'][index]['id']
            for i in range(n_boxes -1):
                remAnnList = [i for i in range(len(self.imidToAnnList[imid])) if self.imidToAnnList[imid][i][1] not in sampAnnList]
                if len(remAnnList):
                    ei = random.choice(remAnnList)
                    clsExtra = self.imidToAnnList[imid][ei][0]
                    aidExtra = self.imidToAnnList[imid][ei][1]
                    ebox, ebLabel, eboxData = self.getBboxForClass(index, clsExtra, annId=aidExtra)
                    sampAnnList.add(eboxData['id'])
                else:
                    ebox, ebLabel, eboxData = self.randomBBoxSample(imid, max_area=0.2, prevBoxList=all_boxes) # Extra 10% to make the sampling easier
                all_boxes.append(ebox)
                all_boxData.append(eboxData)
                all_bboxlabel.append(ebLabel)
        e_boxes = all_boxes[n_fg_boxes:]
        fg_boxes = all_boxes[:n_fg_boxes]

        allBoxImgs = []
        allBoxMask = []
        for i,fgb in enumerate(fg_boxes):
            sampbbox = [int(bc) for bc in fgb]
            boxCrop = np.zeros((out_img_size[1], out_img_size[0],3))
            boxMask = torch.zeros(1, out_img_size[1], out_img_size[0])
            src_patch = np.array(image)[sampbbox[1]:sampbbox[1]+sampbbox[3], sampbbox[0]:sampbbox[0]+sampbbox[2],:]
            src_mask = np.ones(list(src_patch.shape[:2]))

            scale_ratio = min(1.0, float(out_img_size[0])/float(sampbbox[2]), float(out_img_size[1])/sampbbox[3])
            targ_boxsize = [int(sampbbox[2]*scale_ratio), int(sampbbox[3]*scale_ratio)]
            targ_boxloc = [out_img_size[0]//2 - targ_boxsize[0]//2, # Center the patch in the destination
                           out_img_size[1]//2 - targ_boxsize[1]//2]
            src_patch = np.array(FN.resize(Image.fromarray(src_patch), (targ_boxsize[1],targ_boxsize[0])))
            src_mask = torch.tensor(np.array(FN.resize(Image.fromarray(src_mask), (targ_boxsize[1],targ_boxsize[0]), interpolation=Image.NEAREST)))

            boxCrop[targ_boxloc[1]: targ_boxloc[1]+targ_boxsize[1], targ_boxloc[0]: targ_boxloc[0]+targ_boxsize[0],:] = src_patch
            boxMask[0,targ_boxloc[1]: targ_boxloc[1]+targ_boxsize[1], targ_boxloc[0]: targ_boxloc[0]+targ_boxsize[0]] = src_mask
            boxCrop = Image.fromarray(boxCrop.astype(np.uint8))
            allBoxImgs.append(boxCrop)
            allBoxMask.append(boxMask)

        boxImgOut = torch.cat([(allBoxMask[i]*self.transform[-1](allBoxImgs[i])).unsqueeze(0) for i in range(len(allBoxImgs))], dim=0)
        boxMaskOut = torch.stack(allBoxMask)
        bboxLabelOut = torch.FloatTensor(np.stack(all_bboxlabel[:n_fg_boxes]))

        #if self.get_kp_class:
        #    sampbboxOrig = sampbboxOrig + self.imid2kpClusts[cocoid, annId]
        if get_img:
            mask = torch.zeros(1, src_img_size[1], src_img_size[0])
            for ebox in all_boxes:
                mask[:,int(ebox[1]):int(ebox[1]+ebox[3]),int(ebox[0]):int(ebox[0]+ebox[2])] = 1.

            if self.use_highres_patch:
                out_image_size = (int(src_img_size[0]*image_size[1] / src_img_size[0]), int(src_img_size[1]*image_size[0] / src_img_size[1]))
                image = FN.resize(image, (out_image_size[1], out_image_size[0]))
                mask = torch.tensor(np.array(FN.resize(Image.fromarray(mask[0].numpy().astype(np.uint8)), (out_image_size[1],out_image_size[0]), interpolation=Image.NEAREST )).astype(np.float),dtype=torch.float32)[None,::]
                if self.use_gt_mask:
                    mask = torch.cat([mask, mask], dim=0)

            #import ipdb; ipdb.set_trace()#src_img_size
            sampbbox = []
            for ebox in all_boxes:
                sampbbox.append([float(ebox[0])/src_img_size[0], float(ebox[1])/src_img_size[1], float(ebox[2])/src_img_size[0], float(ebox[3])/src_img_size[1]])

            return boxImgOut, boxMaskOut, index, self.transform[-1](image), mask, torch.FloatTensor(sampbbox), bboxLabelOut
        else:
            # Create Mask
            #patch = FN.resized_crop(image, sampbbox[1], sampbbox[0], sampbbox[3],sampbbox[2], (self.bbox_out_size, self.bbox_out_size))
            return boxImgOut, boxMaskOut, index, torch.FloatTensor(bboxLabel)

    #Bdd100k
    def getPatchAndCanvasbyIndexAndclass(self, index, cid, annId=None):
        imD = self.dataset['images'][index]
        image = Image.open(os.path.join(self.image_path, imD['file_name']))

        if annId is None:
            _, annId = random.choice(self.imidToAnnList[imD['id']])
        annIndex = self.annIdToIdx[annId]


        currCls = self.selected_attrs[cid[0]]
        if image.mode != 'RGB':
            #print image.mode
            image = image.convert('RGB')
        c_imgsize = image.size

        sampbbox, bboxLabel, sbox = self.getBboxForClass(index, currCls, annId=annId)
        annId = sbox['id']
        cocoid = self.dataset['images'][index]['id']
        bboxId = cid

        label = self.dataset['images'][index]['label']

        # Apply transforms to the image.
        if not self.use_highres_patch:
            image = self.transform[0](image)
            src_img_size = (self.out_img_size, self.out_img_size)
            out_img_size = (self.out_img_size, self.out_img_size)
        else:
            src_img_size = c_imgsize#self.dataset['images'][index]['imgSize']
            out_img_size = (self.out_img_size, self.out_img_size)

        # Now do the flipping
        hflip = 0
        if self.randHFlip and random.random()>0.5:
            hflip = 1
            image = FN.hflip(image)
            sampbbox[0] = c_imgsize[0]-(sampbbox[0]+sampbbox[2])

        #Convert BBox to actual co-ordinates
        sampbbox = [int(bc) for bc in sampbbox]
        # Prepare the patch, no Resizing now
        boxCrop = np.zeros((out_img_size[1], out_img_size[0],3))
        boxMask = torch.zeros(1, out_img_size[1], out_img_size[0])
        src_patch = np.array(image)[sampbbox[1]:sampbbox[1]+sampbbox[3], sampbbox[0]:sampbbox[0]+sampbbox[2],:]
        src_mask = np.ones(list(src_patch.shape[:2]))

        scale_ratio = min(1.0, float(out_img_size[0])/float(sampbbox[2]), float(out_img_size[1])/sampbbox[3])
        targ_boxsize = [int(sampbbox[2]*scale_ratio), int(sampbbox[3]*scale_ratio)]
        targ_boxloc = [out_img_size[0]//2 - targ_boxsize[0]//2, # Center the patch in the destination
                       out_img_size[1]//2 - targ_boxsize[1]//2]
        src_patch = np.array(FN.resize(Image.fromarray(src_patch), (targ_boxsize[1],targ_boxsize[0])))
        src_mask = torch.tensor(np.array(FN.resize(Image.fromarray(src_mask), (targ_boxsize[1],targ_boxsize[0]), interpolation=Image.NEAREST)))

        boxCrop[targ_boxloc[1]: targ_boxloc[1]+targ_boxsize[1], targ_boxloc[0]: targ_boxloc[0]+targ_boxsize[0],:] = src_patch
        boxMask[0,targ_boxloc[1]: targ_boxloc[1]+targ_boxsize[1], targ_boxloc[0]: targ_boxloc[0]+targ_boxsize[0]] = src_mask
        boxCrop = Image.fromarray(boxCrop.astype(np.uint8))


        #patch = FN.resized_crop(image, sampbbox[1], sampbbox[0], sampbbox[3],sampbbox[2], (self.bbox_out_size, self.bbox_out_size))
        # Now resize to bboxImage and mask to the right sizes

        if self.apply_random_affine:
            # max_scale, min_scale
            min_box_size = 64#self.min_box_size
            sc_range = [ np.maximum(float(min_box_size)   /float(targ_boxsize[1]), float(min_box_size)   /float(targ_boxsize[0])),
                         np.minimum(float(out_img_size[1])/float(targ_boxsize[1]), float(out_img_size[0])/float(targ_boxsize[0]))]
            #sc = 0.5
            dg = np.random.randint(-180,180)
            #dg = np.random.randint(-45,45)
            sh = np.random.randint(-30,30)
            #sh = np.random.randint(-45,45)
            # -----------------------------------------------------------------------
            # check if the object goes out of bounds when rotated and scale appropriately.
            # Do all computations in normalized co-ordinates.
            # top right = (1,1), bottom left = (-1,-1), center = (0,0)
            # -----------------------------------------------------------------------
            # compute co-ordinates of top-right diagonal of the box
            od_x, od_y = targ_boxsize[0]/out_img_size[0], targ_boxsize[1]/out_img_size[1]
            # new co-ordinates of the diag is given as:
            rad_ang = np.radians(dg)
            # for top-right diag
            nd_x_1 = od_x * np.cos(rad_ang) + od_y * np.sin(rad_ang) + 1e-8
            nd_y_1 = od_y * np.cos(rad_ang) - od_x * np.sin(rad_ang) + 1e-8
            # for top-left diag
            nd_x_2 = -od_x * np.cos(rad_ang) + od_y * np.sin(rad_ang) + 1e-8
            nd_y_2 = od_y * np.cos(rad_ang) + od_x * np.sin(rad_ang) + 1e-8
            # Compute maximum allowed scaling such that all the four co-ordintes are between -1 and 1.
            max_scale = min(1./abs(nd_x_1), 1./abs(nd_y_1), 1./abs(nd_x_2), 1./abs(nd_y_2))

            sc_range[0] =  min(sc_range[0],max_scale)
            sc_range[1] =  min(sc_range[1],max_scale)
            sc = np.random.rand()*(sc_range[1] - sc_range[0]) + sc_range[0]


            # New co-ordinates after clockwise roatations
            affineParams = torch.FloatTensor([dg, sh, sc])
            boxAffine = FN.affine(boxCrop, dg, (0,0), sc,sh, resample=Image.BILINEAR)
            boxMaskAffine = torch.FloatTensor(np.asarray(FN.affine(Image.fromarray(boxMask.numpy()[0]), dg, (0,0), sc,sh, resample=Image.NEAREST)))[None,::]
        elif self.apply_randcolor_transform:
            transformed = self.color_augment(image=np.array(image), patch=np.array(boxCrop))
            image = Image.fromarray(transformed['image'])
            boxAffine = Image.fromarray(transformed['patch'])
            boxMaskAffine = boxMask
            affineParams = torch.FloatTensor([0., 0., 0.])
        else:
            affineParams = torch.FloatTensor([0., 0., 0.])


        #if self.bbox_out_size != self.out_image_size:
        #   boxMask = FN.resize(boxMask, (self.bbox_out_size, self.bbox_out_size))

        # Create Mask
        #patch = FN.resized_crop(image, sampbbox[1], sampbbox[0], sampbbox[3],sampbbox[2], (self.bbox_out_size, self.bbox_out_size))

        mask = torch.zeros(1+3*self.getBoxMap, src_img_size[1], src_img_size[0])
        mask[:,sampbbox[1]:sampbbox[1]+sampbbox[3],sampbbox[0]:sampbbox[0]+sampbbox[2]] = 1.
        if self.n_boxes > 1 and len(extra_boxes):
            for box in extra_boxes:
                box = [int(bc*self.out_img_size) for bc in box]
                mask[0,box[1]:box[1]+box[3],box[0]:box[0]+box[2]] = 1.

        if self.boxrotate:
            mask = torch.FloatTensor(np.asarray(self.rotateTrans(Image.fromarray(mask.numpy()[0]))))[None,::]
        if self.use_gt_mask:
            mask = torch.cat([mask, mask], dim=0)

        if self.use_highres_patch:
            image = self.transform[0](image)
            mask = torch.cat([torch.tensor(np.array(self.mask_trans(Image.fromarray(mask[i].numpy().astype(np.uint8)))).astype(np.float),dtype=torch.float32)[None,::] for i in range(mask.shape[0])] ,dim=0)

        classVec = torch.LongTensor([bboxId]) if not self.get_all_inst else torch.LongTensor(np.array([self.valid_ids[index], bboxId[0], annId]))

        #if self.get_kp_class:
        #    sampbbox = sampbbox + self.imid2kpClusts[cocoid, annId]

        final_boxImg  = torch.cat([boxMask*self.transform[-1](boxCrop), boxMaskAffine*self.transform[-1](boxAffine)],dim=0)
        final_boxMask = torch.cat([boxMask, boxMaskAffine],dim=0)
        if self.get_rand_mask:
            boxImgRand, boxMaskRand, _, _ = self.getRandMaskByClass(currCls, image_size=self.out_img_size) #targ_split[data_iter[(c_idx + ridx)%totalItems]]
            final_boxImg  = torch.cat([final_boxImg, boxImgRand], dim=0)
            final_boxMask = torch.cat([final_boxMask, boxMaskRand], dim=0)

        return self.transform[-1](image), \
               torch.FloatTensor(label), \
               final_boxImg, \
               torch.FloatTensor(bboxLabel), \
               mask, \
               torch.IntTensor(sampbbox), \
               classVec, \
               final_boxMask, \
               affineParams

    #Bdd100k
    def getfilename_bycocoid(self, cocoid):
        return os.path.basename(self.dataset['images'][self.imgId2idx[cocoid]]['file_name'])

class PascalDatasetBBoxSample(Dataset):
    def __init__(self, transform, mode, select_attrs=[], datafile='instances_trainval_200712.json', out_img_size=128, bbox_out_size=64,
                 balance_classes=0, onlyrandBoxes=False, max_object_size=0., min_box_size = 16, max_with_union=True, use_gt_mask=False,
                 boxrotate=0, n_fg_boxes = 1, n_extra_boxes= 0, square_resize=0, filter_by_mincooccur = -1., only_indiv_occur = 0., patch_mode=0,
                 use_highres_patch=0, get_all_inst=False, remove_invalid = 0, color_or_affine='affine', get_kp_class=0, get_rand_mask=0,
                 detectorMode=0, canvas_image_size=416, filter_class_instances = []):

        self.transform = transform
        self.mode = mode
        datafile = 'instances_trainval_200712.json' if mode=='train' else 'instances_uniqueval_2012.json'
        self.image_path = os.path.join('data','pascalVoc','images')
        self.n_fg_boxes = n_fg_boxes
        self.n_extra_boxes= n_extra_boxes
        self.dataset = json.load(open(os.path.join('data','pascalVoc','annotations',datafile),'r'))
        self.num_data = len(self.dataset['images'])
        self.attr2idx = {}
        self.idx2attr = {}
        self.catid2attr = {}
        self.out_img_size = out_img_size
        self.square_resize = square_resize
        self.bbox_out_size = bbox_out_size
        self.filter_by_mincooccur = filter_by_mincooccur
        self.only_indiv_occur = only_indiv_occur
        #self.selected_attrs = ['person', 'book', 'car', 'bird', 'chair'] if select_attrs== [] else select_attrs
        self.selected_attrs = select_attrs
        self.get_kp_class = get_kp_class
        self.get_rand_mask = get_rand_mask
        self.detectorMode = detectorMode
        self.canvas_image_size = canvas_image_size

        self.apply_random_affine = True if color_or_affine=='affine' else False
        self.apply_randcolor_transform = True if color_or_affine=='color' else False
        if self.apply_randcolor_transform:
            self.color_augment = Compose([RGBShift(r_shift_limit=30, g_shift_limit=20, b_shift_limit=30), RandomBrightnessContrast(),HueSaturationValue(hue_shift_limit=0, sat_shift_limit=10, val_shift_limit=40), RandomGamma()],p=0.99, additional_targets={'patch':'image'})

        if len(select_attrs) == 0 or (select_attrs[0] == 'all'):
            self.selected_attrs = [attr['name'] for attr in self.dataset['categories']]
        self.filter_class_instances = set(filter_class_instances)
        print('Removing instances of classes ', filter_class_instances)
        self.balance_classes = balance_classes
        self.onlyrandBoxes = onlyrandBoxes
        self.max_object_size = max_object_size
        self.max_with_union= max_with_union
        self.min_box_size = min_box_size
        #self.min_minbox_size = 10
        self.use_gt_mask = use_gt_mask
        self.boxrotate = boxrotate
        self.patch_mode = patch_mode
        self.use_highres_patch = use_highres_patch
        self.remove_invalid= remove_invalid
        if self.boxrotate:
            self.rotateTrans = transforms.Compose([transforms.RandomRotation(boxrotate,resample=Image.NEAREST)])
        if use_gt_mask == 1:
            self.mask_trans =transforms.Compose([transforms.Resize(out_img_size if not square_resize else [out_img_size, out_img_size] , interpolation=Image.NEAREST), transforms.CenterCrop(out_img_size)])

        self.getBoxMap = 0

        if self.getBoxMap:
            self.classTocolor = cm.get_cmap('Spectral')
            self.cNorm = matplotlib.colors.Normalize(vmin=0., vmax=len(self.selected_attrs))
        self.randHFlip = 'Flip' in transform

        print('Start preprocessing dataset..!')
        self.preprocess()
        print('Finished preprocessing dataset..!')
        self.imgId2idx = {imid:i for i,imid in enumerate(self.valid_ids)}

        self.get_all_inst = get_all_inst
        self.num_data = len(self.dataset['images']) if not get_all_inst else len(self.all_imgIdandAnn)

    #PASCALVOC
    def preprocess(self):
        for i, attr in enumerate(self.dataset['categories']):
            self.attr2idx[attr['name']] = i
            self.idx2attr[i] = attr['name']
            self.catid2attr[attr['id']] = attr['name']

        self.sattr_to_idx = {att:i for i, att in enumerate(self.selected_attrs)}

        #--------------------------------------------------------
        # Keep only the required annotations
        # 1. required classes
        # 2. based on occluded and truncated filters
        #--------------------------------------------------------
        selset = set(self.selected_attrs) - self.filter_class_instances
        self.dataset['annotations'] = [ann
                                       for ann in self.dataset['annotations']
                                       if (self.catid2attr[ann['category_id']] in selset)
                                       ]
        self.imidToIndex = {img['id']:i for i,img in enumerate(self.dataset['images'])}

        print('%d annotations left after category filters'%(len(self.dataset['annotations'])))

        fixed_ann = []
        for i, ann in enumerate(self.dataset['annotations']):
            # Correct BBox for Resize(of smaller edge) and CenterCrop
            cbbox = ann['bbox']
            # Largest side of the box will be resized to image_size
            box_size = (cbbox[2], cbbox[3])
            img = self.dataset['images'][self.imidToIndex[ann['image_id']]]
            boxArea = (box_size[0] * box_size[1])/(img['height']*img['width'])

            sc_ratio = min(1.0, self.out_img_size/(float(box_size[0]+1e-8)), self.out_img_size / float(box_size[1]+1e-8))
            new_box_size = (int(box_size[0]*sc_ratio) ,  int(box_size[1]*sc_ratio))
            if (boxArea <= self.max_object_size) and (max(new_box_size) >= self.min_box_size) and (min(new_box_size) >= 3) and (not ann['iscrowd']):# and ((not self.remove_invalid) or ann['valid']):
                ann['box_label'] = np.zeros(max(len(self.selected_attrs),1))
                ann['box_label'][self.sattr_to_idx[self.catid2attr[ann['category_id']]]] = 1.
                ann['cls'] = self.catid2attr[ann['category_id']]
                fixed_ann.append(ann)

        self.dataset['annotations'] = fixed_ann
        print('%d annotations left after size filters'%(len(self.dataset['annotations'])))

        self.catsInImg = defaultdict(set)
        self.imidToAnnList = defaultdict(list)
        self.attToannId = defaultdict(list)
        self.annIdToIdx = {}
        for i, ann in enumerate(self.dataset['annotations']):
            self.imidToAnnList[ann['image_id']].append((ann['cls'], ann['id']))
            self.annIdToIdx[ann['id']] = i
            self.catsInImg[ann['image_id']].add(ann['cls'])
            self.attToannId[ann['cls']].append(ann['id'])

        self.dataset['images'] = [img for img in self.dataset['images'] if img['id'] in self.imidToAnnList and len(self.imidToAnnList[img['id']]) > 0]
        self.valid_ids= []
        self.imidToIndex = {}
        for i,img in enumerate(self.dataset['images']):
            self.dataset['images'][i]['label'] = np.zeros(max(len(self.selected_attrs),1))
            self.catsInImg[img['id']] = list(self.catsInImg[img['id']])
            self.dataset['images'][i]['label'][[self.sattr_to_idx[cat] for cat in self.catsInImg[img['id']]]] = 1
            self.valid_ids.append(img['id'])
            self.imidToIndex[img['id']] = i
        print('%d images left after empty img filtering'%(len(self.valid_ids)))

    #PASCALVOC
    def __getitem__(self, index):
        # In this situation ignore index and sample classes uniformly
        if self.detectorMode:
            returnvals = self.getCanvasImg(index=index, image_size=self.canvas_image_size)
        else:
            if self.get_all_inst:
                index, currCls, annId = self.all_imgIdandAnn[index]
            else:
                if self.balance_classes==1:
                    currCls = random.choice(list(self.attToannId.keys()))
                    annId = random.choice(self.attToannId[currCls])
                    index = self.imidToIndex[self.dataset['annotations'][self.annIdToIdx[annId]]['image_id']]
                else:#if self.balance_classes==2:
                    imid = self.dataset['images'][index]['id']
                    currCls = random.choice(self.catsInImg[imid])

                annId= None

            cid = [self.sattr_to_idx[currCls]] if currCls != 'bg' else [0]
            returnvals = self.getPatchAndCanvasbyIndexAndclass(index, cid, annId=annId)

        return tuple(returnvals)

    #PASCALVOC
    def __len__(self):
        return self.num_data

    #PASCALVOC
    def getfilename(self, index):
        return self.dataset['images'][index]['file_name']

    #PASCALVOC
    def randomBBoxSample(self, imid, max_area = -1, prevBoxList=[], imgSize=[]):
        # With 50% chance sample from background or foreground
        # Minimum size
        minLen = 0.05
        maxLen = 0.3
        maxIou = 0.3
        cbboxList = [self.dataset['annotations'][self.annIdToIdx[aid]] for acls,aid in self.imidToAnnList[imid]]
        n_t = 0
        while 1:
            # sample a random background box
            cbid = None
            tL_x, tL_y = random.uniform(0,1.-minLen-0.01), random.uniform(0,1.-minLen-0.01)
            l_x = random.uniform(minLen, min(1.-tL_x,maxLen))
            l_y = random.uniform(minLen, min(1.-tL_y,maxLen))
            sbox = [tL_x, tL_y, l_x, l_y]
            # Prepare label for this box
            bboxLabel = np.zeros(max(len(self.selected_attrs),1))
            #if len(cbboxList):
            noOverlap = True
            for bb in cbboxList+prevBoxList:
                ibox = [float(sbox[0]*imgSize[0]), float(sbox[1]*imgSize[1]), float(sbox[2]*imgSize[0]), float(sbox[3]*imgSize[1])]
                iou, aInb, bIna = computeIOU(ibox, bb['bbox'])
                if iou > maxIou or aInb >0.8:
                    noOverlap = False
                #if bIna > 0.8 and bb['cid'] > 0:
                #    bboxLabel[self.sattr_to_idx[self.catid2attr[bb['cid']]]] = 1
            if (noOverlap and ((max_area < 0) or ((sbox[2]*sbox[3])< max_area))) or (n_t>10):
                return ibox, bboxLabel, cbid
            n_t += 1

    #PASCALVOC
    def getBboxForClass(self, index, cls, annId=None):
        imid = self.dataset['images'][index]['id']
        if annId is None:
            cbboxList = [self.dataset['annotations'][self.annIdToIdx[aid]] for acls,aid in self.imidToAnnList[imid] if acls==cls]
        else:
            cbboxList = [self.dataset['annotations'][self.annIdToIdx[annId]]]
        if len(cbboxList):
            cbid = random.randrange(len(cbboxList))
            sbox = cbboxList[cbid]
            return copy(sbox['bbox']),sbox['box_label'], sbox
        else:
            print(index, self.dataset['images'][index], cls, annId)
            assert(0)

    #PASCALVOC
    def getCanvasImg(self, cls=None, index=None, image_size=128):
        if type(image_size) is not list:
            image_size = [image_size, image_size]
        if cls is None and index is None:
            assert(0)
        elif (cls is None) and (index is not None):
            imgId = self.dataset['images'][index]['id']
            cls = random.choice(self.catsInImg[imgId])
            annId = None
        elif index is None:
            annId = random.choice(self.attToannId[cls])
            index = self.imidToIndex[self.dataset['annotations'][self.annIdToIdx[annId]]['image_id']]

        #----------------------------------------------------
        # Read and load the image
        #----------------------------------------------------
        imD = self.dataset['images'][index]
        image = Image.open(os.path.join(self.image_path, imD['file_path'], imD['file_name']))
        if image.mode != 'RGB':
            #print image.mode
            image = image.convert('RGB')
        c_imgsize = image.size
        src_img_size = c_imgsize
        out_img_size = (self.out_img_size, self.out_img_size)

        #----------------------------------------------------
        # Sample the required number of boxes
        #----------------------------------------------------
        all_boxes = []
        all_boxData = []
        all_bboxlabel = []
        n_boxes = self.n_fg_boxes + self.n_extra_boxes
        imid = self.dataset['images'][index]['id']
        sampAnnList = set()
        j=0
        for i in range(n_boxes):
            remAnnList = [i for i in range(len(self.imidToAnnList[imid])) if self.imidToAnnList[imid][i][1] not in sampAnnList]
            if len(remAnnList):
                ei = random.choice(remAnnList)
                clsExtra = self.imidToAnnList[imid][ei][0]
                aidExtra = self.imidToAnnList[imid][ei][1]
                ebox, ebLabel, eboxData = self.getBboxForClass(index, clsExtra, annId=aidExtra)
                sampAnnList.add(eboxData['id'])
                all_boxData.append(eboxData)
            else:
                ebox, ebLabel, eboxData = self.randomBBoxSample(imid, max_area=0.3, prevBoxList=all_boxData, imgSize=src_img_size) # Extra 10% to make the sampling easier
                all_boxData.append({'bbox':ebox, 'cid':-1})
            all_boxes.append(ebox)
            all_bboxlabel.append(ebLabel)

        e_boxes = all_boxes[self.n_fg_boxes:]
        fg_boxes = all_boxes[:self.n_fg_boxes]

        allBoxImgs = []
        allBoxMask = []
        for i,fgb in enumerate(fg_boxes):
            sampbbox = [int(bc) for bc in fgb]
            boxCrop = np.zeros((out_img_size[1], out_img_size[0],3))
            boxMask = torch.zeros(1, out_img_size[1], out_img_size[0])
            src_patch = np.array(image)[sampbbox[1]:sampbbox[1]+sampbbox[3], sampbbox[0]:sampbbox[0]+sampbbox[2],:]
            src_mask = np.ones(list(src_patch.shape[:2]))

            scale_ratio = min(1.0, float(out_img_size[0])/float(sampbbox[2]), float(out_img_size[1])/sampbbox[3])
            targ_boxsize = [int(sampbbox[2]*scale_ratio), int(sampbbox[3]*scale_ratio)]
            targ_boxloc = [out_img_size[0]//2 - targ_boxsize[0]//2, # Center the patch in the destination
                           out_img_size[1]//2 - targ_boxsize[1]//2]
            src_patch = np.array(FN.resize(Image.fromarray(src_patch), (targ_boxsize[1],targ_boxsize[0])))
            src_mask = torch.tensor(np.array(FN.resize(Image.fromarray(src_mask), (targ_boxsize[1],targ_boxsize[0]), interpolation=Image.NEAREST)))

            boxCrop[targ_boxloc[1]: targ_boxloc[1]+targ_boxsize[1], targ_boxloc[0]: targ_boxloc[0]+targ_boxsize[0],:] = src_patch
            boxMask[0,targ_boxloc[1]: targ_boxloc[1]+targ_boxsize[1], targ_boxloc[0]: targ_boxloc[0]+targ_boxsize[0]] = src_mask
            boxCrop = Image.fromarray(boxCrop.astype(np.uint8))
            allBoxImgs.append(boxCrop)
            allBoxMask.append(boxMask)

        boxImgOut = torch.cat([(allBoxMask[i]*self.transform[-1](allBoxImgs[i])).unsqueeze(0) for i in range(len(allBoxImgs))], dim=0)
        boxMaskOut = torch.stack(allBoxMask)
        bboxLabelOut = torch.FloatTensor(np.stack(all_bboxlabel[:self.n_fg_boxes]))

        #if self.get_kp_class:
        #    sampbboxOrig = sampbboxOrig + self.imid2kpClusts[cocoid, annId]
        mask = torch.zeros(1, src_img_size[1], src_img_size[0])
        for ebox in all_boxes:
            mask[:,int(ebox[1]):int(ebox[1]+ebox[3]),int(ebox[0]):int(ebox[0]+ebox[2])] = 1.
        if self.use_highres_patch:
            gain = float(max(image_size)) / float(max(src_img_size))
            canvas_image_size = (int(src_img_size[0]*gain), int(src_img_size[1]*gain))
            #image = FN.resize(image, (canvas_image_size[1], canvas_image_size[0]))
            image = Image.fromarray(cv2.resize(np.array(image), (canvas_image_size[0], canvas_image_size[1]), interpolation=cv2.INTER_AREA))
            mask = torch.tensor(np.array(FN.resize(Image.fromarray(mask[0].numpy().astype(np.uint8)), (canvas_image_size[1],canvas_image_size[0]), interpolation=Image.NEAREST )).astype(np.float),dtype=torch.float32)[None,::]
            if self.use_gt_mask:
                mask = torch.cat([mask, mask], dim=0)

        #import ipdb; ipdb.set_trace()#src_img_size
        sampbbox = []
        for ebox in all_boxes:
            sampbbox.append([float(ebox[0])/src_img_size[0], float(ebox[1])/src_img_size[1], float(ebox[2])/src_img_size[0], float(ebox[3])/src_img_size[1]])

        image = self.transform[-1](image)[None,::]
        cs = image.shape[2:]

        # TODO
        ns = image_size
        pad = [(ns[1]-cs[1])//2, math.ceil((ns[1]-cs[1])/2),(ns[0]-cs[0])//2, math.ceil((ns[0]-cs[0])/2)]
        image = torch.nn.functional.pad(image, pad, value=0.)[0]
        mask = torch.nn.functional.pad(mask[None,::], pad, value=0.)[0]

        return image, mask, torch.FloatTensor(pad), torch.IntTensor(list(cs)), boxImgOut, boxMaskOut, torch.FloatTensor(sampbbox), bboxLabelOut, torch.IntTensor([index])

    #PASCALVOC
    def getRandMaskByClass(self, cls=None, index=None, get_img=False, image_size=128, n_fg_boxes=1, n_extra_boxes=0):
        if type(image_size) is not list:
            image_size = [image_size, image_size]
        if cls is None and index is None:
            assert(0)
        elif (cls is None) and (index is not None):
            imgId = self.dataset['images'][index]['id']
            cls = random.choice(self.catsInImg[imgId])
            annId = None
        elif index is None:
            annId = random.choice(self.attToannId[cls])
            index = self.imidToIndex[self.dataset['annotations'][self.annIdToIdx[annId]]['image_id']]

        imD = self.dataset['images'][index]
        image = Image.open(os.path.join(self.image_path, imD['file_name']))
        if image.mode != 'RGB':
            #print image.mode
            image = image.convert('RGB')
        c_imgsize = image.size

        src_img_size = c_imgsize
        out_img_size = (self.out_img_size, self.out_img_size)


        sampbboxOrig, bboxLabel, sbox = self.getBboxForClass(index, cls, annId=annId)
        all_boxes = [sampbboxOrig]
        all_boxData = [sbox]
        all_bboxlabel = [bboxLabel]
        n_boxes = n_fg_boxes + n_extra_boxes
        if n_boxes > 1:
            sampAnnList = set([sbox['id']])
            imid = self.dataset['images'][index]['id']
            for i in range(n_boxes -1):
                remAnnList = [i for i in range(len(self.imidToAnnList[imid])) if self.imidToAnnList[imid][i][1] not in sampAnnList]
                if len(remAnnList):
                    ei = random.choice(remAnnList)
                    clsExtra = self.imidToAnnList[imid][ei][0]
                    aidExtra = self.imidToAnnList[imid][ei][1]
                    ebox, ebLabel, eboxData = self.getBboxForClass(index, clsExtra, annId=aidExtra)
                    sampAnnList.add(eboxData['id'])
                else:
                    ebox, ebLabel, eboxData = self.randomBBoxSample(imid, max_area=0.2, prevBoxList=all_boxes) # Extra 10% to make the sampling easier
                all_boxes.append(ebox)
                all_boxData.append(eboxData)
                all_bboxlabel.append(ebLabel)
        e_boxes = all_boxes[n_fg_boxes:]
        fg_boxes = all_boxes[:n_fg_boxes]

        allBoxImgs = []
        allBoxMask = []
        for i,fgb in enumerate(fg_boxes):
            sampbbox = [int(bc) for bc in fgb]
            boxCrop = np.zeros((out_img_size[1], out_img_size[0],3))
            boxMask = torch.zeros(1, out_img_size[1], out_img_size[0])
            src_patch = np.array(image)[sampbbox[1]:sampbbox[1]+sampbbox[3], sampbbox[0]:sampbbox[0]+sampbbox[2],:]
            src_mask = np.ones(list(src_patch.shape[:2]))

            scale_ratio = min(1.0, float(out_img_size[0])/float(sampbbox[2]), float(out_img_size[1])/sampbbox[3])
            targ_boxsize = [int(sampbbox[2]*scale_ratio), int(sampbbox[3]*scale_ratio)]
            targ_boxloc = [out_img_size[0]//2 - targ_boxsize[0]//2, # Center the patch in the destination
                           out_img_size[1]//2 - targ_boxsize[1]//2]
            src_patch = np.array(FN.resize(Image.fromarray(src_patch), (targ_boxsize[1],targ_boxsize[0])))
            src_mask = torch.tensor(np.array(FN.resize(Image.fromarray(src_mask), (targ_boxsize[1],targ_boxsize[0]), interpolation=Image.NEAREST)))

            boxCrop[targ_boxloc[1]: targ_boxloc[1]+targ_boxsize[1], targ_boxloc[0]: targ_boxloc[0]+targ_boxsize[0],:] = src_patch
            boxMask[0,targ_boxloc[1]: targ_boxloc[1]+targ_boxsize[1], targ_boxloc[0]: targ_boxloc[0]+targ_boxsize[0]] = src_mask
            boxCrop = Image.fromarray(boxCrop.astype(np.uint8))
            allBoxImgs.append(boxCrop)
            allBoxMask.append(boxMask)

        boxImgOut = torch.cat([(allBoxMask[i]*self.transform[-1](allBoxImgs[i])).unsqueeze(0) for i in range(len(allBoxImgs))], dim=0)
        boxMaskOut = torch.stack(allBoxMask)
        bboxLabelOut = torch.FloatTensor(np.stack(all_bboxlabel[:n_fg_boxes]))

        #if self.get_kp_class:
        #    sampbboxOrig = sampbboxOrig + self.imid2kpClusts[cocoid, annId]
        if get_img:
            mask = torch.zeros(1, src_img_size[1], src_img_size[0])
            for ebox in all_boxes:
                mask[:,int(ebox[1]):int(ebox[1]+ebox[3]),int(ebox[0]):int(ebox[0]+ebox[2])] = 1.

            if self.use_highres_patch:
                out_image_size = (int(src_img_size[0]*image_size[1] / src_img_size[0]), int(src_img_size[1]*image_size[0] / src_img_size[1]))
                image = FN.resize(image, (out_image_size[1], out_image_size[0]))
                mask = torch.tensor(np.array(FN.resize(Image.fromarray(mask[0].numpy().astype(np.uint8)), (out_image_size[1],out_image_size[0]), interpolation=Image.NEAREST )).astype(np.float),dtype=torch.float32)[None,::]
                if self.use_gt_mask:
                    mask = torch.cat([mask, mask], dim=0)

            #import ipdb; ipdb.set_trace()#src_img_size
            sampbbox = []
            for ebox in all_boxes:
                sampbbox.append([float(ebox[0])/src_img_size[0], float(ebox[1])/src_img_size[1], float(ebox[2])/src_img_size[0], float(ebox[3])/src_img_size[1]])

            return boxImgOut, boxMaskOut, index, self.transform[-1](image), mask, torch.FloatTensor(sampbbox), bboxLabelOut
        else:
            # Create Mask
            #patch = FN.resized_crop(image, sampbbox[1], sampbbox[0], sampbbox[3],sampbbox[2], (self.bbox_out_size, self.bbox_out_size))
            return boxImgOut, boxMaskOut, index, torch.FloatTensor(bboxLabel)

    #PASCALVOC
    def getPatchAndCanvasbyIndexAndclass(self, index, cid, annId=None):
        imD = self.dataset['images'][index]
        image = Image.open(os.path.join(self.image_path, imD['file_name']))

        if annId is None:
            _, annId = random.choice(self.imidToAnnList[imD['id']])
        annIndex = self.annIdToIdx[annId]


        currCls = self.selected_attrs[cid[0]]
        if image.mode != 'RGB':
            #print image.mode
            image = image.convert('RGB')
        c_imgsize = image.size

        sampbbox, bboxLabel, sbox = self.getBboxForClass(index, currCls, annId=annId)
        annId = sbox['id']
        cocoid = self.dataset['images'][index]['id']
        bboxId = cid

        label = self.dataset['images'][index]['label']

        # Apply transforms to the image.
        if not self.use_highres_patch:
            image = self.transform[0](image)
            src_img_size = (self.out_img_size, self.out_img_size)
            out_img_size = (self.out_img_size, self.out_img_size)
        else:
            src_img_size = c_imgsize#self.dataset['images'][index]['imgSize']
            out_img_size = (self.out_img_size, self.out_img_size)

        # Now do the flipping
        hflip = 0
        if self.randHFlip and random.random()>0.5:
            hflip = 1
            image = FN.hflip(image)
            sampbbox[0] = c_imgsize[0]-(sampbbox[0]+sampbbox[2])

        #Convert BBox to actual co-ordinates
        sampbbox = [int(bc) for bc in sampbbox]
        # Prepare the patch, no Resizing now
        boxCrop = np.zeros((out_img_size[1], out_img_size[0],3))
        boxMask = torch.zeros(1, out_img_size[1], out_img_size[0])
        src_patch = np.array(image)[sampbbox[1]:sampbbox[1]+sampbbox[3], sampbbox[0]:sampbbox[0]+sampbbox[2],:]
        src_mask = np.ones(list(src_patch.shape[:2]))

        scale_ratio = min(1.0, float(out_img_size[0])/float(sampbbox[2]), float(out_img_size[1])/sampbbox[3])
        targ_boxsize = [int(sampbbox[2]*scale_ratio), int(sampbbox[3]*scale_ratio)]
        targ_boxloc = [out_img_size[0]//2 - targ_boxsize[0]//2, # Center the patch in the destination
                       out_img_size[1]//2 - targ_boxsize[1]//2]
        src_patch = np.array(FN.resize(Image.fromarray(src_patch), (targ_boxsize[1],targ_boxsize[0])))
        src_mask = torch.tensor(np.array(FN.resize(Image.fromarray(src_mask), (targ_boxsize[1],targ_boxsize[0]), interpolation=Image.NEAREST)))

        boxCrop[targ_boxloc[1]: targ_boxloc[1]+targ_boxsize[1], targ_boxloc[0]: targ_boxloc[0]+targ_boxsize[0],:] = src_patch
        boxMask[0,targ_boxloc[1]: targ_boxloc[1]+targ_boxsize[1], targ_boxloc[0]: targ_boxloc[0]+targ_boxsize[0]] = src_mask
        boxCrop = Image.fromarray(boxCrop.astype(np.uint8))


        #patch = FN.resized_crop(image, sampbbox[1], sampbbox[0], sampbbox[3],sampbbox[2], (self.bbox_out_size, self.bbox_out_size))
        # Now resize to bboxImage and mask to the right sizes

        if self.apply_random_affine:
            # max_scale, min_scale
            min_box_size = 64#self.min_box_size
            sc_range = [ np.maximum(float(min_box_size)   /float(targ_boxsize[1]), float(min_box_size)   /float(targ_boxsize[0])),
                         np.minimum(float(out_img_size[1])/float(targ_boxsize[1]), float(out_img_size[0])/float(targ_boxsize[0]))]
            #sc = 0.5
            dg = np.random.randint(-180,180)
            #dg = np.random.randint(-45,45)
            sh = np.random.randint(-30,30)
            #sh = np.random.randint(-45,45)
            # -----------------------------------------------------------------------
            # check if the object goes out of bounds when rotated and scale appropriately.
            # Do all computations in normalized co-ordinates.
            # top right = (1,1), bottom left = (-1,-1), center = (0,0)
            # -----------------------------------------------------------------------
            # compute co-ordinates of top-right diagonal of the box
            od_x, od_y = targ_boxsize[0]/out_img_size[0], targ_boxsize[1]/out_img_size[1]
            # new co-ordinates of the diag is given as:
            rad_ang = np.radians(dg)
            # for top-right diag
            nd_x_1 = od_x * np.cos(rad_ang) + od_y * np.sin(rad_ang) + 1e-8
            nd_y_1 = od_y * np.cos(rad_ang) - od_x * np.sin(rad_ang) + 1e-8
            # for top-left diag
            nd_x_2 = -od_x * np.cos(rad_ang) + od_y * np.sin(rad_ang) + 1e-8
            nd_y_2 = od_y * np.cos(rad_ang) + od_x * np.sin(rad_ang) + 1e-8
            # Compute maximum allowed scaling such that all the four co-ordintes are between -1 and 1.
            max_scale = min(1./abs(nd_x_1), 1./abs(nd_y_1), 1./abs(nd_x_2), 1./abs(nd_y_2))

            sc_range[0] =  min(sc_range[0],max_scale)
            sc_range[1] =  min(sc_range[1],max_scale)
            sc = np.random.rand()*(sc_range[1] - sc_range[0]) + sc_range[0]


            # New co-ordinates after clockwise roatations
            affineParams = torch.FloatTensor([dg, sh, sc])
            boxAffine = FN.affine(boxCrop, dg, (0,0), sc,sh, resample=Image.BILINEAR)
            boxMaskAffine = torch.FloatTensor(np.asarray(FN.affine(Image.fromarray(boxMask.numpy()[0]), dg, (0,0), sc,sh, resample=Image.NEAREST)))[None,::]
        elif self.apply_randcolor_transform:
            transformed = self.color_augment(image=np.array(image), patch=np.array(boxCrop))
            image = Image.fromarray(transformed['image'])
            boxAffine = Image.fromarray(transformed['patch'])
            boxMaskAffine = boxMask
            affineParams = torch.FloatTensor([0., 0., 0.])
        else:
            affineParams = torch.FloatTensor([0., 0., 0.])


        #if self.bbox_out_size != self.out_image_size:
        #   boxMask = FN.resize(boxMask, (self.bbox_out_size, self.bbox_out_size))

        # Create Mask
        #patch = FN.resized_crop(image, sampbbox[1], sampbbox[0], sampbbox[3],sampbbox[2], (self.bbox_out_size, self.bbox_out_size))

        mask = torch.zeros(1+3*self.getBoxMap, src_img_size[1], src_img_size[0])
        mask[:,sampbbox[1]:sampbbox[1]+sampbbox[3],sampbbox[0]:sampbbox[0]+sampbbox[2]] = 1.
        if self.n_boxes > 1 and len(extra_boxes):
            for box in extra_boxes:
                box = [int(bc*self.out_img_size) for bc in box]
                mask[0,box[1]:box[1]+box[3],box[0]:box[0]+box[2]] = 1.

        if self.boxrotate:
            mask = torch.FloatTensor(np.asarray(self.rotateTrans(Image.fromarray(mask.numpy()[0]))))[None,::]
        if self.use_gt_mask:
            mask = torch.cat([mask, mask], dim=0)

        if self.use_highres_patch:
            image = self.transform[0](image)
            mask = torch.cat([torch.tensor(np.array(self.mask_trans(Image.fromarray(mask[i].numpy().astype(np.uint8)))).astype(np.float),dtype=torch.float32)[None,::] for i in range(mask.shape[0])] ,dim=0)

        classVec = torch.LongTensor([bboxId]) if not self.get_all_inst else torch.LongTensor(np.array([self.valid_ids[index], bboxId[0], annId]))

        #if self.get_kp_class:
        #    sampbbox = sampbbox + self.imid2kpClusts[cocoid, annId]

        final_boxImg  = torch.cat([boxMask*self.transform[-1](boxCrop), boxMaskAffine*self.transform[-1](boxAffine)],dim=0)
        final_boxMask = torch.cat([boxMask, boxMaskAffine],dim=0)
        if self.get_rand_mask:
            boxImgRand, boxMaskRand, _, _ = self.getRandMaskByClass(currCls, image_size=self.out_img_size) #targ_split[data_iter[(c_idx + ridx)%totalItems]]
            final_boxImg  = torch.cat([final_boxImg, boxImgRand], dim=0)
            final_boxMask = torch.cat([final_boxMask, boxMaskRand], dim=0)

        return self.transform[-1](image), \
               torch.FloatTensor(label), \
               final_boxImg, \
               torch.FloatTensor(bboxLabel), \
               mask, \
               torch.IntTensor(sampbbox), \
               classVec, \
               final_boxMask, \
               affineParams

    #PASCALVOC
    def getfilename_bycocoid(self, cocoid):
        return os.path.basename(self.dataset['images'][self.imgId2idx[cocoid]]['file_name'])

class ADE20k(Dataset):
    def __init__(self, transform, split, select_attrs=[], out_img_size=128, bbox_out_size=64,
                 max_object_size=0., max_with_union=True, use_gt_mask=False,
                 boxrotate=0, n_boxes = 1, square_resize=0) :
        self.image_path = os.path.join('data','ade20k')
        self.transform = transform
        self.split = split
        self.n_boxes = n_boxes
        self.iouThresh = 0.5
        datafile = 'train.odgt' if split == 'train' else 'validation.odgt'
        self.datafile = os.path.join('data','ade20k',datafile)
        self.dataset = [json.loads(x.rstrip()) for x in open(self.datafile, 'r')]
        self.num_data = len(self.dataset)
        clsData = open('data/ade20k/object150_info.csv','r').read().splitlines()
        self.clsidx2attr = {i:ln.split(',')[-1] for i, ln in enumerate(clsData[1:])}
        self.clsidx2Stuff = {i:int(ln.split(',')[-2]) for i, ln in enumerate(clsData[1:])}
        self.validCatIds = set([i for i in self.clsidx2Stuff if not self.clsidx2Stuff[i]])
        self.maskSample = 'nonStuff'
        self.out_img_size = out_img_size
        self.square_resize = square_resize
        self.bbox_out_size = bbox_out_size
        #self.selected_attrs = ['person', 'book', 'car', 'bird', 'chair'] if select_attrs== [] else select_attrs
        self.selected_attrs = ['background']
        self.max_object_size = max_object_size
        self.max_with_union= max_with_union
        self.use_gt_mask = use_gt_mask
        self.boxrotate = boxrotate
        if self.boxrotate:
            self.rotateTrans = transforms.Compose([transforms.RandomRotation(boxrotate,resample=Image.NEAREST)])
        if use_gt_mask == 1:
            self.mask_transform = transforms.Compose([transforms.Resize(out_img_size if not square_resize else [out_img_size, out_img_size] , interpolation=Image.NEAREST), transforms.CenterCrop(out_img_size)])

        self.valid_ids = []
        for i,img in enumerate(self.dataset):
            imid = int(os.path.basename(img['fpath_img']).split('.')[0].split('_')[-1])
            self.dataset[i]['image_id'] = imid
            self.valid_ids.append(imid)

        self.randHFlip = 'Flip' in transform

        print('Start preprocessing dataset..!')
        print('Finished preprocessing dataset..!')
        self.imgId2idx = {imid:i for i,imid in enumerate(self.valid_ids)}

    #ADE20k
    def randomBBoxSample(self, max_area = -1):
        # With 50% chance sample from background or foreground
        # Minimum size
        minLen = 0.1
        maxLen = 0.7
        maxIou = 0.3
        cbboxList = []
        n_t = 0
        while 1:
            # sample a random background box
            cbid = None
            tL_x, tL_y = random.uniform(0,1.-minLen-0.01), random.uniform(0,1.-minLen-0.01)
            l_x = random.uniform(minLen, min(1.-tL_x,maxLen))
            l_y = random.uniform(minLen, min(1.-tL_y,maxLen))
            sbox = [tL_x, tL_y, l_x, l_y]
            # Prepare label for this box
            bboxLabel = np.zeros(max(len(self.selected_attrs),1))
            #if len(cbboxList):
            if ((max_area < 0) or ((sbox[2]*sbox[3])< max_area) or (n_t>5)):
                return sbox, bboxLabel, cbid
            n_t += 1

    #ADE20k
    def __getitem__(self, index):
        # In this situation ignore index and sample classes uniformly
        returnvals = self.getbyIndexAndclass(index)

        return tuple(returnvals)

    #ADE20k
    def getbyIdAndclass(self, imgid, cls, hflip=0):
        index = self.imgId2idx[imgid]
        cid = [self.sattr_to_idx[cls]] if cls != 'bg' else [0]
        returnvals = self.getbyIndexAndclass(index, cid)
        return tuple(returnvals)

    #ADE20k
    def getbyIndexAndclass(self, index,cls=None):

        imgDb = self.dataset[index]
        image_id = imgDb['image_id']
        image = Image.open(os.path.join(self.image_path,imgDb['fpath_img']))
        if image.mode != 'RGB':
            #print image.mode
            image = image.convert('RGB')
        cid = [0]

        sampbbox, bboxLabel, cbid = self.randomBBoxSample(0.5)
        extra_boxes = []
        if self.n_boxes > 1:
            # Sample random number of boxes between 1 and n_boxes
            c_nbox = np.random.randint(0,self.n_boxes)
            c_area = sampbbox[2]*sampbbox[3]
            for i in xrange(c_nbox):
                # Also stop at total area > 50%
                if c_area < 0.5:
                    bsamp, _, _ = self.randomBBoxSample(0.6-c_area) # Extra 10% to make the sampling easier
                    extra_boxes.append(bsamp)
                    c_area += bsamp[2]*bsamp[3]
                else:
                    break

        label = np.ones(max(len(self.selected_attrs),1))

        # Apply transforms to the image.
        image = self.transform[0](image)
        # Now do the flipping
        hflip = 0
        if self.randHFlip and random.random()>0.5:
            hflip = 1
            image = FN.hflip(image)
            sampbbox[0] = 1.0-(sampbbox[0]+sampbbox[2])
        if self.use_gt_mask==1:
            # Use GT masks as input
            gtMask = self.getGTMaskInp(index, hflip=hflip)

        #Convert BBox to actual co-ordinates
        sampbbox = [int(bc*self.out_img_size) for bc in sampbbox]
        boxCrop = FN.resized_crop(image, sampbbox[1], sampbbox[0], sampbbox[3],sampbbox[2], (self.bbox_out_size, self.bbox_out_size))
        # Create Mask
        mask = torch.zeros(1,self.out_img_size,self.out_img_size)
        mask[0,sampbbox[1]:sampbbox[1]+sampbbox[3],sampbbox[0]:sampbbox[0]+sampbbox[2]] = 1.
        if self.n_boxes > 1 and len(extra_boxes):
            for box in extra_boxes:
                box = [int(bc*self.out_img_size) for bc in box]
                mask[0,box[1]:box[1]+box[3],box[0]:box[0]+box[2]] = 1.

        if self.boxrotate:
            mask = torch.FloatTensor(np.asarray(self.rotateTrans(Image.fromarray(mask.numpy()[0]))))[None,::]
        if self.use_gt_mask:
            mask = torch.cat([mask, gtMask], dim=0)

        return self.transform[-1](image), torch.FloatTensor(label), torch.FloatTensor(bboxLabel), torch.FloatTensor(bboxLabel), mask, torch.IntTensor(sampbbox), torch.LongTensor(cid)

    #ADE20k
    def __len__(self):
        return self.num_data

    #ADE20k
    def getfilename(self, index):
        return os.path.basename(self.dataset[index]['fpath_img'])

    #ADE20k
    def getfilename_bycocoid(self, cocoid):
        return os.path.basename(self.dataset[self.imgId2idx[cocoid]]['fpath_img'])

    #ADE20k
    def getcocoid(self, index):
        return self.dataset[index]['image_id']

    #ADE20k
    def getGTMaskInp(self, index, cls=None, hflip=False, mask_type=None):
        imgDb = self.dataset[index]
        segmImg = np.array(Image.open(os.path.join(self.image_path,imgDb['fpath_segm'])))-1
        presentClass = np.unique(segmImg)
        validClass = map(lambda x: x in self.validCatIds, presentClass)
        chosenIdx = np.random.choice(presentClass[validClass]) if np.sum(validClass) > 0 else -10
        if chosenIdx < 0:
            maskTotal = np.zeros((self.out_img_size,self.out_img_size))
            sampbbox, bboxLabel, cbid = self.randomBBoxSample(0.5)
            sampbbox = [int(bc*self.out_img_size) for bc in sampbbox]
            maskTotal[sampbbox[1]:sampbbox[1]+sampbbox[3],sampbbox[0]:sampbbox[0]+sampbbox[2]] = 1.
        else:
            maskTotal = (segmImg == chosenIdx).astype(np.float)
        if hflip:
            maskTotal = maskTotal[:,::-1]

        mask = torch.FloatTensor(np.asarray(self.mask_transform(Image.fromarray(np.clip(maskTotal,0,1)))))[None,::]

        return mask

class CocoMaskDataset(Dataset):
    def __init__(self, transform, mode, select_attrs=[], balance_classes=0, n_masks_perclass=-1):
        self.data_path = os.path.join('data','coco')
        self.transform = transform if transform is not None else lambda x: x
        self.mode = mode
        filename = 'instances_train2014.json' if mode=='train' else  'instances_val2014.json'
        self.dataset =  COCOTool(os.path.join(self.data_path, filename))
        self.selected_attrs = ['person', 'book', 'car', 'bird', 'chair'] if select_attrs== [] else select_attrs
        valid_ids = []
        for catid in self.dataset.getCatIds(self.selected_attrs):
            valid_ids.extend(self.dataset.getImgIds(catIds=catid))
        self.valid_ids = list(set(valid_ids))
        self.imgId2idx = {imid:i for i,imid in enumerate(self.valid_ids)}
        self.num_data = len(self.valid_ids)
        self.attr2idx = {}
        self.idx2attr = {}
        self.catid2attr = {}
        self.nc = 1
        self.balance_classes = balance_classes
        self.n_masks_perclass = n_masks_perclass

        self.preprocess()
        print('Loaded Mask Data')

    #COCOMaskDataset
    def preprocess(self):
        for atid in self.dataset.cats:
            self.catid2attr[self.dataset.cats[atid]['id']] = self.dataset.cats[atid]['name']

        self.sattr_to_idx = {att:i for i, att in enumerate(self.selected_attrs)}
        self.labels = {}
        self.catsInImg = {}
        self.imgSizes = {}
        self.validCatIds = self.dataset.getCatIds(self.selected_attrs)

        selset = set(self.selected_attrs)
        for i, imgid in enumerate(self.valid_ids):
            self.labels[i] = np.zeros(len(selset))
            self.labels[i][[self.sattr_to_idx[self.catid2attr[ann['category_id']]] for ann in self.dataset.imgToAnns[imgid] if self.catid2attr[ann['category_id']] in selset]] = 1.
            self.catsInImg[i] = list(set([ann['category_id'] for ann in self.dataset.imgToAnns[imgid] if self.catid2attr[ann['category_id']] in selset]))
            self.imgSizes[i] =  [self.dataset.imgs[imgid]['height'], self.dataset.imgs[imgid]['width']]


        if self.balance_classes:
            self.attToImgId = defaultdict(set)
            for i, imgid in enumerate(self.valid_ids):
                if len(self.catsInImg[i]):
                    for attid in self.catsInImg[i]:
                        self.attToImgId[self.catid2attr[attid]].add(i)
                else:
                    import ipdb; ipdb.set_trace()

            self.attToImgId = {k:list(v) for k,v in self.attToImgId.items()}
            for ann in self.attToImgId:
                    shuffle(self.attToImgId[ann])
            if self.n_masks_perclass >0:
                self.attToImgId = {k:v[:self.n_masks_perclass] for k,v in self.attToImgId.items()}

    #COCOMaskDataset
    def __getitem__(self, index):
        #image = Image.open(os.path.join(self.image_path,self.dataset['images'][index]['filepath'], self.dataset['images'][index]['filename']))
        # In this situation ignore index and sample classes uniformly
        if self.balance_classes:
            currCls = random.choice(self.selected_attrs)
            index = random.choice(self.attToImgId[currCls])

        maskTotal = np.zeros((self.imgSizes[index][0], self.imgSizes[index][1]))
        label = np.zeros(len(self.selected_attrs))
        if len(self.catsInImg[index]):
            # Randomly sample an annotation
            currObjId = random.choice(self.catsInImg[index])
            for ann in self.dataset.loadAnns(self.dataset.getAnnIds(self.valid_ids[index], currObjId)):
                cm = self.dataset.annToMask(ann)
                maskTotal[:cm.shape[0],:cm.shape[1]] += cm
            label[self.sattr_to_idx[self.catid2attr[currObjId]]] = 1.

        mask = torch.FloatTensor(np.asarray(self.transform(Image.fromarray(np.clip(maskTotal,0,1)))))[None,::]

        return mask, torch.FloatTensor(label)

    #COCOMaskDataset
    def __len__(self):
        return self.num_data

    #COCOMaskDataset
    def getfilename(self, index):
        return self.dataset['images'][index]['filename']

    #COCOMaskDataset
    def getbyIdAndclass(self, imgid, cls, hflip=0):
        if (imgid not in self.imgId2idx):
            maskTotal = np.zeros((128,128))
        else:
            index= self.imgId2idx[imgid]
            catId = self.dataset.getCatIds(cls)
            maskTotal = np.zeros((self.imgSizes[index][0], self.imgSizes[index][1]))
            if len(self.catsInImg[index]) and (catId[0] in self.catsInImg[index]):
                # Randomly sample an annotation
                for ann in self.dataset.loadAnns(self.dataset.getAnnIds(self.valid_ids[index], catId)):
                    cm = self.dataset.annToMask(ann)
                    maskTotal[:cm.shape[0],:cm.shape[1]] += cm
            if hflip:
                maskTotal = maskTotal[:,::-1]

        mask = torch.FloatTensor(np.asarray(self.transform(Image.fromarray(np.clip(maskTotal,0,1)))))[None,::]

        return mask

    #COCOMaskDataset
    def getbyImgAnnId(self, imgid, annId, hflip=0):
        index= self.imgId2idx[imgid]
        maskTotal = np.zeros((self.imgSizes[index][0], self.imgSizes[index][1]))
        ann = self.dataset.loadAnns(annId)
        cm = self.dataset.annToMask(ann[0])
        maskTotal[:cm.shape[0],:cm.shape[1]] += cm
        if hflip:
            maskTotal = maskTotal[:,::-1]
        mask = torch.FloatTensor(np.asarray(self.transform(Image.fromarray(np.clip(maskTotal,0,1)))))[None,::]
        return mask

    #COCOMaskDataset
    def getbyClass(self, cls):
        allMasks = []
        for c in cls:
            curr_obj = self.selected_attrs[c]
            catId = self.dataset.getCatIds(curr_obj)
            index = random.choice(self.attToImgId[curr_obj])
            maskTotal = np.zeros((self.imgSizes[index][0], self.imgSizes[index][1]))
            if len(self.catsInImg[index]):
                # Randomly sample an annotation
                for ann in self.dataset.loadAnns(self.dataset.getAnnIds(self.valid_ids[index], catId)):
                    cm = self.dataset.annToMask(ann)
                    maskTotal[:cm.shape[0],:cm.shape[1]] += cm
            maskTotal = torch.FloatTensor(np.asarray(self.transform(Image.fromarray(np.clip(maskTotal,0,1)))))[None,::]
            allMasks.append(maskTotal[None,::])

        return torch.cat(allMasks,dim=0)

    #COCOMaskDataset
    def getbyIdAndclassBatch(self, imgid, cls, hFlips = None):
        allMasks = []
        for i,c in enumerate(cls):
            curr_obj = self.selected_attrs[c]
            catId = self.dataset.getCatIds(curr_obj)
            if (imgid[i] not in self.imgId2idx):
                maskTotal = np.zeros((128,128))
            else:
                index = self.imgId2idx[imgid[i]]
                maskTotal = np.zeros((self.imgSizes[index][0], self.imgSizes[index][1]))
                if len(self.catsInImg[index]) and (catId[0] in self.catsInImg[index]):
                    # Randomly sample an annotation
                    for ann in self.dataset.loadAnns(self.dataset.getAnnIds(imgid[i], catId)):
                        cm = self.dataset.annToMask(ann)
                        maskTotal[:cm.shape[0],:cm.shape[1]] += cm
                if (hFlips is not None) and hFlips[i] == 1:
                    maskTotal = maskTotal[:,::-1]
            maskTotal = torch.FloatTensor(np.asarray(self.transform(Image.fromarray(np.clip(maskTotal,0,1)))))[None,::]
            allMasks.append(maskTotal[None,::])

        return torch.cat(allMasks,dim=0)


def get_loader(image_path, metadata_path, crop_size, image_size, batch_size, dataset='CelebA', mode='train',
               select_attrs=[], datafile='datasetBoxAnn.json', bboxLoader=False, bbox_size = 64,
               randomrotate=0, randomscale=(0.5, 0.5), loadMasks=False, balance_classes=0, onlyrandBoxes=False,
               max_object_size=0., min_box_size=16, n_masks=-1, imagenet_norm=False, use_gt_mask = False, n_fg_boxes = 1,
               n_extra_boxes = 0, square_resize = 0, filter_by_mincooccur = -1., only_indiv_occur = 0, patch_mode = 0,
               use_highres_patch=0, get_all_inst = False, remove_invalid = 0, color_or_affine='affine', get_kp_class = 0,
               get_rand_mask= 0, patch_with_bg=False):
    """Build and return data loader."""

    transList = [transforms.Resize(image_size if not square_resize else [image_size, image_size]), transforms.CenterCrop(image_size)] if not loadMasks else [transforms.Resize(image_size if not square_resize else [image_size, image_size], interpolation=Image.NEAREST), transforms.RandomCrop(image_size)]
    if mode == 'train':
        transList.extend([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        transList.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if imagenet_norm:
        transList[-1] = transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    if loadMasks:
        transform = transforms.Compose(transList[:-2])
    elif bboxLoader:
        # Split the transforms into 3 parts.
        # First is applied on the entire image before cropping
        # Second part consists of random augments which needs special handling
        # second is applied to convert image to tensor applied sperately to image and crop
        transform = [transforms.Compose(transList[:2]), 'Flip' if mode=='train' else None, transforms.Compose(transList[-2:])]
    else:
        transform = transforms.Compose(transList)

    if loadMasks:
        if dataset == 'coco':
            dataset = CocoMaskDataset(transform, mode, select_attrs=select_attrs, balance_classes=balance_classes,
                    n_masks_perclass=n_masks)
        elif dataset == 'mrcnn':
            dataset = MRCNN_MaskDataset(transform, mode, select_attrs=select_attrs, balance_classes=balance_classes,
                    n_masks_perclass=n_masks)
        elif dataset == 'sdi':
            dataset = SDI_MaskDataset(transform, mode, select_attrs=select_attrs, balance_classes=balance_classes,
                    n_masks_perclass=n_masks)
        elif dataset == 'pascal':
            dataset = PascalMaskDataset(transform, mode, select_attrs=select_attrs, balance_classes=balance_classes,
                    n_masks_perclass=n_masks)
    else:
        if dataset == 'CelebA':
            dataset = CelebDataset(image_path, metadata_path, transform, mode, select_attrs=select_attrs)
        elif dataset == 'RaFD':
            dataset = ImageFolder(image_path, transform)
        elif dataset == 'coco':
            if bboxLoader:
                dataset = CocoDatasetBBoxSample(transform, mode, select_attrs, datafile, image_size, bbox_size,
                        balance_classes=balance_classes, onlyrandBoxes=onlyrandBoxes, max_object_size=max_object_size,
                        min_box_size = min_box_size, use_gt_mask = use_gt_mask, boxrotate = randomrotate, n_fg_boxes = n_fg_boxes,
                        n_extra_boxes = n_extra_boxes, square_resize = square_resize, filter_by_mincooccur = filter_by_mincooccur,
                        only_indiv_occur = only_indiv_occur, patch_mode = patch_mode, use_highres_patch = use_highres_patch,
                        get_all_inst=get_all_inst, remove_invalid = remove_invalid, color_or_affine=color_or_affine,
                        get_kp_class = get_kp_class, get_rand_mask=get_rand_mask, patch_with_bg=patch_with_bg)
            else:
                dataset = CocoDataset(transform, mode, select_attrs=select_attrs, datafile=datafile,
                        out_img_size=image_size, balance_classes=balance_classes)
        elif dataset == 'bdd100k':
            dataset = BDD100kDataset(transform, mode, select_attrs, datafile, image_size, bbox_size,
                    balance_classes=balance_classes, onlyrandBoxes=onlyrandBoxes, max_object_size=max_object_size,
                    min_box_size = min_box_size, use_gt_mask = use_gt_mask, boxrotate = randomrotate, n_boxes = n_boxes,
                    square_resize = square_resize, filter_by_mincooccur = filter_by_mincooccur, only_indiv_occur = only_indiv_occur,
                    patch_mode = patch_mode, use_highres_patch = use_highres_patch, get_all_inst=get_all_inst, remove_invalid = remove_invalid,
                    color_or_affine=color_or_affine, get_kp_class = get_kp_class, get_rand_mask=get_rand_mask )
        elif dataset == 'places2':
            dataset = Places2DatasetBBoxSample(transform, mode, select_attrs, datafile, image_size, bbox_size,
                    balance_classes=balance_classes, onlyrandBoxes=onlyrandBoxes, max_object_size=max_object_size,
                    use_gt_mask = use_gt_mask, boxrotate = randomrotate, n_boxes = n_boxes)
        elif dataset == 'ade20k':
            dataset = ADE20k(transform, mode, select_attrs, image_size, bbox_size, max_object_size=max_object_size,
                    use_gt_mask = use_gt_mask, boxrotate= randomrotate, n_boxes = n_boxes, square_resize = square_resize)
        elif dataset == 'flickrlogo':
            dataset = FlickrLogoBBoxSample(transform, mode, select_attrs, datafile, image_size, bbox_size,
                      balance_classes=balance_classes, onlyrandBoxes=onlyrandBoxes, max_object_size=max_object_size,
                      use_gt_mask = use_gt_mask, boxrotate = randomrotate, n_boxes = n_boxes)
        elif dataset == 'belgalogo':
            dataset = BelgaLogoBBoxSample(transform, mode, select_attrs, datafile, image_size, bbox_size,
                      balance_classes=balance_classes, onlyrandBoxes=onlyrandBoxes, max_object_size=max_object_size,
                      use_gt_mask = use_gt_mask, boxrotate = randomrotate, n_boxes = n_boxes)
        elif dataset == 'outofcontext':
            dataset = OutofContextBBoxSample(transform, mode, select_attrs, datafile, image_size, bbox_size,
                      balance_classes=balance_classes, onlyrandBoxes=onlyrandBoxes, max_object_size=max_object_size,
                      use_gt_mask = use_gt_mask, boxrotate = randomrotate, n_boxes = n_boxes)
        elif dataset == 'unrel':
            dataset = UnrelBBoxSample(transform, mode, select_attrs, datafile, image_size, bbox_size,
                      balance_classes=balance_classes, onlyrandBoxes=onlyrandBoxes, max_object_size=max_object_size,
                      use_gt_mask = use_gt_mask, boxrotate = randomrotate, n_boxes = n_boxes)
        elif dataset == 'pascalVoc':
            dataset = PascalDatasetBBoxSample(transform, mode, select_attrs, datafile, image_size, bbox_size,
                    balance_classes=balance_classes, onlyrandBoxes=onlyrandBoxes, max_object_size=max_object_size,
                    min_box_size = min_box_size, use_gt_mask = use_gt_mask, boxrotate = randomrotate, n_boxes = n_boxes,
                    square_resize = square_resize, filter_by_mincooccur = filter_by_mincooccur, only_indiv_occur = only_indiv_occur,
                    patch_mode = patch_mode, use_highres_patch = use_highres_patch, get_all_inst=get_all_inst, remove_invalid = remove_invalid,
                    color_or_affine=color_or_affine, get_kp_class = get_kp_class, get_rand_mask=get_rand_mask )
        elif dataset == 'mnist':
            dataset = MNISTDatasetBBoxSample(transform, mode, select_attrs, image_size, bbox_size,
                    randomrotate=randomrotate, scaleRange=randomscale)
        elif dataset == 'celebbox':
            dataset = MNISTDatasetBBoxSample(transform, mode, select_attrs, image_size, bbox_size,
                    randomrotate=randomrotate, scaleRange=randomscale, squareAspectRatio=True, use_celeb=True)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=16 if not loadMasks else 2 if image_size==32 else 6, pin_memory=True)
    return data_loader

def get_dataset(image_path, metadata_path, crop_size, image_size, dataset='CelebA', split='train', select_attrs=[],
                datafile='datasetBoxAnn.json', bboxLoader=False, bbox_size = 64, randomrotate=0,
                randomscale=(0.5, 0.5), loadMasks=False, balance_classes=0, onlyrandBoxes=False, max_object_size=0., min_box_size = 16,
                n_masks=-1, imagenet_norm=False, use_gt_mask = False, mode='test', n_fg_boxes = 1, n_extra_boxes = 0, square_resize = 0,
                filter_by_mincooccur = -1., only_indiv_occur = 0, patch_mode = 0, use_highres_patch = 0, get_all_inst=False,
                remove_invalid = 0, color_or_affine='affine', get_kp_class = 0, get_rand_mask=0, detectorMode = 0,
                canvas_image_size = 416, filter_class_instances= [], patch_with_bg=False, only_selIds = []):
    """Build and return data loader."""

    transList = [transforms.Resize(image_size if not square_resize else [image_size, image_size]), transforms.CenterCrop(image_size)] if not loadMasks else [transforms.Resize(image_size if not square_resize else [image_size, image_size], interpolation=Image.NEAREST), transforms.RandomCrop(image_size)]
    if mode == 'train':
        transList.extend([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        if loadMasks:
            transList[-1] = transforms.CenterCrop(image_size)
        transList.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if imagenet_norm:
        transList[-1] = transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))

    if loadMasks:
        transform = transforms.Compose(transList[:-2])
    elif bboxLoader:
        # Split the transforms into 3 parts.
        # First is applied on the entire image before cropping
        # Second part consists of random augments which needs special handling
        # second is applied to convert image to tensor applied sperately to image and crop
        transform = [transforms.Compose(transList[:2]), 'Flip' if mode=='train' else None, transforms.Compose(transList[-2:])]
    else:
        transform = transforms.Compose(transList)

    if loadMasks:
        if dataset == 'coco':
            dataset = CocoMaskDataset(transform, split, select_attrs=select_attrs, balance_classes=balance_classes, n_masks_perclass=n_masks)
        elif dataset == 'mrcnn':
            dataset = MRCNN_MaskDataset(transform, split, select_attrs=select_attrs, balance_classes=balance_classes,
                    n_masks_perclass=n_masks)
        elif dataset == 'sdi':
            dataset = SDI_MaskDataset(transform, mode, select_attrs=select_attrs, balance_classes=balance_classes,
                    n_masks_perclass=n_masks)
        elif dataset == 'pascal':
            dataset = PascalMaskDataset(transform, split, select_attrs=select_attrs, balance_classes=balance_classes, n_masks_perclass=n_masks)
    else:
        if dataset == 'CelebA':
            dataset = CelebDataset(image_path, metadata_path, transform, split)
        elif dataset == 'RaFD':
            dataset = ImageFolder(image_path, transform)
        elif dataset == 'coco':
            if bboxLoader:
                dataset = CocoDatasetBBoxSample(transform, split, select_attrs, datafile, image_size, bbox_size,
                                                balance_classes=balance_classes, onlyrandBoxes=onlyrandBoxes,
                                                max_object_size=max_object_size, min_box_size = min_box_size, use_gt_mask = use_gt_mask,
                                                boxrotate= randomrotate, n_fg_boxes = n_fg_boxes, n_extra_boxes = n_extra_boxes,
                                                square_resize = square_resize, filter_by_mincooccur = filter_by_mincooccur, only_indiv_occur = only_indiv_occur,
                                                patch_mode = patch_mode, use_highres_patch = use_highres_patch, get_all_inst=get_all_inst,
                                                remove_invalid = remove_invalid, color_or_affine=color_or_affine, get_kp_class=get_kp_class,
                                                get_rand_mask=get_rand_mask, detectorMode = detectorMode, canvas_image_size=canvas_image_size,
                                                filter_class_instances=filter_class_instances, patch_with_bg=patch_with_bg, only_selIds = only_selIds)
            else:
                dataset = CocoDataset(transform, split, select_attrs=select_attrs, datafile=datafile,
                                      out_img_size=image_size, balance_classes=balance_classes)
        elif dataset == 'bdd100k':
            dataset = BDD100kDataset(transform, split, select_attrs, datafile, image_size, bbox_size,
                    balance_classes=balance_classes, onlyrandBoxes=onlyrandBoxes, max_object_size=max_object_size,
                    min_box_size = min_box_size, use_gt_mask = use_gt_mask, boxrotate = randomrotate, n_fg_boxes = n_fg_boxes, n_extra_boxes = n_extra_boxes,
                    square_resize = square_resize, filter_by_mincooccur = filter_by_mincooccur, only_indiv_occur = only_indiv_occur,
                    patch_mode = patch_mode, use_highres_patch = use_highres_patch, get_all_inst=get_all_inst, remove_invalid = remove_invalid,
                    color_or_affine=color_or_affine, get_kp_class = get_kp_class, get_rand_mask=get_rand_mask, filter_class_instances=filter_class_instances,
                    detectorMode = detectorMode, canvas_image_size=canvas_image_size)
        elif dataset == 'places2':
            dataset = Places2DatasetBBoxSample(transform, split, select_attrs, datafile, image_size, bbox_size,
                    balance_classes=balance_classes, onlyrandBoxes=onlyrandBoxes, max_object_size=max_object_size,
                    use_gt_mask = use_gt_mask, boxrotate = randomrotate, n_boxes = n_boxes)
        elif dataset == 'ade20k':
            dataset = ADE20k(transform, split, select_attrs, image_size, bbox_size, max_object_size=max_object_size,
                    use_gt_mask = use_gt_mask, boxrotate= randomrotate, n_boxes = n_boxes, square_resize = square_resize)
        elif dataset == 'flickrlogo':
                dataset = FlickrLogoBBoxSample(transform, split, select_attrs, datafile, image_size, bbox_size,
                        balance_classes=balance_classes, onlyrandBoxes=onlyrandBoxes, max_object_size=max_object_size,
                        use_gt_mask = use_gt_mask, boxrotate = randomrotate, n_boxes = n_boxes)
        elif dataset == 'outofcontext':
            dataset = OutofContextBBoxSample(transform, mode, select_attrs, datafile, image_size, bbox_size,
                      balance_classes=balance_classes, onlyrandBoxes=onlyrandBoxes, max_object_size=max_object_size,
                      use_gt_mask = use_gt_mask, boxrotate = randomrotate, n_boxes = n_boxes)
        elif dataset == 'unrel':
            dataset = UnrelBBoxSample(transform, mode, select_attrs, datafile, image_size, bbox_size,
                      balance_classes=balance_classes, onlyrandBoxes=onlyrandBoxes, max_object_size=max_object_size,
                      use_gt_mask = use_gt_mask, boxrotate = randomrotate, n_boxes = n_boxes)
        elif dataset == 'belgalogo':
                dataset = BelgaLogoBBoxSample(transform, split, select_attrs, datafile, image_size, bbox_size,
                        balance_classes=balance_classes, onlyrandBoxes=onlyrandBoxes, max_object_size=max_object_size,
                        use_gt_mask = use_gt_mask, boxrotate = randomrotate, n_boxes = n_boxes)
        elif dataset == 'pascalVoc':
            dataset = PascalDatasetBBoxSample(transform, split, select_attrs, datafile, image_size, bbox_size,
                    balance_classes=balance_classes, onlyrandBoxes=onlyrandBoxes, max_object_size=max_object_size,
                    min_box_size = min_box_size, use_gt_mask = use_gt_mask, boxrotate = randomrotate, n_fg_boxes = n_fg_boxes, n_extra_boxes = n_extra_boxes,
                    square_resize = square_resize, filter_by_mincooccur = filter_by_mincooccur, only_indiv_occur = only_indiv_occur,
                    patch_mode = patch_mode, use_highres_patch = use_highres_patch, get_all_inst=get_all_inst, remove_invalid = remove_invalid,
                    color_or_affine=color_or_affine, get_kp_class = get_kp_class, get_rand_mask=get_rand_mask, filter_class_instances=filter_class_instances,
                    detectorMode = detectorMode, canvas_image_size=canvas_image_size)
        elif dataset == 'mnist':
            dataset = MNISTDatasetBBoxSample(transform, split, select_attrs, image_size, bbox_size, randomrotate=randomrotate, scaleRange=randomscale)
        elif dataset == 'celebbox':
            dataset = MNISTDatasetBBoxSample(transform, split, select_attrs, image_size, bbox_size, randomrotate=randomrotate, scaleRange=randomscale, squareAspectRatio=True, use_celeb=True)

    return dataset
