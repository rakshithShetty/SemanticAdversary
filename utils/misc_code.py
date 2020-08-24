import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from utils.utils import show
from sklearn.cluster import KMeans
from removalmodels.stargan import kp2gaussian
import torch
import cv2
from utils.data_loader_stargan import get_dataset
from torch.nn import functional as F

def visualize_keypoints(kp, colors):
    #kp = kp/(kp.max(dim=2)[0].max(dim=2)[0].unsqueeze(-1).unsqueeze(-1))
    n_keypoints = kp.shape[1]
    kp_img = (kp[:,:,None,:,:]*colors.view(1,n_keypoints,3,1,1)).sum(dim=1)
    kp_img = (kp_img -  colors.mean(dim=0)[:,None,None])
    return kp_img

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

    stacked = stacked.astype(np.uint8)
    #stacked = cv2.cvtColor(stacked.astype(np.uint8), cv2.COLOR_BGR2RGB)
    return stacked

n_kp = 15
n_centers = 50
cmap = matplotlib.cm.get_cmap('tab20b')
kp_colors = torch.FloatTensor(cmap(np.arange(n_kp)/float(n_kp-1.))[:,:3])
im_sz = 64

#kpfile = 'kp_dumps/keypoints_noRRegGPp5Pmask_Allimage.npz'
kpfile = 'kp_dumps/keypoints_hResRot180AffineKPnoDecaySz64_Allimage.npz'
kpData = np.load(kpfile)

kpFeat = kpData['kp_m'].reshape((kpData['kp_m'].shape[0],-1))
kmeans = KMeans(n_clusters=n_centers, init='k-means++').fit(kpFeat)
kMeanCenters = torch.FloatTensor(kmeans.cluster_centers_.reshape(n_centers,n_kp,2))

centerGImg = kp2gaussian({'mean':kMeanCenters, 'var': torch.zeros(kMeanCenters.shape[:2], dtype=kMeanCenters.dtype)[:,:,None,None] + 0.003}, (im_sz, im_sz), kp_variance='single')
kpCenterImg = visualize_keypoints(centerGImg, kp_colors)

padimg = np.zeros((im_sz,5,3),dtype=np.uint8) + 128

# Show all the centroid keypoints
n_col = 5
n_rows = n_centers//n_col
all_rows = []
for i in range(n_rows):
    all_rows.append(make_image([kpCenterImg[i*n_col + c][None,::] for c in range(n_col)],padimg=padimg))
    all_rows.append(np.zeros((5,all_rows[-1].shape[1],3),dtype=np.uint8) + 128)

final_img = np.vstack(all_rows)

plt.imshow(final_img); plt.show()

# Show all the centroid keypoints with example poses
n_col = 2
n_examples = 5
n_rows = n_centers//n_col
all_rows = []
for i in range(n_rows):
    rowImg = []
    for c in range(n_col):
        index = i*n_col + c
        examples = []
        randex = np.random.choice(np.where(kmeans.labels_==index)[0],5)
        rowImg.append(kpCenterImg[index][None,::])
        exImg = kp2gaussian({'mean':torch.FloatTensor(kpData['kp_m'][randex]), 'var': torch.FloatTensor(kpData['kp_v'][randex])}, (im_sz, im_sz), kp_variance='single')
        exImg = visualize_keypoints(exImg, kp_colors)
        rowImg.extend([exImg[e:e+1] for e in range(n_examples)])
    all_rows.append(make_image(rowImg,padimg=padimg))
    all_rows.append(np.zeros((5,all_rows[-1].shape[1],3),dtype=np.uint8) + 128)

final_img = np.vstack(all_rows)
plt.imshow(final_img); plt.show()

dataset = get_dataset('', '', 128, 128, 'coco', 'train',
                      select_attrs=['person'], datafile='datasetComplete.json', bboxLoader=1,
                      bbox_size = 128, randomrotate = 0, max_object_size=1.,
                      use_gt_mask = 1, n_boxes = 0, onlyrandBoxes= False, patch_mode = True,
                      use_highres_patch = 1, get_all_inst = True, min_box_size=80, remove_invalid=True)


kpData= dict(kpData)
kpData.update({'kmeans_labels':kmeans.labels_, 'kmeans_centers':kmeans.cluster_centers_.reshape(n_centers,n_kp,2)})

cluster_counts = np.bincount(kpData['kmeans_labels'])
cls_index = np.bincount(kpData['kmeans_labels']).argsort()[::-1]
n_col = 2
n_examples = 20
n_rows = n_centers//n_col
all_rows = []
for i in range(n_rows):
    rowImg = []
    for c in range(n_col):
        index = i*n_col + c
        examples = []
        randex = np.random.choice(np.where(kmeans.labels_==cls_index[index])[0],n_examples)
        rowImg.append(kpCenterImg[cls_index[index]][None,::])
        for e in randex:
            imIndex = dataset.imgId2idx[int(kpData['kp_info'][e][0])]
            imCls = [dataset.sattr_to_idx[kpData['kp_info'][e][1]]]
            _, _, boxImg, _, _, _, _, _, _  = dataset.getPatchAndCanvasbyIndexAndclass(imIndex, imCls, annId = int(kpData['kp_info'][e][2]))
            rowImg.append(F.adaptive_avg_pool2d(boxImg[:3][None,::],(im_sz,im_sz)))
    all_rows.append(make_image(rowImg,padimg=padimg))
    all_rows.append(np.zeros((5,all_rows[-1].shape[1],3),dtype=np.uint8) + 128)

final_img = np.vstack(all_rows)

fig, ax1 = plt.subplots()
ax1.imshow(final_img)
ax1.set_yticks(np.arange(n_rows) * (im_sz+5) + im_sz/2 + 3)
ax1.set_yticklabels([cluster_counts[cls_index[2*i]] for i in range(n_centers//2)])
extra_axis = matplotlib.axis.YAxis(ax1)
extra_axis.tick_right()
#ax2 = ax1.twinx()
extra_axis.set_ticks(np.arange(n_rows) * (im_sz+5) + im_sz/2 + 3)
extra_axis.set_label_position('right')
extra_axis.set_ticklabels([cluster_counts[cls_index[2*i+1]] for i in range(n_centers//2)])
ax1.add_artist(extra_axis)
plt.show()



np.savez(kpfile.split('.')[0]+'_withKmean.npz',**kpData)



for i,img in enumerate(data['images']):
    if img['split'] is not 'train':
        for j,bbox in enumerate(img['bboxAnn']):
            if bbox['cid'] == 1:
                data['images'][i]['bboxAnn'][j]['valid'] = 1 if (((img['cocoid'], bbox['id']) in idAndAnnIdtoValid) and idAndAnnIdtoValid[img['cocoid'], bbox['id']]) else 0



from sklearn.cluster import KMeans
#----------------------------
# feature clustering
#----------------------------
kpfile = 'kp_dumps/keypoints_hResRot180AffineKPnoDecaySz64_Allimage.npz'
kpData = np.load(kpfile)
kpData = dict(kpData)
train_feat = kpData['kp_feat'].mean(axis=1)
kmeans = KMeans(n_clusters=n_centers, init='k-means++').fit(train_feat)

padimg = np.zeros((im_sz,5,3),dtype=np.uint8) + 128
kpData= dict(kpData)
kpData.update({'feat_kmeans_labels':kmeans.labels_, 'feat_kmeans_centers':kmeans.cluster_centers_})

cluster_counts = np.bincount(kpData['feat_kmeans_labels'])
cls_index = np.bincount(kpData['feat_kmeans_labels']).argsort()[::-1]
n_col = 2
n_examples = 20
n_rows = n_centers//n_col
all_rows = []
for i in range(n_rows):
    rowImg = []
    for c in range(n_col):
        index = i*n_col + c
        examples = []
        randex = np.random.choice(np.where(kmeans.labels_==cls_index[index])[0],n_examples)
        for e in randex:
            imIndex = dataset.imgId2idx[int(kpData['kp_info'][e][0])]
            imCls = [dataset.sattr_to_idx[kpData['kp_info'][e][1]]]
            _, _, boxImg, _, _, _, _, _, _  = dataset.getPatchAndCanvasbyIndexAndclass(imIndex, imCls, annId = int(kpData['kp_info'][e][2]))
            rowImg.append(F.adaptive_avg_pool2d(boxImg[:3][None,::],(im_sz,im_sz)))
    all_rows.append(make_image(rowImg,padimg=padimg))
    all_rows.append(np.zeros((5,all_rows[-1].shape[1],3),dtype=np.uint8) + 128)

final_img = np.vstack(all_rows)

fig, ax1 = plt.subplots()
ax1.imshow(final_img)
ax1.set_yticks(np.arange(n_rows) * (im_sz+5) + im_sz/2 + 3)
ax1.set_yticklabels([cluster_counts[cls_index[2*i]] for i in range(n_centers//2)])
extra_axis = matplotlib.axis.YAxis(ax1)
extra_axis.tick_right()
#ax2 = ax1.twinx()
extra_axis.set_ticks(np.arange(n_rows) * (im_sz+5) + im_sz/2 + 3)
extra_axis.set_label_position('right')
extra_axis.set_ticklabels([cluster_counts[cls_index[2*i+1]] for i in range(n_centers//2)])
ax1.add_artist(extra_axis)
plt.show()


import os

out_dir = '/home/rshetty/mydocs/MyPlots/AddObjects/' + 'second_'+kpfile.split('/')[-1].split('.')[0]
if not os.path.exists(out_dir):
    os.makedirs(out_dir)



classes = np.unique(kpData['kp_info'][:,1])

kmeans_labels = {}
kmeans_centers = {}
kpData.update({'kmeans_labels':np.zeros(len(kpData['kp_info'])), 'kmeans_centers':{}})
dataset = get_dataset('', '', 128, 128, 'coco', 'train',
                      select_attrs=['all'], datafile='datasetComplete.json', bboxLoader=1,
                      bbox_size = 128, randomrotate = 0, max_object_size=1.,
                      use_gt_mask = 1, n_boxes = 0, onlyrandBoxes= False, patch_mode = True,
                      use_highres_patch = 1, get_all_inst = True, min_box_size=80)

for ci in range(len(classes)):
    print('currrent class is %s'%(classes[ci]))
    cls =classes[ci]
    cls_samples = (kpData['kp_info'][:,1] == cls)
    kpFeat = kpData['kp_m'][cls_samples].reshape((cls_samples.sum(),-1))
    kmeans = KMeans(n_clusters=n_centers, init='k-means++').fit(kpFeat)
    kMeanCenters = torch.FloatTensor(kmeans.cluster_centers_.reshape(n_centers,n_kp,2))
    kpData['kmeans_labels'][cls_samples] = kmeans.labels_
    kpData['kmeans_centers'][cls] = kmeans.cluster_centers_

    centerGImg = kp2gaussian({'mean':kMeanCenters, 'var': torch.zeros(kMeanCenters.shape[:2], dtype=kMeanCenters.dtype)[:,:,None,None] + 0.003}, (im_sz, im_sz), kp_variance='single')
    kpCenterImg = visualize_keypoints(centerGImg, kp_colors)

    cluster_counts = np.bincount(kmeans.labels_)
    clust_index = cluster_counts.argsort()[::-1]
    n_col = 2
    n_examples = 20
    n_rows = n_centers//n_col
    all_rows = []
    for i in range(n_rows):
        rowImg = []
        for c in range(n_col):
            index = i*n_col + c
            examples = []
            randex = np.random.choice(np.where(kmeans.labels_==clust_index[index])[0],n_examples)
            rowImg.append(kpCenterImg[clust_index[index]][None,::])
            for e in randex:
                imIndex = dataset.imgId2idx[int(kpData['kp_info'][cls_samples][e,0])]
                imCls = [dataset.sattr_to_idx[cls]]
                _, _, boxImg, _, _, _, _, _, _  = dataset.getPatchAndCanvasbyIndexAndclass(imIndex, imCls, annId = int(kpData['kp_info'][cls_samples][e,2]))
                rowImg.append(F.adaptive_avg_pool2d(boxImg[:3][None,::],(im_sz,im_sz)))
        all_rows.append(make_image(rowImg,padimg=padimg))
        all_rows.append(np.zeros((5,all_rows[-1].shape[1],3),dtype=np.uint8) + 128)

    final_img = np.vstack(all_rows)

    fig, ax1 = plt.subplots()
    ax1.imshow(final_img)
    ax1.set_yticks(np.arange(n_rows) * (im_sz+5) + im_sz/2 + 3)
    ax1.set_yticklabels([cluster_counts[clust_index[2*i]] for i in range(n_centers//2)])
    extra_axis = matplotlib.axis.YAxis(ax1)
    extra_axis.tick_right()
    #ax2 = ax1.twinx()
    extra_axis.set_ticks(np.arange(n_rows) * (im_sz+5) + im_sz/2 + 3)
    extra_axis.set_label_position('right')
    extra_axis.set_ticklabels([cluster_counts[clust_index[2*i+1]] for i in range(n_centers//2)])
    ax1.add_artist(extra_axis)
    plt.savefig(os.path.join(out_dir, 'kmeans_KP_%s_nCl%d.pdf'%(cls, n_centers)), dpi=1200)




kpData.update({'feat_kmeans_labels':np.zeros(len(kpData['kp_info'])), 'feat_kmeans_centers':{}})
for ci in range(len(classes)):
    print('currrent class is %s'%(classes[ci]))
    cls =classes[ci]
    cls_samples = (kpData['kp_info'][:,1] == cls)
    kpFeat = kpData['kp_feat'][cls_samples].mean(axis=1)
    kmeans = KMeans(n_clusters=n_centers, init='k-means++').fit(kpFeat)
    #kMeanCenters = torch.FloatTensor(kmeans.cluster_centers_.reshape(n_centers,n_kp,2))
    kpData['feat_kmeans_labels'][cls_samples] = kmeans.labels_
    kpData['feat_kmeans_centers'][cls] =  kmeans.cluster_centers_

    #centerGImg = kp2gaussian({'mean':kMeanCenters, 'var': torch.zeros(kMeanCenters.shape[:2], dtype=kMeanCenters.dtype)[:,:,None,None] + 0.003}, (im_sz, im_sz), kp_variance='single')
    #kpCenterImg = visualize_keypoints(centerGImg, kp_colors)

    cluster_counts = np.bincount(kmeans.labels_)
    clust_index = cluster_counts.argsort()[::-1]
    n_col = 2
    n_examples = 20
    n_rows = n_centers//n_col
    all_rows = []
    for i in range(n_rows):
        rowImg = []
        for c in range(n_col):
            index = i*n_col + c
            examples = []
            randex = np.random.choice(np.where(kmeans.labels_==clust_index[index])[0],n_examples)
            for e in randex:
                imIndex = dataset.imgId2idx[int(kpData['kp_info'][cls_samples][e,0])]
                imCls = [dataset.sattr_to_idx[cls]]
                _, _, boxImg, _, _, _, _, _, _  = dataset.getPatchAndCanvasbyIndexAndclass(imIndex, imCls, annId = int(kpData['kp_info'][cls_samples][e,2]))
                rowImg.append(F.adaptive_avg_pool2d(boxImg[:3][None,::],(im_sz,im_sz)))
        all_rows.append(make_image(rowImg,padimg=padimg))
        all_rows.append(np.zeros((5,all_rows[-1].shape[1],3),dtype=np.uint8) + 128)

    final_img = np.vstack(all_rows)

    fig, ax1 = plt.subplots()
    ax1.imshow(final_img)
    ax1.set_yticks(np.arange(n_rows) * (im_sz+5) + im_sz/2 + 3)
    ax1.set_yticklabels([cluster_counts[clust_index[2*i]] for i in range(n_centers//2)])
    extra_axis = matplotlib.axis.YAxis(ax1)
    extra_axis.tick_right()
    #ax2 = ax1.twinx()
    extra_axis.set_ticks(np.arange(n_rows) * (im_sz+5) + im_sz/2 + 3)
    extra_axis.set_label_position('right')
    extra_axis.set_ticklabels([cluster_counts[clust_index[2*i+1]] for i in range(n_centers//2)])
    ax1.add_artist(extra_axis)
    plt.savefig(os.path.join(out_dir, 'kmeans_feat_%s_nCl%d.pdf'%(cls, n_centers)), dpi=1200)


np.savez(kpfile.split('.')[0]+'_withKmean.npz',**kpData)

ci =0
cls =classes[ci]
cls_samples = (kpData['kp_info'][:,1] == cls)
kpFeat = kpData['kp_m'][cls_samples].reshape((cls_samples.sum(),-1))
kmeans = KMeans(n_clusters=n_centers, init='k-means++').fit(kpFeat)
allCenters = kpData['kmeans_centers'].item()
kpDataVal.update({'kmeans_labels':np.zeros(len(kpDataVal['kp_info'])), 'kmeans_centers':{}})
for ci in range(len(classes)):
    print('currrent class is %s'%(classes[ci]))
    cls =classes[ci]
    cls_samples = (kpDataVal['kp_info'][:,1] == cls)
    kpFeat = kpDataVal['kp_m'][cls_samples].reshape((cls_samples.sum(),-1))
    kmeans.cluster_centers_ = allCenters[cls].view(n_centers, -1)

    kpDataVal['kmeans_labels'][cls_samples] = kmeans.predict(kpFeat)
    kpDataVal['kmeans_centers'][cls] = kmeans.cluster_centers_


allCenters = kpData['feat_kmeans_centers'].item()
kpDataVal.update({'feat_kmeans_labels':np.zeros(len(kpData['kp_info'])), 'feat_kmeans_centers':{}})
for ci in range(len(classes)):
    print('currrent class is %s'%(classes[ci]))
    cls =classes[ci]
    cls_samples = (kpDataVal['kp_info'][:,1] == cls)
    kpFeat = kpDataVal['kp_feat'][cls_samples].mean(axis=1)
    kmeans.cluster_centers_ = allCenters[cls].view(n_centers, -1)
    kpDataVal['feat_kmeans_labels'][cls_samples] = kmeans.predict(kpFeat)
    kpDataVal['feat_kmeans_centers'][cls] =  kmeans.cluster_centers_

    #centerGImg = kp2gaussian({'mean':kMeanCenters, 'var': torch.zeros(kMeanCenters.shape[:2], dtype=kMeanCenters.dtype)[:,:,None,None] + 0.003}, (im_sz, im_sz), kp_variance='single')
    #kpCenterImg = visualize_keypoints(centerGImg, kp_colors)

for ci in range(len(classes)):
    print('currrent class is %s'%(classes[ci]))
    cls =classes[ci]
    cls_samples = (kpData['kp_info'][:,1] == cls)
    labels = kpData['kmeans_labels'][cls_samples].astype(int)
    cluster_counts = np.bincount(labels)
    clust_index = cluster_counts.argsort()[::-1]
    n_col = 2
    n_examples = 20
    n_rows = n_centers//n_col
    all_rows = []
    for i in range(n_rows):
        rowImg = []
        for c in range(n_col):
            index = i*n_col + c
            examples = []
            randex = np.random.choice(np.where(labels==clust_index[index])[0],n_examples)
            for e in randex:
                imIndex = dataset.imgId2idx[int(kpData['kp_info'][cls_samples][e,0])]
                imCls = [dataset.sattr_to_idx[cls]]
                _, _, boxImg, _, _, _, _, _, _  = dataset.getPatchAndCanvasbyIndexAndclass(imIndex, imCls, annId = int(kpData['kp_info'][cls_samples][e,2]))
                rowImg.append(F.adaptive_avg_pool2d(boxImg[:3][None,::],(im_sz,im_sz)))
        all_rows.append(make_image(rowImg,padimg=padimg))
        all_rows.append(np.zeros((5,all_rows[-1].shape[1],3),dtype=np.uint8) + 128)

    final_img = np.vstack(all_rows)

    fig, ax1 = plt.subplots()
    ax1.imshow(final_img)
    ax1.set_yticks(np.arange(n_rows) * (im_sz+5) + im_sz/2 + 3)
    ax1.set_yticklabels([(clust_index[2*i],cluster_counts[clust_index[2*i]]) for i in range(n_centers//2)])
    extra_axis = matplotlib.axis.YAxis(ax1)
    extra_axis.tick_right()
    #ax2 = ax1.twinx()
    extra_axis.set_ticks(np.arange(n_rows) * (im_sz+5) + im_sz/2 + 3)
    extra_axis.set_label_position('right')
    extra_axis.set_ticklabels([(cluster_counts[clust_index[2*i+1]],clust_index[2*i+1]) for i in range(n_centers//2)])
    ax1.add_artist(extra_axis)
    plt.savefig(os.path.join(out_dir, 'kmeans_KP_%s_nCl%d.pdf'%(cls, n_centers)), dpi=1200)
    plt.close('all')


np.savez(kpfile.split('.')[0]+'_withKmean.npz',**kpData)


from collections import defaultdict
bad_classes = defaultdict(list)

from albumentations import (
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)

import numpy as np




from shutil import copyfile
coconames = open('tools/yolov3/data/coco.names').read().splitlines()
catToId = {cls:i for i,cls in enumerate(coconames)}
from os.path import join


"bicycle" "bus" "car" "motorcycle" "person" "traffic light" "stop sign" "train" "truck"

"bike" "bus" "car" "motor" "person" "traffic light" "traffic sign" "train" "truck"
