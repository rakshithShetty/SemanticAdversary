from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as geo
from shapely.ops import cascaded_union
import torch

def intersectBox(bboxA, bboxB):

    # Co-ordinates are of the form, [topleft X, topleft Y, width, height]
    topLeft = [max(bboxA[0],bboxB[0]), max(bboxA[1], bboxB[1])]
    botRight = [min(bboxA[0]+bboxA[2],bboxB[0]+bboxB[2]), min(bboxA[1]+bboxA[3], bboxB[1]+bboxB[3])]
    inter = max(botRight[0] - topLeft[0],0) * max(botRight[1] - topLeft[1],0)

    return inter

def intersectBoxBatch(bboxA, bboxB):
    m = bboxA.shape[0]
    n = bboxB.shape[0]

    # Co-ordinates are of the form, [topleft X, topleft Y, width, height]
    topLeft = [torch.max(bboxA[:,:,0], bboxB[:,:,0]), torch.max(bboxA[:,:,1], bboxB[:,:,1])]
    botRight= [torch.min(bboxA[:,:,0]+bboxA[:,:,2], bboxB[:,:,0]+bboxB[:,:,2]), torch.min(bboxA[:,:,1]+bboxA[:,:,3], bboxB[:,:,1]+bboxB[:,:,3])]
    inter = torch.clamp(botRight[0] - topLeft[0],min=0)*torch.clamp(botRight[1] - topLeft[1],min=0)
    return inter

#def intersectNP(bboxA, bboxB):
#    # Co-ordinates are of the form, [topleft X, topleft Y, width, height]
#    topLeft = np.maximum(bboxA[:2],bboxB[:2])
#    botRight = np.minimum(bboxA[:2]+bboxA[2:],bboxB[:2]+bboxB[2:])
#    inter = np.prod(np.clip(botRight-topLeft,0, None))
#    return inter


#Compute How much of boxB is in BoxA
def computeContainment(bboxA, bboxB):
    inter = intersectBox(bboxA, bboxB)
    aA = bboxA[2] * bboxA[3]
    aB = bboxB[2] * bboxB[3]
    return (inter/aB), inter, aA, aB

def computeIOUBatch(bboxA, bboxB):
    m = bboxA.shape[0]
    n = bboxB.shape[0]
    ba = bboxA.unsqueeze(1).repeat_interleave(n,1)
    bb = bboxB.unsqueeze(0).repeat_interleave(m,0)
    inter = intersectBoxBatch(ba, bb)
    aA = ba[:,:,2] * ba[:,:,3]
    aB = bb[:,:,2] * bb[:,:,3]
    aU = (aA + aB) - inter
    return (inter/aU), (inter/aA), (inter/aB)

def computeIOU(bboxA, bboxB):
    inter = intersectBox(bboxA, bboxB)
    aA = bboxA[2] * bboxA[3]
    aB = bboxB[2] * bboxB[3]
    aU = aA + aB - inter
    return (inter/aU), (inter/aA), (inter/aB)

def computeUnionArea(boxes):
    boxes = [geo.box(bb[0],bb[1],bb[0]+bb[2], bb[1]+bb[3]) for bb in boxes]
    return cascaded_union(boxes).area

def show(img):
    if type(img) == torch.Tensor:
        if img.device.type == 'cuda':
            npimg = img.data.cpu().numpy()
        else:
            npimg = img.numpy()
    else:
        npimg = img
    if len(npimg.shape) == 2:
        npimg = npimg[None,::]
    if npimg.shape[0] == 1:
        npimg = npimg[[0,0,0]]
    plt.imshow(((np.transpose(npimg, (1,2,0))+1.0)*255./2.0).astype(np.uint8), interpolation='nearest')
    plt.show()


def toletterbox(img, height=416, color=(0., 0., 127.5)):
    # Resize a rectangular image to a padded square
    shape = img.shape[2:]  # shape = [height, width]
    ratio = float(height) / max(shape)  # ratio  = old / new
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    dw = (height - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
    return img, ratio, dw, dh


