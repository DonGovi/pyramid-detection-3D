import os
import sys
import time
import math
import torch
import torch.nn as nn
import numpy as np


def meshgrid(z, y, x):
    '''
    Return meshgrid in range z & y & x

    Args:
      z: (int) first dim range;
      y: (int) second dim range;
      x: (int) third dim range;
    
    Return:
      (tensor) meshgird, sized [z*y*x, 3]

    Example:
      >> meshgrid(1,2,3)
      0 0 0
      0 0 1
      0 0 2
      0 1 0
      0 1 1
      0 1 2
    '''
    a = torch.arange(0, z)
    b = torch.arange(0, y)
    c = torch.arange(0, x)
    zzz = a.view(-1,1).repeat(1, y*x).view(-1,1)
    yyy = b.view(-1,1).repeat(z, 1, x).view(-1,1)
    xxx = c.view(-1,1).repeat(z*y, 1).view(-1,1)
    return torch.cat((zzz,yyy,xxx), 1)


def change_box_order(boxes, order):
    '''
    Change box order between (zmin,ymin,xmin,zmax,ymax,xmax) and (zcenter,ycenter,xcenter,depth,height,width)

    Args:
      boxes: (tensor) bounding boxs, sized [N, 6].
      order: (str) either "zyxzyx2zyxdhw" or "zyxdhw2zyxzyx"

    Return:
      (tensor) coverted bounding boxes, sized[N, 6]
    '''

    assert order in ["zyxzyx2zyxdhw", "zyxdhw2zyxzyx"]
    a = boxes[:, :3]    # first half    zyx in zyxdhw, zyx_min in zyxzyx
    b = boxes[:, 3:]    # last half     dhw in zyxdhw, zyx_max in zyxzyx

    if order == "zyxzyx2zyxdhw":
        return torch.cat(((a+b)/2, b-a+1), 1)
    else:
        return torch.cat((a-b/2, a+b/2), 1)


def box_iou(box1, box2, order="zyxzyx"):
    '''
    Compute the intersection over union of two set of boxes.
    The default box order is (zmin, ymin, xmin, zmax, ymax, xmax)

    Args:
      box1: (tensor) bounding boxs, sized [N, 6].
      box2: (tensor) bounding boxs, sized [M, 6].
      order: (str) either "zyxzyx2zyxdhw" or "zyxdhw2zyxzyx"

    Return:
      (tensor) iou, sized [N, M].
    '''

    if order == "zyxdhw":
        box1 = change_box_order(box1, "zyxdhw2zyxzyx")
        box2 = change_box_order(box2, "zyxdhw2zyxzyx")

    N = box1.size(0)
    M = box2.size(0)
    # compute the bigger edge between the min edge of the two boxes (top, front, left)
    tfl = torch.max(box1[:, :3].unsqueeze(1).expand(N,M,3),      # [N,3] -> [N,1,3] -> [N,M,3]
                   box2[:, :3].unsqueeze(0).expand(N,M,3))      # [M,3] -> [1,M,3] -> [N,M,3]
    # compute the smaller edge between the max edge of the two boxes (bottem, back, right)
    bbr = torch.min(box1[:, 3:].unsqueeze(1).expand(N,M,3),      # [N,3] -> [N,1,3] -> [N,M,3]
                   box2[:, 3:].unsqueeze(0).expand(N,M,3))      # [M,3] -> [1,M,3] -> [N,M,3]

    dhw = (bbr - tfl).clamp(min=0)   # [N,M,3]
    inter = dhw[:,:,0] * dhw[:,:,1] * dhw[:,:,2]   #[N,M]
    
    area1 = (box1[:,3] - box1[:,0]) * (box1[:,4] - box1[:,1]) * (box1[:,5] - box1[:,2])     #[N,]
    area2 = (box2[:,3] - box2[:,0]) * (box2[:,4] - box2[:,1]) * (box2[:,5] - box2[:,2])     #[M,]
    area1 = area1.unsqueeze(1).expand_as(inter)   # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)   # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)
    return iou

def box_distance(box1, box2, order="zyxzyx"):
    '''
    Calculate distance between two boxes set
    Args:
      box1: (tensor) bounding boxes, sized [N, 6]
      box2: (tensor) bounding boxes, sized [M, 6]
      order: (str) "zyxzyx" or "zyxdhw"
    Returns:
      distances: (tensor) distance between every boxes in different set [N, M]
    '''
    if order == "zyxzyx":
        box1 = change_box_order(box1, "zyxzyx2zyxdhw")
        box2 = change_box_order(box2, "zyxzyx2zyxdhw")
    N = box1.size(0)
    M = box2.size(0)
    center1 = box1[:,:3].unsqueeze(1).expand(N,M,3)      # [N,3] -> [N,1,3] -> [N,M,3]
    center2 = box2[:,:3].unsqueeze(0).expand(N,M,3)      # [M,3] -> [1,M,3] -> [N,M,3]
    distance = center2 - center1   #[N,M,3]
    distance = torch.sqrt(torch.pow(distance[:,:,0],2) + torch.pow(distance[:,:,1],2) + torch.pow(distance[:,:,2],2))    # [N,M]
    
    return distance  
    
     
def box_nms(bboxes, scores, threshold=0.5):
    '''
    Non-maximum Suppression

    Args:
      bboxes: (tensor) bounding boxes, sized [N,6] -> (zmin, ymin, xmin, zmax, ymax, xmax)
      scores: (tensor) bboxes scores, sized [N,]
      threshold: (float) overlap threshold
      mode: (str) "union" or "min"

    Returns:
      keep: (tensor) selected indices
    '''
    bboxes = bboxes.numpy()
    scores = scores.numpy()

    z1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x1 = bboxes[:,2]
    z2 = bboxes[:,3]
    y2 = bboxes[:,4]
    x2 = bboxes[:,5]

    areas = (z2 - z1) * (y2 - y1) * (x2 - x1)    # [N,]
    order = scores.argsort()[::-1]            # sort scores by descending
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)

        zz1 = np.maximum(z1[i], z1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx1 = np.maximum(x1[i], x1[order[1:]])
        zz2 = np.minimum(z2[i], z2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])

        d = np.maximum(0.0, zz2 - zz1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        w = np.maximum(0.0, xx2 - xx1 + 1)
        inter = d * h * w               # [N, ]
        ovr = inter / (areas[i] + areas[order[1:]] - inter)     # [N, ]

        inds = np.where(ovr < threshold)[0]
        order = order[inds + 1]

    return torch.LongTensor(keep)

def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)  # [D,D]
    return y[labels]            # [N,D]

def box_cover(box1, box2, labels, order="zyxzyx"):
    '''
    find boxes in box2 that completely cover box1

    Args:
      box1: (FloatTensor) [N, 6] 
      box2: (FloatTensor) [M, 6] 
      labels: (LongTensor) [N, ]
      order: (string) zyxzyx or zyxdhw

    Returns:
      keep: (LongTensor) [M, ]  class label for each box2
    '''
    if order == "zyxdhw":
        box1 = change_box_order(box1, "zyxdhw2zyxzyx")
        box2 = change_box_order(box2, "zyxdhw2zyxzyx")

    keep = []
    for m in range(box2.size(0)):
        cls_m = []
        for n in range(box1.size(0)):
            if box1[n, 0] > box2[m, 0] and box1[n, 1] > box2[m, 1] and box1[n, 2] > box2[m, 2] and \
               box1[n, 3] < box2[m, 3] and box1[n, 4] < box2[m, 4] and box1[n, 5] < box2[m, 5]:
                # the anchor fully covers the gt
                cls_m.append(labels[n])
        if len(cls_m)==1:
            # if and only if the anchor fully contains one gt
            keep_m = cls_m[0] + 1
        else:
            keep_m = 0
        keep.append(keep_m)
    keep = torch.LongTensor(keep)
    return keep
