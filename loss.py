from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import one_hot_embedding
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, num_classes=3):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma

    def focal_loss(self, x, y, size_average=False):
        '''Focal loss.

        Args:
          x: (tensor) sized [N,C].
          y: (tensor) sized [N,].

        Return:
          (tensor) focal loss.
        '''
        target = y.view(-1,1)
        logpt = F.log_softmax(x, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1-pt)**self.gamma * logpt
        if size_average == True:
            return loss.mean()
        else:
            return loss.sum()
        

    def focal_loss_alt(self, x, y, size_average=False):
        '''Focal loss alternative.

        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].

        Return:
          (tensor) focal loss.
        '''
        w = Variable(torch.Tensor([1.,1.,1.])).cuda()
        loss = F.cross_entropy(x, y, weight=None, size_average=size_average)
        return loss

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
          loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 6].
          loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 6].
          cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, 3].
          cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].

        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        '''
        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.data.long().sum()

        ################################################################
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        ################################################################
        mask = pos.unsqueeze(2).expand_as(loc_preds)       # [N, #anchors]->[N, #anchors,1]->[N,#anchors,6]
        masked_loc_preds = loc_preds[mask].view(-1,6)      # [#pos,6]
        masked_loc_targets = loc_targets[mask].view(-1,6)  # [#pos,6]
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)

        ################################################################
        # cls_loss = FocalLoss(loc_preds, loc_targets)
        ################################################################
        pos_neg = cls_targets > -1  # exclude ignored anchors
        num_pos_neg = pos_neg.data.long().sum()      # number of background and classes anchors
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        #print(cls_preds[mask].size())
        masked_cls_preds = cls_preds[mask].view(-1, self.num_classes+1)
        cls_loss = self.focal_loss_alt(masked_cls_preds, cls_targets[pos_neg], size_average=False)

        #print('loc_loss: %.3f | cls_loss: %.3f' % (loc_loss.data[0]/num_pos, 
         #                                          cls_loss.data[0]/num_pos_neg), end=' | ')

        loc_loss_avg = loc_loss/num_pos
        cls_loss_avg = cls_loss/num_pos_neg
        loss = loc_loss_avg + cls_loss_avg
        # print("loc_num: %d | cls_num: %d" % (num_pos, num_classes))

        return loss, loc_loss_avg, cls_loss_avg



