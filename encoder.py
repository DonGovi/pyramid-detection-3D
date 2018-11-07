import torch
from utils import box_iou, box_nms, change_box_order, meshgrid
from bbox import BoundingBox, Annotation
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


class DataEncoder(object):
    def __init__(self):
        super(DataEncoder, self).__init__()
        self.anchor_areas = [8, 16, 32]     # length of the longest edge:  p3
        self.aspect_ratios = [0.8, 1, 1.5]    # ratios between z and xy --> z/x
        self.scale_ratios = [1.2, 1, 0.8]   # scale ratios
        self.num_levels = len(self.anchor_areas)
        self.num_anchors = len(self.aspect_ratios) * len(self.scale_ratios)
        self.anchor_edges = self.calc_anchor_edges()

    def calc_anchor_edges(self):
        '''
        Compute edges length of all anchors
        '''
        anchor_edges = []
        for area in self.anchor_areas:
            for ar in self.aspect_ratios:
                if ar < 1:
                    width = area
                    height = area
                    depth = int(width * ar)    # 0.5 * width
                else:
                    depth = area
                    width = int(depth / ar)    # 0.5 * depth
                    height = int(depth / ar)
                for sr in self.scale_ratios:
                    anchor_depth = depth * sr
                    anchor_height = height * sr
                    anchor_width = width * sr
                    anchor_edges.append((anchor_depth, anchor_height, anchor_width))

        return torch.Tensor(anchor_edges).view(self.num_levels, self.num_anchors, 3)     #(1, 9, 3)

    def get_anchor_boxes(self, input_size):
        '''
        Compute anchor boxes for each feature map

        Args:
          input_size: (tensor) model input size of (z, h, w).

        Returns:
          boxes: (tensor) anchor boxes for each feature map. Each of size [#anchors, 6],
                      where #anchors = fm_z * fm_h * fm_w * #anchors_per_cell
        '''
        fm_sizes = [(input_size / pow(2, i + 1)).ceil() for i in range(self.num_levels)]
        # Compute every size in p3
        boxes = []
        for i in range(self.num_levels):
            fm_size = fm_sizes[i]
            grid_size = (input_size / fm_size).floor()
            fm_d, fm_h, fm_w = int(fm_size[0]), int(fm_size[1]), int(fm_size[2])
            zyx = meshgrid(fm_d, fm_h, fm_w) + 0.5   # [fm_d * fm_h * fm_w, 3]
            zyx = (zyx * grid_size).view(fm_d, fm_h, fm_w, 1, 3).expand(fm_d, fm_h, fm_w, 9, 3)
            dhw = self.anchor_edges[i].view(1, 1, 1, 9, 3).expand(fm_d, fm_h, fm_w, 9, 3)
            box = torch.cat([zyx, dhw], 4)     # (fm_d, fm_h, fm_w, 9, 6) 
            boxes.append(box.view(-1, 6))   # (num_levels, fm_d*fm_h*fm_w*9, 6)
        return torch.cat(boxes, 0) # (num_levels*fm_d*fm_h*fm_w*9, 6) -> (36864, 6)

    def encode(self, boxes, labels, input_size):
        '''
        Encode target bounding boxes and class labels.fm_d

        Implement the Faster RCNN box coder in 3D image:
          tz = (z - anchor_z) / anchor_d
          ty = (y - anchor_y) / anchor_h
          tx = (x - anchor_x) / anchor_w
          td = log(d / anchor_d)
          th = log(h / anchor_h)
          tx = log(w / anchor_w)

        Args:
          boxes: (tensor) bounding boxes of (zmin, ymin, xmin, zmax, ymax, xmax), sized [#obj, 6]
          labels: (tensor) object class labels, sized [#obj,]
          input_size: (int/tuple) model input size of (d, h, w)

        Returns:
          loc_targets: (tensor) encoded boudning boxes, sized [#anchors, 6]
          cls_targets: (tensor) encoded class labels, sized [#anchors,]
        '''
        if isinstance(input_size, int):
            input_size = torch.Tensor([input_size, input_size, input_size])
        else:
            input_size = torch.Tensor(input_size) 
        anchor_boxes = self.get_anchor_boxes(input_size)     # (z, y, x, d, h, w)
        boxes = change_box_order(boxes, 'zyxzyx2zyxdhw')
        #print(boxes.size())
        ious = box_iou(anchor_boxes, boxes, order="zyxdhw")    # num_anchors x objects
        max_ious, max_ids = ious.max(1)       # find the best object for each anchor, return ious_value and object index
        best_ious, best_ids = ious.max(0)    # find the best anchor for each object, return ious_value and anchor index
        boxes = boxes[max_ids]
        #print(boxes.size())

        loc_zyx = (boxes[:, :3] - anchor_boxes[:, :3]) / anchor_boxes[:, 3:]
        loc_dhw = boxes[:, 3:] / anchor_boxes[:, 3:]
        loc_dhw = loc_dhw.numpy()
        loc_dhw = np.log(loc_dhw)
        loc_dhw = torch.from_numpy(loc_dhw)
        loc_targets = torch.cat([loc_zyx, loc_dhw], 1)

        cls_targets = 1 + labels[max_ids]   # the background class = 0, so +1 for object classes
        #print(cls_targets.size())
        cls_targets[max_ious<0.4] = 0
        
        for i in range(best_ids.size()[0]):
            cls_targets[best_ids[i]] = 1 + labels[i]	

        ig_num = cls_targets.size()[0] - 100
        cls_targets_array = cls_targets.numpy()
        neg_idx = np.where(cls_targets_array == 0)
        if ig_num > len(neg_idx[0]):
            ig_num -= (ig_num - len(neg_idx[0])) 
        ig_idx = np.random.choice(neg_idx[0], ig_num, replace=False)
        cls_targets_array[ig_idx] = -1
        cls_targets = torch.from_numpy(cls_targets_array)
        '''
        ignore = (max_ious > 0.15) & (max_ious < 0.4)
        cls_targets[ignore] = -1
         
        for i in range(best_ids.size()[0]):
            cls_targets[best_ids[i]] = 1 + labels[i]	
        '''
        return loc_targets, cls_targets


    def decode(self, loc_preds, cls_preds, input_size):
        '''
        Decode outputs back to bounding box locations and class labels.

        Args:
          loc_preds: (tensor) predicted locations, sized [#anchors, 6]
          cls_preds: (tensor) predicted class labels, sized [#anchors, #classes]
          input_size: (int/tuple) model input size of (z, h, w)

        Return:
          boxes: (tensor) decode box locations, sozed [#obj, 6]
          labels: (tensor) class labels for each box, sized [#obj,]
        '''
        CLS_THRESH = 0.75
        NMS_THRESH = 0.05
        if isinstance(input_size, int):
            input_size = torch.Tensor([input_size, input_size, input_size])
        else:
            input_size = torch.Tensor(input_size)
        anchor_boxes = self.get_anchor_boxes(input_size)

        loc_zyx = loc_preds[:, :3]
        loc_dhw = loc_preds[:, 3:]
        zyx = loc_zyx * anchor_boxes[:, 3:] + anchor_boxes[:, :3]
        dhw = loc_dhw.exp() * anchor_boxes[:, 3:]
        boxes = torch.cat([zyx-dhw/2, zyx+dhw/2], 1)    # [#anchors, 6]
        scores, labels = F.softmax(Variable(cls_preds), dim=1).data.max(1)    # [#anchors,] the best class for each anchor
        obj_idx = (labels > 0)
        obj_num = obj_idx.long().sum()
        if obj_num == 0:
            #print("Not found any object")
            return 0, 0, 0, 0, 0, 0
        else: 
            obj_mask = obj_idx.unsqueeze(1).expand_as(boxes)
            masked_scores = scores[obj_idx]
            masked_labels = labels[obj_idx]
            #print(masked_scores, masked_labels)
            masked_boxes = boxes[obj_mask].view(-1, 6)
            ids = (masked_scores > CLS_THRESH)
            if ids.long().sum() == 0:
                return 0, 0, 0, 0, 0, 0
            else:
                box_ids = ids.unsqueeze(1).expand_as(masked_boxes)
                obj_boxes = masked_boxes[box_ids].view(-1,6)
                obj_scores = masked_scores[ids]
                obj_labels = masked_labels[ids]
                
                ais_ids = (obj_labels == 1)
                if ais_ids.long().sum() == 0:
                    ais_pred_boxes = 0
                    ais_pred_scores = 0
                    ais_pred_labels = 0
                else:
                    # print(ais_ids.long().sum())
                    ais_ids = ais_ids.nonzero().squeeze()
                    #ais_masks = ais_ids.unsqueeze(1).expand_as(obj_boxes)
                    ais_labels = obj_labels[ais_ids]
                    ais_scores = obj_scores[ais_ids]
                    ais_boxes = obj_boxes[ais_ids]
                    #print(ais_boxes.size())
                    ais_keep = box_nms(ais_boxes, ais_scores, threshold=NMS_THRESH)
                    ais_pred_labels = ais_labels[ais_keep]
                    ais_pred_scores = ais_scores[ais_keep]
                    ais_pred_boxes = ais_boxes[ais_keep]
 
                mia_ids = (obj_labels == 2)
                if mia_ids.long().sum() == 0:
                    mia_pred_boxes = 0
                    mia_pred_scores = 0
                    mia_pred_labels = 0
                else:
                    mia_ids = mia_ids.nonzero().squeeze()
                    #mia_masks = mia_ids.unsqueeze(1).expand_as(obj_boxes)
                    mia_labels = obj_labels[mia_ids]
                    mia_scores = obj_scores[mia_ids]
                    mia_boxes = obj_boxes[mia_ids]
                    mia_keep = box_nms(mia_boxes, mia_scores, threshold=NMS_THRESH)
                    mia_pred_boxes = mia_boxes[mia_keep]
                    mia_pred_scores = mia_scores[mia_keep]
                    mia_pred_labels = mia_labels[mia_keep]
                    #keep = box_nms(masked_boxes[ids], masked_scores[ids], threshold=NMS_THRESH)
                return ais_pred_boxes, ais_pred_scores, ais_pred_labels, mia_pred_boxes, mia_pred_scores, mia_pred_labels


if __name__ == '__main__':
    import os
    import pandas as pd
    import numpy as np
    
    encoder = DataEncoder()
    #patient_ID = "P05471399_000"
    label_file = "/home/youkun/sph_samples/ais_and_mia/samples_bbox.csv"
    label_df = pd.read_csv(label_file)
    for i in range(label_df.shape[0]):
        sample_id = label_df.loc[i, "sample_id"] 
        df = label_df[label_df["sample_id"] == sample_id]
        image_shape = np.array([64, 64, 64])
        label = Annotation(image_shape=image_shape, label_dataframe=df)
        box_arr, label_arr = label.build_annotations()
        box_tensor = torch.Tensor(box_arr)
        label_tensor = torch.LongTensor(label_arr)
        #print(box_tensor, label_tensor)
        loc_target, cls_target = encoder.encode(box_tensor, label_tensor, 64)
        #print(loc_target.size(), cls_target.size())
        pos = cls_target>0
        num_pos = pos.long().sum()
        if num_pos == 0:
            print(sample_id)
