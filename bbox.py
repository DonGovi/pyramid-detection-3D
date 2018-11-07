# -*-coding:utf-8-*-

import augmentation_tools as aug
import pandas as pd
import numpy as np
import torch


# import torch.utils.data as data


class BoundingBox(object):
    def __init__(self, edge_array, image_depth, image_height, image_width, label):
        super(BoundingBox, self).__init__()
        self.top = edge_array[0]  # z_min
        self.front = edge_array[1]  # y_min
        self.left = edge_array[2]  # x_min
        self.bottom = edge_array[3]  # z_max
        self.back = edge_array[4]  # y_max
        self.right = edge_array[5]  # x_max
        self.depth = image_depth
        self.height = image_height
        self.width = image_width
        self.label = label
    
    def get_boxes(self):
        box = [self.top, self.front, self.left, self.bottom, self.back, self.right]
        # print(box, self.label)
        return box, self.label
    
    def filp(self, flip_dim):
        if flip_dim == "d":
            top = self.depth - self.bottom
            bottom = self.depth - self.top
            self.top = top
            self.bottom = bottom
        elif flip_dim == "h":
            front = self.height - self.back
            back = self.height - self.front
            self.front = front
            self.back = back
        elif flip_dim == "w":
            left = self.width - self.right
            right = self.width - self.left
            self.left = left
            self.right = right
        
        return self


class Annotation(object):
    '''
    Get tumor annotation list of one scan case
    '''
    
    def __init__(self, image_shape, label_dataframe, flip_dim=None):
        super(Annotation, self).__init__()
        self.patient_df = label_dataframe
        self.flip_dim = flip_dim
        self.depth = image_shape[0]
        self.height = image_shape[1]
        self.width = image_shape[2]
        self.ann_list = self.build_annotations()
    
    def build_annotations(self):
        box_list = []
        label_list = []
        '''
        cols_name and iloc index in dataframe:
        class,max_x,max_y,max_z,min_x,min_y,min_z,sample_id
          0    1      2     3     4     5     6      7
        '''
        for i in range(self.patient_df.shape[0]):
            # print(self.patient_df.iloc[i])
            if isinstance(self.patient_df.iloc[i, 1], float):  # judge the nodule has label
                edge_array = np.zeros(6, dtype=np.int32)
                # print(edge_array)
		
                edge_array[0] = max(0., int(self.patient_df.iloc[i, 6])-2)    # z_min
                edge_array[1] = max(0., int(self.patient_df.iloc[i, 5])-3)    # y_min
                edge_array[2] = max(0., int(self.patient_df.iloc[i, 4])-3)    # x_min
                edge_array[3] = min(int(self.patient_df.iloc[i, 3])+2, 63.)   # z_max
                edge_array[4] = min(int(self.patient_df.iloc[i, 2])+3, 63.)   # y_max
                edge_array[5] = min(int(self.patient_df.iloc[i, 1])+3, 63.)   # x_max
		'''
                edge_array[0] = int(self.patient_df.iloc[i, 6])    # z_min
                edge_array[1] = int(self.patient_df.iloc[i, 5])    # y_min
                edge_array[2] = int(self.patient_df.iloc[i, 4])    # x_min
                edge_array[3] = int(self.patient_df.iloc[i, 3])    # z_max
                edge_array[4] = int(self.patient_df.iloc[i, 2])    # y_max
                edge_array[5] = int(self.patient_df.iloc[i, 1])    # x_max
        ''' 
                label = self.patient_df.iloc[i, 0]
                # print(edge_array)
                box = BoundingBox(edge_array, self.depth, self.height, self.width, label)
                if not self.flip_dim == None:
                    box = box.filp(self.flip_dim)
                bbox, label = box.get_boxes()
                # print(bbox, label)
                box_list.append(bbox)
                label_list.append(label)
        
        box_arr = np.array(box_list, dtype=np.float32)
        label_arr = np.array(label_list, dtype=np.int64)
        return torch.FloatTensor(box_arr), torch.LongTensor(label_arr)


if __name__ == '__main__':
    patient_ID = "P05471399_000"
    label_file = "E:/sph_samples/samples_128/samples_bbox.csv"
    label_df = pd.read_csv(label_file)
    df = label_df[label_df["sample_id"] == patient_ID]
    # print(df.iloc[0,7])
    image_shape = np.array([128, 128, 128])
    label = Annotation(image_shape=image_shape, label_dataframe=df, flip_dim="d")
    box_tsr, label_tsr = label.build_annotations()
    print(box_tsr, label_tsr)
