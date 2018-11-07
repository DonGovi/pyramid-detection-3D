#-*-coding:utf-8-*-

import numpy as np
import os
from plot_nodule import plot_arr
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_opening, binary_closing
from skimage import measure, color
import config

def merge_result(result_file, result_path):
    result_arr = np.load(os.path.join(result_path, result_file))
    #print(result_arr.shape)
    label_arr = np.zeros((result_arr.shape[1], result_arr.shape[2], result_arr.shape[3]), dtype=np.float32)
    result_arr = result_arr[0, ...]
    for z in range(label_arr.shape[0]):
        for y in range(label_arr.shape[1]):
            for x in range(label_arr.shape[2]):
                if result_arr[z,y,x,0] >= result_arr[z,y,x,1]:
                    label_arr[z,y,x] = 0
                else:
                    label_arr[z,y,x] = 1
    print(label_arr.shape)

    return label_arr

class Result(object):
    """docstring for Result"""
    def __init__(self, filename, pic_save_path, arr_save_path, origin_path):
        super(Result, self).__init__()
        self.filename = filename
        self.pic_save_path = pic_save_path
        self.arr_save_path = arr_save_path
        self.origin_path = origin_path
    
    def merge(self):
        self.origin = np.load(os.path.join(self.origin_path, self.filename)) # batch_size, 32, 64, 64, channel
        #print(self.origin.shape)
        label = np.zeros((32, 64, 64), dtype=np.int32)
        self.origin = self.origin[0, ...]  # 32, 64, 64, channel
        #print(self.origin.shape)
        for z in range(label.shape[0]):
            for y in range(label.shape[1]):
                for x in range(label.shape[2]):
                    if self.origin[z,y,x,0] >= self.origin[z,y,x,1]:
                        label[z,y,x] = 0
                    else:
                        label[z,y,x] = 1
        #np.save(os.path.join(self.arr_save_path, self.filename), self.label)
        #self.name = self.filename.split(".")[0]
        #plot_arr(self.label, os.path.join(self.pic_save_path, self.name+".png"))
        return label

    def process(self):
        label = self.merge()
        #mask = np.copy(label)
        for i in range(label.shape[0]):
            label_slice = label[i, ...]
            mask_slice = measure.label(label_slice, connectivity=1)
            seed_val = mask_slice[mask_slice.shape[0]//2, mask_slice.shape[1]//2]
            '''
            if np.max(mask_slice) == 1:
                label[i, ...] = label_slice
                continue
            '''
            if label_slice[mask_slice.shape[0]//2, mask_slice.shape[1]//2] == 1:
                label_slice[mask_slice != seed_val] = 0
            else:
                label_slice = 0
            label[i, ...] = label_slice
        #seed = mask[mask.shape[0]//2, mask.shape[1]//2, mask.shape[2]//2]
        #label[mask != seed] = 0
        name = self.filename.split('.')[0]
        np.save(os.path.join(self.arr_save_path, self.filename), label)
        plot_arr(label, os.path.join(self.pic_save_path, name+".png"))





if __name__ == '__main__':
    cfg = config.Config()
    pic_save_path = os.path.join(cfg.p_v3_path, "prob_pictures/")
    arr_save_path = os.path.join(cfg.p_v3_path, "probs/")
    origin_path = os.path.join(cfg.p_v3_path, "results/")
    #file = "001_probs_P10377452.npy"
    file_list = os.listdir(origin_path)
    for file in file_list:
        print("Processing %s" % file)
        result = Result(filename=file, pic_save_path=pic_save_path, arr_save_path=arr_save_path, 
                    origin_path=origin_path)
        result.process()
