#-*-coding:utf-8-*-

import numpy as np
import os
from plot_nodule import plot_arr

def process(sample_file, sample_path, sample_save_path, pic_save_path, threhold=-150.0):
    print("Trying to process %s..." % sample_file)
    array = np.load(os.path.join(sample_path, sample_file))
    array[array >= threhold] = -1000.0

    file_name = sample_file.split(".")[0]
    #print(file_name)
    np.save(os.path.join(sample_save_path, file_name+".npy"), array)
    plot_arr(array, os.path.join(pic_save_path, file_name+".png"))






if __name__ == '__main__':
    sph_path = "D:/SPH_data/padding_samples/"
    sample_path = os.path.join(sph_path, "samples_v1/")
    sample_out_path = os.path.join(sph_path, "samples_v2/")
    picture_out_path = os.path.join(sph_path, "pictures_v2/")
    sample_list = os.listdir(sample_path)
    for sample_file in sample_list:
        process(sample_file, sample_path, sample_out_path, picture_out_path)
