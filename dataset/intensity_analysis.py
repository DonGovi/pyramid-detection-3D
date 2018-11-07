#-*-coding:utf-8-*-

import numpy as np
from sph_sample import read_scan
import os
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    sph_path = "D:/SPH_data/"
    data_path = os.path.join(sph_path, "extract_data/")
    label_file = os.path.join(sph_path, "annotations/label.csv")
    sample_path = os.path.join(sph_path, "padding_samples/samples_v1/")
    '''
    nodule_df = pd.read_csv(label_file)
    patient_id = nodule_df.loc[1, "patient_ID"]
    print(patient_id)
    coord = nodule_df.loc[1, "coord_XYZ"]
    coord_x = int(coord.split(" ")[0])
    coord_y = int(coord.split(" ")[1])
    coord_z = int(coord.split(" ")[2])
    image_arr = read_scan(os.path.join(data_path, patient_id))  # zyx
    print(image_arr[coord_z, coord_y-4:coord_y+4, coord_x-4:coord_x+4])
    '''
    sample_file = os.path.join(sample_path, "nodule_001_P10377452.npy")
    sample_arr = np.load(sample_file)
    patch = sample_arr[8, 16:48, 40:48]
    plt.subplot(111)
    plt.imshow(patch, cmap="gray")
    plt.show()

    print(patch)

