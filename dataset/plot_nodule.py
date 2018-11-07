#-*-coding:utf-8-*-

import numpy as np
import matplotlib.pyplot as plt
import os
from augmentation_tools import AugMethod3D

def plot_arr(array, save_name):
    #array[array>=threhold] = 1
    #array[array<threhold] = 0
    fig, axes = plt.subplots(8, 8, figsize=(64, 64))
    count = 0
    for i in range(4):
        for j in range(8):
            axes[i,j].imshow(array[count,...], cmap="gray")
            count += 1
    #fig.tight_layout()
    plt.imshow(save_name)
    plt.close('all')


if __name__ == '__main__':
    sample_path = "D:/SPH_data/samples/"
    output_path = "D:/SPH_data/sample_pictures/"

    sample_list = os.listdir(sample_path)
    for sample in sample_list:
        array = np.load(os.path.join(sample_path, sample))
        save_name = os.path.join(output_path, sample.split(".")[0] + ".png")
        plot_arr(array, save_name)