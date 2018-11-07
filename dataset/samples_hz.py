#-*-coding:utf-8-*-

import augmentation_tools as at
import config
import os
from sph_sample import read_scan
import cv2
import pandas as pd
import scipy.ndimage.interpolation
import numpy as np


def resample(image, old_spacing, new_spacing=np.array([1, 1, 1])):
    resize_factor = old_spacing / new_spacing.astype(float)
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = old_spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing

def sample(cfg, cube_size=64):
    label_df = pd.read_csv(cfg.label_file)
    for i in label_df.index:
        case = label_df.loc[i, "patient_ID"]
        case_path = os.path.join(cfg.extract_path, case+"/")
        print(case_path)
        if os.path.exists(case_path):
            scan_arr, old_spacing = read_scan(case_path)
            origin_shape = np.array(scan_arr.shape, dtype=np.float32)
            scan_arr, new_spacing = resample(scan_arr, old_spacing)
            new_shape = np.array(scan_arr.shape, dtype=np.float32)
            coord_xyz = label_df.loc[i, "coord_XYZ"].split(" ")
            coord_x = int(float(coord_xyz[0])/origin_shape[2] * new_shape[2])
            coord_y = int(float(coord_xyz[1])/origin_shape[1] * new_shape[1])
            coord_z = int(float(coord_xyz[2])/origin_shape[0] * new_shape[0])
            edge_z_min = max(0, coord_z-cube_size//2)
            edge_z_max = min(scan_arr.shape[0], coord_z+cube_size//2)
            edge_y_min = max(0, coord_y-cube_size//2)
            edge_y_max = min(scan_arr.shape[1], coord_y+cube_size//2)
            edge_x_min = max(0, coord_x-cube_size//2)
            edge_x_max = min(scan_arr.shape[2], coord_x+cube_size//2)
            nodule_arr = scan_arr[edge_z_min:edge_z_max,
                                  edge_y_min:edge_y_max,
                                  edge_x_min:edge_x_max]
            nodule_arr = (nodule_arr + 1000.0) / 1400.0
            nodule_arr[nodule_arr > 1] = 1
            nodule_arr[nodule_arr < 0] = 0
            nodule_arr *= 255
            if nodule_arr.shape != [cube_size, cube_size, cube_size]:
                if nodule_arr.shape[0] < cube_size:
                    padding_arr = np.zeros((cube_size-nodule_arr.shape[0],
                                            nodule_arr.shape[1], 
                                            nodule_arr.shape[2]),dtype=np.float32)
                    if coord_z-cube_size//2 < 0:
                        nodule_arr = np.concatenate([padding_arr, nodule_arr], axis=0)
                    else:
                        nodule_arr = np.concatenate([nodule_arr, padding_arr], axis=0)
                if nodule_arr.shape[1] < cube_size:
                    padding_arr = np.zeros((nodule_arr.shape[0],
                                            cube_size-nodule_arr.shape[1],
                                            nodule_arr.shape[2]), dtype=np.float32)
                    if coord_y-cube_size//2 < 0:
                        nodule_arr = np.concatenate([padding_arr, nodule_arr], axis=1)
                    else:
                        nodule_arr = np.concatenate([nodule_arr, padding_arr], axis=1)
                if nodule_arr.shape[2] < cube_size:
                    padding_arr = np.zeros((nodule_arr.shape[0],
                                            nodule_arr.shape[1],
                                            cube_size-nodule_arr.shape[2]), dtype=np.float32)
                    if coord_x-cube_size//2 < 0:
                        nodule_arr = np.concatenate([padding_arr, nodule_arr], axis=2)
                    else:
                        nodule_arr = np.concatenate([nodule_arr, padding_arr], axis=2)

            flat_image = at.AugMethod3D.trans_cube_2_flat(nodule_arr)
            save_name = os.path.join(cfg.fhz_path, "nodule_"+str(i).zfill(3)+"_"+case)
            cv2.imwrite(save_name+".png", flat_image)
            cv2.destroyAllWindows()




if __name__ == '__main__':
    cfg = config.Config()
    sample(cfg)