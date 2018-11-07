#-*-coding:utf-8-*-

import numpy as np
import pandas as pd
import dicom
import os
import math
import plot_nodule as pn
import SimpleITK as sitk
import matplotlib.pyplot as plt
import config


def read_scan(scan_path):
    dcm_list = os.listdir(scan_path)
    slice_dict = {}
    for dcm_file in dcm_list:
        reader = dicom.read_file(os.path.join(scan_path,dcm_file))
        num = reader.data_element("InstanceNumber").value
        slice_dict[dcm_file] = num
        del reader, num
    slice_dict = sorted(slice_dict.items(), key=lambda x:x[1])
    dcm_file = slice_dict[0][0]
    example_reader = dicom.read_file(os.path.join(scan_path,dcm_file))
    spacing_z = example_reader.SliceThickness
    spacing_xy = example_reader.PixelSpacing
    #print(spacing_xy)
    spacing_x = float(spacing_xy[0])
    spacing_y = float(spacing_xy[1])
    old_spacing = np.array([spacing_z, spacing_y, spacing_x], dtype=np.float32)
    example = sitk.ReadImage(os.path.join(scan_path,dcm_file))
    example_slice = sitk.GetArrayFromImage(example)
    #print(slice_dict)
    image_array = np.zeros((len(slice_dict), example_slice.shape[1], example_slice.shape[2]), dtype=np.float32)

    for i in range(len(slice_dict)):
        dcm_file = slice_dict[i][0]
        image = sitk.ReadImage(os.path.join(scan_path,dcm_file))
        image_slice = sitk.GetArrayFromImage(image)   #zyx
        image_array[i, ...] = image_slice[0, ...]
        #print(image_array.shape)
        del image, image_slice

    del slice_dict

    return image_array, old_spacing
    

def sample_sph(cfg, threhold=-150.0):
    label_df = pd.read_csv(cfg.label_file)
    #count = 0
    non_case = []
    for i in label_df.index:
        scan_path = label_df.loc[i, "patient_ID"]
        print("Start sample nodule_%d of %s" % (int(i), scan_path))
        if os.path.exists(os.path.join(cfg.extract_path, scan_path)) and int(i) not in cfg.deviant_index:
            scan_array = read_scan(os.path.join(cfg.extract_path, scan_path))

            nodule_coords = label_df.loc[i, "coord_XYZ"].split(" ")
            nodule_x = int(nodule_coords[0])
            nodule_y = int(nodule_coords[1])
            nodule_z = int(nodule_coords[2]) 
            edge_z_min = nodule_z-cfg.cube_size//4
            edge_z_max = nodule_z+cfg.cube_size//4
            edge_y_min = nodule_y-cfg.cube_size//2
            edge_y_max = nodule_y+cfg.cube_size//2
            edge_x_min = nodule_x-cfg.cube_size//2
            edge_x_max = nodule_x+cfg.cube_size//2
            print(edge_z_min, edge_z_max, edge_y_min, edge_y_max, edge_x_min, edge_x_max)
            nodule_array = scan_array[edge_z_min:edge_z_max, edge_y_min:edge_y_max, edge_x_min:edge_x_max]
            
            nodule_64_array = np.ones((32, 64, 64), dtype=np.float32)
            nodule_64_array *= -1000.0
            padding_z_min = (nodule_64_array.shape[0] - nodule_array.shape[0]) // 2
            padding_z_max = padding_z_min + nodule_array.shape[0]
            padding_y_min = (nodule_64_array.shape[1] - nodule_array.shape[1]) // 2
            padding_y_max = padding_y_min + nodule_array.shape[1]
            padding_x_min = (nodule_64_array.shape[2] - nodule_array.shape[2]) // 2
            padding_x_max = padding_x_min + nodule_array.shape[2]

            nodule_64_array[padding_z_min:padding_z_max, 
                            padding_y_min:padding_y_max,
                            padding_x_min:padding_x_max] = nodule_array
            nodule_64_array[nodule_64_array > threhold] = -1000.0
            
            save_name = "nodule_"+str(int(i)).zfill(3)+"_"+scan_path
            picture_path = os.path.join(cfg.p_v3_path, "sample_pictures/")
            output_path = os.path.join(cfg.p_v3_path, "samples/")
            if not os.path.exists(picture_path):
                os.makedirs(picture_path)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            print("Save nodule %d of %s" % (int(i), scan_path))
            pn.plot_arr(nodule_64_array, picture_path+save_name+".png")
            np.save(output_path+save_name+".npy", nodule_64_array)
            #count += 1
            del scan_array, nodule_array, nodule_64_array
        else:
            non_case.append(scan_path)

    return non_case



if __name__ == '__main__':

    cfg = config.Config()
    non_case = sample_sph(cfg)
    print(non_case)
    '''
    file = "D:/SPH_data/extract_data/0800419438/"
    image_array = read_scan(file)
    '''
    
