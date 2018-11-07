#-*-coding:utf-8-*-

import sph_sample as ss
from plot_nodule import plot_arr
import numpy as np
import pandas as pd
import config
import dicom
import os
import shutil


def extract_scans(data_path, extract_path, seriesUID):

            #main_seriesUID = find_main_scan(case, data_path)
    #os.mkdir(extract_path, mode=0o777)
    dicom_files = os.listdir(data_path)
    for dcm in dicom_files:
        reader = dicom.read_file(data_path+"/"+dcm)
        sliceUID = reader.data_element("SeriesInstanceUID").value
        if sliceUID == seriesUID:
            shutil.copyfile(data_path+"/"+dcm, extract_path+"/"+dcm)
        else:
            continue

def sample_deviant(case, cfg):
    label_df = pd.read_csv(os.path.join(cfg.annotation_path, "label.csv"))
    case_df = label_df[label_df["patient_ID"] == case]
    print(case_df)
    count = 1
    scan_path = os.path.join(cfg.extract_path, case+"/")
    image_array = ss.read_scan(scan_path)
    print(image_array.shape)
    for i in case_df.index:
        coord_x = int(case_df.loc[i, "coord_XYZ"].split(" ")[0])
        coord_y = int(case_df.loc[i, "coord_XYZ"].split(" ")[1])
        coord_z = int(case_df.loc[i, "coord_XYZ"].split(" ")[2])
        edge_z_min = coord_z-cfg.cube_size//4
        edge_z_max = coord_z+cfg.cube_size//4
        edge_y_min = coord_y-cfg.cube_size//2
        edge_y_max = coord_y+cfg.cube_size//2
        edge_x_min = coord_x-cfg.cube_size//2
        edge_x_max = coord_x+cfg.cube_size//2
        nodule_array = image_array[edge_z_min:edge_z_max,
                                  edge_y_min:edge_y_max,
                                  edge_x_min:edge_x_max]
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
        save_name = os.path.join(cfg.p_v2_path, "deviant/", case+"_"+str(count).zfill(2))
        print(save_name)
        np.save(save_name+".npy", nodule_64_array)
        plot_arr(nodule_64_array, save_name+".png")
        count += 1



if __name__ == '__main__':
    seriesUID = "1.3.12.2.1107.5.1.4.65018.30000016050712025118100482153"
    cfg = config.Config()
    data_path = os.path.join(cfg.origin_path, "0800441455/")
    extract_path = os.path.join(cfg.extract_path, "0800441455/")
    case = "0800441455"
    sample_deviant(case, cfg)


