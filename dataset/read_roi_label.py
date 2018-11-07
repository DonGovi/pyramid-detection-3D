
import os
import cv2
import numpy as np
import pandas as pd
import shutil
import config
import sph_sample as ss
import math
from plot_nodule import plot_arr
import scipy.ndimage.interpolation


def move_file(roi_path):
    file_list = os.listdir(roi_path)

    raw_path = os.path.join(roi_path, "raw/")
    predict_path = os.path.join(roi_path, "predict/")
    if not os.path.exists(raw_path):
        os.makedirs(raw_path)
    if not os.path.exists(predict_path):
        os.makedirs(predict_path)

    for file in file_list:
        attri = file.split("_")[3]
        if attri == "rawimage.png":
            shutil.move(roi_path+file, raw_path+file)
        else:
            shutil.move(roi_path+file, predict_path+file)

class Pulad(object):
    """docstring for Pulad (pulmonary adenocarcinoma)"""
    def __init__(self, label_record, predict_file):
        self.origin_coord = label_record.loc["coord_XYZ"]
        self.patient_id = label_record.loc["patient_ID"]
        self.predict_file = predict_file

    @staticmethod
    def resample(image, old_spacing, new_spacing=np.array([1, 1, 1])):
        resize_factor = old_spacing / new_spacing.astype(float)
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        new_spacing = old_spacing / real_resize_factor

        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

        return image, new_spacing

    def get_newcoord(self):
        # get center coords of tumor after resampling
        origin_x = float(self.origin_coord.split(" ")[0])
        origin_y = float(self.origin_coord.split(" ")[1])
        origin_z = float(self.origin_coord.split(" ")[2])
        old_image, old_spacing = ss.read_scan(os.path.join(cfg.extract_path, self.patient_id))  # zyx
        new_image, new_spacing = Pulad.resample(old_image, old_spacing)   #zyx
        
        if not os.path.exists(cfg.resample_path):
            os.makedirs(cfg.resample_path)
        if not os.path.exists(os.path.join(cfg.resample_path, self.patient_id+".npy")):
            print("Saving resampled CT data %s" % self.patient_id)
            np.save(os.path.join(cfg.resample_path, self.patient_id+".npy"), new_image)
        
        new_z = int(origin_z/old_image.shape[0] * new_image.shape[0])
        new_y = int(origin_y/old_image.shape[1] * new_image.shape[1])
        new_x = int(origin_x/old_image.shape[2] * new_image.shape[2])

        return np.array([new_z, new_y, new_x])

    def bbox_resample(self):
        # get Bounding box coordinates after resampling
        flat_arr_rgb = cv2.imread(self.predict_file)  # (512, 512, 3)
        # this array is a flatten array with shape of (64,64,64)
        flat_arr = np.max(flat_arr_rgb, axis=2)  #(512, 512)
        flat_arr = np.clip(flat_arr, 0, 1)
        n_row = math.sqrt(cfg.cube_size)
        n_col = n_row
        label_arr = np.zeros((cfg.cube_size,cfg.cube_size,cfg.cube_size), dtype=np.int32)
        for i in range(cfg.cube_size):
            row = i // n_row
            col = i % n_col
            label_arr[..., i] = flat_arr[int(row*cfg.cube_size):int(row*cfg.cube_size+cfg.cube_size),
                                         int(col*cfg.cube_size):int(col*cfg.cube_size+cfg.cube_size)]
        index = np.where(label_arr==1)  # tuple (z_index_array, y_index_array, x_index_array)
        if np.max(label_arr) != 0:
            z_min = np.min(index[0])
            z_max = np.max(index[0])
            y_min = np.min(index[1])
            y_max = np.max(index[1])
            x_min = np.min(index[2])
            x_max = np.max(index[2])
            min_coord = np.array([z_min, y_min, x_min], dtype=np.int32)
            max_coord = np.array([z_max, y_max, x_max], dtype=np.int32)
            center_coord = np.array([33,33,33], dtype=np.int32)
            tumor_center = self.get_newcoord()
            min_edge = tumor_center + (min_coord - center_coord)
            max_edge = tumor_center + (max_coord - center_coord)
            cv2.destroyAllWindows()
        else:
            print("%s wasn't segmented" % self.predict_file)
            tumor_center = np.zeros(3)
            min_edge = np.zeros(3)
            max_edge = np.zeros(3)
        return tumor_center, min_edge, max_edge 



def main(cfg):

    label_df = pd.read_csv(cfg.label_file, encoding="utf-8")
    label_df["new_z"] = None
    label_df["new_y"] = None
    label_df["new_x"] = None
    label_df["min_z"] = None
    label_df["min_y"] = None
    label_df["min_x"] = None
    label_df["max_z"] = None
    label_df["max_y"] = None
    label_df["max_x"] = None

    predict_list = os.listdir(os.path.join(cfg.roi_results, "predict/"))
    for predict_name in predict_list:
        #if int(predict_name.split("_")[1]):
        i = int(predict_name.split("_")[1])
        case_name = label_df.loc[i, "patient_ID"]
        print("Starting processing %dth tumor of %s" % (int(i),case_name))
        if os.path.exists(os.path.join(cfg.extract_path, case_name)):
            label_record = label_df.loc[i]
            #file_name = "nodule_" + str(i).zfill(3) + "_" + case_name + "_predict.png"
            predict_file = os.path.join(cfg.roi_results, "predict/", predict_name)
            tumor = Pulad(label_record, predict_file)
            tumor_center, min_edge, max_edge = tumor.bbox_resample()
            label_df.loc[i, "new_z"] = tumor_center[0]
            label_df.loc[i, "new_y"] = tumor_center[1]
            label_df.loc[i, "new_x"] = tumor_center[2]
            label_df.loc[i, "min_z"] = min_edge[0]
            label_df.loc[i, "min_y"] = min_edge[1]
            label_df.loc[i, "min_x"] = min_edge[2]
            label_df.loc[i, "max_z"] = max_edge[0]
            label_df.loc[i, "max_y"] = max_edge[1]
            label_df.loc[i, "max_x"] = max_edge[2]
            
        if i % 10 == 0:
            label_df.to_csv(cfg.annotation_path+"bbox_label.csv", index=False, encoding="utf-8")
            

    label_df.to_csv(cfg.annotation_path+"bbox_label.csv", index=False, encoding="utf-8")



if __name__ == '__main__':
    cfg = config.Config()
    main(cfg)
