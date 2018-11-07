#-*-coding:utf-8-*-

import os

class Config(object):
    """docstring for Config"""
    def __init__(self):
        self.data_path = "D:/SPH_data/"
        self.p_sample_path = os.path.join(self.data_path, "padding_samples/")
        self.p_v1_path = os.path.join(self.p_sample_path, "v1/")
        self.p_v2_path = os.path.join(self.p_sample_path, "v2/")
        self.p_v3_path = os.path.join(self.p_sample_path, "v3/")
        self.annotation_path = os.path.join(self.data_path, "annotations/")
        self.resample_path = os.path.join(self.data_path, "resample_data/")
        self.origin_path = os.path.join(self.data_path, "data/")
        self.extract_path = os.path.join(self.data_path, "extract_data/")
        self.fhz_path = os.path.join(self.data_path, "samples_for_fhz/")
        self.label_file = os.path.join(self.annotation_path, "label.csv")
        self.roi_results = os.path.join(self.data_path, "unannotated_cube_predict/")
        self.cube_size = 64
        self.deviant_index = [17, 18, 167]