#-*-coding:utf-8-*-


import SimpleITK as sitk
import numpy as np
import pandas as pd
import dicom
import os
import shutil



sph_data_path = "D:/SPH_data/extra/"
sph_extract_path = "D:/SPH_data/extract_data/"

def find_main_scan(case, sph_data_path):
    dcm_list = os.listdir(os.path.join(sph_data_path,case))
    #print(dcm_list)
    series_uids = []
    count = {}
    for dcm in dcm_list:
        reader = dicom.read_file(os.path.join(sph_data_path,case,dcm))
        seriesUID = reader.data_element("SeriesInstanceUID").value
        if seriesUID in series_uids:
            count[seriesUID] += 1
        else:
            count[seriesUID] = 1
            series_uids.append(seriesUID)

    main_seriesUID = max(count.items(), key=lambda x: x[1])[0]

    return main_seriesUID

def extract_scans(data_path, extract_path):
    scans_list = os.listdir(data_path)
    for case in scans_list:
        if os.path.exists(os.path.join(extract_path, case)):
            print("patient %s exist" % case)
            continue
        else:
            print("starting find patient %s's scans..." % case)
            main_seriesUID = find_main_scan(case, data_path)
            os.mkdir(os.path.join(extract_path, case), mode=0o777)
            dicom_files = os.listdir(os.path.join(data_path, case))
            for dcm in dicom_files:
                reader = dicom.read_file(data_path+case+"/"+dcm)
                seriesUID = reader.data_element("SeriesInstanceUID").value
                if seriesUID == main_seriesUID:
                    shutil.copyfile(data_path+case+"/"+dcm, extract_path+case+"/"+dcm)
                else:
                    continue
            print("patient %s finish" % case)



if __name__ == "__main__":
    
