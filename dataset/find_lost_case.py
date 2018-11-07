#-*-coding:utf-8-*-

import pandas as pd
import numpy as np
import os
import config

if __name__ == '__main__':
    cfg = config.Config()
    label_df = pd.read_csv(cfg.label_file)
    drop_dup_df = label_df.drop_duplicates("patient_ID")
    patient_list = drop_dup_df["patient_ID"].tolist()
    sample_list = os.listdir("p_v2_path")

