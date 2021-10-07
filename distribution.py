import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from data_parser import get_data_partition



df_ecg= pd.DataFrame()
path='/Muse2021/data/c3_muse_stress/'
vid2partition, partition2vid = get_data_partition(path+'metadata/partition.csv')
feature ='feature_segments/resp'
for video_id,type in vid2partition.items():
    feature_file = os.path.join(path, 'feature_segments/resp/', str(video_id) + '.csv')
    print()
    assert os.path.exists(
        feature_file), f'Error: no available {feature_file}'
    df = pd.read_csv(feature_file)

    df_ecg= pd.concat([df_ecg, df["resp"]], ignore_index=True)
    df_ecg.to_csv('resp.csv')
    




