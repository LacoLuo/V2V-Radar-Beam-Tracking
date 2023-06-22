import os 
import numpy as np
import pandas as pd
import scipy.io as sio
import random
import torch
from torch.utils.data import Dataset

def create_samples(data_paths, len_of_sequence, x_size, shuffle=False):
    df = pd.read_csv(data_paths, index_col='index')
    radar_features = df.iloc[:, (len_of_sequence+1)-x_size:len_of_sequence+1].to_numpy(str)
    labels = df.iloc[:, -1].to_numpy(int)
    data_samples = list(zip(radar_features, labels))

    if shuffle:
        random.shuffle(data_samples)

    return data_samples

class DataFeed(Dataset):
    def __init__(self, data_path, len_of_sequence, x_size, init_shuffle=True, normalize=False):
        self.samples = create_samples(data_path, len_of_sequence, x_size, shuffle=init_shuffle)
        
        self.data_path = data_path
        self.x_size = x_size
        self.normalize = normalize
    
    def __len__(self):
        return len(self.samples)

    def _track_Tx(self, tx_infos, object_infos):
        radar_infos = np.zeros((self.x_size, 3))
        radar_infos[0] = tx_infos[0]

        prev_tx_infos = radar_infos[0]
        for i in range(1, self.x_size):
            diff_ranges = np.abs(object_infos[i][:, 0]-prev_tx_infos[0])
            diff_velocitys = np.abs(object_infos[i][:, 1]-prev_tx_infos[1])
            normalized_diff = diff_ranges/50 + diff_velocitys/108
            
            tx_idx = np.argmin(normalized_diff)
            # If the current Tx is too far from the previous Tx on the Range-Doppler map, it is a missing data
            if diff_ranges[tx_idx] > 5 or diff_velocitys[tx_idx] > 10:
                radar_infos[i] = prev_tx_infos
            else:
                radar_infos[i] = prev_tx_infos = object_infos[i][tx_idx]
        return radar_infos

    def __getitem__(self, idx):
        sample = self.samples[idx]
        radar_features = sample[0]
        label = sample[1]
        
        tx_infos = np.zeros((self.x_size, 3))
        object_infos = list()
        for i in range(len(radar_features)):
            radar_feature = sio.loadmat(os.path.join(os.path.dirname(self.data_path), radar_features[i]))
            tx_infos[i, 0] = radar_feature['tx_range']
            tx_infos[i, 1] = radar_feature['tx_velocity']
            tx_infos[i, 2] = radar_feature['tx_angle']
            object_infos.append(np.concatenate((radar_feature['obj_range'].T, radar_feature['obj_velocity'].T, radar_feature['obj_angle'].T), axis=1))

        radar_infos = self._track_Tx(tx_infos, object_infos)

        if self.normalize:
            radar_infos[:, 0] = radar_infos[:, 0] / 50
            radar_infos[:, 1] = (radar_infos[:, 1] + 54) / (54 * 2)
            radar_infos[:, 2] = (radar_infos[:, 2] + 90) / (90 * 2)

        return radar_infos, torch.tensor(label)
