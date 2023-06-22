import os 
import numpy as np
import pandas as pd
import scipy.io as sio
import random
import torch
from torch.utils.data import Dataset

def create_samples(data_paths, len_of_sequence, x_size, shuffle=False):
    df = pd.read_csv(data_paths, index_col='index')
    radar_feature = df.iloc[:, (len_of_sequence+1)-x_size:(len_of_sequence+1)].to_numpy(str)
    labels = df.iloc[:, -1].to_numpy(int)
    data_samples = list(zip(radar_feature, labels))

    if shuffle:
        random.shuffle(data_samples)

    return data_samples

class DataFeed(Dataset):
    def __init__(self, data_path, feature, len_of_sequence, x_size, init_shuffle=True, normalize=False):
        self.samples = create_samples(data_path, len_of_sequence, x_size, shuffle=init_shuffle)

        self.data_path = data_path
        self.feature = feature
        self.x_size = x_size
        self.normalize = normalize
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        radar_features = sample[0]
        label = sample[1]
        
        if self.feature == "RD":
            radar_maps = torch.zeros((self.x_size, 256, 128))
        elif self.feature == "DA":
            radar_maps = torch.zeros((self.x_size, 128, 64))
        elif self.feature == "RA":
            radar_maps = torch.zeros((self.x_size, 256, 64))

        prev_beams = torch.zeros((self.x_size,))
        for i in range(len(radar_features)):
            radar_feature = sio.loadmat(os.path.join(os.path.dirname(self.data_path), radar_features[i])) ## The path might need to be customized
            if self.feature == "RD":
                radar_map = radar_feature['range_doppler']
                if self.normalize:
                    radar_map = (radar_map - radar_map.min()) / (radar_map.max() - radar_map.min())
                radar_maps[i] = torch.from_numpy(radar_map)
            elif self.feature == "RA":
                radar_map = radar_feature['range_angle']
                if self.normalize:
                    radar_map = (radar_map - radar_map.min()) / (radar_map.max() - radar_map.min())
                radar_maps[i] = torch.from_numpy(radar_map)
            elif self.feature == "DA":
                radar_map = radar_feature['doppler_angle']
                if self.normalize:
                    radar_map = (radar_map - radar_map.min()) / (radar_map.max() - radar_map.min())
                radar_maps[i] = torch.from_numpy(radar_map)
            prev_beam = radar_feature['current_beam'][0][0]
            prev_beams[i] = torch.tensor(prev_beam % 64, requires_grad=False)


        return radar_maps, prev_beams, torch.tensor(label, requires_grad=False)
