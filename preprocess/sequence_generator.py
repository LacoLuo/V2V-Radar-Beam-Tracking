import os
import argparse
import tqdm
import copy
import numpy as np 
import pandas as pd
import seaborn as sns
import scipy.io as sio
import matplotlib.pyplot as plt
from RadarKPI import Radar_KPI

def get_preprocessed_radar_data(csv_frame):
    num_of_data = len(csv_frame)
    data_labels = np.empty((0, 1), dtype=int)
    radar_idxs = np.empty((0, 1), dtype=int)
    for i in tqdm.tqdm(np.arange(len(csv_frame)), desc="Preprocessed Radar Data"):
        data = sio.loadmat(csv_frame.iloc[i]['radar_feature'])
        beam_idx = data['current_beam'] % 64
        radar_idx = data['current_beam'] // 64
        data_labels = np.append(data_labels, beam_idx)
        radar_idxs = np.append(radar_idxs, radar_idx)

    return data_labels, radar_idxs

class TimeSeriesGenerator:
    def __init__(self, csv_file, x_size, y_size, delay, seed, seq_index_column):
        self.radar_kpi = Radar_KPI()

        self.csv_filename = csv_file.split('.')[0]
        self.csv_frame = pd.read_csv(csv_file, index_col='index')
        assert len(self.csv_frame) == self.csv_frame.index.max()

        self.data_labels, self.radar_idxs = get_preprocessed_radar_data(self.csv_frame)

        self.sequences = self.csv_frame['seq_idx'].unique()
        self.num_sequences = len(self.csv_frame['seq_idx'].unique())

        self.seq_start = []
        self.seq_end = []
        self._extract_seq_start_end(seq_index_column)

        self.x_size = x_size
        self.y_size = y_size
        self.delay = delay

        self.data_list_x = np.empty((0, x_size), dtype=int) # indices of input samples
        self.data_list_y = np.empty((0, y_size), dtype=int)
        self.data_list_seq = np.empty((0, y_size), dtype=int)

        self._generate_indices()

        self.data_list_labels = self.data_labels[self.data_list_y - 1]
        self.data_list_x_labels = self.data_labels[self.data_list_x - 1]

        # Shuffling indices
        self.num_datapoints = len(self.data_list_y)
        self.data_idx = np.arange(self.num_datapoints)
        rng = np.random.default_rng(seed)
        #rng.shuffle(self.data_idx)

        # Shuffling sequences
        self.seq_idx = np.arange(self.num_sequences)
        rng = np.random.default_rng(seed)
        rng.shuffle(self.seq_idx)

    def _extract_seq_start_end(self, seq_index_column):
        for i in np.arange(self.num_sequences):
            data_indices = self.csv_frame[self.csv_frame[seq_index_column] == self.sequences[i]].index
            self.seq_start.append(data_indices.min())
            self.seq_end.append(data_indices.max())

    def _generate_indices(self):
        for i in range(len(self.seq_start)):
            x_start_ind = self.seq_start[i]
            x_end_ind = x_start_ind + self.x_size + 0

            y_start_ind = x_end_ind + self.delay + 0
            y_end_ind = y_start_ind + self.y_size

            while y_end_ind <= self.seq_end[i] + 1:
                radar_diff = 0
                for j in np.arange(x_start_ind, y_end_ind-1, 1):
                    curr_radar_idx = self.radar_idxs[j - 1]
                    next_radar_idx = self.radar_idxs[j]
                    if curr_radar_idx != next_radar_idx:
                        radar_diff = 1
                        break
                if radar_diff == 1: # remove cross-radar sequence
                    x_start_ind += 1
                    x_end_ind += 1
                    y_start_ind += 1
                    y_end_ind += 1
                    continue

                self.data_list_x = np.vstack((self.data_list_x, np.arange(x_start_ind, x_end_ind, 1)))
                self.data_list_y = np.vstack((self.data_list_y, np.arange(y_start_ind, y_end_ind, 1)))
                self.data_list_seq = np.append(self.data_list_seq, i)
                x_start_ind += 1
                x_end_ind += 1
                y_start_ind += 1
                y_end_ind += 1

    def take_by_idx(self, idx):
        new_dataset = copy.copy(self)
        new_dataset.data_idx = new_dataset.data_idx[idx]
        new_dataset.num_datapoints = len(new_dataset.data_idx)
        return new_dataset

    def __len__(self):
        return self.num_datapoints

    def save_split_files(self, split=(0.6, 0.3, 0.1), data_path_csv_column=None, split_names=('train', 'val', 'test'), sequence_split=False, label_name='beam_index'):
            assert len(split) == len(split_names)

            idx_list = []
            if sequence_split:
                num_sequences = self.num_sequences

                init_idx = 0
                for i in range(len(split)-1):
                    end_idx = init_idx + int(num_sequences * split[i])
                    idx_list.append(np.where(np.in1d(self.data_list_seq[self.data_idx], self.seq_idx[init_idx:end_idx])))
                    init_idx = end_idx
                idx_list.append(np.where(np.in1d(self.data_list_seq[self.data_idx], self.seq_idx[init_idx:])))
            else:
                num_datapoints = len(self)
                init_idx = 0
                for i in range(len(split)-1):
                    end_idx = init_idx + int(num_datapoints * split[i])
                    idx_list.append(np.arange(init_idx, end_idx))
                    init_idx = end_idx
                idx_list.append(np.arange(init_idx, num_datapoints))

            for n, name in enumerate(split_names):
                self.take_by_idx(idx_list[n]).save_file(file_nametag=name,
                                                        data_path_csv_column=data_path_csv_column,
                                                        label_name=label_name,
                                                        shuffled=True)
                if name == 'test':
                    self.take_by_idx(idx_list[n]).calculate_sample_hold_accuracy(name)

    def save_file(self, data_path_csv_column=None, label_name='beam_index', shuffled=False, file_nametag=''):

        save_filename = self.csv_filename + '_series' + '_' + file_nametag + '.csv'

        df_x = pd.DataFrame(self.csv_frame[data_path_csv_column].to_numpy(str)[self.data_list_x - 1],
                            columns=[data_path_csv_column + '_%i' % (i + 1) for i in range(self.x_size)])
        df_y = pd.DataFrame(self.data_list_labels,
                            columns=['%s_%i' % (label_name, i + 1) for i in range(self.data_list_labels.shape[1])])

        df = pd.concat([df_x, df_y], axis=1)
        df.index.name = 'index'
        df.index += 1

        if shuffled:
            df = df.iloc[self.data_idx]
            df.index.name = 'data_index'
            df = df.reset_index()
            df.index.name = 'index'
            df.index += 1

        filename = save_filename
        df.to_csv(filename)
        print('%i data points are saved to %s' % (len(df), filename))

    def calculate_sample_hold_accuracy(self, tag):
        sample_hold_top1_acc = np.zeros((self.x_size,))
        sample_hold_top3_acc = np.zeros((self.x_size))
        sample_hold_top5_acc = np.zeros((self.x_size))
        beam_diff = list()
        for i in self.data_idx:
            label = self.data_list_labels[i, -1]
            for j in range(self.x_size):
                beam_index = self.data_labels[self.data_list_x[i][j]-1]
                if beam_index == label:
                    sample_hold_top1_acc[j] += 1
                if beam_index == label or beam_index-1 == label or beam_index+1 == label:
                    sample_hold_top3_acc[j] +=1
                if beam_index == label or beam_index-1 == label or beam_index+1 == label or beam_index-2 == label or beam_index+2 == label:
                    sample_hold_top5_acc[j] +=1
                if j == 0:
                    beam_diff.append(np.abs(label-beam_index))

        print(f'Top-1 Accuracy of Sample-Hold on {tag} data: {np.round(sample_hold_top1_acc / self.num_datapoints, 2)}')
        print(f'Top-3 Accuracy of Sample-Hold on {tag} data: {np.round(sample_hold_top3_acc / self.num_datapoints, 2)}')
        print(f'Top-5 Accuracy of Sample-Hold on {tag} data: {np.round(sample_hold_top5_acc / self.num_datapoints, 2)}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Generate sequence dataset")
    parser.add_argument(
            "-f", "--csv_file", required=True, type=str,
            help="csv filename")
    parser.add_argument(
            "-x", "--x_size", required=True, type=int,
            help="size of input samples")
    parser.add_argument(
            "-y", "--y_size", required=True, type=int,
            help="size of label samples")
    parser.add_argument(
            "-d", "--delay", default=-1, type=int,
            help="number of samples between x and y sequences")
    args = parser.parse_args()

    csv_file = args.csv_file

    x_size = args.x_size
    y_size = args.y_size
    delay = args.delay

    rng_seed = 0

    seq_index_column = 'seq_idx'
    data_path_csv_column = 'radar_feature'

    label_name = 'beam_index'

    sequence_split = False

    x = TimeSeriesGenerator(csv_file=csv_file,
                            x_size=x_size,
                            y_size=y_size,
                            seed=rng_seed,
                            delay=delay,
                            seq_index_column=seq_index_column)
    x.save_file(data_path_csv_column=data_path_csv_column)
    x.calculate_sample_hold_accuracy('all')

    x.save_split_files(split=(0.7, 0, 0.3),
                       data_path_csv_column=data_path_csv_column,
                       label_name=label_name,
                       sequence_split=sequence_split)
