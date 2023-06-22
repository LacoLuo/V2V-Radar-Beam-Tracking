import os
import argparse
import tqdm
import torch
import numpy as np 
import pandas as pd
import scipy.io as sio
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from CFAR import CFAR_CA
from RadarKPI import Radar_KPI, Radar_Cube, Range_Doppler, Range_Angle, Doppler_Angle 

def detect_objects(range_doppler_map, min_num_of_objects=5):
    # Prepare CFAR
    threshold = 5
    cell_size = 7
    guard_size = 4
    CFAR_obj = CFAR_CA(threshold=threshold, cell=[cell_size, cell_size], guard=[guard_size, guard_size], square=True, detection_method='+')

    # Prepare DBSCAN
    eps = 7.5
    min_samples = 2
    DBSCAN_obj = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None)

    while threshold >= 1.25:
        try:
            # Perform CFAR
            range_doppler_cfar = CFAR_obj.detect(range_doppler_map)
            idx_range, idx_doppler = np.where(range_doppler_cfar==True)
            points = np.concatenate((np.expand_dims(idx_range, axis=1), np.expand_dims(idx_doppler, axis=1)), axis=1)

            # Clustering
            clusters = DBSCAN_obj.fit(points)
            labels = np.unique(clusters.labels_)
            centroids = np.zeros((len(labels), 2), dtype=int) # (range_idx, doppler_idx)
            for idx, label in enumerate(labels):
                points_in_cluster = points[np.where(clusters.labels_ == label)]
                power = [range_doppler_map[point[0], point[1]] for point in points_in_cluster]
                centroids[idx] = points_in_cluster[np.argmax(power)]

            if centroids.shape[0] >= min_num_of_objects:
                break
            else:
                threshold -= 0.25
                CFAR_obj = CFAR_CA(threshold=threshold, cell=[cell_size, cell_size], guard=[guard_size, guard_size], square=True, detection_method='+')
        except:
            threshold -= 0.25
            CFAR_obj = CFAR_CA(threshold=threshold, cell=[cell_size, cell_size], guard=[guard_size, guard_size], square=True, detection_method='+')

    return centroids

def estimate_angles(centroids, radar_cube):
    angles = np.zeros(len(centroids))
    idx_to_delete = list()
    for i, centroid in enumerate(centroids):
        if centroid[0] <= 12.5 and centroid[1] <= 71 and centroid[1] >= 59: # remove car body
            angles[i] = 90
            idx_to_delete.append(i)
            continue
        
        angle_slice = torch.abs(radar_cube).numpy(force=True)[:, centroid[0], centroid[1]]
        angle_bin_idx = np.argmax(angle_slice) - 32 # -N/2 <= idx < N/2, N=64
        angles[i] = np.arcsin(angle_bin_idx * 2 / 64) / np.pi * 180
    
    if len(angles) > 1 and len(angles) != len(idx_to_delete):
        angles = np.delete(angles, idx_to_delete)
        centroids = np.delete(centroids, idx_to_delete, 0)

    return centroids, angles

def preprocess(args):
    count_per_radar = [0] * 4
    count_per_beam_per_radar = np.zeros((4, 64))
    # Prepare GPU device
    device = torch.device('cuda:0')

    # Read the csv file
    csv_file = 'scenario38_selected.csv'
    csv_filename = csv_file.split('.')[0]

    df = pd.read_csv(csv_file)
    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Get Radar KPI
    radar_kpi = Radar_KPI()

    # Process the beam power first
    optimal_beams = []
    for i, row in enumerate(tqdm.tqdm(df.itertuples(index=False), total=len(df))):
        # Get the receive power of each direction: [64,]
        unit1_pwr1 = np.loadtxt(row.unit1_pwr1, dtype=float)
        unit1_pwr2 = np.loadtxt(row.unit1_pwr2, dtype=float)
        unit1_pwr3 = np.loadtxt(row.unit1_pwr3, dtype=float)
        unit1_pwr4 = np.loadtxt(row.unit1_pwr4, dtype=float)

        # Estimate the optimal beam
        unit1_pwr = np.concatenate((unit1_pwr1, unit1_pwr2, unit1_pwr3, unit1_pwr4))
        optimal_beam = np.argmax(unit1_pwr)
        optimal_beams.append(optimal_beam)
        count_per_radar[optimal_beam//64] += 1
        count_per_beam_per_radar[optimal_beam//64, optimal_beam%64] += 1

    # Process radar data
    radar_feature_column = []
    seq_idx_column = []
    beam_diff = []
    count = 0
    for index, row in enumerate(tqdm.tqdm(df.iloc[:-1].itertuples(index=False), total=len(df)-1)):
        mdic = {}

        if args.sided_radar:
            if optimal_beams[index] // 64 == 0 or optimal_beams[index] // 64 == 2:
                continue

        # Get the radar data of each direction: [# of antennas, # of samples per chirp, # of chirps] = [4, 256, 128]
        unit1_radar1 = sio.loadmat(row.unit1_radar1)['data']
        unit1_radar2 = sio.loadmat(row.unit1_radar2)['data']
        unit1_radar3 = sio.loadmat(row.unit1_radar3)['data']
        unit1_radar4 = sio.loadmat(row.unit1_radar4)['data']
        unit1_radar = [unit1_radar1, unit1_radar2, unit1_radar3, unit1_radar4]

        # Pick the radar data with optimal beam
        unit1_radar = torch.from_numpy(unit1_radar[optimal_beams[index]//64]).to(device)

        # Estimate the radar cube: [64, 256, 128]
        radar_cube = Radar_Cube(unit1_radar, n=64, remove_mean=True)
        
        # Estimate Range-Doppler map: [256, 128]
        range_doppler_map = Range_Doppler(radar_cube, mean=args.mean, log_scale=args.log)
        RD_max_mean_ratio = (torch.max(range_doppler_map) / torch.mean(range_doppler_map)).numpy(force=True)
        range_doppler_map = range_doppler_map.numpy(force=True)
        mdic["range_doppler"] = range_doppler_map
        
        # Estimate Range-Angle map: [256, 64]
        range_angle_map = Range_Angle(radar_cube, mean=args.mean, log_scale=args.log).numpy(force=True)
        mdic["range_angle"] = range_angle_map
        
        # Estimate Doppler-Angle map: [128, 64]
        doppler_angle_map = Doppler_Angle(radar_cube, mean=args.mean, log_scale=args.log).numpy(force=True)
        mdic["doppler_angle"] = doppler_angle_map
        
        # Record the current and next beams
        mdic["current_beam"] = optimal_beams[index]
        mdic["next_beam"] = optimal_beams[index+1]

        # Detect objects in the radar map
        centroids = detect_objects(range_doppler_map)

        # Estimate the detected angle of each object
        centroids, angles = estimate_angles(centroids, radar_cube)

        # Get the angle, range and velocity of the objects
        obj_angle = angles
        obj_range = centroids[:, 0] * radar_kpi.range_res
        obj_velocity = centroids[:, 1] * radar_kpi.velocity_res - radar_kpi.velocity_max

        # Record the detected objects
        mdic["objects"] = centroids
        mdic["obj_angle"] = obj_angle
        mdic["obj_range"] = obj_range
        mdic["obj_velocity"] = obj_velocity

        # Use area of interest for Tx detection
        angle_of_angle_bins = np.arcsin((np.arange(0, 64) - 32) * 2 / 64) / np.pi * 180
        angle_of_optimal_beam = radar_kpi.angle_of_beams[optimal_beams[index] % 64]
        idx_to_remove = np.where((angle_of_angle_bins < angle_of_optimal_beam - 5) | (angle_of_angle_bins > angle_of_optimal_beam + 5))
        radar_cube[idx_to_remove] = 0
        range_doppler_map = Range_Doppler(radar_cube, mean=args.mean, log_scale=args.log).numpy(force=True)
        centroids = detect_objects(range_doppler_map, min_num_of_objects=2)

        # Estimate the detected angle of each Tx candidate
        centroids, angles = estimate_angles(centroids, radar_cube)

        # Estimate the object of the transmitter
        angle_of_optimal_beam = radar_kpi.angle_of_beams[optimal_beams[index] % 64]
        tx_idx = (np.abs(angles - angle_of_optimal_beam)).argmin()
        tx_point = centroids[tx_idx]

        # Get the angle, range and velocity of the transmitter
        tx_angle = angles[tx_idx]
        tx_range = tx_point[0] * radar_kpi.range_res
        tx_velocity = tx_point[1] * radar_kpi.velocity_res - radar_kpi.velocity_max

        # Record the detected Tx
        mdic["tx"] = tx_point
        mdic["tx_angle"] = tx_angle
        mdic["tx_range"] = tx_range
        mdic["tx_velocity"] = tx_velocity

        # Save the processed data
        sio.savemat(os.path.join(out_dir, f"{row.abs_index}.mat"), mdic)

        radar_feature_column.append(os.path.join(out_dir, f"{row.abs_index}.mat"))
        seq_idx_column.append(row.seq_index)
        
        # Calculate the number of non-changing samples
        beam_diff.append(abs(optimal_beams[index+1]-optimal_beams[index]))
        if beam_diff[-1] == 0:
            count += 1

    df_out = pd.DataFrame(list(zip(seq_idx_column, radar_feature_column)), columns=['seq_idx', 'radar_feature'])
    df_out.index.name = 'index'
    df_out.index += 1
    df_out.to_csv(csv_filename + '_preprocess' + '.csv')
    print(f"Number of non-changing samples: {count}/{len(beam_diff)}")

    return row, out_dir ## For testing

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Preprocess the FMCW radar data")
    parser.add_argument(
            "-o", "--out_dir", required=True, type=str,
            help="path of output directory")
    parser.add_argument(
            "-s", "--sided_radar", action='store_true',
            help="use data of radar on the side")
    parser.add_argument(
            "--log", action='store_true',
            help="use log-scale")
    parser.add_argument(
            "--mean", action='store_true',
            help="")
    args = parser.parse_args()

    row, out_dir = preprocess(args)

