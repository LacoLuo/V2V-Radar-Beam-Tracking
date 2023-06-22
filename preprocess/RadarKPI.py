import torch
import numpy as np 
import scipy.io as sio

class Radar_KPI: ## FMCW radar
    def __init__ (self):
        # Radar configuration
        self.N_r = 4 # number of receiver antennas

        # Frame configuration
        self.S = 15 # chirp slope (MHz/us)
        self.f_0 = 77 # starting frequency (GHz)
        self.F_s = 5 # sampling frequency (samples/us)
        self.N_sample = 256 # number of ADC samples per chirp
        self.N_chirp = 128 # number of chirp loops per frame
        self.T_PRI = 65 # chirp repetition interval (us)
        
        # Derived parameters
        self.T_active = self.N_sample / self.F_s # active chirp duration (us)
        self.BW = self.S * self.T_active # chirp bandwidth (MHz)
        self.f_c = self.f_0 + 1e-3 * self.BW / 2 # center frequency (GHz)
        
        # Range KPI
        self.range_res = 3 * 1e8 / (2 * self.BW * 1e6) # range resolution (m)
        self.range_max = self.N_sample * self.range_res # maximum unambiguous detectable range (m)
        
        # Doppler KPI
        self.velocity_res = (3 * 1e8 / (self.f_c * 1e9)) / (2 * self.T_PRI * 1e-6 * self.N_chirp) * 3.6 # velocity resolution (km/hr)
        self.velocity_max = (3 * 1e8 / (self.f_c * 1e9)) / (4 * self.T_PRI * 1e-6) * 3.6 # maximum unambiguous detectable velocity (km/hr)

        # Codebook configuration
        self.codebook_pattern = sio.loadmat('./beam_codebook.mat')['codebook_pattern']
        self.measurement_offset_angle = 4 * np.pi / 180
        self.angle_start = 0 - self.measurement_offset_angle
        self.angle_end = np.pi - self.measurement_offset_angle
        self.num_of_angle = self.codebook_pattern.shape[1]
        self.angle_of_beams = 90 - np.arange(self.angle_start, self.angle_end, (self.angle_end - self.angle_start) / self.num_of_angle)[np.argmax(self.codebook_pattern, axis=1)] / np.pi * 180 ## degree
    
    def print_KPI(self):
        print(f'Active chirp duration = {self.T_active} us\n',
              f'Chirp bandwidth = {self.BW} MHz\n',
              f'Center frequency = {self.f_c} GHz\n',
              f'Range resolution = {self.range_res} m\n',
              f'Maximum range = {self.range_max} m\n',
              f'Velocity resolution = {self.velocity_res} km/hr\n',
              f'Maximum velocity = {self.velocity_max} km/hr')

def Radar_Cube(radar_data, n=64, remove_mean=True):
    # Perform Range-DFT
    range_DFT = torch.fft.fft(radar_data, dim=1) ## [4, 256, 128]
    # Remove DC offset (relatively static objects)
    if remove_mean:
        range_DFT  = range_DFT - torch.mean(range_DFT, dim=2, keepdim=True)
    # Perform Doppler-DFT
    doppler_DFT = torch.fft.fft(range_DFT, dim=2) ## [4, 256, 128]
    # Perform Angle-DFT (Radar Cube)
    angle_DFT = torch.fft.fft(doppler_DFT, n=n, dim=0) ## [64, 256, 128]
    radar_cube = torch.fft.fftshift(angle_DFT, dim=(0, 2))
    return radar_cube

def Range_Doppler(radar_cube, mean=False, log_scale=False):
    if mean:
        range_doppler_map = torch.mean(torch.abs(radar_cube), dim=0) ## [256, 128]
    else:
        range_doppler_map = torch.sum(torch.abs(radar_cube), dim=0) ## [256, 128]
    if log_scale:
        range_doppler_map = torch.log2(1 + range_doppler_map)
    return range_doppler_map

def Range_Angle(radar_cube, mean=False, log_scale=False):
    if mean:
        range_angle_map = torch.t(torch.mean(torch.abs(radar_cube), dim=2)) ## [256, 64]
    else:
        range_angle_map = torch.t(torch.sum(torch.abs(radar_cube), dim=2)) ## [256, 64]
    if log_scale:
        range_angle_map = torch.log2(1 + range_angle_map)
    return range_angle_map

def Doppler_Angle(radar_cube, mean=False, log_scale=False):
    if mean:
        doppler_angle_map = torch.t(torch.mean(torch.abs(radar_cube), dim=1)) ## [128, 64]
    else:
        doppler_angle_map = torch.t(torch.sum(torch.abs(radar_cube), dim=1)) ## [128, 64]
    if log_scale:
        doppler_angle_map = torch.log2(1 + doppler_angle_map)
    return doppler_angle_map
