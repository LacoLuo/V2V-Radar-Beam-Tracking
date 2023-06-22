# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 14:01:02 2021

@author: demir
"""

import numpy as np
from scipy import ndimage

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class CFAR_CA:
    def __init__(self, threshold = 2, cell = [2, 2], guard = [1, 1], square=True, detection_method = '+'):
        
        self.threshold = threshold
        self.detection_method = detection_method
        cell_size = np.array(cell)
        guard_size = np.array(guard)
        
        if square:
            CFAR_filter = np.ones(tuple(2*(cell_size+guard_size)+1))
            idx = np.ix_(*[np.arange(cell_size[i], cell_size[i]+2*guard_size[i]+1) for i in range(len(CFAR_filter.shape))])
            CFAR_filter[idx] = 0
        else: # Plus sign type of filter
            CFAR_filter = np.zeros(tuple(2*(cell_size+guard_size)+1))
            for i in range(len(CFAR_filter.shape)):
                key_elem = np.arange(cell_size[i], -cell_size[i], -1)-1
                idx_sh = np.tile(cell_size+guard_size, (len(key_elem), 1)).T
                idx_sh[i] = key_elem
                idx = np.ix_(*idx_sh)
                CFAR_filter[idx] = 1

        CFAR_filter = CFAR_filter/np.sum(CFAR_filter)
        
        self.filter = CFAR_filter
    
    def visualize_filter(self):
        filter_dim = len(self.filter.shape)
        if filter_dim <= 3 and filter_dim >= 1:
            f = self.filter
            while len(f.shape) < 3:
                f = np.expand_dims(f, axis=-1)
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            cmap = plt.get_cmap('seismic')
            norm = plt.Normalize(f.min(), f.max())
            ax.voxels(np.ones_like(f), facecolors=cmap(norm(f)), edgecolor='black', alpha=0.5)
            ax.set_box_aspect((f.shape[0], f.shape[1], f.shape[2]))
            plt.title('CA-CFAR Filter')
            plt.show()
        else:
            raise NotImplementedError
    
    def detect(self, input_data, mode='mirror'):
        if self.detection_method == '+':
            detection_th = self.threshold + ndimage.convolve(input_data, self.filter, mode='mirror')
        elif self.detection_method == '*':
            detection_th = self.threshold * ndimage.convolve(input_data, self.filter, mode='mirror')
        else:
            raise NotImplementedError
        return input_data>detection_th
    
