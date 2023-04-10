#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 09:17:06 2021

@author: katarina
"""

import os
import torch
import numpy as np
import pandas as pd
import h5py
import random
import time
from torch.utils.data import Dataset, random_split, ConcatDataset
from sklearn.model_selection import KFold, train_test_split
from joblib import Parallel, delayed
from tqdm import tqdm

# Random seed is kept 'fixed'
def reinit_rng(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
     

class Spect_Dataset(Dataset):
    
    def __init__(self, config, mode, overlap, n=-1, subset=None, train_idx=None, val_idx=None, transforms=None):
        # Random split is kept 'fixed'
        seed_nr = 0
        reinit_rng(seed_nr)
        # Set up paths
        self.config = config
        self.mode = mode
        self.classes = config.classes
        self.subset = subset
        self.overlap = overlap
        self.filepath = self.config.data_path
        self.subset = subset
        self.cv_fold = self.config.cv_fold
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.transforms = transforms
        
        if self.classes != 'all':
            self.files = [f for f in sorted(os.listdir(self.filepath)) if not f.startswith('.')]
            self.PD_path = self.config.PD_path
            PD_files = get_PD_patients(self.PD_path)
            new_files = []
            for i in self.files:
                if i not in PD_files:
                    new_files.append(i)
            self.filenames = new_files
        
        else:
            self.filenames = [f for f in sorted(os.listdir(self.filepath)) if not f.startswith('.')]
        overlap_flag = False
        
        # Select less data?
        if (n > 0):
            self.filenames = self.filenames[:n]
        self.num_records = len(self.filenames)
        
        # Select subset of data with training/validation/test mode (70:10:20)
        if self.config.stratify == True:
            self.labels = get_h5_label(self.filenames, self.filepath)
            k_fold_dataset, test_dataset = train_test_split(self.filenames, test_size=0.2, random_state=seed_nr, stratify=self.labels)
        else:
            train_size = int(0.7 * len(self.filenames))
            dev_size = int(0.1 * len(self.filenames))
            test_size = len(self.filenames) - train_size - dev_size
            train_all = self.filenames
            train_dataset, dev_dataset, test_dataset = random_split(self.filenames, [train_size, dev_size, test_size])
            k_fold_dataset = ConcatDataset([train_dataset, dev_dataset])
        
        if mode == 'train':
            self.dataset = train_dataset
            if overlap == True:
                overlap_flag = True
        elif mode == 'dev':
            self.dataset = dev_dataset
        elif mode == 'test':
            self.dataset = test_dataset
        elif self.cv_fold is not None:
            if mode == 'k_fold_train':
                self.dataset = k_fold_dataset=[k_fold_dataset[i] for i in self.train_idx]
            elif mode == 'k_fold_val':
                self.dataset = k_fold_dataset=[k_fold_dataset[i] for i in self.val_idx]
        elif mode == 'all':
            self.dataset = train_all

        # Set up data variables
        self.epoch_size = config.epoch_size
        self.n_channels = config.n_channels
        self.eeg_channels = self.config.eeg_channels
        self.emg_channels = self.config.emg_channels

        # Get data indexes
        self.n_jobs = 4
        self.spects = {}
        self.attrs = {}
        self.hyp = {}
        
        # Paralell loop to speed up loading of data sizes
        print(f'Number of recordings: {self.num_records}')
        data = ParallelExecutor(n_jobs=self.n_jobs, prefer='threads')(total=len(self.dataset))(delayed(get_h5_size)(
            filename=os.path.join(self.filepath, record)) for record in self.dataset
            )
        #ssc = ParallelExecutor(n_jobs=self.n_jobs, prefer='threads')(total=len(self.dataset))(delayed(get_h5_ssc)(
            #filename=os.path.join(self.filepath, record)) for record in self.dataset
            #)
        #for record, (data_size, attrs), hyp in zip(self.dataset, data, ssc):
            #if overlap_flag == True:
                #self.spects[record] = {'length': data_size,
                                 #'reduced_length': (int(data_size // self.epoch_size)*2)-1}
            #else:
                #self.spects[record] = {'length': data_size,
                                     #'reduced_length': int(data_size // self.epoch_size)}
        for record, (data_size, attrs) in zip(self.dataset, data):
            if overlap_flag == True:
                self.spects[record] = {'length': data_size,
                                 'reduced_length': (int(data_size // self.epoch_size)*2)-1}
            else:
                self.spects[record] = {'length': data_size,
                                     'reduced_length': int(data_size // self.epoch_size)}
            self.attrs[record] = attrs
            #self.hyp[record] = hyp
        
        # Adding tuples to list
        self.indexes = []
        for i, record in enumerate(self.spects.keys()):
            patient = (i, record)
            for j in np.arange(self.spects[record]['reduced_length']):
                # Overlap train data by 50%
                if overlap_flag == True:
                    start = int((j * self.epoch_size)/2)
                    end = int(start+self.epoch_size)
                    slce = (start, end)
                else:
                    slce = (j * self.epoch_size, (j + 1) * self.epoch_size)
                self.indexes.append((patient, slce))
        # self.indexes = [((i, record), (j * self.epoch_size, (j + 1) * self.epoch_size)) for i, record in enumerate(self.spects.keys())
        #                 for j in np.arange(self.spects[record]['reduced_length'])]
        
    def load_h5(self, filename, position):
        with h5py.File(os.path.join(self.filepath, filename),"r") as f:
            # Extract data chunk
            if self.subset == 'eeg':
                # C3, C4, F3, F4, O1, O2
                data = np.array(f['Spect'][:, position[0]:position[1], :self.eeg_channels])
            elif self.subset == 'emg':
                data = np.array(f['Spect'][:, position[0]:position[1], self.eeg_channels:self.eeg_channels+self.emg_channels])
            elif self.subset == 'eog':
                data = np.array(f['Spect'][:, position[0]:position[1], self.eeg_channels+self.emg_channels:])
            elif self.subset == 'wo_tibia':
                data = np.concatenate((np.array(f['Spect'][:, position[0]:position[1], :6]),np.array(f['Spect'][:, position[0]:position[1], 8:])), axis=-1)
            else:
                data = np.array(f['Spect'][:, position[0]:position[1], :])
            # Get power spectral density (thought this was done previously)    
            #data = 10 * np.log10(data)
            data[data == -np.inf] = 0
            # Pytorch wants data in form (Nc, H, W)
            data = torch.from_numpy(np.transpose(data, (2, 0, 1)))
            if self.transforms is not None:
                data = self.transforms(data)

        return data

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        # Get record ID and epoch position
        record = self.indexes[idx][0][1]
        position = [self.indexes[idx][1][0], self.indexes[idx][1][1]]
        # Extract data and attribures
        data = self.load_h5(record, position)
        attrs = self.attrs[record]
        # Select label
        if self.classes == 'all':
            label = attrs['label']
        else:
            if (attrs['label'] == [0,0]).all():
                label = np.asarray([0])
            else:
                label = np.asarray([1])
        # Output dict
        out = {'fid': record,
                'position': position,
                'data': data,
                'label': torch.from_numpy(label.astype(np.float32)),
                'all_attrs': attrs}
        return out
        

def get_h5_ssc(filename):
    time.sleep(0.05)
    with h5py.File(filename, 'r') as h5:
        ssc = h5['hypno'][:]
    return ssc

def get_h5_size(filename):
    with h5py.File(filename, 'r') as h5:
        data_size = h5['Spect'].shape[1]
        attrs = {}
        for k, v in h5.attrs.items():
            attrs[k] = v
    return data_size, attrs

def get_h5_label(filenames, filepath):
    labels = []
    for i in range(0,len(filenames)):
        j = os.path.join(filepath, filenames[i])
        with h5py.File(j, 'r') as h5:
            label = h5.attrs['label']
            if (label == [0, 0]).all():
                labels.append(0)
                continue
            if (label == [0, 1]).all():
                labels.append(1)
                continue
            if (label == [1, 0]).all():
                labels.append(0)
                continue
            if (label == [1, 1]).all():
                labels.append(1)
                continue
    return labels

def get_PD_patients(csv_path):
    PD_csv = pd.read_csv(csv_path, usecols=[1])
    PD_files = []
    for i in range(len(PD_csv.index)):
        PD_files.append(PD_csv['PD Patient'][i])
    return PD_files


all_bar_funcs = {'tqdm': lambda args: lambda x: tqdm(x, **args),
                 'False': lambda args: iter,
                 'None': lambda args: iter}


def ParallelExecutor(use_bar='tqdm', **joblib_args):

    def aprun(bar=use_bar, **tq_args):

        def tmp(op_iter):
            if str(bar) in all_bar_funcs.keys():
                bar_func = all_bar_funcs[str(bar)](tq_args)
            else:
                raise ValueError('Value %s not supported as bar type' % bar)
            return Parallel(**joblib_args)(bar_func(op_iter))

        return tmp

    return aprun
