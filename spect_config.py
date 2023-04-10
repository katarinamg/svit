#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 09:40:04 2021

@author: katarina
"""

import os

class spect_config():
    
    def __init__(self, parameters):
        
        self.data_path = parameters.data_path
        self.epoch_size = parameters.epoch_size
        self.n_channels = parameters.n_channels
        self.dropout_1_1 = parameters.dropout1_1
        self.dropout_1_2 = parameters.dropout1_2
        self.dropout_2 = parameters.dropout2
        self.eeg_channels = 6
        self.emg_channels = 3
        
        # Model parameters
        self.learning_rate = parameters.lr #00001 #0.0001
        self.num_epochs = 10
        
        self.classes = 'rbd' #rbd
        self.PD_path = os.path.expanduser('~/Desktop/Thesis/PD_test_patient_list.csv')
        #self.PD_path = os.path.expanduser('~/Desktop/Thesis/Data/PD_RBD_patient_list.csv')
        self.cv_fold = 5
        self.stratify = False
        
    


        