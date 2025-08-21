#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 12:18:11 2021

@author: katarina
"""

# Functions for extracting spectrogram and hynpnogram data (preprocessing)

import pandas as pd
import numpy as np
import torch
import re
import datetime
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
from fractions import Fraction
from scipy.signal import resample_poly
from datetime import timedelta
from tqdm import tqdm
from scipy.signal import resample_poly, iirfilter, sosfiltfilt, sosfreqz

def psg_highpass(cutoff, fs, order=5, plot_opt = 0):
    nyq = 0.5 * fs
    if isinstance(cutoff, list):
        normal_cutoff = [x / nyq for x in cutoff]
        btype = 'bandpass'
    else:
        normal_cutoff = cutoff / nyq
        btype = 'highpass'

    sos = iirfilter(order, normal_cutoff, rp = 1, rs = 40, btype=btype, analog=False, output='sos', ftype='ellip')
    if plot_opt == 1:
        w, h = sosfreqz(sos, 2**12)
        plt.plot(w*fs/(2*np.pi), 20 * np.log10(abs(h)))
        plt.title('Filter frequency response')
        plt.xlabel('Frequency [radians / second]')
        plt.ylabel('Amplitude [dB]')
        if btype == 'highpass':
            plt.xlim(0, cutoff*2)
            plt.axvline(cutoff, color='green') # cutoff frequency
        else:
            plt.xlim(0, cutoff[-1] + (cutoff[-1] - cutoff[0])*0.5)
            plt.axvline(cutoff[0], color='green') # cutoff frequency
            plt.axvline(cutoff[-1], color='green') # cutoff frequency
        plt.margins(0, 0.1)
        plt.grid(which='both', axis='both')
        plt.show()
    return sos

def psg_highpass_filter(data, cutoff, fs, order=5, plot_opt = 0):
    sos = psg_highpass(cutoff, fs, order=order, plot_opt=plot_opt)
    y = sosfiltfilt(sos, data)
    return y

def get_mean_std(loader):
    mean = 0.
    std = 0.
    for batch in loader:
        x = batch['data']
        batch_samples = x.size(0) # batch size (the last batch can have smaller size!)
        x_squeeze = x.view(batch_samples, x.size(1), -1)
        mean += x_squeeze.mean(2).sum(0)
        std += x_squeeze.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    return mean, std

def extract_marker_t(marker_path, des_fs):
    
    # Init variables
    result = None
    l_off = None
    l_on = None
    FMT = '%H:%M:%S,%f'
    
    # Open marker file and find times
    with open(marker_path, encoding='utf8', errors='ignore') as f:
        for line in f:
            result = re.search('((\d*),)', line)
            if result is not None:
                l_off = result.group(2)
            result = re.search(',(\d*)', line)
            if result is not None:
                l_on = result.group(1)
                
    l_off = int(l_off)
    l_on = int(l_on)
    
    return l_off, l_on

def extract_marker_t_during(header):
    
    # Init variables
    result1 = None
    result2 = None
    l_off = None
    l_on = None

    for el in header['annotations']:
        el = str(el)
        result1 = re.search('([0-9.]*),\s-[0-9],\s.([Ll]ights\s[Oo]n.|LIGHTS\sON.)', el)
        if result1 is not None:
            l_on = result1.group(1)
        result2 = re.search('([0-9.]*),\s-[0-9],\s.([Ll]ights\s[Oo]ff.|LIGHTS\sOUT.)', el)
        if result2 is not None:
            l_off = result2.group(1)
    if l_off is not None and l_on is not None:            
        l_off = int(float(l_off))
        l_on = int(float(l_on))
    
    return l_off, l_on

def extract_marker_t_STAGES(csv_path, ssc):
    
    # Init variables
    l_off = None
    l_on = None

    df = pd.read_csv(csv_path, header=None)
    for i in range(len(df.index)):
        start_rec = pd.to_datetime(df.iloc[1][0])
        if df.iloc[i][2] == ' LightsOn':
            l_on = pd.to_datetime(df.iloc[i][0])
        if df.iloc[i][2] == ' LightsOff':
            l_off = pd.to_datetime(df.iloc[i][0])
        
    if l_off is None and len(ssc) > 0:  
        first_sleep_list = [i for i, n in enumerate(ssc) if (n != 1)]
        l_off = pd.to_datetime(df.iloc[first_sleep_list[0]][0])
    if l_on is None and len(ssc) > 0:
        last_sleep_list = [i for i,n in enumerate(list(reversed(ssc))) if (n !=1)]
        idx = len(ssc) - last_sleep_list[0] - 1
        l_on = pd.to_datetime(df.iloc[idx][0])
    l_off = int(float((l_off-start_rec).total_seconds()))
    l_on= int(float((l_on-start_rec).total_seconds()))   

    return l_off, l_on

def extract_hypnogram(hypnogram_path, l_off, l_on):
    # Extracts text based hypnogram and returns numbered list
    # representing sleep stages.
    # INPUTS: hypnogram_file = file path, start_t = time at lights off in seconds
    # format, end_t = time at lights on in datetime format
    # OUTPUTS: list of sleep stages
    # Wake = 1, REM = 0, N1 = -1, N2 = -2, N3 = -3
    
    # Initialise hypnogram
    hypno = []
    
    # read in file
    df = pd.read_csv(hypnogram_path,
             header=None,
             names=['state'])
    
    if l_on == 0:
        l_on = df.index[len(df.index)-1]
    
    # add code to list for relevant time period
    for i in range(len(df.index)):
        if l_off == 0:
            sleep_map = [0,-1,-2,-3]
            if df.iloc[i]['state'] in sleep_map:
                l_off = df.index[i]*30
                break
            else:
                l_off = None
        
        if l_off is not None and l_on != 0:
            t = df.index[i]
            seconds = t*30
            if seconds >= l_off and seconds <= l_on:
                hypno.append(df.iloc[i]['state'])
        else:
            hypno = None

    
    return hypno, l_off, l_on

def get_ssc_STAGES(csv_file):
    
    df = pd.read_csv(csv_file, header=None)
    ssc_dur = list(df.iloc[:, 1])
    ssc_raw = list(df.iloc[:, 2])
    ssc = []
    is_ssc_event = False
    for (dur, event) in zip(ssc_dur, ssc_raw):
        # Check for nan (missing column)
        if dur != dur:
            continue
        try:
            dur = float(dur)
        except:
            continue
        # Edit false dur
        if dur == 0:
            n_stage = 1
        elif dur == 2592000:
            n_stage = 1
        else:
            n_stage = int(dur // 30)
    
        if event == ' Wake':
            num_stage = 1
            is_ssc_event = True
        elif event == ' UnknownStage':
            num_stage = 1
            is_ssc_event = True
        elif event == ' Stage1':
            num_stage = -1
            is_ssc_event = True
        elif event == ' Stage2':
            num_stage = -2
            is_ssc_event = True
        elif event == ' Stage3':
            num_stage = -3
            is_ssc_event = True
        elif event == ' REM':
            num_stage = 0
            is_ssc_event = True
        else:
            is_ssc_event = False
        if is_ssc_event:
            for i in range(n_stage):
                ssc.append(num_stage)

    return ssc

def extract_hypnogram_during(hypnogram_file, start, l_off, l_on):
    # Extracts text based hypnogram and returns numbered list
    # representing sleep stages.
    # INPUTS: hypnogram_file = file path, start_t = time at lights off in datetime
    # format, end_t = time at lights on in datetime format
    # OUTPUTS: list of sleep stages
    # Wake = 1, REM = 0, N1 = -1, N2 = -2, N3 = -3
    
    # initialise hypnogram
    hypno = []
    have_time_annotations = False
    
    # read in file
    h, df = read_file(hypnogram_file)
    if l_off is not None and l_on is not None:
        # find time to lights off and on
        time_change_loff = datetime.timedelta(seconds=l_off)
        t_loff = start + time_change_loff
        l_off_seconds = (t_loff.hour * 60 + t_loff.minute) * 60 + t_loff.second
        time_change_lon = datetime.timedelta(seconds=l_on)
        t_lon = start + time_change_lon
        l_on_seconds = (t_lon.hour * 60 + t_lon.minute) * 60 + t_lon.second
        
        have_time_annotations = True
    
    # coded mapping
    mapping = {}
    for i, el in enumerate(df['state'].unique()):
        if el == 'Wake' or el == 'WAKE':
            mapping[el] = 1
        elif el == 'REM' or el == 'Rem':
            mapping[el] = 0
        elif el == 'N1':
            mapping[el] = -1
        elif el == 'N2':
            mapping[el] = -2
        elif el == 'N3':
            mapping[el] = -3
        else:
            mapping[el] = -4
    
    # apply mapping to dataframe
    df['coded_state'] = df.apply(lambda row: mapping[row.state], axis=1)
    
    # add code to list for relevant time period
    if have_time_annotations:
        for i in range(len(df.index)):
            t = df.index[i]
            seconds = (t.hour * 60 + t.minute) * 60 + t.second
            if seconds >= l_off_seconds:
                hypno.append(df.iloc[i]['coded_state'])
            elif seconds <= l_on_seconds:
                hypno.append(df.iloc[i]['coded_state'])
            else:
                continue
    else:
        start_rec = df.index[0]
        start_rec_sec = (start_rec.hour * 60 + start_rec.minute) * 60 + start_rec.second
        first_sleep_list = [i for i, n in enumerate(df['coded_state']) if (n!=1)]
        l_off_datetime = df.index[first_sleep_list[0]]
        l_off_sec = (l_off_datetime.hour * 60 + l_off_datetime.minute) * 60 + l_off_datetime.second
        l_off = l_off_sec - start_rec_sec
        last_sleep_list = [i for i,n in enumerate(list(reversed(df['coded_state'])))]
        idx = len(df.index) - last_sleep_list[0] - 1
        l_on_datetime = df.index[idx]
        l_on_sec = (l_on_datetime.hour * 60 + l_on_datetime.minute) * 60 + l_on_datetime.second
        l_on = 86400 - (start_rec_sec - l_on_sec)
        
    return hypno, l_off, l_on

'''
def extract_hypnogram(hypnogram_path, l_off, l_on):
    # Extracts text based hypnogram and returns numbered list
    # representing sleep stages.
    # INPUTS: hypnogram_file = file path, start_t = time at lights off in datetime
    # format, end_t = time at lights on in datetime format
    # OUTPUTS: list of sleep stages
    # Wake = 1, REM = 0, N1 = -1, N2 = -2, N3 = -3
    
    # initialise hypnogram
    hypno = []
    
    # read in file
    h, df = read_file(hypnogram_file)
    
    # coded mapping
    mapping = {}
    for i, el in enumerate(df['state'].unique()):
        if el == 'Wake':
            mapping[el] = 1
        elif el == 'REM':
            mapping[el] = 0
        elif el == 'N1':
            mapping[el] = -1
        elif el == 'N2':
            mapping[el] = -2
        elif el == 'N3':
            mapping[el] = -3
        else:
            mapping[el] = -4
    
    # apply mapping to dataframe
    df['coded_state'] = df.apply(lambda row: mapping[row.state], axis=1)
    
    # add code to list for relevant time period
    for i in range(len(df.index)):
        t = df.index[i]
        seconds = (t.hour * 60 + t.minute) * 60 + t.second
        if seconds >= l_off:
            hypno.append(df.iloc[i]['coded_state'])
        elif seconds <= l_on:
            hypno.append(df.iloc[i]['coded_state'])
        else:
            continue
    
    return hypno
'''
def read_file(path):
    # read hypnogram file in as pandas dataframe
    # INPUTS: path to hypnogram file
    # OUTPUTS: header of file and body (df)
    with open(path, 'r') as f:
        lines = f.readlines()
        newline_ix = lines.index('\n')
        header = lines[:newline_ix]
        df = pd.read_csv(path,
                         delimiter='; ',
                         header=None,
                         names=['time', 'state'],
                         index_col='time',
                         skiprows=1 + len(header))
        df.index = pd.to_datetime(df.index, format='%H:%M:%S,%f', utc=True).time
    return (header, df)

def channel_alias_dict():
    # Simple dict for chennl aliases
    # INPUTS: None
    # OUTPUTS: channel_alias (aliases for channels which have been referenced)
              # channel alias_ref1 and channel_alias_ref2 - if patient has both of these
              # then the patient has that signal but it is unreferenced and referencing must
              # be implemented 
    channel_alias = {'C3': ['c3-a2','EEG C3-A2'], 'C4': ['c4-a1','EEG C4-A1'],
                     'F3': ['f3-a2','EEG F3-A2'], 'F4': ['f4-a1','EEG F4-A1'],
                     'O1': ['o1-a2','EEG O1-A2'], 'O2': ['o2-a1','EEG O2-A1'],
                     'EOGR': ['eogr-a1','EOGH-A1','EEG EOGH-A1','EEG EOGH-A1-Ref','E2'],
                     'EOGL': ['eogl-a2','EOGL-A2','EEG EOGV-A2','EEG EOGV-A2-Ref','E1'],
                     'TIBR': ['TIBH','TibR','EEG TIBH-Ref','tibr','EMG TIBH','TIBH-Gnd','PLMr','LMr'],
                     'TIBL':['TIBV','TibL','EEG TIBV-Ref','tibl','EMG TIBV','TIBV-Gnd','PLMl','LMl'],
                     'ECG': ['ECG','ECG EKG','ecg','ECG II', 'ECG 2','EKG','EKG-Gnd','EMG EKG'],
                     'CHIN': ['CHIN','Chin','chin','EMG Chin','EMG CHIN','EMG','Chin-Gnd'],
                     'NASALPRES': ['Nasal','Air Nasal','Nasal-Gnd','EEG Nasal-Ref','nasal','Flow Pres','Pressure Flow','Flow Pr'],
                     'ABD': ['Abdomen','Resp Abdomen','RIP Abdom','Abdomen-Gnd','abdomen','RIP Abdomen','Resp/Abd','EEG Abd-Ref'],
                     'THO': ['Thorax','Resp Thorax','RIP Thora','Thorax-Gnd','thorax','RIP Thorax','EEG Tho-Ref'],
                     'SaO2':['SpO2','SaO2 SpO2','SAO2-Gnd','SaO2 SAO2','sao2','SAO2','SPO2']}
    
    channel_alias_ref1 = {'C3':['C3','EEG C3-Ref','C3-Ref'],
                        'C4':['C4','EEG C4-Ref','C4-Ref'],
                        'F3':['F3','EEG F3-Ref','F3-Ref'],
                        'F4':['F4','EEG F4-Ref','F4-Ref'],
                        'O1':['O1','EEG O1-Ref','O1-Ref'],
                        'O2':['O2','EEG O2-Ref','O2-Ref'],
                        'EOGR':['EOGH','EEG EOGH-Ref','EOGH-Ref','EOGr'],
                        'EOGL':['EOGV','EEG EOGV-Ref','EOGV-Ref','EOGl'],
                        'TIBR':[],
                        'TIBL':[],
                        'ECG':[],
                        'CHIN':['EMG2','EMG+','ChinR', 'Chin1'],
                        'NASALPRES':[],
                        'ABD':[],
                        'THO':[],
                        'SaO2':[]}
    
    channel_alias_ref2 = {'C3':['A2','A2-Ref','EEG A2-Ref','M2'],
                        'C4':['A1','A1-Ref','EEG A1-Ref','M1'],
                        'F3':['A2','A2-Ref','EEG A2-Ref','M2'],
                        'F4':['A1','A1-Ref','EEG A1-Ref','M1'],
                        'O1':['A2','A2-Ref','EEG A2-Ref','M2'],
                        'O2':['A1','A1-Ref','EEG A1-Ref','M1'],
                        'EOGR':['A1','A1-Ref','EEG A1-Ref','M1'],
                        'EOGL':['A2','A2-Ref','EEG A2-Ref','M2'],
                        'TIBR':[],
                        'TIBL':[],
                        'ECG':[],
                        'CHIN':['EMG1','EMG-','ChinL', 'Chin2', 'Chin3'],
                        'NASALPRES':[],
                        'ABD':[],
                        'THO':[],
                        'SaO2':[]}
    
    
    return channel_alias, channel_alias_ref1, channel_alias_ref2

def channel_alias_during():
    # Simple dict for chennl aliases
    # INPUTS: None
    # OUTPUTS: channel_alias (aliases for channels which have been referenced)
              # channel alias_ref1 and channel_alias_ref2 - if patient has both of these
              # then the patient has that signal but it is unreferenced and referencing must
              # be implemented 
    channel_alias = {'C3': ['C3:M2','M2:C3'], 'C4': ['C4:M1','M1:C4'],
                     'F3': ['F3:M2', 'M2:F3'], 'F4': ['F4:M1','M1:F4'],
                     'O1': ['O1:M2', 'M2:O1'], 'O2': ['O2:M1','M1:O2'],
                     'EOGR': ['E2:M1'],
                     'EOGL': ['E1:M2'],
                     'TIBR': ['RAT','PLMr', 'PLMr.'], # 
                     'TIBL':['LAT', 'PLMl','PLMl.'], # 
                     'CHIN': ['Chin']}
    
    channel_alias_ref1 = {'C3':['C3'],
                        'C4':['C4'],
                        'F3':['F3'],
                        'F4':['F4'],
                        'O1':['O1'],
                        'O2':['O2'],
                        'EOGR':['E2', 'EOG2'],
                        'EOGL':['E1', 'EOG1'],
                        'TIBR':[],
                        'TIBL':[],
                        'CHIN':['Chin1']}
    
    channel_alias_ref2 = {'C3':['A2','M2'],
                        'C4':['A1','M1'],
                        'F3':['A2','M2'],
                        'F4':['A1','M1'],
                        'O1':['A2','M2'],
                        'O2':['A1','M1'],
                        'EOGR':['A1','M1'],
                        'EOGL':['A2','M2'],
                        'TIBR':[],
                        'TIBL':[],
                        'CHIN':['Chin2', 'Chin3']}
    
    
    return channel_alias, channel_alias_ref1, channel_alias_ref2

    
'''
def has_channels(export_chan, channel_labels, channel_alias, channel_alias_ref1, channel_alias_ref2):
    
    # Init variables
    has_chan = np.zeros((len(export_chan)))
    is_referenced = False
    channel_names = {}
    #channel_reference = {}
    
    for j in range(len(export_chan)):
        has_ref1 = 0
        has_ref2 = 0
        for c in range(len(channel_labels)):
            label = channel_labels[c]
            if label in channel_alias[export_chan[j]]:
                has_chan[j] = 1
                is_referenced = True
                channel_names[export_chan[j]] = label
                continue
            if len(channel_alias_ref1[export_chan[j]]) != 0:
                if label in channel_alias_ref1[export_chan[j]]:
                    has_ref1 = 1
                    ref_label = label
                if label in channel_alias_ref2[export_chan[j]]:
                    has_ref2 = 1
            if has_ref1 == 1 and has_ref2 == 1:
                has_chan[j] = 1
                channel_names[export_chan[j]] = ref_label
                continue
            
    return has_chan, is_referenced, channel_names
'''

def has_channels(export_chan, channel_labels, channel_alias, channel_alias_ref1, channel_alias_ref2):
    
    # Init variables
    has_chan = np.zeros((len(export_chan)))
    channel_names = {}
    channel_reference = {}
    
    for j in range(len(export_chan)):
        has_ref1 = 0
        has_ref2 = 0
        for c in range(len(channel_labels)):
            label = channel_labels[c]
            if label in channel_alias[export_chan[j]]:
                has_chan[j] = 1
                channel_reference[label] = 'referenced'
                channel_names[export_chan[j]] = label
                continue
            if len(channel_alias_ref1[export_chan[j]]) != 0:
                if label in channel_alias_ref1[export_chan[j]]:
                    has_ref1 = 1
                    ref_label = label
                if label in channel_alias_ref2[export_chan[j]]:
                    has_ref2 = 1
            if has_ref1 == 1 and has_ref2 == 1:
                has_chan[j] = 1
                channel_names[export_chan[j]] = ref_label
                channel_reference[ref_label] = 'not referenced'
                continue
            
    return has_chan, channel_reference, channel_names

def reference_EEG(f, eeg_signal, channel_labels, current_ch):
    
    # Function to reference an EEG signal
    # INPUTS: 
    # channel_labels = list of channels of current patient
    # eeg_signal = Current channel signal
    # current_ch = name of current channel
    # f = edfreader object
    # OUPUTS:
    # Referenced EEG signal (minus contralteral ref signal)
    
    refed_eeg_signal = None
    
    reference_map1 = ['A1', 'M1', 'A1-Ref', 'EEG A1-Ref']
    reference_map2 = ['A2', 'M2', 'A2-Ref', 'EEG A2-Ref']
    chin_ref_map = ['EMG1','EMG-','ChinL']
    
    # Find index of both reference channels so that they can be read
    for k in range(len(channel_labels)):
        if channel_labels[k] in reference_map1:
            ref1_idx = channel_labels.index(channel_labels[k])
        if channel_labels[k] in reference_map2:
            ref2_idx = channel_labels.index(channel_labels[k])
        if channel_labels[k] in chin_ref_map:
            chin_ref = channel_labels.index(channel_labels[k])
            
    channel_map1 = ['C4', 'F4', 'O2', 'EOGR']
    channel_map2 = ['C3', 'F3', 'O1', 'EOGL']
    
    if current_ch in channel_map1:
        ref_signal1 = f.readSignal(ref1_idx)
        refed_eeg_signal = eeg_signal - ref_signal1
    elif current_ch in channel_map2:
        ref_signal2 = f.readSignal(ref2_idx)
        refed_eeg_signal = eeg_signal - ref_signal2
    elif current_ch =='CHIN':
        chin_ref_signal = f.readSignal(chin_ref)
        refed_eeg_signal = eeg_signal - chin_ref_signal
    else:
        print('Referencing Error: No match for current channel', current_ch)
    
    return refed_eeg_signal

def change_fs(signal, des_fs, fs):
    
    resample_frac = Fraction(des_fs/fs).limit_denominator(1000)
    resampled_sig = resample_poly(signal, resample_frac.numerator, resample_frac.denominator)
    
    return resampled_sig

def get_cosine_schedule(optimizer: torch.optim.Optimizer, num_epochs:int, steps_per_epoch:int):
    '''
    Function to create cosine decay learning rate

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer used.
    num_epochs : int
        Number of epochs.
    steps_per_epoch : int
        How many training steps there are per epoch.

    Returns
    -------
    lr_scheduler
        Learning rate scheduler with cosine decay function.

    '''
    
    num_training_steps = num_epochs * steps_per_epoch
    warm_up_steps = 0
    
    def lr_lambda(current_step):
        if current_step < warm_up_steps:
            return float(current_step) / float(max(1, warm_up_steps))
        progress = (float(current_step - warm_up_steps) / 
                    float(max(1, num_training_steps - warm_up_steps)))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, -1)

def load_ckp(path, model, optimizer):
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #best_model = checkpoint['best_model']

    return model, optimizer, checkpoint

def class_F1_binary(pred, labels):
    '''
    Function which computes the weighted F1 score and F1 score per class

    Parameters
    ----------
    pred : tensor
        Predicted labels.
    labels : tensor
        Actual labels.

    Returns
    -------
    F1_class : Dict
        F1 score for each individual class.
    tot_weighted_F1 : float
        Weighted F1 score over all classes.

    '''
    F1_class = {}
    b, counts = np.unique(labels, axis=0, return_counts=True)
    num_samples = []

    for subset in ['CC', 'RBD']:
        if subset == 'CC':
            C = torch.tensor([0])
        if subset == 'RBD':
            C = torch.tensor([1])
    
        total_pred = (pred == C)
        lbl = (labels == C)
        TP = (total_pred*lbl).prod(axis=1).sum()
        FP = total_pred.prod(axis=1).sum()-TP
    
        FN = 0
        for i, j in zip(labels, pred):
            if (i==C).prod().item() == 1:
                if (j==i).prod().item() != 1:
                    FN += 1  
    
        # Precision (TP/FP+TP) # Recall (TP/FN+TP)
        if TP.item() == 0:
            pres = 0
            recall = 0
        else:
            pres = TP.item()/(FP.item()+TP.item())
            recall = TP.item()/(FN+TP.item())
    
        # F1 score per class
        if pres == 0 and recall == 0:
            F1_class[subset] = 0
        else:
            F1_class[subset] = 2*(pres*recall)/(pres+recall)
    
        for i,ele in enumerate(torch.tensor(b)):
            if (ele == C).prod().item() == 1:
                num_samples.append(counts[i])
                
    # Combined F1 weighted avg
    weighted_F1 = []
    for i, j in zip(F1_class.values(), num_samples):
        weighted_F1.append(i*j)
    tot_weighted_F1 = sum(weighted_F1)/sum(num_samples)
    
    return F1_class, tot_weighted_F1

def class_F1(pred, labels):
    '''
    Function which computes the weighted F1 score and F1 score per class

    Parameters
    ----------
    pred : tensor
        Predicted labels.
    labels : tensor
        Actual labels.

    Returns
    -------
    F1_class : Dict
        F1 score for each individual class.
    tot_weighted_F1 : float
        Weighted F1 score over all classes.

    '''
    F1_class = {}
    b, counts = np.unique(labels, axis=0, return_counts=True)
    num_samples = []

    for subset in ['PD', 'RBD', 'PD_RBD', 'CC']:
        if subset == 'PD':
            C = torch.tensor([1,0])
        if subset == 'RBD':
            C = torch.tensor([0,1])
        if subset == 'PD_RBD':
            C = torch.tensor([1,1])
        if subset == 'CC':
            C = torch.tensor([0,0])
    
        total_pred = (pred == C)
        lbl = (labels == C)
        TP = (total_pred*lbl).prod(axis=1).sum()
        FP = total_pred.prod(axis=1).sum()-TP
    
        FN = 0
        for i, j in zip(labels, pred):
            if (i==C).prod().item() == 1:
                if (j==i).prod().item() != 1:
                    FN += 1  
    
        # Precision (TP/FP+TP) # Recall (TP/FN+TP)
        if TP.item() == 0:
            pres = 0
            recall = 0
        else:
            pres = TP.item()/(FP.item()+TP.item())
            recall = TP.item()/(FN+TP.item())
    
        # F1 score per class
        if pres == 0 and recall == 0:
            F1_class[subset] = 0
        else:
            F1_class[subset] = 2*(pres*recall)/(pres+recall)
    
        for i,ele in enumerate(torch.tensor(b)):
            if (ele == C).prod().item() == 1:
                num_samples.append(counts[i])
                
    # Combined F1 weighted avg
    weighted_F1 = []
    for i, j in zip(F1_class.values(), num_samples):
        weighted_F1.append(i*j)
    tot_weighted_F1 = sum(weighted_F1)/sum(num_samples)
    
    return F1_class, tot_weighted_F1

def conf_mat(all_predicted, all_target):
    
    true_lbl = []
    pred_lbl = []
    
    for target in all_target:
        if (target.tolist()  == [0, 0]):
            true_lbl.append(0)
            continue
        if (target.tolist() == [0, 1]):
            true_lbl.append(1)
            continue
        if (target.tolist()  == [1, 0]):
            true_lbl.append(2)
            continue
        if (target.tolist() == [1, 1]):
            true_lbl.append(3)
            continue
    
    for pred in all_predicted:
        if (pred.tolist() == [0, 0]):
            pred_lbl.append(0)
            continue
        if (pred.tolist() == [0, 1]):
            pred_lbl.append(1)
            continue
        if (pred.tolist() == [1, 0]):
            pred_lbl.append(2) #2
            continue
        if (pred.tolist() == [1, 1]):
            pred_lbl.append(3) #3
            continue
    
    true_lbl = np.array(true_lbl)
    pred_lbl = np.array(pred_lbl)
    
    # Stack pred and true
    stacked = np.stack((true_lbl, pred_lbl), axis=-1)
    
    # Init conf mat
    cmt = torch.zeros(4,4, dtype=torch.int64) #4,4    
    
    for p in stacked:
        tl, pl = p.tolist()
        cmt[tl, pl] = cmt[tl, pl] + 1
    
    return cmt

def conf_mat_binary(all_predicted, all_target):
    
    true_lbl = []
    pred_lbl = []
    
    for target in all_target:
        if (target.tolist()  == [0]):
            true_lbl.append(0)
            continue
        if (target.tolist() == [1]):
            true_lbl.append(1)

    
    for pred in all_predicted:
        if (pred.tolist() == [0]):
            pred_lbl.append(0)
            continue
        if (pred.tolist() == [1]):
            pred_lbl.append(1)
            continue
    
    true_lbl = np.array(true_lbl)
    pred_lbl = np.array(pred_lbl)
    
    # Stack pred and true
    stacked = np.stack((true_lbl, pred_lbl), axis=-1)
    
    # Init conf mat
    cmt = torch.zeros(2,2, dtype=torch.int64) #4,4    
    
    for p in stacked:
        tl, pl = p.tolist()
        cmt[tl, pl] = cmt[tl, pl] + 1
    
    return cmt

def plot_confusion_matrix(cm, title, normalize=False, cmap=plt.cm.Blues):
    if normalize:
        cm = cm.numpy()
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    classes = ['CC', 'RBD', 'PD', 'PD+RBD']
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 1.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('conf_mat.png', dpi=800)


def get_weight_vector(dataset):
    
    class_vector = []
    for el in tqdm(dataset, 'Extracting class vector'):
        label = el['all_attrs']['label'].astype(np.int32)
        if (label == [0, 0]).all():
            class_vector.append(0)
            continue
        if (label == [0, 1]).all():
            class_vector.append(1)
            continue
        if (label == [1, 0]).all():
            class_vector.append(2)
            continue
        if (label == [1, 1]).all():
            class_vector.append(3)
            continue
    uniques, counts = np.unique(class_vector, return_counts=True)
    uniques = uniques.tolist()
    all_count = sum(counts)
    weight_vector = []
    for c in class_vector:
        weight_vector.append(all_count / counts[uniques.index(c)])
    return torch.tensor(weight_vector)

def get_weight_vector_binary(dataset):
    
    class_vector = []
    for el in tqdm(dataset, 'Extracting class vector'):
        label = el['all_attrs']['label'].astype(np.int32)
        if (label == [0, 0]).all():
            class_vector.append(0)
            continue
        if (label == [0, 1]).all():
            class_vector.append(1)
            continue
        if (label == [1, 0]).all():
            class_vector.append(0)
            continue
        if (label == [1, 1]).all():
            class_vector.append(1)
            continue
    uniques, counts = np.unique(class_vector, return_counts=True)
    uniques = uniques.tolist()
    all_count = sum(counts)
    weight_vector = []
    for c in class_vector:
        weight_vector.append(all_count / counts[uniques.index(c)])
    return torch.tensor(weight_vector)


def output_per_patient(patient_outputs, pd_thresh, rbd_thresh):
    
    patient_outputs[patient_outputs[:, 0] >= pd_thresh, 0] = 1
    patient_outputs[patient_outputs[:, 0] < pd_thresh, 0] = 0
    patient_outputs[patient_outputs[:, 1] >= rbd_thresh, 1] = 1
    patient_outputs[patient_outputs[:, 1] < rbd_thresh, 1] = 0
    
    return patient_outputs

def output_per_patient_binary(patient_outputs, rbd_thresh):
    
    patient_outputs[patient_outputs >= rbd_thresh] = 1
    patient_outputs[patient_outputs < rbd_thresh] = 0
    
    return patient_outputs

def optimize_thresh(pred_outputs, labels):

    # Trigger thresholds
    granularity = 21
    thresh_range = np.linspace(0.0, 1.0, granularity)
    grid = np.hstack((np.repeat(thresh_range, granularity).reshape(-1, 1),
                      np.tile(np.linspace(0.0, 1.0, granularity), granularity).reshape(-1, 1)))
    pd_thresh = 0.0
    rbd_thresh = 0.0
    
    # Pred_outputs are the sigmoid of the outputs
    dev_F1 = 0.0
    acc = 0
    # For each combination of thresholds, check the accuracy and F1
    for el in grid:
        pred_labels = torch.clone(pred_outputs)
        pred_labels[pred_labels[:, 0] >= el[0], 0] = 1
        pred_labels[pred_labels[:, 0] < el[0], 0] = 0
        pred_labels[pred_labels[:, 1] >= el[1], 1] = 1
        pred_labels[pred_labels[:, 1] < el[1], 1] = 0
        F1_class, this_F1 = class_F1(pred_labels, labels)
        this_acc = ((labels == pred_labels).prod(axis=1).sum().item())/len(labels)
        if this_F1 > dev_F1:
            acc = this_acc
            dev_F1 = this_F1
            dev_F1_class = F1_class
            prediction = pred_labels
            pd_thresh = el[0]
            rbd_thresh = el[1]
            print(f'opt_acc: {acc}, opt_F1: {dev_F1}, pd_thresh: {pd_thresh}, rbd_thresh: {rbd_thresh}')
    
    return prediction, pd_thresh, rbd_thresh, dev_F1, dev_F1_class, acc

def cohort_predictions(all_fids, all_labels, all_pred):
    dcsm_labels = []
    dcsm_pred = []
    stdf_labels = []
    stdf_pred = []
    for idx,fid in enumerate(all_fids):
        if fid.startswith('DCSM'):
            dcsm_labels.append(all_labels[idx])
            dcsm_pred.append(all_pred[idx])
        else:
            stdf_labels.append(all_labels[idx])
            stdf_pred.append(all_pred[idx])
    dcsm_pred = torch.tensor(dcsm_pred).reshape(len(dcsm_pred),1)
    dcsm_labels = torch.tensor(dcsm_labels).reshape(len(dcsm_pred),1)
    stdf_pred = torch.tensor(stdf_pred).reshape(len(stdf_pred),1)
    stdf_labels = torch.tensor(stdf_labels).reshape(len(stdf_labels),1)
    
    return dcsm_pred, dcsm_labels, stdf_pred, stdf_labels
    
def patient_logits(dev_ds, dev_probs):   
        
    # Get real labels (lbl_outputs) from dev dataset
    lbl_outputs = []
    pred_outputs = []
    # Take mean over night of logits and sigmoid it
    for i in dev_ds.dataset:
        if (dev_ds.attrs[i]['label'] == [0,0]).all() or (dev_ds.attrs[i]['label'] == [1,0]).all():
            lbl_outputs.append([torch.tensor([0])])
        else:
            lbl_outputs.append([torch.tensor([1])])
        dev_probs[i]['p_RBD'] = torch.sigmoid(torch.tensor(np.mean(dev_probs[i]['p_RBD'])))
        pred_outputs.append([dev_probs[i]['p_RBD']])
    pred_outputs = torch.tensor(pred_outputs)
    lbl_outputs = torch.tensor(lbl_outputs) 
    
    # Trigger thresholds
    granularity = 21 #21
    thresh_range = np.linspace(0.0, 1.0, granularity)
    rbd_thresh = 0.0
    
    # Pred_outputs are the sigmoid of the outputs
    dev_F1 = 0.0
    acc = 0
    # For each combination of thresholds, check the accuracy and F1
    for el in thresh_range:
        pred_labels = torch.clone(pred_outputs)
        pred_labels[pred_labels >= el] = 1
        pred_labels[pred_labels < el] = 0
        F1_class, this_F1 = class_F1_binary(pred_labels, lbl_outputs)
        this_acc = ((lbl_outputs == pred_labels).prod(axis=1).sum().item())/len(lbl_outputs)
        if this_F1 > dev_F1:
            acc = this_acc
            dev_F1 = this_F1
            dev_F1_class = F1_class
            prediction = pred_labels
            rbd_thresh = el
    return acc, dev_F1, dev_F1_class, prediction, rbd_thresh

def label_smoothing(labels, smoothing_factor):
    smoothed_labels = (1 - smoothing_factor) * labels + 0.1 / 2
    return smoothed_labels
            