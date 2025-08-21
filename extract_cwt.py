import pywt
import os
import numpy as np
import pyedflib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
from functions import *

def main():
    
    # Data path
    #path_dur = '/oak/stanford/groups/mignot/projects/irbd_during_v2'
    #path_control = '/oak/stanford/groups/mignot/psg/STAGES/deid/STNF'
    path_control = '/Volumes/KAT/Data/stanford_irbd_cohort/edf_control/edf_control'
    path_dur = '/Volumes/KAT/Data/stanford_irbd_cohort/edf_control/edf_irbd'
    csv_path = '/Volumes/KAT/Data/stanford_irbd_cohort/edf_control/csv_control'
    
    # Load the dataframe with patient
    data_xl = pd.read_csv('/Volumes/KAT/Data/stanford_irbd_cohort/edf_control/stanford_irbd_control_ds.csv', index_col='ID')
    
    # See how many patients are already extracted so it can continue 
    #path_done = '/scratch/users/kmgunter/Patient_Spect_New/'
    path_done = os.path.expanduser('~/Desktop/Thesis/Patient_Spect_CWT_1hz/')
    
    # Iterate through folders and files
    for subdir, dirs, files in os.walk(path_done):
        folder = files
        
    data_xl.drop(data_xl[data_xl.Dianosis == 'CIDP'].index, inplace=True)
    data_xl.drop(data_xl[data_xl.Dianosis == 'M'].index, inplace=True)
    data_xl.index = data_xl.index.astype(str, copy=False)
    
    # # Extract labels as np array
    labels = np.zeros((len(data_xl.index), 2)) #data_xl.index
    diagnosis = {'Control': [0,0], 'I': [0,1], 'D': [0,1], 'P':[1,1]}
    for i in range(len(data_xl.index)):
        labels[i][:] = diagnosis[data_xl.iloc[i]['Dianosis']]   
        
    # Loop through each patient folder       
    for i in range(len(data_xl.index)): #data_xl.index
        dur = False
        filename = None 
        if data_xl.index[i]+'.h5' in folder:
           continue
        # Define data path
        if data_xl.index[i].startswith('STNF'):
            path = path_control
        else:
            path = path_dur
            dur = True
            
        for f in os.listdir(path):
            if f.startswith(data_xl.index[i]):
                filename = path+'/'+f
        if filename == None:
            print('No file found for patient ID: '+data_xl.index[i])
            continue
        try:
            header = pyedflib.highlevel.read_edf_header(filename, read_annotations=True)
            f = pyedflib.EdfReader(filename)
        except:
            print('Unable to open edf:', data_xl.index[i])
            continue
        
        # Channel labels
        channel_labels = f.getSignalLabels()
        # Sampling frequencies
        fss = f.getSampleFrequencies()
        
        # Determine if channels are present and if referenced or not
        [channel_alias, channel_alias_ref1, channel_alias_ref2] = channel_alias_during()
        # 'Normal' labels of channels we want to extract
        export_chan = ['C3','C4','F3','F4','O1','O2','TIBR','TIBL','CHIN','EOGR','EOGL']
        [has_chan, channel_reference, channel_names] = has_channels(export_chan, channel_labels, channel_alias, channel_alias_ref1, channel_alias_ref2)
        # If file does not contain all channels we want, drop patient
        if sum(has_chan) != len(export_chan):
            continue
        
        # Sampling freq, epoch time in sec, epoch in samples (init variables)
        des_fs = 250
        epoch_t = 30
        epoch_s = epoch_t
        
        # Extract time points for signal from lights.txt
        if dur == True:
            startdate = header['startdate']
            start_t, end_t = extract_marker_t_during(header)
            #hypnogram_path = '/oak/stanford/groups/mignot/projects/irbd_during_v2/SLEEP_PROFILES/'+data_xl.index[i][:5]
            hypnogram_path = '/Volumes/KAT/Data/stanford_irbd_cohort/edf_control/SLEEP PROFILES/'+data_xl.index[i][:5]
            hypno_file = None
            try:
                during_hyp_files = os.listdir(hypnogram_path)
                for file in during_hyp_files:
                    if file.startswith('Sleep profile -') or file.startswith(data_xl.index[i]):
                        hypno_file = file
            except:
                print('no hypnogram file for: ', data_xl.index[i])
            if hypno_file is not None:
                hypnogram_file = hypnogram_path+'/'+hypno_file
                hypno, start_t, end_t = extract_hypnogram_during(hypnogram_file, startdate, start_t, end_t)
            else:
                hypno = None
        else:
            hypno = get_ssc_STAGES(csv_path+'/'+data_xl.index[i][:-4]+'.csv')
            start_t, end_t = extract_marker_t_STAGES(csv_path+'/'+data_xl.index[i][:-4]+'.csv', hypno)
            hypno = hypno[start_t//30:end_t//30]
            
        # Number of samples to time on/off
        if start_t is not None and end_t is not None:
            samples_loff = round(start_t*des_fs)
            samples_lon = round(end_t*des_fs)
        else:
            samples_loff = 0
            samples_lon = -2
            
        # Init EOG signals 
        eogr_signal = None
        eogl_signal = None
        emg_signal = None
        
        # Extract channel signals and iteratively add to spec stack
        chn_no = 0
        all_spec = []
        eog_channel = False
        emg_channel = False
        for l in channel_names.values():
            print(l)
            eeg_idx = channel_labels.index(l)
            current_ch = list(channel_names.keys())[list(channel_names.values()).index(l)]
            eeg_signal = f.readSignal(eeg_idx)
            
            # Reference signals which require it
            #if is_referenced is False:
                #eeg_signal = reference_EEG(f, eeg_signal, channel_labels, current_ch)
            if channel_reference[l] == 'not referenced':
                eeg_signal = reference_EEG(f, eeg_signal, channel_labels, current_ch)
            # Change sampling freq. if needed here
            # Find fs of current channel
            fs = fss[eeg_idx]
            
            # Filter EMG channels
            if current_ch in ['TIBR','TIBL','CHIN']:
                emg_channel = True
                emg_signal = eeg_signal
                #eeg_signal = psg_highpass_filter(eeg_signal, 10, fs=fs, order=5, plot_opt = 0)
                
            # If current fs is not same as the desired fs, resample signal
            if fs != des_fs:
                eeg_signal = change_fs(eeg_signal, des_fs, fs)
            if current_ch == 'EOGR':
                eogr_signal = eeg_signal
                eog_channel = True
            if current_ch == 'EOGL':
                eogl_signal = eeg_signal
                eog_channel = True
            if current_ch in ['TIBR','TIBL','CHIN']:
                emg_channel = True
                emg_signal = eeg_signal
                
            # Check unit of measurement and change if needed
            unit = f.getPhysicalDimension(eeg_idx)    
            if unit == 'mV':
                eeg_signal = eeg_signal/1000
                
            # Create stack of frames using each eeg signal (cut signal to time here)
            if eog_channel == True:
                widths = [6.5, 7, 7.5, 8, 9, 10, 12, 15, 17, 20, 25, 30, 35, 40, 50, 60, 70, 80, 120, 180, 260, 350, 450, 700, 1000]
                spec, freqs = pywt.cwt(eeg_signal[samples_loff:samples_lon+1], widths, 'morl', sampling_period=1/250)
                eog_channel = False
            elif emg_channel == True:
                widths = [2, 2.2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 12, 13, 15, 17, 20]
                spec, freqs = pywt.cwt(emg_signal[samples_loff:samples_lon+1], widths, 'morl', sampling_period=1/250)
                emg_channel = False
            else:
                widths = [6, 6.5, 7, 7.5, 8, 9, 10, 12, 15, 17, 20, 25, 30, 35, 40, 50, 60, 70, 80, 120, 180, 260, 350, 400, 650]
                spec, freqs = pywt.cwt(eeg_signal[samples_loff:samples_lon+1], widths, 'morl', sampling_period=1/250)

            # DOWNSAMPLE HERE
            power = (abs(spec)) ** 2
            power_log2 = np.log2(power)
            power_log2[power_log2 < -80] = -80 

            window_size = 250
            Mov_Avg = np.zeros([25,int(power_log2.shape[1]/250)])
            j=0

            indices = np.arange(0, (power_log2.shape[1])-1, 250)
            for p in indices:
                this_window = power_log2[:,p : p + window_size]
                window_average = np.sum(this_window,axis=1) / window_size
                window_average = window_average.reshape(-1,1)
                Mov_Avg[:,j] = window_average[:,0]
                j+=1
                
            # Stack spects from each EEG channel (only consider given freq. range)
            if chn_no == 0:
                all_spec = Mov_Avg
                chn_no = 1
                
            elif current_ch == 'EOGR' or current_ch == 'EOGL':
                all_spec = np.dstack((all_spec, Mov_Avg))
                
            else:
                all_spec = np.dstack((all_spec, Mov_Avg))
                
            if eogr_signal is not None and eogl_signal is not None:
                comb_eog = eogr_signal - eogl_signal
                widths = [6.5, 7, 7.5, 8, 9, 10, 12, 15, 17, 20, 25, 30, 35, 40, 50, 60, 70, 80, 120, 180, 260, 350, 450, 700, 1000]
                eog_spec, freqs = pywt.cwt(comb_eog[samples_loff:samples_lon+1], widths, 'morl', sampling_period=1/250)
                eog_power = (abs(eog_spec)) ** 2
                eog_power_log2 = np.log2(eog_power)
                eog_power_log2[eog_power_log2 < -80] = -80 

                window_size = 250
                eog_Mov_Avg = np.zeros([25,int(eog_power.shape[1]/250)])
                j=0

                indices = np.arange(window_size//2, (eog_power_log2.shape[1])-window_size//2, 250)
                eog_power_log2 = np.pad(eog_power_log2, ((0,0),(window_size//2, window_size-1-window_size//2)), mode='edge')
                for m in indices:
                    this_window = eog_power_log2[:,m : m + window_size]
                    window_average = np.sum(this_window,axis=1) / window_size
                    window_average = window_average.reshape(-1,1)
                    eog_Mov_Avg[:,j] = window_average[:,0]
                    j+=1
                    
                all_spec = np.dstack((all_spec, eog_Mov_Avg))
        plt.clf()    
        f.close()
        
        #output_path = '/scratch/users/kmgunter/Patient_Spect_New/'
        output_path = os.path.expanduser('~/Desktop/Thesis/Patient_Spect_CWT_1hz/')
        output_filename = output_path+data_xl.index[i]+'.h5'
        with h5py.File(output_filename, "w") as output_f:
        #Save spect and hypnogram for each patient as seperate file
            output_f.create_dataset('Spect', data = all_spec, dtype='f4', chunks = ((len(freqs), epoch_s, len(channel_names))))
            output_f.create_dataset('hypno', data = hypno, dtype='f4')
            output_f.attrs['label'] = labels[i][:] 
            output_f.attrs['ID'] =  data_xl.index[i]
    return 

if __name__ == '__main__':
    main()