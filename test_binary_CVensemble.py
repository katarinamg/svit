#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 00:59:16 2022

@author: katarina
"""

import torch
import numpy as np
import json
import warnings
import argparse
import copy
from functions import class_F1_binary, load_ckp, conf_mat_binary, output_per_patient_binary, plot_confusion_matrix, cohort_predictions
from torch.utils.data import DataLoader
from torch import nn
from spect_dataset import *
from spect_config import spect_config
from models import CNN_model1, dilated_CNN
from transformer_4l import *
from scipy.stats import gmean

warnings.filterwarnings("ignore", message="divide by zero encountered")

def main():
    
       # Arguments are handled by the argparse module
    parser = argparse.ArgumentParser()
    
     # Arguments and their default values
    parser.add_argument(
        "-data_path", 
        help = "Define input data path.", 
        type = str,
        default = os.path.expanduser('~/Desktop/Thesis/Patient_Spect_CWT/'),

        )
    parser.add_argument(
        "-model_path", 
        help = "Define input model path.", 
        type = str,
        default = os.path.expanduser('~/Desktop/Thesis/model_outputs'),

        )
    parser.add_argument(
        "-model", 
        help = "Define model.", 
        type = str,
        default = 'simpleCNN',

        )
    parser.add_argument(
        "-epoch_size", 
        help = "Specify length of epoch (s).", 
        type = int,
        default = 30*5,
        
        )
    parser.add_argument(
        "-n_channels", 
        help = "Number of PSG channels as input", 
        type = int,
        default = 12,
    )
    parser.add_argument(
        "-dropout1_1", 
        help = "1D Drop out percentage as float (e.g. 0.1)", 
        type = float,
        default = 0.7,

    )
    parser.add_argument(
        "-dropout1_2", 
        help = "1D Drop out percentage as float (e.g. 0.1)", 
        type = float,
        default = 0.1,

    )
    parser.add_argument(
        "-dropout2", 
        help = "2D Drop out percentage as float (e.g. 0.1)", 
        type = float,
        default = 0.1,

    )
    parser.add_argument(
        "-subset", 
        help = "subset of channels", 
        type = str,
        default = None,

    )
    parser.add_argument(
        "-lr", 
        help = "Learning rate as float (e.g. 0.1)", 
        type = float,
        default = 0.00001,

    )
    
    args = parser.parse_args()
    subset = args.subset
    
    config = spect_config(args) 
    batch_size = 64
    test_ds = Spect_Dataset(config, 'test', overlap=False, n = -1, subset=subset)
    test_dl = DataLoader(test_ds, shuffle=False, batch_size=batch_size, num_workers=8)
    
    # Import model
    modelstr = args.model
    if modelstr == 'transformer':
        model = EEGTransformer(config)
    elif modelstr == 'dilatedCNN':
        model = dilated_CNN(config)
    else:
        model = CNN_model1(config)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    patients = []
    for i in test_ds.dataset:
        patients.append(i)
    print(patients)
    
    # Average thresholds
    rbd_thresholds = []
    
    #Create dict to track performance based on class
    total_probs = {}
    all_pred_ensemble = {}
    classes = {'PD+CC': [0], 'RBD+PD': [1]}
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    loss_ensemble = 0
    '''
    kfold_models = [0,1,2,3,4]
    
    
    mean_loss=[]
    for i in kfold_models:
        PATH = os.path.expanduser('~/Desktop/Thesis/kitkatty1/transformer_binary_2023_kfold'+str(i)+'_aug.pt')
        best_model, optimizer, checkpoint = load_ckp(PATH, model, optimizer)
        mean_loss.append(np.mean(checkpoint['dev_loss_log']))
    
    print(np.mean(mean_loss))
    '''
    kfold_models = [0,1,2,3,4]
    for i in kfold_models:
        
        print('Fold:' + str(i))   
        #PATH = 'transformer_binary_2023_kfold'+str(i)+'_aug.pt'
        PATH = args.model_path+modelstr+'_binary_2023_kfold4_'+subset+'_aug_'+str(args.epoch_size)+'.pt'
        best_model, optimizer, checkpoint = load_ckp(PATH, model, optimizer)
        rbd_thresholds.append(checkpoint['rbd_thresh'])
        
        # Create patient dict to collect all outputs for a patient
        probs_fold = {p: {'p_RBD':[]} for p in patients}
        
        # Init variables
        loss_list = []
        lbl_total = 0
        correct = 0
        k = 0
        
        # Test Loop
        best_model.eval()
        with torch.no_grad():
            for batch in test_dl:
                batch_IDS = batch['fid']
                data = batch['data']
                labels = batch['label'] 
                
                # Run the forward pass
                pred_labels = best_model(data)
                loss = criterion(pred_labels, labels)
                loss_list.append(loss.item())
                
                # Collect all logits for each patient 
                for b_ID, b_o in zip(batch_IDS, pred_labels):
                    probs_fold[b_ID]['p_RBD'].append(b_o.item())
                
                lbl_total += labels.size(0)
                
                # Collect all IDs, outputs, and labels
                if k == 0:
                    all_fids = batch_IDS
                    all_pred = pred_labels
                    all_labels = labels
                    k = 1
                else:
                    all_fids.extend(batch_IDS)
                    all_pred = torch.cat((all_pred, pred_labels))
                    all_labels = torch.cat((all_labels, labels))
                    
        all_pred_ensemble['fold'+str(i)] = all_pred
        total_probs['fold'+str(i)] = probs_fold
        print('Loss fold',str(i),': ',np.mean(loss_list))
        
    # Get average predictions
    """
    avg_ensemble_pred = torch.zeros_like(all_pred_ensemble['fold0'])
    for i in all_pred_ensemble:
        avg_ensemble_pred += all_pred_ensemble[i]
    avg_ensemble_pred = avg_ensemble_pred/len(kfold_models)
    
    loss_ensemble = criterion(avg_ensemble_pred, all_labels)
    
    # Threshold logits
    avg_ensemble_pred[avg_ensemble_pred >= 0] = 1
    avg_ensemble_pred[avg_ensemble_pred < 0] = 0
    """
    
    avg_ensemble_pred = {}
    for i in all_pred_ensemble:
        avg_ensemble_pred[i] = (torch.sigmoid(all_pred_ensemble[i])).numpy()
    avg_p_ensemble = np.stack((avg_ensemble_pred['fold0'],avg_ensemble_pred['fold1'], avg_ensemble_pred['fold2'], avg_ensemble_pred['fold3'], avg_ensemble_pred['fold4']))
    gmean_ensemble = torch.from_numpy(gmean(avg_p_ensemble, axis=0))
    ensemble_criterion = torch.nn.BCELoss()
    loss_ensemble = ensemble_criterion(gmean_ensemble, all_labels)
    
    # Threshold logits
    gmean_ensemble[gmean_ensemble >= 0.5] = 1
    gmean_ensemble[gmean_ensemble < 0.5] = 0
                
    # Accuracy
    correct += (gmean_ensemble == all_labels).prod(axis=1).sum().item()
    test_acc = correct / len(all_labels) * 100
    
    #Collect the correct predictions for each class
    for label, prediction in zip(all_labels, gmean_ensemble):
        if (label == prediction).prod().item() == 1:
            for key in classes:
                if (label == torch.tensor(classes[key])).prod().item() == 1:
                    correct_pred[key] += 1
                    break
        for key in classes:
            if (label == torch.tensor(classes[key])).prod().item() == 1:
                total_pred[key] += 1
                break
                        
    # Accuracy per class and overall
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} % (no. of samples = {})".format(classname,
                                                accuracy, total_pred[classname]))

    print(f'Loss: {loss_ensemble}, Overall Ensemble Accuracy: {test_acc}')
    

    # Collect per cohort labels and predictions            
    dcsm_pred, dcsm_labels, stdf_pred, stdf_labels = cohort_predictions(all_fids, all_labels, gmean_ensemble)
         
    # Take mean over night of logits and sigmoid it
    pred_outputs = []
    # Save actual patient labels
    lbl_outputs = []
    
    avg_rbd_threshold = np.mean(rbd_thresholds)
    avg_probs_all = {p: {'p_RBD':[]} for p in patients}
    
    for p in avg_probs_all:
        for fold in total_probs:
            avg_probs_all[p]['p_RBD'].append(total_probs[fold][p]['p_RBD'])
    for p in avg_probs_all:
        multiple_lists = avg_probs_all[p]['p_RBD']
        arrays = [np.array(x) for x in multiple_lists]
        avg_probs_all[p]['p_RBD']=[np.mean(k) for k in zip(*arrays)]
    
    for l in test_ds.dataset:
        if (test_ds.attrs[l]['label'] == [0,0]).all() or (test_ds.attrs[l]['label'] == [1,0]).all():
            lbl_outputs.append([torch.tensor([0])])
            avg_probs_all[l]['ytrue'] = 0
        else:
            lbl_outputs.append([torch.tensor([1])])
            avg_probs_all[l]['ytrue'] = 1
        p_RBD_mean = torch.sigmoid(torch.tensor(np.mean(avg_probs_all[l]['p_RBD'])))
        if p_RBD_mean >= avg_rbd_threshold:
            avg_probs_all[l]['y_pred'] = 1
        else:
            avg_probs_all[l]['y_pred'] = 0
        pred_outputs.append([p_RBD_mean])
    pred_outputs = torch.tensor(pred_outputs)
    lbl_outputs = torch.tensor(lbl_outputs)    
    
    # Output per patient based on optimised thresholds
    patient_outputs = output_per_patient_binary(pred_outputs, avg_rbd_threshold)
      
    return correct, lbl_total, gmean_ensemble, all_pred, all_labels, patient_outputs, lbl_outputs, avg_probs_all, dcsm_pred, dcsm_labels, stdf_pred, stdf_labels, modelstr, subset, args.epoch_size
    
if __name__ == '__main__':
    correct, lbl_total, gmean_ensemble, all_pred, all_labels, patient_outputs, lbl_outputs, avg_probs_all, dcsm_pred, dcsm_labels, stdf_pred, stdf_labels, modelstr, subset, epoch_size = main()
    # Confusion mat
    
    cmt = conf_mat_binary(gmean_ensemble, all_labels)
    
    # F1 score per class & Weighted Avg F1 (on patient basis)
    F1_class_pp, weighted_F1_pp = class_F1_binary(patient_outputs, lbl_outputs)  
    print('F1_class_pp', F1_class_pp)
    print('Weighted_F1_pp', weighted_F1_pp)
    # F1 score per class & Weighted Avg F1 (on epoch basis)
    F1_class, weighted_F1 = class_F1_binary(gmean_ensemble, all_labels) 
    print('F1_class', F1_class)
    print('Weighted F1', weighted_F1)
    # F1 score per class & weighted Avg F1 seperated by cohort (on epoch)
    F1_class_stdf, weighted_F1_stdf = class_F1_binary(stdf_pred, stdf_labels) 
    print('F1_class stanford', F1_class_stdf)
    print('Weighted stanford', weighted_F1_stdf)
    F1_class_dcsm, weighted_F1_dcsm = class_F1_binary(dcsm_pred, dcsm_labels) 
    print('F1_class DCSM', F1_class_dcsm)
    print('Weighted DCSM', weighted_F1_dcsm)

    with open('F1_class_pp_'+modelstr+'_'+subset+'_'+epoch_size+'.json', 'w') as fp:
        json.dump(F1_class_pp, fp)
    with open('F1_class_epoch_'+modelstr+'_'+subset+'_'+epoch_size+'.json', 'w') as fp:
        json.dump(F1_class, fp)
    with open('STFD_F1_class_epoch_'+modelstr+'_'+subset+'_'+epoch_size+'.json', 'w') as fp:
        json.dump(F1_class_stdf, fp)
    with open('DCSM_F1_class_epoch_'+modelstr+'_'+subset+'_'+epoch_size+'.json', 'w') as fp:
        json.dump(F1_class_dcsm, fp)
    line = 'F1 PER PATIENT: ' + str(F1_class_pp) + ' Weighted F1 PATIENT: ' + str(weighted_F1_pp) + ' F1 PER EPOCH: ' + str(F1_class) + ' WEIGHTED F1 EPOCH: ' + str(weighted_F1) + ' F1 EPOCH STFD: ' + str(F1_class_stdf) + ' WEIGHTED F1 EPOCH STFD: ' + str(weighted_F1_stdf) + ' F1 EPOCH DCSM: ' + str(F1_class_dcsm) + ' WEIGHTED F1 EPOCH DCSM: ' + str(weighted_F1_dcsm)
    with open('output_'+modelstr+'_binary_2023_kfold'+'_'+str(subset)+'_aug_'+str(epoch_size)+'.txt', "w") as outfile:
            outfile.write(line)
    
    with open(modelstr+'_binary_2022_kfold_ensemble_probs_'+str(subset)+'_'+str(epoch_size)+'.json', 'w') as fp:
        json.dump(avg_probs_all, fp)