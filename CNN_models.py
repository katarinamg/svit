#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 15:53:20 2021

@author: katarina
"""
import torch
import torch.nn as nn

class CNN_model1(nn.Module):
    # This model is based on matteo et al.
    # DEF _INIT_(SELF, [LIST OF HYPERPARAMETERS])
    def __init__(self, config):
        super(CNN_model1, self).__init__()
        self.config = config
        self.nc = self.config.n_channels
        self.do1 = self.config.dropout_1_1
        self.do2 = self.config.dropout_2
        self.expansion1 = self.nc*4
        self.classes = config.classes
        
        self.layer1 = torch.nn.Sequential(
            
            torch.nn.BatchNorm2d(self.nc), 
            torch.nn.Conv2d(self.nc, self.expansion1, kernel_size=(25,1), stride=1, padding=0), #kernel_size=(96,1)
            torch.nn.ReLU())
        
        self.layer2 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(self.expansion1),
            torch.nn.Conv2d(self.expansion1, self.expansion1*4, kernel_size=(1,4), stride=1, padding=0),
            torch.nn.ReLU())
        self.fc1 = torch.nn.Linear(1 * 147 * self.expansion1*4, 1024)
        self.fc2 = torch.nn.Linear(1024, 256)
        if self.classes != 'all':
            self.fc3 = torch.nn.Linear(256, 1)
        else:
            self.fc3 = torch.nn.Linear(256, 2)
        self.dropout_conv = torch.nn.Dropout2d(self.do2)
        self.dropout = torch.nn.Dropout(self.do1)
        
    
    
    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x) 

        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x)) 
        output = self.fc3(torch.relu(self.dropout((self.fc2(x))))) 

        return output
    

class dilated_CNN(nn.Module):
    
    def __init__(self, config):
        super(dilated_CNN, self).__init__()
        
        self.config = config
        self.nc = self.config.n_channels
        self.do1_1 = self.config.dropout_1_1
        self.do1_2 = self.config.dropout_1_2
        self.do2 = self.config.dropout_2
        self.exp = int((self.nc*25)/2) 
        self.classes = config.classes
        
        self.layer1 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(self.nc),
            torch.nn.Conv2d(self.nc, self.exp, kernel_size=(25,1), stride=(1,1), padding=0, dilation=(1, 1)),
            torch.nn.ReLU())
        
        self.layer2 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(self.exp),
            torch.nn.Conv2d(self.exp, self.exp, kernel_size=(1,3), stride=(1,2), padding=0, dilation=(1, 1)),
            torch.nn.ReLU())
        
        self.layer3 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(self.exp),
            torch.nn.Conv2d(self.exp, self.exp, kernel_size=(1,3), stride=(1,1), padding=0, dilation=(1, 1)),
            torch.nn.ReLU())
    
        self.layer4 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(self.exp),
            torch.nn.Conv2d(self.exp, self.exp, kernel_size=(1,3), stride=(1,2), padding=0, dilation=(1, 2)),
            torch.nn.ReLU())
        
        self.layer5 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(self.exp),
            torch.nn.Conv2d(self.exp, self.exp, kernel_size=(1,3), stride=(1,1), padding=0, dilation=(1, 4)),
            torch.nn.ReLU())
        
        self.layer6 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(self.exp),
            torch.nn.Conv2d(self.exp, self.exp, kernel_size=(1,3), stride=(1,1), padding=0, dilation=(1, 8)),
            torch.nn.ReLU())
        
        self.layer7 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(self.exp),
            torch.nn.Conv2d(self.exp, self.exp, kernel_size=(1,3), stride=(1,1), padding=0, dilation=(1, 80)), #dilation=(1, 16)
            torch.nn.ReLU())
        
        self.layer8 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(self.exp),
            torch.nn.Conv2d(self.exp, self.exp, kernel_size=(1,3), stride=(1,1), padding=0, dilation=(1, 1)),
            torch.nn.ReLU())
        
        self.layer9 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(self.exp),
            torch.nn.Conv2d(self.exp, self.exp, kernel_size=(1,1), stride=(1,1), padding=0, dilation=(1, 1)),
            torch.nn.ReLU())
        
        self.fc1 = torch.nn.Linear(1 * 186 * self.exp, 4096)
        self.fc2 = torch.nn.Linear(4096, 2048)
        self.fc3 = torch.nn.Linear(2048, 1024)
        self.fc4 = torch.nn.Linear(1024, 256)
        if self.classes != 'all':
            self.fc5 = torch.nn.Linear(256, 1)
        else:
            self.fc5 = torch.nn.Linear(256, 2)
        self.dropout1 = torch.nn.Dropout(self.do1_1)
        self.dropout1_2 = torch.nn.Dropout(self.do1_2)
        self.dropout2 = torch.nn.Dropout2d(self.do2)
    
    def forward(self, x):
        
        # Conv layers
        x = self.layer1(x)
        x = self.layer2(x) 
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        
        # Flatten x [batch_size x num_features]
        x = x.view(x.size(0), -1)
        # Fully connected layers
        x = torch.relu(self.fc1(x)) 
        x = torch.relu(self.dropout1_2(self.fc2(x))) 
        x = torch.relu(self.fc3(x)) 
        x = torch.relu(self.dropout1(self.fc4(x)))
        output = self.fc5(x)
        
        return output
      
    