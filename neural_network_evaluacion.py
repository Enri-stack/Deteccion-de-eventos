# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 12:39:54 2022

@author: Paula
"""

import numpy as np


def sigmoid(x):
    return (1.0/(1.0 + np.exp((-1)*(x))))

  
class FIR_EVAL:
    
    def __init__(self, num_layers, num_neurons, FIR_lenghts):
        
        self.num_layers=num_layers
        self.num_neurons=num_neurons
        self.FIR_lenghts=FIR_lenghts
        
#FUNCION FORWARD

    def forward(self, l, inputs, activity, weights, threshold):
               
        
       if l<(self.num_layers-1): #Para la primera iteracion no mueve los valores en los filtro FIR.
                for i in range(self.num_neurons[l]):
                    
                    for k in range(self.FIR_lenghts[l+1]-1, 0, -1):  
                         activity[l][i][k]=activity[l][i][k-1]
                    
                    activity[l][i][0]=0
                    for j in range(self.num_neurons[l-1]):
                        activity[l][i][0]=activity[l][i][0] + np.dot(inputs[l][i][j], weights[l][i][j])   
                                           
                    activity[l][i][0]=activity[l][i][0]-threshold[l][i]
                    activity[l][i][0]=sigmoid(activity[l][i][0])
                        
                for j in range(self.num_neurons[l+1]):
                    
                    inputs[l+1][j]= activity[l]
              
       else:
           
            for i in range(self.num_neurons[l]):
                    
                    activity[l][i][0]=0 
                    for j in range(self.num_neurons[l-1]):
                        activity[l][i][0]=activity[l][i][0] + np.dot(inputs[l][i][j], weights[l][i][j])   
                                           
                    activity[l][i]=activity[l][i]-threshold[l][i]
                    activity[l][i]=sigmoid(activity[l][i])
                                   
                    
                    inputs[l+1]= activity[l]           
            
       return inputs         