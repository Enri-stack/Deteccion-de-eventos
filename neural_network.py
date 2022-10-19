# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 14:58:21 2021

@author: Paula
"""

import numpy as np


def sigmoid(x):
    return (1.0/(1.0 + np.exp((-1)*(x))))

  
class FIR_MLP:
    
    def __init__(self, num_layers, num_neurons, FIR_lenghts,learning_rate, momentum):
        
        self.num_layers=num_layers
        self.num_neurons=num_neurons
        self.FIR_lenghts=FIR_lenghts
        self.learning_rate=learning_rate
        self.momentum=momentum
        
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
                
       
#FUNCION BACKWARD
   
    def backward(self, l, error, inputs, activity, weights, increment_weights, delta, threshold):
        
        
        for j in range(self.num_neurons[l-1]): #numero neuronas de la capa en la que estoy 
            
            for i in range(self.FIR_lenghts[l-1]-1,0,-1):    
                
                delta[l-1][j][i]=delta[l-1][j][i-1] 
                
            #Calcula delta para la capa de salida.
            if l==self.num_layers:  
                
                delta[l-1][j][0]= error[j] 
                
            #Calcula delta para el resto de capas de la red neuronal
            else: 
                
                                    
                delta[l-1][j][0]=0
                for i in range(self.num_neurons[l]): 
                    
                   delta[l-1][j][0]= delta[l-1][j][0]+np.dot(delta[l][i],weights[l][i][j])*np.dot(activity[l-1][j], 1-activity[l-1][j])
                
                                           
            for i in range(self.num_neurons[l-2]): #nuerones capa anterior
                
                for k in range(self.FIR_lenghts[l-1]): # FIR
                    
                    increment_weights[l-1][j][i][k]= (-self.learning_rate * delta[l-1][j][k] * inputs[l-1][j][i][k]) + (self.momentum * increment_weights[l-1][j][i][k])

    
            #update thresholds
            threshold[l-1][j]=threshold[l-1][j]+ (self.learning_rate * delta[l-1][j][0])
            
        
    
            