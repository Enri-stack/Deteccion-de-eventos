# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 14:58:04 2021

@author: Paula
"""

import numpy as np  

import os 


#Lee los parametros del fichero .txt
def readParameters(sFileName):
   
    lParameterList = []
    hFileHandler = open(sFileName)
    
    for line in hFileHandler:
        
        lParameterList.append(line.strip()) #lee linea por linea el fichero 
     
    return  lParameterList  
 
def load_weights_and_thresholds():
    
    results_dir=r'\Users\Paula\Documents\PFG\experiments' #directory with the weights and thresholds saved in the training 
    
    weights=np.load(os.path.join(results_dir,r'epoch_6_weights.npy'), allow_pickle=True) #the name of the .npy file must be the same as the file saved in the training proccess
    weights=weights.tolist()
    
    
    threshold=np.load(os.path.join(results_dir,r'epoch_6_threshold.npy'), allow_pickle=True) #the name of the .npy file must be the same as the file saved in the training proccess
    threshold=threshold.tolist()
    
    return weights, threshold    
    
def init_weights_and_threshold(num_neurons, FIR_lenghts):  
         
    weights=[]
    threshold=[]
    
    #Pesos
    weights0=np.array([[0]])
 
    weights1=[]
    weights2=[]
    eachneuron1=[]
    eachneuron2=[]
   
    for i in range(num_neurons[1]):
        eachneuron1=[]
        for j in range(num_neurons[0]):
           eachneuron1.append((0.14*np.random.random((FIR_lenghts[1],)) -0.07))
        weights1.append(eachneuron1)
     
    for i in range(num_neurons[2]): 
        eachneuron2=[]
        for j in range(num_neurons[1]):
            eachneuron2.append((0.14*np.random.random((FIR_lenghts[2],)) -0.07))
        weights2.append(eachneuron2)
    
 
    weights.append(weights0)
    weights.append(weights1)
    weights.append(weights2)
    

    #Threshold
    threshold0=np.array([0])
    
    threshold1=[]
    threshold2=[]
    
    for i in range(num_neurons[1]):
        threshold1.append(np.zeros(1)) 
    
    for i in range(num_neurons[2]):        
        threshold2.append(np.zeros(1))
        
    
    threshold.append(threshold0)
    threshold.append(threshold1)
    threshold.append(threshold2)  
    
    return  weights, threshold 
                

def init_inputs_and_activity(num_neurons, FIR_lenghts):

    activity=[] 
    inputs=[]
    
    inputs0=[]
    inputs1=[]
    inputs2=[]
    inputs3=[]
    eachneuron1=[]
    eachneuron2=[]
    
    for i in range (num_neurons[0]):
        inputs0.append(np.zeros(FIR_lenghts[1]))
        
     
    for i in range(num_neurons[1]):  
       eachneuron1=[]
       for j in range(num_neurons[0]):
           eachneuron1.append(np.zeros(FIR_lenghts[1]))
       inputs1.append(eachneuron1)
    
    for i in range(num_neurons[2]): 
       eachneuron2=[]     
       for j in range(num_neurons[1]):
           eachneuron2.append(np.zeros(FIR_lenghts[2]))
       inputs2.append(eachneuron2)
    
    for i in range(num_neurons[2]): 
        inputs3.append(np.zeros(1))
    

    inputs.append(inputs0)
    inputs.append(inputs1)
    inputs.append(inputs2)
    inputs.append(inputs3)
    
    #Salida de la neurona
    activity0=np.array([0]).tolist()

    activity1=[]
    activity2=[]
    
    
    for i in range(num_neurons[1]):  
        activity1.append(np.zeros(FIR_lenghts[2])) 
                
            
    for i in range(num_neurons[2]):
        activity2.append(np.array([0]).tolist())


    activity.append(activity0)
    activity.append(activity1)
    activity.append(activity2)
    
    return inputs, activity


def init_delta_and_incWeights(num_neurons, FIR_lenghts):
      
    increment_weights=[]
    delta=[]
    
    #incremento de los pesos
    increment_weights0=np.array([0])
       
    increment_weights1=[]
    increment_weights2=[]
    eachneuron1=[]
    eachneuron2=[]
    
    for i in range(num_neurons[1]):
        eachneuron1=[]
        for j in range(num_neurons[0]):
            eachneuron1.append(np.zeros(FIR_lenghts[1]))
        increment_weights1.append(eachneuron1)
         
    for i in range(num_neurons[2]):
        eachneuron2=[]
        for j in range(num_neurons[1]):
            eachneuron2.append(np.zeros(FIR_lenghts[2]))
        increment_weights2.append(eachneuron2)

    increment_weights.append(increment_weights0)
    increment_weights.append(increment_weights1)
    increment_weights.append(increment_weights2)
   
    #deltas
    delta0=np.array([0])

    delta1=[]
    delta2=[]
    
    for i in range(num_neurons[1]):  
        delta1.append(np.zeros(FIR_lenghts[1]))

    for i in range(num_neurons[2]):
        delta2.append(np.zeros(FIR_lenghts[2]))


    delta.append(delta0)
    delta.append(delta1)
    delta.append(delta2)

  
    return delta, increment_weights