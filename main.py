# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 19:24:16 2021

@author: Paula
"""

import numpy as np
import os
import sys #funciones que interactuan con el interprete
import neural_network as fir_mlp
import parameters as par
import matplotlib.pyplot as plt #funciona como matlab (crea figura, traza lineas en un area de trazado etc)



def get_filenames_list(feature_dir):
    filenames_list=[]
    for filename in os.listdir(feature_dir):
        filenames_list.append(filename)
    return filenames_list 

def main(argv):   
    
    #Parameter file
    ParFile = 'C:\\Users\\Paula\\Documents\\PFG\\inputParameters.txt'
    sFeatureDir = r'C:\Users\Paula\Documents\PFG\caracteristicas_npy'
    sLabelDir = r'C:\Users\Paula\Documents\PFG\eventos_npy'
    sResultsDir = r'C:\Users\Paula\Documents\PFG\Resultados'
    
    # FIR-MLP settings     
    NumNeurons = [] 
    FIRlenghts = [] 
    
    #read initial parameters 
    NetworkParameters = par.readParameters(ParFile)
    print(NetworkParameters)

    iNumLayers = int(NetworkParameters[0]) #numero de capas
    
    
    for iLayer in range(1, iNumLayers + 1):
        NumNeurons.append(int(NetworkParameters[iLayer])) #numero de neuronas 
    FIRlenghts.append(1) #longitud de los filtros sinapticos
        
    for iLayer in range(iNumLayers + 1, 2 * iNumLayers):
        FIRlenghts.append(int(NetworkParameters[iLayer]))
    
    rLearningRate = float(NetworkParameters[2 * iNumLayers]) #tasa de aprendizaje
    rMomentum = float(NetworkParameters[2 * iNumLayers + 1]) #momentum
    
    
    #create neural network instance
    oNN = fir_mlp.FIR_MLP(iNumLayers, NumNeurons, FIRlenghts, rLearningRate, rMomentum)
    
    filenames_list=get_filenames_list(sFeatureDir)
    filenames_events=get_filenames_list(sLabelDir)
       

    epochs=6 #Number of epochs
    
       
    inputs, activity =par.init_inputs_and_activity(NumNeurons, FIRlenghts)
    weights, threshold=par.init_weights_and_threshold(NumNeurons, FIRlenghts)                
    delta, increment_weights =par.init_delta_and_incWeights(NumNeurons, FIRlenghts)
   
        
    for epoch_cnt in range(epochs): 

         
          
          
          
          
                     
          print("\n*****************************************************")
          print('*********************EPOCH nº {}**********************'.format(epoch_cnt+1)) 
          print("*****************************************************")
            
          """for file_cnt in range(len(filenames_list)): """  
                
          if epoch_cnt==0:
              x=1
              y=21 #SERIA 16
              z=1
          if epoch_cnt==1:
              x=40
              y=20
              z=-1
          if epoch_cnt==2:
              x=2
              y=42
              z=2
          if epoch_cnt==3:
              x=1
              y=41
              z=2
          if epoch_cnt==4:
              x=40
              y=0
              z=-2
          if epoch_cnt==5:
              x=40
              y=1
              z=-2
              
             
          
          frame_errorTotal=[]
          for r in range(x,y,z):
                
               print('\n\n{}: Training of file {}'.format(r,filenames_list[r-1]))  
               feature_file = np.load(os.path.join(sFeatureDir, filenames_list[r-1])) 
                
               label_file = np.load(os.path.join(sLabelDir,  filenames_events[r-1])) 

               for r in range(len(feature_file[0])):    
                    
                    frame=feature_file[:,r ]#Quitar el 3 para hacerlo con todas  las filas
                    frame=np.squeeze(frame)
                    label=label_file[:, r ]
                        
                    for j in range(NumNeurons[0]): 
                        
                        inputs[0][j][0]=frame[j]
                        
                    for i in range(NumNeurons[1]):
                    
                       inputs[1][i]= inputs[0]
                        
                    for l in range(1, len(NumNeurons)):
                        
                      inputs=oNN.forward(l, inputs, activity, weights, threshold)
                      
                    for k in range(NumNeurons[0]):
                        for i in range(FIRlenghts[l-1]-1, 0, -1):  #He puesto l-1
                          inputs[0][k][i]=inputs[0][k][i-1]
                    
                                                          
                    output=(np.squeeze(inputs[3]))
                        
                    label=label_file[:, r ] 
                    
                    error= output-label #-error
                    
                    error_real=[]     
                    for i in range(len(output)):
                       error_real.append(float(-label[i]*np.log(output[i])-(1-label[i])*np.log(1-output[i])))
                    
                    #n=len(output)    
                    
                    #error_real=label-output

                    errorTotal=np.sum(error_real)
                    frame_errorTotal.append(errorTotal)
                           

                   
                    
                   

                    for l in range(iNumLayers, 1, -1):  

                      oNN.backward(l, error,inputs, activity, weights, increment_weights, delta, threshold)
                      
                    for l in range(1, len(NumNeurons)):
                        
                        for j in range(NumNeurons[l]):
                            
                            for i in range(NumNeurons[l-1]):
                            
                              #update weights
                              weights[l][j][i]=  weights[l][j][i]+increment_weights[l][j][i]
                              
                              
          
          plt.plot(range(len(frame_errorTotal)), frame_errorTotal, color='b')
          plt.ylim([-1, 20])
          plt.ylabel('Cost')
          plt.xlabel('frames')
          plt.tight_layout()
          plt.grid()
          plt.title('epoch_eval: %i' %epoch_cnt) 
          sname= epoch_cnt+1
          plt.savefig(os.path.join(sResultsDir, str(sname) + '_plot_eval') + '.png')
          plt.close()    
          
    # plt.plot(range(len(frame_errorTotal)), frame_errorTotal, color='b')
    # plt.ylim([-1, 20])
    # plt.ylabel('Cost')
    # plt.xlabel('frames')
    # plt.tight_layout()
    # plt.grid()
    # plt.title('epoch_eval: %i' %epoch_cnt) 
    # sname= epoch_cnt+1
    # plt.savefig(os.path.join(sResultsDir, str(sname) + '_plot_eval') + '.png')
    # plt.close()             
          
          #save weights and thresholds
          np.save(os.path.join(sResultsDir, 'epoch_' + str(epoch_cnt+1) + '_weights.npy'), weights, allow_pickle=True )
          np.save(os.path.join(sResultsDir, 'epoch_' + str(epoch_cnt+1) + '_threshold.npy'), threshold, allow_pickle=True )
                    

if __name__ == '__main__':
    main(sys.argv)        
        
        
        
        
        
        
        
        
        
        
        
        