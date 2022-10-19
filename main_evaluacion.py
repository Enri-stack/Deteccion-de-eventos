# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 18:43:57 2022

@author: Paula
"""
 

import numpy as np
import os
import matplotlib.pyplot as plt #funciona como matlab (crea figura, traza lineas en un area de trazado etc)
import sys #funciones que interactuan con el interprete
import neural_network_evaluacion as fir_eval
import parameters as par

def get_filenames_list(feature_dir):
   filenames_list=[]
   for filename in os.listdir(feature_dir):
       filenames_list.append(filename)         
   return filenames_list

def main(argv):
    
        sParFile = 'C:\\Users\\Paula\\Documents\\PFG\\inputParameters.txt'
        sResultsDir = r'C:\Users\Paula\Documents\PFG\Resultados'
        eval_feature_dir=r'C:\Users\Paula\Documents\PFG\caracteristicas_eval_npy'
        eval_label_dir=r'\Users\Paula\Documents\PFG\eventos_eval_npy'
       
        # FIR-MLP settings     
        NumNeurons = [] 
        FIRlenghts = [] 
        
        #read initial parameters   
        lNetworkParams = par.readParameters(sParFile)
        print(lNetworkParams)     
        iNumLayers = int(lNetworkParams[0]) 
        
        for iLayer in range(1, iNumLayers + 1):
            NumNeurons.append(int(lNetworkParams[iLayer])) 
        
        FIRlenghts.append(0)
        for iLayer in range(iNumLayers + 1, 2 * iNumLayers):
            FIRlenghts.append(int(lNetworkParams[iLayer]))
        
        oNN = fir_eval.FIR_EVAL(iNumLayers, NumNeurons, FIRlenghts)
        
    
        print("\n**********PREDICTION**********\n")
        
        print("\nAnalysing evaluation dataset...\n")
        
        filenames_list=get_filenames_list(eval_feature_dir) 
        filenames_events=get_filenames_list(eval_label_dir)
         
        weights, threshold=par.load_weights_and_thresholds() 
        inputs, activity =par.init_inputs_and_activity(NumNeurons, FIRlenghts)
        
        
               
            
        errorTotal=0
        
          

        
        for file_cnt in range(len(filenames_list)):
            fileError=[]
            
            output_Clearthroat=[]
            output_Cough=[]
            output_Doorslam=[]
            output_Drawer=[]
            output_Keyboard=[]
            output_Keys=[]
            output_Knock=[]
            output_Laughter=[]
            output_Pageturn=[]
            output_Phone=[]
            output_Speech=[]
            output_Noise=[]
            
            numClearthroat_real=0
            numCough_real=0
            numDoorslam_real=0
            numDrawer_real=0
            numKeyboard_real=0
            numKeys_real=0
            numKnock_real=0
            numLaughter_real=0
            numPageturn_real=0
            numPhone_real=0
            numSpeech_real=0
            numNoise_real=0
            
            numEvents=0
         
          
            numClearthroat_true=0
            numCough_true=0
            numDoorslam_true=0
            numDrawer_true=0
            numKeyboard_true=0
            numKeys_true=0
            numKnock_true=0
            numLaughter_true=0
            numPageturn_true=0
            numPhone_true=0
            numSpeech_true=0
            numNoise_true=0
            
            
            cnt_polifonicos=0
            cnt_monofonicos=0
            
            frame_errorTotal=[]
            
            
            print('\n\n{}: {}'.format(file_cnt+1, filenames_list[file_cnt]))
            feature_file = np.load(os.path.join(eval_feature_dir, filenames_list[file_cnt])) 
            #feature_file = np.load(os.path.join(eval_feature_dir, filenames_list[file_cnt-1])) 
                
            label_file = np.load(os.path.join(eval_label_dir,  filenames_events[file_cnt]))  
            
                          
                        

            for r in range(len(feature_file[file_cnt])):  
                               
                
                frame=feature_file[:, r ]
                frame=np.squeeze(frame)
                label=label_file[:, r ]
                 
                for j in range(NumNeurons[0]): 
                        
                        inputs[0][j][0]=frame[j]
                        
                for i in range(NumNeurons[1]):
                    
                       inputs[1][i]= inputs[0]
                        
                for l in range(1, len(NumNeurons)):
                        
                      inputs=oNN.forward(l, inputs, activity, weights, threshold)
                      
                for k in range(NumNeurons[0]):
                    for i in range(FIRlenghts[l-1]-1, 0, -1):  
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
                           

                output_Clearthroat.append(output[0])
                output_Cough.append(output[1])
                output_Doorslam.append(output[2])
                output_Drawer.append(output[3])
                output_Keyboard.append(output[4])
                output_Keys.append(output[5])
                output_Knock.append(output[6])
                output_Laughter.append(output[7])
                output_Pageturn.append(output[8])
                output_Phone.append(output[9])
                output_Speech.append(output[10])
                output_Noise.append(output[11])
                             
                
                numEvents=numEvents+1 #contabiliza el numero de eventos
                
                
                
                if label[0] ==1:
                            numClearthroat_real=numClearthroat_real+1
                if label[1] ==1:
                            numCough_real=numCough_real+1
                if label[2] ==1:
                            numDoorslam_real=numDoorslam_real+1
                if label[3] ==1:
                            numDrawer_real=numDrawer_real+1
                if label[4] ==1:
                            numKeyboard_real=numKeyboard_real+1
                if label[5] ==1:
                            numKeys_real=numKeys_real+1
                if label[6] ==1:
                            numKnock_real=numKnock_real+1
                if label[7] ==1:
                            numLaughter_real=numLaughter_real+1
                if label[8] ==1:
                            numPageturn_real=numPageturn_real+1
                if label[9] ==1:
                            numPhone_real=numPhone_real+1
                if label[10] ==1:
                            numSpeech_real=numSpeech_real+1
                if label[11] ==1:
                            numNoise_real=numNoise_real+1
                            
                
                x=[]
                
                for p in range(len(output)):
                  a=output[p]
                 
                  if a>output[11]:
                      x.append(1)
                  else: 
                      x.append(0)                    
             
                            
                if x[0] == 1 and label[0] == 1:
                    numClearthroat_true=numClearthroat_true+1   
                if x[1] == 1 and label[1] == 1:
                    numCough_true=numCough_true+1  
                if x[2] == 1 and label[2] == 1:
                    numDoorslam_true=numDoorslam_true+1
                if x[3] == 1 and label[3] == 1:
                    numDrawer_true=numDrawer_true+1
                if x[4] == 1 and label[4] == 1:
                    numKeyboard_true=numKeyboard_true+1
                if x[5] == 1 and label[5] == 1:
                    numKeys_true=numKeys_true+1
                if x[6] == 1 and label[6] == 1:
                    numKnock_true=numKnock_true+1
                if x[7] == 1 and label[7] == 1:
                    numLaughter_true=numLaughter_true+1
                if x[8] == 1 and label[8] == 1:
                    numPageturn_true=numPageturn_true+1
                if x[9] == 1 and label[9] == 1:
                    numPhone_true=numPhone_true+1
                if x[10] == 1 and label[10] == 1:
                    numSpeech_true=numSpeech_true+1
                if x[0]==0 and x[1]==0 and x[2]==0 and x[3]==0 and x[4]==0 and x[5]==0 and x[6]==0 and x[7]==0 and x[8]==0 and x[9]==0 and x[10]==0 and x[11] == 0 and label[11] == 1:
                    numNoise_true=numNoise_true+1
                
             
                     
                
                
            plt.plot(range(len(frame_errorTotal)), frame_errorTotal, color='b')
            plt.ylim([0, 9])
            plt.ylabel('Cost')
            plt.xlabel('frames')
            plt.tight_layout()
            plt.grid()
            plt.title('epoch_eval: %i' %file_cnt) 
            sname= file_cnt+1
            plt.savefig(os.path.join(sResultsDir, str(sname) + '_plot_eval') + '.png')
            plt.close()
            
            plt.plot(range(len(output_Clearthroat)), output_Clearthroat, color='b')
            #plt.xlim([0,100])
            plt.ylim([0, 1])
            plt.tight_layout()
            plt.grid()
            sname= file_cnt+1
            plt.savefig(os.path.join(sResultsDir, str(sname) + '_plot_Clearthroat') + '.png')
            plt.close()
            
            plt.plot(range(len(output_Cough)), output_Cough, color='b')
            #plt.xlim([0,100])
            plt.ylim([0, 1])
            plt.tight_layout()
            plt.grid()
            sname= file_cnt+1
            plt.savefig(os.path.join(sResultsDir, str(sname) + '_plot_Cough') + '.png')
            plt.close()
            
            plt.plot(range(len(output_Doorslam)), output_Doorslam, color='b')
            #plt.xlim([0,100])
            plt.ylim([0, 1])
            plt.tight_layout()
            plt.grid()
            sname= file_cnt+1
            plt.savefig(os.path.join(sResultsDir, str(sname) + '_plot_Doorslam') + '.png')
            plt.close()
            
            plt.plot(range(len(output_Drawer)), output_Drawer, color='b')
            #plt.xlim([0,100])
            plt.ylim([0, 1])
            plt.tight_layout()
            plt.grid()
            sname= file_cnt+1
            plt.savefig(os.path.join(sResultsDir, str(sname) + '_plot_Drawer') + '.png')
            plt.close()
            
            plt.plot(range(len(output_Keyboard)), output_Keyboard, color='b')
            #plt.xlim([0,100])
            plt.ylim([0, 1])
            plt.tight_layout()
            plt.grid()
            sname= file_cnt+1
            plt.savefig(os.path.join(sResultsDir, str(sname) + '_plot_Keyboard') + '.png')
            plt.close()
            
            plt.plot(range(len(output_Keys)), output_Keys, color='b')
            #plt.xlim([0,100])
            plt.ylim([0, 1])
            plt.tight_layout()
            plt.grid() 
            sname= file_cnt+1
            plt.savefig(os.path.join(sResultsDir, str(sname) + '_plot_Keys') + '.png')
            plt.close()
            
            plt.plot(range(len(output_Knock)), output_Knock, color='b')
            #plt.xlim([0,100])
            plt.ylim([0, 1])
            plt.tight_layout()
            plt.grid()
            sname= file_cnt+1
            plt.savefig(os.path.join(sResultsDir, str(sname) + '_plot_Knock') + '.png')
            plt.close()
            
            plt.plot(range(len(output_Laughter)), output_Laughter, color='b')
           # plt.xlim([0,100])
            plt.ylim([0, 1])
            plt.tight_layout()
            plt.grid()
            sname= file_cnt+1
            plt.savefig(os.path.join(sResultsDir, str(sname) + '_plot_Laughter') + '.png')
            plt.close()
            
            plt.plot(range(len(output_Noise)), output_Noise, color='b')
            #plt.xlim([0,100])
            plt.ylim([0, 1])
            plt.tight_layout()
            plt.grid()
            sname= file_cnt+1
            plt.savefig(os.path.join(sResultsDir, str(sname) + '_plot_Noise') + '.png')
            plt.close()
            
            plt.plot(range(len(output_Pageturn)), output_Pageturn, color='b')
            #plt.xlim([0,100])
            plt.ylim([0, 1])
            plt.tight_layout()
            plt.grid()
            sname= file_cnt+1
            plt.savefig(os.path.join(sResultsDir, str(sname) + '_plot_Pageturn') + '.png')
            plt.close()
            
            plt.plot(range(len(output_Phone)), output_Phone, color='b')
            #plt.xlim([0,100])
            plt.ylim([0, 1])
            plt.tight_layout()
            plt.grid()
            sname= file_cnt+1
            plt.savefig(os.path.join(sResultsDir, str(sname) + '_plot_Phone') + '.png')
            plt.close()
            
            plt.plot(range(len(output_Speech)), output_Speech, color='b')
           # plt.xlim([0,100])
            plt.ylim([0, 1])
            plt.tight_layout()
            plt.grid()
            sname= file_cnt+1
            plt.savefig(os.path.join(sResultsDir, str(sname) + '_plot_Speech') + '.png')
            plt.close()
            
            #save prediction stadistics
            
            file_path=  os.path.join(sResultsDir, 'prediction_'+ str(file_cnt) + '_predictionStadistics.txt')
    
            file = open(file_path, "w")
            file.write('\n****************************PREDICTION STADISTICS******************************' + '\n\n\n') 
            
            
            file.write('\n****************************STADISTICS OF EACH ACOUSTIC EVENT******************************' + '\n\n') 

            file.write('CLEARTHROAT' + '\n')
            file.write('-Number of clearthroat: %i' %numClearthroat_real + '\n')
            file.write('-Number of clearthroat true: %i' %numClearthroat_true + '\n')
            if(numClearthroat_real!=0):
                file.write('-Percentage os success: {0:.2f} %  '.format((numClearthroat_true/numClearthroat_real)*100) + '\n\n\n')
            else:
                file.write('\n\n\n')
                
            file.write('COUGHT' + '\n')
            file.write('-Number of Cough: %i'  % numCough_real + '\n')
            file.write('-Number of Cough true: %i'  % numCough_true + '\n')
            if(numCough_real!=0):
                file.write('-Percentage os success: {0:.2f} %  '.format((numCough_true/numCough_real)*100)+ '\n\n\n')
            else:
                file.write('\n\n\n')
                
            file.write('DOORSLAM' + '\n')
            file.write('-Number of doorslam: %i' %numDoorslam_real + '\n')
            file.write('-Number of doorslam true: %i' %numDoorslam_true + '\n')
            if(numDoorslam_real!=0):
                file.write('-Percentage os success: {0:.2f} %  '.format((numDoorslam_true/numDoorslam_real)*100) + '\n\n\n')
            else:
                file.write('\n\n\n')
                
            file.write('DRAWER' + '\n')
            file.write('-Number of drawer: %i' %numDrawer_real + '\n')
            file.write('-Number of drawer true: %i' %numDrawer_true + '\n')
            if(numDrawer_real!=0):
                file.write('-Percentage os success: {0:.2f} %  '.format((numDrawer_true/numDrawer_real)*100) + '\n\n\n')
            else:
                file.write('\n\n\n')
                
            file.write('KEYBOARD' + '\n')
            file.write('-Number of keyboard: %i' %numKeyboard_real + '\n')
            file.write('-Number of keyboard true: %i' %numKeyboard_true + '\n')
            if(numKeyboard_real!=0):
                file.write('-Percentage os success: {0:.2f} %  '.format((numKeyboard_true/numKeyboard_real)*100) + '\n\n\n')
            else:
                file.write('\n\n\n')
                
            file.write('KEYS' + '\n')
            file.write('-Number of keys: %i' %numKeys_real + '\n')
            file.write('-Number of keys true: %i' %numKeys_true + '\n')
            if(numKeys_real!=0):
             file.write('-Percentage os success: {0:.2f} %  '.format((numKeys_true/numKeys_real)*100) + '\n\n\n')
            else:
                file.write('\n\n\n')
                
            file.write('KNOCK' + '\n')
            file.write('-Number of knock: %i' %numKnock_real + '\n')
            file.write('-Number of knock true: %i' %numKnock_true + '\n')
            if(numKnock_real!=0):
                file.write('-Percentage os success: {0:.2f} %  '.format((numKnock_true/numKnock_real)*100) + '\n\n\n')
            else:
                file.write('\n\n\n')
                
            file.write('LAUGHTER' + '\n')
            file.write('-Number of laughter:%i ' %numLaughter_real + '\n')
            file.write('-Number of laughter true:%i ' %numLaughter_true + '\n')
            if(numLaughter_real!=0):
                file.write('-Percentage os success: {0:.2f} %  '.format((numLaughter_true/numLaughter_real)*100) + '\n\n\n')
            
            file.write('PAGETURN' + '\n')
            file.write('-Number of pageturn: %i' %numPageturn_real + '\n')
            file.write('-Number of pageturn true: %i' %numPageturn_true + '\n')
            if(numPageturn_real!=0):
                file.write('-Percentage os success: {0:.2f} %  '.format((numPageturn_true/numPageturn_real)*100) + '\n\n\n')
            else:
                file.write('\n\n\n')
                
            file.write('PHONE' + '\n')
            file.write('-Number of phone:%i ' %numPhone_real + '\n')
            file.write('-Number of phone true:%i ' %numPhone_true + '\n')
            if(numPhone_real!=0):
                file.write('-Percentage os success: {0:.2f} %  '.format((numPhone_true/numPhone_real)*100) + '\n\n\n')
            else:
                file.write('\n\n\n')
                
            file.write('SPEECH' + '\n')
            file.write('-Number of speech:%i ' %numSpeech_real + '\n')
            file.write('-Number of speech true:%i ' %numSpeech_true + '\n')
            if(numSpeech_real!=0):
                file.write('-Percentage os success: {0:.2f} %  '.format((numSpeech_true/numSpeech_real)*100) + '\n\n\n')
            else:
                file.write('\n\n\n')
                
            file.write('NOISE' + '\n')
            file.write('-Number of noise: %i' %numNoise_real + '\n')
            file.write('-Number of noise true: %i' %numNoise_true + '\n')
            if(numNoise_real!=0):
                file.write('-Percentage os success: {0:.2f} %  '.format((numNoise_true/numNoise_real)*100) + '\n\n\n')
            else:
                file.write('\n\n\n')
                
                        
            file.write('-Number of analized frames: %i ' %numEvents + '\n\n\n')
            file.write('-Percentage os success: {0:.2f} %  '.format(((numClearthroat_true+numCough_true+numDoorslam_true+numDrawer_true+numKeyboard_true+numKeys_true+numKnock_true+numLaughter_true+numPageturn_true+ numPhone_true+numSpeech_true+numNoise_true)/numEvents)*100)  + '\n')
            
            
            file.write("*********************************************************************************************************************\n")
                
            file.close()   
                        
        
            fileError+=frame_errorTotal
            
if __name__ == '__main__':
    main(sys.argv) 