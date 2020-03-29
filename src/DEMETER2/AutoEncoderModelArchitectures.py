'''
Created on Mar 10, 2020

@author: andrewdavidson aedavids@ucsc.edu
'''

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam

###############################################################################
class AutoEncoderModelArchitectures(object):
    '''
    specifying multiple deep architectures in juypter notebooks gets messy.
    
    evaluating multiple deep architectures in python gets messey
    '''

    ###############################################################################
    def __init__(self, trainingDataShape, encodingDimensions=19, 
                 optimizer='adam', loss='mean_squared_error',
                 metrics=['accuracy', 'mse']):
        '''
        https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
        https://www.tensorflow.org/api_docs/python/tf/keras/losses
        '''
        self.trainingDataShape = trainingDataShape
        self.numFeatures = trainingDataShape[1]
        self.ed = encodingDimensions
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

    ###############################################################################        
    def debugArch(self ):
        '''
        https://www.tensorflow.org/api_docs/python/tf/keras/metrics
    
        why is validation error less than test error?
        
        aedwip 
        https://arxiv.org/abs/1511.07289 should we be using 'elu' activation
        '''
        autoencoderModel = Sequential()
    
        # adding an input layer this way instead of using input_shape argument
        # of first layer makes it easier to change the arch
        autoencoderModel.add(Input(shape=(self.numFeatures,), name="input"))
    
        autoencoderModel.add(Dense( 3, activation='relu', name="e064"))
    
        autoencoderModel.add(Dense(self.ed,    activation='relu', name="bottleneck"))
    
        # decoder part of model
        autoencoderModel.add(Dense( 3, activation='relu', name="d064"))

    
        # do not use sigmoid all predictions will be zero. 
        #autoencoderModel.add(Dense(numFeatures,    activation='sigmoid', name="dOut"))
        autoencoderModel.add(Dense(self.numFeatures,    activation='relu', name="dOut"))
    
        # set up our optimizer
        # https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
        autoencoderModel.compile(loss=self.loss, optimizer=self.optimizer,  metrics=self.metrics)
      
        return autoencoderModel
         
    ###############################################################################
    def __str__(self, *args, **kwargs):
        #return object.__str__(self, *args, **kwargs)
        fmt1 = "traningDataShape:{} \nnumFeatures:{} \nencodingDimensions:{}"
        fmt2 = "\noptimizer:{} \nloss:{} \nmetrics:{}"
        fmt = fmt1 + fmt2
        ret = fmt.format(self.trainingDataShape,
                   self.numFeatures,
                    self.ed,
                    self.optimizer,
                    self.loss,
                    self.metrics
                    )
        return ret
        
    ###############################################################################        
    def simpleArch(self ):
        '''
        https://www.tensorflow.org/api_docs/python/tf/keras/metrics
    
        '''
        autoencoderModel = Sequential()
    
        # adding an input layer this way instead of using input_shape argument
        # of first layer makes it easier to change the arch
        autoencoderModel.add(Input(shape=(self.numFeatures,), name="input"))
    
        autoencoderModel.add(Dense( 64, activation='relu', name="e064"))
    
        autoencoderModel.add(Dense(self.ed,    activation='relu', name="bottleneck"))
    
        # decoder part of model
        autoencoderModel.add(Dense( 64, activation='relu', name="d064"))

    
        # do not use sigmoid all predictions will be zero. 
        #autoencoderModel.add(Dense(numFeatures,    activation='sigmoid', name="dOut"))
        autoencoderModel.add(Dense(self.numFeatures,    activation='relu', name="dOut"))
    
        # set up our optimizer
        # https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
        autoencoderModel.compile(loss=self.loss, optimizer=self.optimizer,  metrics=self.metrics)
      
        return autoencoderModel
    
    ###############################################################################    
    def l3Arch(self ):
        '''
        https://www.tensorflow.org/api_docs/python/tf/keras/metrics
    
        '''            
        autoencoderModel = Sequential()
    
        # adding an input layer this way instead of using input_shape argument
        # of first layer makes it easier to change the arch
        autoencoderModel.add(Input(shape=(self.numFeatures,), name="input"))
    
        autoencoderModel.add(Dense( 128, activation='relu', name="e128"))
        autoencoderModel.add(Dense( 64, activation='relu', name="e064"))
    
        autoencoderModel.add(Dense(self.ed,    activation='relu', name="bottleneck"))
    
        # decoder part of model
        autoencoderModel.add(Dense( 64, activation='relu', name="d064"))
        autoencoderModel.add(Dense( 128, activation='relu', name="d128"))
    
        # do not use sigmoid all predictions will be zero
        autoencoderModel.add(Dense(self.numFeatures,    activation='relu', name="dOut"))
    
        # set up our optimizer
        # https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
        autoencoderModel.compile(loss=self.loss, optimizer=self.optimizer,  metrics=self.metrics)
        
        return autoencoderModel  
    
    ###############################################################################    
    def l5Arch(self ):
        '''
        https://www.tensorflow.org/api_docs/python/tf/keras/metrics
    
        '''    
        autoencoderModel = Sequential()
    
        # adding an input layer this way instead of using input_shape argument
        # of first layer makes it easier to change the arch
        autoencoderModel.add(Input(shape=(self.numFeatures,), name="input"))
    
        autoencoderModel.add(Dense( 256, activation='relu', name="e256"))
        autoencoderModel.add(Dense( 128, activation='relu', name="e128"))
        autoencoderModel.add(Dense( 64, activation='relu', name="e064"))
    
        autoencoderModel.add(Dense(self.ed,    activation='relu', name="bottleneck"))
    
        # decoder part of model
        autoencoderModel.add(Dense( 64, activation='relu', name="d064"))
        autoencoderModel.add(Dense( 128, activation='relu', name="d128"))
        autoencoderModel.add(Dense( 256, activation='relu', name="d256"))
    
        # do not use sigmoid all predictions will be zero. 
        #autoencoderModel.add(Dense(numFeatures,    activation='sigmoid', name="dOut"))
        autoencoderModel.add(Dense(self.numFeatures,    activation='relu', name="dOut"))
    
        # set up our optimizer
        # https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
        autoencoderModel.compile(loss=self.loss, optimizer=self.optimizer,  metrics=self.metrics)
        
        return autoencoderModel          
    