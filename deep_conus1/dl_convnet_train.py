
#####################################################################################
#####################################################################################
#
# Author: Maria J. Molina
# National Center for Atmospheric Research
#
# Script to split data into training and testing sets, and standardize, for deep learning model training. 
#
#
#####################################################################################
#####################################################################################


#----------------------------------------------------------


import matplotlib.pyplot as plt

import keras
import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session

from keras.models import Model, save_model, load_model, Sequential
from keras.layers import Dense, Activation, Conv2D, Input, AveragePooling2D, MaxPooling2D
from keras.layers import SpatialDropout2D, Flatten, LeakyReLU, Dropout, BatchNormalization, LeakyReLU
from keras.optimizers import SGD, Adam
from keras.regularizers import l2

from sklearn.metrics import mean_squared_error, roc_auc_score

import xarray as xr
import numpy as np
import pandas as pd


#----------------------------------------------------------



class dl_training:
    
    
    def __init__(self, working_directory, dlfile_directory, variables, model_num, 
                 mask=False, climate='current', 
                 conv_1_mapnum=32, conv_2_mapnum=68, conv_3_mapnum=128, 
                 acti_1_func='relu', acti_2_func='relu', acti_3_func='relu',
                 filter_width=5, learning_rate = 0.0001, output_func_and_loss='sigmoid_mse', strides_len=1,
                 validation_split=0.1, batch_size=128, epochs=10, 
                 pool_method='mean', batch_norm=True, spatial_drop=True):
        
        
        """
            Class instantiation of dl_training:
            
            Build and train a deep convolutional neural network using previously created imported data.
            
            
        PARAMETERS
        ----------
        working_directory: directory path to save DL model (str)
        dlfile_directory: directory path where train data is stored (str)
        variables: numpy array of variable name strings
        model_num: number assignment for the model (int)
        climate: current or future climate (str; default current)
        conv_1_mapnum: number of activation (feature) maps in first conv layer (int; default 32)
        conv_2_mapnum: number of activation (feature) maps in second conv layer (int; default 68)
        conv_3_mapnum: number of activation (feature) maps in third conv layer (int; default 128)
        acti_1_func: activation function to apply to first conv layer output (str; default relu)
        acti_2_func: activation function to apply to second conv layer output (str; default relu)
        acti_3_func: activation function to apply to third conv layer output (str; default relu)
        filter_width: width of sliding filter for conv layers (int; default 5)
        learning_rate: learning rate to use for Adam optimizer
        output_func_and_loss: the activation function to apply to the output layer and loss function to use in training (str; 
                              default sigmoid_mse [sigmoid act func and mean squared error])
        strides_len: length of strides when sliding filter (int; default 1)
        validation_split: percent split of training data used for validation (float; default 0.1 [10%])
        batch_size: size of batch used during training (int; default 128)
        epochs: number of epochs to run through during training (int; default 10)
        pool_method: method to use for pooling layers (str; default mean [also have max])
        batch_norm: whether to apply batch normalization after every conv layer (boolean; default True)
        spatial_drop: whether to apply spatial dropout (30%) after every conv layer (boolean; default True)
        
        """
        

        self.working_directory = working_directory
        self.dlfile_directory = dlfile_directory
        
        self.variables = variables
        self.model_num = model_num
        
        self.mask = mask
        if not self.mask:
            self.mask_str = 'nomask'
        if self.mask:
            self.mask_str = 'mask'
        
        self.conv_1_mapnum = conv_1_mapnum
        self.conv_2_mapnum = conv_2_mapnum
        self.conv_3_mapnum = conv_3_mapnum
        
        self.acti_1_func = acti_1_func
        self.acti_2_func = acti_2_func
        self.acti_3_func = acti_3_func
        
        self.filter_width = filter_width
        self.learning_rate = learning_rate
        
        
        self.output_func_and_loss = output_func_and_loss

        if self.output_func_and_loss == 'softmax':
            self.denseshape = 2
            self.loss_func = 'sparse_categorical_crossentropy'

        if self.output_func_and_loss == 'sigmoid_bin':
            self.denseshape = 1        
            self.loss_func = 'binary_crossentropy'
            self.output_activation = 'sigmoid'

        if self.output_func_and_loss == 'sigmoid_mse':
            self.denseshape = 1        
            self.loss_func = 'mean_squared_error'
            self.output_activation = 'sigmoid'
            
        
        self.batch_norm = batch_norm
        self.spatial_drop = spatial_drop
        self.strides_len = strides_len
        
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.epochs = epochs
        
        
        if pool_method == 'mean':
            self.meanpool = True
            self.maxpool = False
            
        if pool_method == 'max':
            self.meanpool = False
            self.maxpool = True
            
            
        if climate != 'current' and climate != 'future':
            raise Exception("Please enter current or future for climate option.")
        if climate == 'current' or climate == 'future':
            self.climate = climate
        
        
        
    def variable_translate(self, variable):
        
        """
            Variable name for the respective filenames.
        """
        
        var = {
               'EU':'EU',
               'EV':'EV',
               'TK':'TK',
               'QVAPOR':'QVAPOR',
               'WMAX':'MAXW',
               'W_vert':'W',
               'PRESS':'P',
               'DBZ':'DBZ',
               'CTT':'CTT',
               'UH25':'UH25',
               'UH03':'UH03',
              }
        
        try:
            out = var[variable]
            return out
        except:
            raise ValueError("Please enter TK, EU, EV, QVAPOR, P, W_vert, or WMAX as variable.")
            
            

    def initiate_session(self):
        
        """
            Initiate GPU session for DL training.
        """
        
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                            # (nothing gets printed in Jupyter, only if you run it standalone)
        sess = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(sess)
        return
    
    
    
    def open_files(self):

        """
            Open the training data files.
        """
        
        datas = {}
        
        for var in self.variables:

            datas[var] = xr.open_dataset(f'/{self.dlfile_directory}/{self.climate}_{self.variable_translate(var).lower()}_{self.mask_str}_dldata_traintest.nc')
            
        return datas

        

    def transpose_load_concat(self, **kwargs):
        
        """
            Eagerly load the training labels and data, reshaping data to have features in final dim.
        """
        
        thedatas = {}
        label = {}
        
        for key, value in kwargs.items():

            thedatas[key] = value.X_train.transpose('a','x','y','features').values
            label = value.X_train_label.values

        if len(kwargs) > 1:
            X_train = np.concatenate(list(thedatas.values()), axis=3)
        if len(kwargs) == 1:
            X_train = np.squeeze(np.asarray(list(thedatas.values())))
        
        return X_train, label


        
    def omit_nans(self, data, label):

        """
            Remove any nans from the training data.
        """
        
        maskarray = np.full(data.shape[0], True)
        
        masker = np.unique(np.argwhere(np.isnan(data))[:,0])
        
        maskarray[masker] = False
        
        traindata = data[maskarray,:,:,:]
        trainlabel = trainlabel[maskarray]
        
        return traindata, trainlabel



    def compile_meanpool_model(self, data):

        """
            Assemble and compile the deep conv neural network.
        """
        
        model = Sequential([

            Conv2D(self.conv_1_mapnum, 
                   (self.filter_width, self.filter_width),
                   input_shape=data.shape[1:], 
                   strides=self.strides_len,
                   padding='same', data_format='channels_last',
                   dilation_rate=1, activation=self.acti_1_func, use_bias=True, 
                   kernel_initializer='glorot_uniform', bias_initializer='zeros', 
                   kernel_regularizer=l2(0.001), bias_regularizer=None, 
                   activity_regularizer=None, kernel_constraint=None, 
                   bias_constraint=None),

            
            BatchNormalization(axis=3, momentum=0.99, epsilon=0.001, 
                                   center=True, scale=True, 
                                   beta_initializer='zeros', gamma_initializer='ones',
                                   moving_mean_initializer='zeros', 
                                   moving_variance_initializer='ones',
                                   beta_regularizer=None, gamma_regularizer=None,
                                   beta_constraint=None, gamma_constraint=None),

            SpatialDropout2D(rate=0.3, data_format='channels_last'),

            AveragePooling2D(pool_size=(2, 2), strides=None, padding='same', 
                                 data_format='channels_last'),
            
            
            Conv2D(self.conv_2_mapnum, 
                   (self.filter_width, self.filter_width),
                   input_shape=data.shape[1:], 
                   strides=self.strides_len,
                   padding='same', data_format='channels_last',
                   dilation_rate=1, activation=self.acti_1_func, use_bias=True, 
                   kernel_initializer='glorot_uniform', bias_initializer='zeros', 
                   kernel_regularizer=l2(0.001), bias_regularizer=None, 
                   activity_regularizer=None, kernel_constraint=None, 
                   bias_constraint=None),

            
            BatchNormalization(axis=3, momentum=0.99, epsilon=0.001, 
                                   center=True, scale=True, 
                                   beta_initializer='zeros', gamma_initializer='ones',
                                   moving_mean_initializer='zeros', 
                                   moving_variance_initializer='ones',
                                   beta_regularizer=None, gamma_regularizer=None,
                                   beta_constraint=None, gamma_constraint=None),

            SpatialDropout2D(rate=0.3, data_format='channels_last'),

            AveragePooling2D(pool_size=(2, 2), strides=None, padding='same', 
                                 data_format='channels_last'),


            Conv2D(self.conv_3_mapnum, 
                   (self.filter_width, self.filter_width),
                   input_shape=data.shape[1:], 
                   strides=self.strides_len,
                   padding='same', data_format='channels_last',
                   dilation_rate=1, activation=self.acti_1_func, use_bias=True, 
                   kernel_initializer='glorot_uniform', bias_initializer='zeros', 
                   kernel_regularizer=l2(0.001), bias_regularizer=None, 
                   activity_regularizer=None, kernel_constraint=None, 
                   bias_constraint=None),


            BatchNormalization(axis=3, momentum=0.99, epsilon=0.001, 
                                   center=True, scale=True, 
                                   beta_initializer='zeros', gamma_initializer='ones',
                                   moving_mean_initializer='zeros', 
                                   moving_variance_initializer='ones',
                                   beta_regularizer=None, gamma_regularizer=None,
                                   beta_constraint=None, gamma_constraint=None),

            SpatialDropout2D(rate=0.3, data_format='channels_last'),

            AveragePooling2D(pool_size=(2, 2), strides=None, padding='same', 
                                 data_format='channels_last'),
            

            Flatten(),

            Dense(units = self.denseshape, activation = self.output_activation, 
                  use_bias=True, 
                  kernel_initializer='glorot_uniform', 
                  bias_initializer='zeros', kernel_regularizer=None, 
                  bias_regularizer=None, activity_regularizer=None, 
                  kernel_constraint=None, bias_constraint=None)

        ])


        model.compile(optimizer=Adam(lr = self.learning_rate), loss = self.loss_func, metrics=['accuracy', 'mean_squared_error', 'mean_absolute_error'])

        return model

    
    
    def compile_dl(self, data):
        
        """
            Compile the convolutional neural network and print model summary.
        """
    
        model = compile_meanpool_model(data)
        model.summary()
        return
        
        
        
    def train_dl(self, model, data, label):
        
        """
            Train the compiled DL model and save output.
        """
        
        history = model.fit(x = data, 
                            y = label, 
                            validation_split = self.validation_split, 
                            batch_size = self.batch_size, 
                            epochs = self.epochs, 
                            shuffle = True)
        
        pd.DataFrame(history.history).to_csv(f'/{self.working_directory}/model_{self.model_num}_{self.climate}.csv')
        save_model(model, f"/{self.working_directory}/model_{self.model_num}_{self.climate}.h5")
        return
    
    

    
    
    