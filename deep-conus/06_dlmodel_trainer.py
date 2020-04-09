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


class DLTraining:

    """Class instantiation of DLTraining:
            
    Build and train a deep convolutional neural network using previously created imported data.
            
    Attributes:
        working_directory (str): Directory path to save DL model.
        dlfile_directory (str): Directory path where train data is stored.
        variables (str): Numpy array of variable name strings. Options include ``EU``, ``EV``, ``TK``, ``QVAPOR``, ``WMAX``, 
                         ``W_vert``,``PRESS``,``DBZ``,``CTT``,``UH25``, and``UH03``.
        model_num (int): Number assignment for the model.
        mask (boolean): Whether to train using the masked data or the non-masked data. Defaults to False.
        climate (str): Whether to train with the ``current`` or ``future`` climate simulations. Defaults to ``current``.
        print_sequential (boolean): Whether to print the sequetial steps occurring during training. Defaults to ``True``.
        conv_1_mapnum (int): Number of activation maps in first conv layer. Defaults to 32.
        conv_2_mapnum (int): Number of activation maps in second conv layer. Defaults to 68.
        conv_3_mapnum (int): Number of activation maps in third conv layer. Defaults to 128.
        acti_1_func (str): Activation function to apply to first conv layer. Defaults to ``relu``.
        acti_2_func (str): Activation function to apply to second conv layer. Defaults to ``relu``.
        acti_3_func (str): Activation function to apply to third conv layer. Defaults to ``relu``.
        filter_width (int): Width of sliding filter to apply to conv layers. Defaults to 5.
        learning_rate (float): Learning rate to use for Adam optimizer. Defaults to 0.0001.
        output_func_and_loss (str): The activation function to apply to the output layer and the loss function to use in training. 
                                    Defaults to ``sigmoid_mse`` [sigmoid activation function and mean squared error loss function]).
        strides_len (int): The length of strides to use when sliding the filter. Defaults to 1.
        validation_split (float): The percent split of training data used for validation. Defaults to 0.1 [e.g., 10% of training data]).
        batch_size (int): Size of batches used during training. Defaults to 128.
        epochs (int): The number of epochs to run through during training. Defaults to 10.
        pool_method (str): Pooling method. Defaults to ``mean`` (also have ``max`` available).
        batch_norm (boolean): Whether to apply batch normalization after every convolutional layer. Defaults to ``True``.
        spatial_drop (boolean): Whether to apply spatial dropout (at 30%) after every convolutional layer. Defaults to ``True``.
        
    Raises:
        Exception: Checks whether correct values were input for ``climate``, ``output_func_and_loss``, and ``pool_method``.
        
    """
    
    def __init__(self, working_directory, dlfile_directory, variables, model_num, 
                 mask=False, climate='current', print_sequential=True,
                 conv_1_mapnum=32, conv_2_mapnum=68, conv_3_mapnum=128, 
                 acti_1_func='relu', acti_2_func='relu', acti_3_func='relu',
                 filter_width=5, learning_rate=0.0001, output_func_and_loss='sigmoid_mse', strides_len=1,
                 validation_split=0.1, batch_size=128, epochs=10, 
                 pool_method='mean', batch_norm=True, spatial_drop=True):

        self.working_directory=working_directory
        self.dlfile_directory=dlfile_directory
        self.variables=variables
        self.model_num=model_num
        
        self.mask=mask
        if not self.mask:
            self.mask_str='nomask'
        if self.mask:
            self.mask_str='mask'

        if climate!='current' and climate!='future':
            raise Exception("Please enter current or future for climate option.")
        else:
            self.climate=climate
            
        self.print_sequential=print_sequential
            
        self.conv_1_mapnum=conv_1_mapnum
        self.conv_2_mapnum=conv_2_mapnum
        self.conv_3_mapnum=conv_3_mapnum
        
        self.acti_1_func=acti_1_func
        self.acti_2_func=acti_2_func
        self.acti_3_func=acti_3_func
        
        self.filter_width=filter_width
        self.learning_rate=learning_rate
        
        self.output_func_and_loss=output_func_and_loss
        if self.output_func_and_loss!='softmax' and self.output_func_and_loss!='sigmoid_bin' and self.output_func_and_loss!='sigmoid_mse':
            raise Exception("``self.output_func_and_loss`` options include ``softmax``, ``sigmoid_bin``, and ``sigmoid_mse``.")
        elif self.output_func_and_loss=='sigmoid_mse':
            self.denseshape=1        
            self.loss_func='mean_squared_error'
            self.output_activation='sigmoid'            
        elif self.output_func_and_loss=='softmax':
            self.denseshape=2
            self.loss_func='sparse_categorical_crossentropy'
        else: #sigmoid_bin
            self.denseshape=1        
            self.loss_func='binary_crossentropy'
            self.output_activation='sigmoid'

        self.strides_len=strides_len
        self.validation_split=validation_split
        self.batch_size=batch_size
        self.epochs=epochs
        
        if pool_method!='mean' and pool_method!='max':
            raise Exception('``pool_method`` options available are ``mean`` and ``max``.')
        else:
            self.pool_method=pool_method
        self.batch_norm=batch_norm
        self.spatial_drop=spatial_drop
        
        
    def variable_translate(self, variable):
        
        """Variable name for the respective filenames.
           
        Args:
            variable (str): The variable to feed into the dictionary.
            
        Returns:
            variable (str): The variable name to use for opening saved files.
            
        Raises:
            ValueError: If provided variable is not available.
            
        """
        var={
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
            out=var[variable]
            return out
        except:
            raise ValueError("Please enter ``TK``, ``EV``, ``EU``, ``QVAPOR``, ``PRESS``, ``W_vert``, ``UH25``, ``UH03``, ``MAXW``, ``CTT``, or ``DBZ`` as variable.")
            

    def initiate_session(self):
        
        """Initiate CPU or GPU session for DL training.
        
        """
        config=tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth=True  # dynamically grow the memory used on the GPU
        config.log_device_placement=True  # to log device placement (on which device the operation ran)
                                            # (nothing gets printed in Jupyter, only if you run it standalone)
        sess=tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(sess)
    
    
    def open_files(self):

        """Open the training data files.
        
        Returns:
            datas: Dictionary of opened Xarray data arrays containing selected variable training data.
            
        """
        datas={}
        for var in self.variables:
            datas[var]=xr.open_dataset(f'/{self.dlfile_directory}/{self.climate}_{self.variable_translate(var).lower()}_{self.mask_str}_dldata_traintest.nc')
        return datas


    def transpose_load_concat(self, **kwargs):
        
        """Eagerly load the training labels and data, reshaping data to have features in final dim.
        
        Args:
            **kwargs: Dictionary containing opened variable training data, which was opened with ``self.open_files()``.
            
        Returns:
            X_train, label: Eagerly loaded training data and labels as a numpy array.
        
        """
        thedatas={}
        for key, value in kwargs.items():
            thedatas[key]=value.X_train.transpose('a','x','y','features').values
            label=value.X_train_label.values
        if len(kwargs) > 1:
            X_train=np.concatenate(list(thedatas.values()), axis=3)
        if len(kwargs)==1:
            X_train=np.squeeze(np.asarray(list(thedatas.values())))
        return X_train, label

        
    def omit_nans(self, data, label):

        """Remove any ``nans`` from the training data.
        
        Args:
            data (numpy array): Training data.
            label (numpy array): Labels for supervised learning.
            
        Returns:
            data (numpy array): Training data with ``nans`` removed.
            label (numpy array): Corresponding labels of data.
        
        """
        maskarray=np.full(data.shape[0], True)
        masker=np.unique(np.argwhere(np.isnan(data))[:,0])
        maskarray[masker]=False
        traindata=data[maskarray,:,:,:]
        trainlabel=label[maskarray]
        return traindata, trainlabel
    
    
    def compile_meanpool_model(self, data):

        """Assemble and compile the deep conv neural network.
        
        Args:
            data (numpy array): Training data with ``nans`` removed.
        
        Returns:
            model (keras.engine.sequential.Sequential): Compiled deep convolutional neural network and prints model summary.
        
        """
        model=Sequential()

        model.add(Conv2D(self.conv_1_mapnum, 
                   (self.filter_width, self.filter_width),
                   input_shape=data.shape[1:], 
                   strides=self.strides_len,
                   padding='same', data_format='channels_last',
                   dilation_rate=1, activation=self.acti_1_func, use_bias=True, 
                   kernel_initializer='glorot_uniform', bias_initializer='zeros', 
                   kernel_regularizer=l2(0.001), bias_regularizer=None, 
                   activity_regularizer=None, kernel_constraint=None, 
                   bias_constraint=None))

        if self.batch_norm:
            model.add(BatchNormalization(axis=3, momentum=0.99, epsilon=0.001, 
                                         center=True, scale=True, 
                                         beta_initializer='zeros', gamma_initializer='ones',
                                         moving_mean_initializer='zeros', 
                                         moving_variance_initializer='ones',
                                         beta_regularizer=None, gamma_regularizer=None,
                                         beta_constraint=None, gamma_constraint=None))

        if self.spatial_drop:
            model.add(SpatialDropout2D(rate=0.3, data_format='channels_last'))

        if self.pool_method=='mean':
            model.add(AveragePooling2D(pool_size=(2, 2), strides=None, padding='same', 
                                       data_format='channels_last'))
        if self.pool_method=='max':
            model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', 
                                       data_format='channels_last'))
            
        model.add(Conv2D(self.conv_2_mapnum, 
                   (self.filter_width, self.filter_width),
                   input_shape=data.shape[1:], 
                   strides=self.strides_len,
                   padding='same', data_format='channels_last',
                   dilation_rate=1, activation=self.acti_1_func, use_bias=True, 
                   kernel_initializer='glorot_uniform', bias_initializer='zeros', 
                   kernel_regularizer=l2(0.001), bias_regularizer=None, 
                   activity_regularizer=None, kernel_constraint=None, 
                   bias_constraint=None))

        if self.batch_norm:
            model.add(BatchNormalization(axis=3, momentum=0.99, epsilon=0.001, 
                                         center=True, scale=True, 
                                         beta_initializer='zeros', gamma_initializer='ones',
                                         moving_mean_initializer='zeros', 
                                         moving_variance_initializer='ones',
                                         beta_regularizer=None, gamma_regularizer=None,
                                         beta_constraint=None, gamma_constraint=None))

        if self.spatial_drop:
            model.add(SpatialDropout2D(rate=0.3, data_format='channels_last'))

        if self.pool_method=='mean':
            model.add(AveragePooling2D(pool_size=(2, 2), strides=None, padding='same', 
                                       data_format='channels_last'))
        if self.pool_method=='max':
            model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', 
                                       data_format='channels_last'))

        model.add(Conv2D(self.conv_3_mapnum, 
                   (self.filter_width, self.filter_width),
                   input_shape=data.shape[1:], 
                   strides=self.strides_len,
                   padding='same', data_format='channels_last',
                   dilation_rate=1, activation=self.acti_1_func, use_bias=True, 
                   kernel_initializer='glorot_uniform', bias_initializer='zeros', 
                   kernel_regularizer=l2(0.001), bias_regularizer=None, 
                   activity_regularizer=None, kernel_constraint=None, 
                   bias_constraint=None))

        if self.batch_norm:
            model.add(BatchNormalization(axis=3, momentum=0.99, epsilon=0.001, 
                                         center=True, scale=True, 
                                         beta_initializer='zeros', gamma_initializer='ones',
                                         moving_mean_initializer='zeros', 
                                         moving_variance_initializer='ones',
                                         beta_regularizer=None, gamma_regularizer=None,
                                         beta_constraint=None, gamma_constraint=None))

        if self.spatial_drop:
            model.add(SpatialDropout2D(rate=0.3, data_format='channels_last'))

        if self.pool_method=='mean':
            model.add(AveragePooling2D(pool_size=(2, 2), strides=None, padding='same', 
                                       data_format='channels_last'))
        if self.pool_method=='max':
            model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', 
                                       data_format='channels_last'))
            
        model.add(Flatten())
            
        #Dense()  #activation? #relu

        model.add(Dense(units=self.denseshape, activation=self.output_activation, 
                  use_bias=True, 
                  kernel_initializer='glorot_uniform', 
                  bias_initializer='zeros', kernel_regularizer=None, 
                  bias_regularizer=None, activity_regularizer=None, 
                  kernel_constraint=None, bias_constraint=None))


        model.compile(optimizer=Adam(lr=self.learning_rate), loss=self.loss_func, metrics=['accuracy', 'mean_squared_error', 'mean_absolute_error'])
        print(model.summary())
        return model
        
        
    def train_dl(self, model, data, label):
        
        """Train the compiled DL model, save the trained model, and save the history and metric information from training to 
        ``self.dl_filedirectory``.
            
        Args: 
            model (keras.engine.sequential.Sequential): Compiled deep convolutional neural network.
            data (numpy array): Training data with ``nans`` removed.
            label (numpy array): Corresponding labels of data.
            
        """
        history=model.fit(x=data, 
                            y=label, 
                            validation_split=self.validation_split, 
                            batch_size=self.batch_size, 
                            epochs=self.epochs, 
                            shuffle=True)
        pd.DataFrame(history.history).to_csv(f'/{self.working_directory}/model_{self.model_num}_{self.climate}.csv')
        save_model(model, f"/{self.working_directory}/model_{self.model_num}_{self.climate}.h5")
    

    def sequence_funcs(self):
        
        """Training of the deep convolutional neural network in sequential steps.
        
        """
        if self.print_sequential:
            print("Initiating session...")
        self.initiate_session()
        if self.print_sequential:
            print("Opening files...")
        data=self.open_files()
        if self.print_sequential:
            print("Generating training data and labels...")
        train_data, label_data=self.transpose_load_concat(**data)
        if self.print_sequential:
            print("Removing nans...")
        train_data, label_data=self.omit_nans(train_data, label_data)
        if self.print_sequential:
            print("Compiling model...")
        model=self.compile_meanpool_model(train_data)
        if self.print_sequential:
            print("Training model...")
        self.train_dl(model, train_data, label_data)
        data=None
        train_data=None
        label_data=None
        
    