import keras
import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
from scipy.ndimage import gaussian_filter
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from metpy.plots import colortables
import matplotlib.colors as colors


class InterpretDLModel:
    
    """Class instantiation of InterpretDLModel:
    
    Here we load the variable data for interpretation of the trained deep convolutional neural network.
    
    Attributes:
        climate (str): Whether analyzing ``current`` or ``future`` climate simulation.
        method (str): Method for parsing data. Options include ``random``, ``month``, ``season``, ``year``.
        variable (str): Variable name for saliency map output. Options include: 
                        ``EU1``, ``EU3``, ``EU5``, ``EU7``, 
                        ``EV1``, ``EV3``, ``EV5``, ``EV7``, 
                        ``TK1``, ``TK3``, ``TK5``, ``TK7``, 
                        ``QVAPOR1``, ``QVAPOR3``, ``QVAPOR5``, ``QVAPOR7``,
                        ``W1``, ``W3``, ``W5``, ``W7``,
                        ``P1``, ``P3``, ``P5``, ``P7``,
                        ``WMAX``, ``DBZ``,``CTT``,``UH25``, and``UH03``.
        working_directory (str): The directory path to where the produced files will be saved and worked from.
        var_directory (str): Directory where the test subset variable data is saved.
        model_directory (str): Directory where the deep learning model is saved.
        model_num (str): The number of the model as it was saved.
        eval_directory (str): Directory where evaluation files will be saved.
        
        mask (boolean): Whether to train using the masked data or the non-masked data. Defaults to ``False``.
        random_choice (int): The integer the respective ``random`` method file was saved as. Defaults to ``None``.
        month_choice (int): Month for analysis. Defaults to ``None``.
        season_choice (str): Three-month season string, if ``method==season`` (e.g., 'DJF'). Defaults to ``None``.
        year_choice (int): Year for analysis. Defaults to ``None``.
        obs_threshold (float): Decimal value that denotes whether model output is a ``1`` or ``0``. Defaults to ``0.5``.
        
    Raises:
        Exceptions: Checks whether correct values were input for ``climate`` and ``method``.
        
    Todo:
        * Add loading and handling of test subsets that were created using the ``month``, ``season``, and ``year`` methods.
        * Add evaluation of UH25 and UH03 for failure cases.
        
    """

    def __init__(self, climate, method, variable, dist_directory, model_directory, model_num, comp_directory, mask=False, 
                 random_choice=None, month_choice=None, season_choice=None, year_choice=None):
        
        if climate!='current' and climate!='future':
            raise Exception("Please enter ``current`` or ``future`` as string for climate period selection.")
        else:
            self.climate=climate
        
        if method!='random' and method!='month' and method!='season' and method!='year':
            raise Exception("Please enter ``random``, ``month``, ``season``, or ``year`` as method.")
        else:
            self.method=method
            
        self.variable=variable
        self.dist_directory=dist_directory
        self.model_directory=model_directory
        self.model_num=model_num
        self.comp_directory=comp_directory
        
        self.mask=mask
        if not self.mask:
            self.mask_str='nomask'
        if self.mask:
            self.mask_str='mask'
    
        self.random_choice=random_choice
        self.month_choice=month_choice 
        self.season_choice=season_choice 
        self.year_choice=year_choice
    
    
    def variable_translate(self):
        
        """Variable name for the respective filenames.
           
        Args:
            variable (str): The variable to feed into the dictionary.
            
        Returns:
            variable (str): The variable name to use for opening saved files.
            
        Raises:
            ValueError: If provided variable is not available.
            
        """
        var={  'EU1':'EU', 'EU3':'EU', 'EU5':'EU', 'EU7':'EU',
               'EV1':'EV', 'EV3':'EV', 'EV5':'EV', 'EV7':'EV',
               'TK1':'TK', 'TK3':'TK', 'TK5':'TK', 'TK7':'TK',
               'QVAPOR1':'QVAPOR', 'QVAPOR3':'QVAPOR', 'QVAPOR5':'QVAPOR', 'QVAPOR7':'QVAPOR', 
               'WMAX':'MAXW',
               'W1':'W', 'W3':'W', 'W5':'W', 'W7':'W',
               'P1':'P', 'P3':'P', 'P5':'P', 'P7':'P',
               'DBZ':'DBZ',
               'CTT':'CTT',
               'UH25':'UH25',
               'UH03':'UH03',
              }
        try:
            out=var[self.variable]
            return out
        except:
            raise ValueError("Please enter ``TK``, ``EV``, ``EU``, ``QVAPOR``, ``PRESS``, ``W_vert``, ``UH25``, ``UH03``, ``MAXW``, ``CTT``, or ``DBZ`` as variable with height AGL appended (1, 3, 5, or 7).")
            
            
    def convert_string_height(self):
        
        """Convert the string variable name's height to integer for indexing mean and standard deviation data.
        
        """
        the_hgt=int(self.variable[-1])
        heights=np.array([1,3,5,7])
        the_indx=np.where(heights==the_hgt)
        return the_indx     
    
    
    def extract_variable_mean_and_std(self):
        
        """Open the file containing mean and std information for the variable.
        
        """
        data=xr.open_dataset(f"/{self.dist_directory}/{self.climate}_{self.variable_translate().lower()}_{self.mask_str}_dldata_traindist.nc")
        self.variable_mean=data.train_mean.values[self.convert_string_height()[0][0]]
        self.variable_std=data.train_std.values[self.convert_string_height()[0][0]]
        
        
    def extract_eu_mean_and_std(self):
        
        """Open the file containing mean and std information for the variable.
        
        """
        data=xr.open_dataset(f"/{self.dist_directory}/{self.climate}_eu_{self.mask_str}_dldata_traindist.nc")
        self.eu_mean=data.train_mean.values[self.convert_string_height()[0][0]]
        self.eu_std=data.train_std.values[self.convert_string_height()[0][0]]
        
        
    def extract_ev_mean_and_std(self):
        
        """Open the file containing mean and std information for the variable.
        
        """
        data=xr.open_dataset(f"/{self.dist_directory}/{self.climate}_ev_{self.mask_str}_dldata_traindist.nc")
        self.ev_mean=data.train_mean.values[self.convert_string_height()[0][0]]
        self.ev_std=data.train_std.values[self.convert_string_height()[0][0]]
    
    
    def extract_dbz_mean_and_std(self):
        
        """Open the file containing mean and std information for the variable.
        
        """
        data=xr.open_dataset(f"/{self.dist_directory}/{self.climate}_dbz_{self.mask_str}_dldata_traindist.nc")
        self.dbz_mean=data['train_mean'].values[0]
        self.dbz_std=data['train_std'].values[0]
    
    
    def extract_model(self):
        
        """Load the DL model from h5 data set.
        
        """
        loaded_model=load_model(f'{self.model_directory}/model_{self.model_num}_current.h5')
        print(loaded_model.summary())
        return loaded_model
    
    
    def extract_variable_and_dbz(self):
        
        """Open the file containing the test data.
        
        """
        return xr.open_dataset(f'{self.comp_directory}/composite_results_{self.mask_str}_model{self.model_num}_{self.method}{self.random_choice}.nc')
    
    
    def extract_variable_index(self, data):
    
        """Find the variable index from the respective test data set.
        
        Args:
            data: Dataset opened with ``extract_variable_and_dbz()``.
        
        """
        return np.where(data.coords['features'].values==self.variable)[0][0]
        
        
    def extract_dbz_index(self, data):
    
        """Find the ``DBZ`` index from the respective test data set.
        
        Args:
            data: Dataset opened with ``extract_variable_and_dbz()``.
        
        """
        return np.where(data.coords['features'].values=='DBZ')[0][0]
    
    
    def extract_EV_index(self, data):
    
        """Find the ``EV`` (v-wind) index from the respective test data set for the corresponding variable height.
        
        Args:
            data: Dataset opened with ``extract_variable_and_dbz()``.
        
        """
        return np.where(data.coords['features'].values=='EV'+self.variable[-1])[0][0]
    
    
    def extract_EU_index(self, data):
    
        """Find the ``EU`` (u-wind) index from the respective test data set for the corresponding variable height.
        
        Args:
            data: Dataset opened with ``extract_variable_and_dbz()``.
        
        """
        return np.where(data.coords['features'].values=='EU'+self.variable[-1])[0][0]
    
    
    def preview_dbz(self, composite_group, input_index, test_data):
        
        """Preview the testing data ``DBZ`` values to help choose the example for ``saliency_preview``.
        
        Args:
            composite_group (str): The subset of the test data based on prediction outcome. Choices include true positive ``tp``, 
                                   true positive > 99% probability ``tp_99``, false positive ``fp``, false positive > 99% probability 
                                   ``fp_99``, false negative ``fn``, false negative < 1% probability ``fn_01``, true negative ``tn``, 
                                   true negative < 1% probability ``tn_01``.
            input_index (int): The example's index to preview.
            dl_model (Keras saved model): The DL model to preview. Layers and activations will be extracted from loaded model.
            test_data (numpy array): The test data to use for saliency map generation.
            train_mean (float): The mean of the chosen variable to reverse standardization.
            train_std (float): The standard deviation of the chosen variable to reverse standardization.
        
        """
        levels=[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]
        cmap = colortables.get_colortable('NWSReflectivity')
        return xr.plot.contourf(test_data[composite_group][input_index, :, :, test.extract_dbz_index(testdata)] * test.dbz_std + test.dbz_mean, 
                                cmap=cmap, levels=levels)
    
    
    def preview_saliency(self, composite_group, input_index, dl_model, test_data):
        
        """Preview the deep learning model input using saliency maps.
        
        Args:
            composite_group (str): The subset of the test data based on prediction outcome. Choices include true positive ``tp``, 
                                   true positive > 99% probability ``tp_99``, false positive ``fp``, false positive > 99% probability 
                                   ``fp_99``, false negative ``fn``, false negative < 1% probability ``fn_01``, true negative ``tn``, 
                                   true negative < 1% probability ``tn_01``.
            input_index (int): The example's index to to preview.
            dl_model (Keras saved model): The DL model to preview. Layers and activations will be extracted from loaded model.
            test_data (numpy array): The test data to use for saliency map generation.
            train_mean (float): The mean of the chosen variable to reverse standardization.
            train_std (float): The standard deviation of the chosen variable to reverse standardization.
            vmin (int): Minimum value for ``pcolormesh`` plot.
            vmax (int): Maximum value for ``pcolormesh`` plot.
            cmap (str): Matplotlib colorbar name for visualization.
        
        """
        testdata=test_data[composite_group]
        
        fig, axes=plt.subplots(4, 8, figsize=(16, 8), sharex=True, sharey=True)
        plt.subplots_adjust(0.02, 0.02, 0.96, 0.94, wspace=0,hspace=0)
        
        for conv_filter, ax in enumerate(axes.ravel()):
            print(conv_filter)
            
            #out_diff=K.abs(dl_model.layers[-5].output[0, 2, 2, conv_filter] - 1)  
            out_diff=K.abs(dl_model.layers[-3].output[0, conv_filter] - 1)     #dense layer that was added
            
            grad=K.gradients(out_diff, [dl_model.input])[0]
            grad/=K.maximum(K.std(grad), K.epsilon())
            iterate=K.function([dl_model.input, K.learning_phase()], [out_diff, grad])
            input_img_data_neuron_grad=np.zeros((1, 32, 32, 20))
            
            input_img_data_neuron=np.copy(testdata[input_index:input_index+1,:,:,:-2])    #change this to -3 once UH03 is resolved!!!!
            out_loss, out_grad=iterate([input_img_data_neuron, 1])

            #DBZ
            levels=[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]
            cmap_dbz = colortables.get_colortable('NWSReflectivity')
            ax.contourf(test_data[composite_group][input_index, :, :, self.extract_dbz_index(testdata)] * self.dbz_std + self.dbz_mean, 
                        cmap=cmap_dbz, levels=levels, alpha=0.2)

            #chosen input variable
            ax.contour(gaussian_filter(-out_grad[0, :, :, self.extract_variable_index(testdata)], 1), 
                       [-3, -2, -1, 1, 2, 3], vmin=-3, vmax=3, cmap="seismic", linewidths=3.0)

            #doing v and u?
            #EV_var=
            #EU_var=
            #ax.quiver(input_img_data_neuron[0, :, :, -2] * train_std + train_mean, input_img_data_neuron[0, :, :, -1] * train_std + train_mean, scale=100)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.text(16, 16, conv_filter, fontsize=14)
        plt.suptitle("Final Convolution Filter Saliency Maps", fontsize=14, y=0.98)
        plt.show()
    
    
    def save_saliency_maps(self, composite_group, indices):
        
        """Save the features using chosen indices to generate final images using the next module.
        
        Args:
            composite_group (str): The subset of the test data based on prediction outcome. Choices include true positive ``tp``, 
                                   true positive > 99% probability ``tp_99``, false positive ``fp``, false positive > 99% probability 
                                   ``fp_99``, false negative ``fn``, false negative < 1% probability ``fn_01``, true negative ``tn``, 
                                   true negative < 1% probability ``tn_01``.
            indices (int): Numpy array of chosen indices.
        
        """
        #.to_netcdf(f'{}{}{}_{composite_group}.nc')
        return
        
        