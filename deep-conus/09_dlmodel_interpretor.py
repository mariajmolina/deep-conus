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


class InterpretDLModel:
    
    """Class instantiation of InterpretDLModel:
    
    Here we load the variable data for interpretation of the trained deep convolutional neural network.
    
    Attributes:
        climate (str): Whether analyzing ``current`` or ``future`` climate simulation.
        method (str): Method for parsing data. Options include ``random``, ``month``, ``season``, ``year``.
        variable (str): Variable name for saliency map output. Options include ``EU``, ``EV``, ``TK``, ``QVAPOR``, ``WMAX``, 
                        ``W_vert``,``PRESS``,``DBZ``,``CTT``,``UH25``, and``UH03``.
                         
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

    def __init__(self, climate, method, variables, var_directory, model_directory, model_num, eval_directory, mask=False, 
                 random_choice=None, month_choice=None, season_choice=None, year_choice=None, obs_threshold=0.5):
        
        if climate!='current' and climate!='future':
            raise Exception("Please enter ``current`` or ``future`` as string for climate period selection.")
        else:
            self.climate=climate
        
        if method!='random' and method!='month' and method!='season' and method!='year':
            raise Exception("Please enter ``random``, ``month``, ``season``, or ``year`` as method.")
        else:
            self.method=method
            
        self.variable=variable
        self.var_directory=var_directory
        self.model_directory=model_directory
        self.model_num=model_num
        self.eval_directory=eval_directory
        
        self.mask=mask
        if not self.mask:
            self.mask_str='nomask'
        if self.mask:
            self.mask_str='mask'
    
        self.random_choice=random_choice 
        
        self.month_choice=month_choice 
        self.season_choice=season_choice 
        self.year_choice=year_choice
        self.obs_threshold=obs_threshold
    

    
    def extract_variable_and_dbz(self):
        
        """Open the file containing the test data.
        
        data=xr.Dataset({
            'tp':(['a','x','y','features'], self.test_data[self.tp_indx,:,:,:].squeeze()),
            'tp_99':(['b','x','y','features'], self.test_data[self.tp_99_indx,:,:,:].squeeze()),
            'fp':(['c','x','y','features'], self.test_data[self.fp_indx,:,:,:].squeeze()),
            'fp_99':(['d','x','y','features'], self.test_data[self.fp_99_indx,:,:,:].squeeze()),
            'fn':(['e','x','y','features'], self.test_data[self.fn_indx,:,:,:].squeeze()),
            'fn_01':(['f','x','y','features'], self.test_data[self.fn_01_indx,:,:,:].squeeze()),
            'tn':(['g','x','y','features'], self.test_data[self.tn_indx,:,:,:].squeeze()),
            'tn_01':(['h','x','y','features'], self.test_data[self.tn_01_indx,:,:,:].squeeze()),
            },
            coords=
            {'features':(['features'], self.variables),
            })
        
        """
        return xr.open_dataset(f'{self.eval_directory}/composite_results_{self.mask_str}_model{self.model_num}_{self.method}{self.random_choice}.nc')
    
    
    def extract_mean_and_std(self):
        
        """Open the file containing mean and std information.
        
            dist_assemble=xr.Dataset({
                'train_mean':(['features'], train_mean),
                'train_std':(['features'], train_std),
                },
                coords=
                {'feature':(['features'],self.attrs_array),
                })
                
            dist_assemble.to_netcdf(f"/{self.working_directory}/{self.climate}_{self.variable_translate().lower()}_{self.mask_str}_dldata_traindist.nc")
        
        """
        return
    
    
    def extract_model(self):
        
        """Load the DL model from h5 data set.
        
        """
        return load_model(f'{self.model_directory}/model_{self.model_num}_{self.climate}.h5')
    
    
    def extract_variable_index(self)
    
        """Find the variable index from the respective test data set.
        
        """
        return
        
    def extract_dbz_index(self)
    
        """Find the variable index from the respective test data set.
        
        """
        return
    
    
    def preview_dbz(self, input_index, dl_model, test_data, train_mean, train_std):
        
        """Preview the testing data dbz values to help choose the example for ``saliency_preview``.
        
        """
        return
    
    
    def preview_saliency(self, input_index, dl_model, test_data, train_mean, train_std, vmin, vmax, cmap):
        
        """Preview the deep learning model input using saliency maps.
        
        Args:
            input_index (int): The example's index to to preview.
            dl_model (Keras saved model): The DL model to preview. Layers and activations will be extracted from loaded model.
            test_data (numpy array): The test data to use for saliency map generation.
            train_mean (float): The mean of the chosen variable to reverse standardization.
            train_std (float): The standard deviation of the chosen variable to reverse standardization.
            vmin (int): Minimum value for ``pcolormesh`` plot.
            vmax (int): Maximum value for ``pcolormesh`` plot.
            cmap (str): Matplotlib colorbar name for visualization.
        
        """
        fig, axes=plt.subplots(4, 8, figsize=(16, 8), sharex=True, sharey=True)
        plt.subplots_adjust(0.02, 0.02, 0.96, 0.94, wspace=0,hspace=0)
        for conv_filter, ax in enumerate(axes.ravel()):
            print(conv_filter)
            out_diff=K.abs(dl_model.layers[-3].output[0, 2, 2, conv_filter] - 1)
            grad=K.gradients(out_diff, [dl_model.input])[0]
            grad/=K.maximum(K.std(grad), K.epsilon())
            iterate=K.function([dl_model.input, K.learning_phase()], [out_diff, grad])
            input_img_data_neuron_grad=np.zeros((1, 32, 32, 20))
            input_img_data_neuron=np.copy(test_data[input_index:input_index+1])    
            out_loss, out_grad=iterate([input_img_data_neuron, 1])
    
            #ax.pcolormesh(input_img_data_neuron[0, :, :, 0], cmap="PuOr")

            #DBZ
            ax.contour(gaussian_filter(-out_grad[0, :, :, 0], 1), [-3, -2, -1, 1, 2, 3], vmin=-3, vmax=3, cmap="RdBu_r")

            #chosen input variable
            ax.pcolormesh(input_img_data_neuron[0, :, :, 0] * train_std + train_mean, vmin=vmin, vmax=vmax, cmap=cmap)

            #doing v and u?
            #ax.quiver(input_img_data_neuron[0, :, :, -2] * train_std + train_mean, input_img_data_neuron[0, :, :, -1] * train_std + train_mean, scale=100)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.text(16, 16, conv_filter, fontsize=14)
        plt.suptitle("Final Convolution Filter Saliency Maps", fontsize=14, y=0.98)
        plt.show()
    
    
    
    
    def save_saliency_maps(self, index):
        
        """Save the features using chosen indices to generate final images using the next module.
        
        """
        return
        
        