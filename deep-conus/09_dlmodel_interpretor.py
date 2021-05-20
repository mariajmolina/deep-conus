import keras
import tensorflow as tf
#from keras import backend as K
from tensorflow.keras import backend as K
#from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
from scipy.ndimage import gaussian_filter
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from metpy.plots import colortables
import matplotlib.colors as colors
tf.compat.v1.disable_eager_execution()

class InterpretDLModel:
    
    """Class instantiation of InterpretDLModel:
    
    Here we load the variable data for interpretation of the trained deep convolutional neural network.
    
    Attributes:
        climate (str): Whether analyzing ``current`` or ``future`` climate simulation.
        variable (str): Variable name for saliency map output. Options include: 
                        ``EU1``, ``EU3``, ``EU5``, ``EU7``, 
                        ``EV1``, ``EV3``, ``EV5``, ``EV7``, 
                        ``TK1``, ``TK3``, ``TK5``, ``TK7``, 
                        ``QVAPOR1``, ``QVAPOR3``, ``QVAPOR5``, ``QVAPOR7``,
                        ``W1``, ``W3``, ``W5``, ``W7``,
                        ``P1``, ``P3``, ``P5``, ``P7``,
                        ``WMAX``, ``DBZ``,``CTT``,``UH25``, and``UH03``.
        dist_directory (str): The directory path where the produced files were saved.
        model_directory (str): Directory where the deep learning model is saved.
        model_num (str): The number of the model as it was saved.
        comp_directory (str): Directory where the composite files were saved.
        mask (boolean): Whether to train using the masked data or the non-masked data. Defaults to ``False``.
        mask_train (boolean): Whether to train using masked state variable data. Defaults to ``False``. Will override ``mask`` to ``True``.
        unbalanced (boolean): Whether training data will be artificially balanced (``False``) or left unbalanced (``True``). Defaults to ``False``. 
        validation (boolean): Whether to extract a validation set from the original unbalanced dataset. Defaults to ``False``. 
        isotonic (boolean): Whether model has an isotonic regression applied to output. Defaults to ``False``.
        random_choice (int): The integer the respective ``random`` method file was saved as. Defaults to ``None``.
        outliers (boolean): Whether evaluating outlier storms. Defaults to ``True``.

    Raises:
        Exceptions: Checks whether correct values were input for ``climate`` and ``method``.
        
    """
    def __init__(self, climate, variable, dist_directory, model_directory, model_num, comp_directory, 
                 mask=False, mask_train=False, unbalanced=False, validation=False, isotonic=False,
                 random_choice=None, outliers=False):
        
        if climate!='current' and climate!='future':
            raise Exception("Please enter ``current`` or ``future`` as string for climate period selection.")
        else:
            self.climate=climate
        self.method='random'
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
        self.mask_train=mask_train
        self.unbalanced=unbalanced
        self.validation=validation
        self.isotonic=isotonic
        self.random_choice=random_choice
        self.outliers = outliers
    
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
        
        """Open the file containing mean and std information for the selected variable.
        
        """
        if not self.unbalanced:
            if not self.validation:
                data=xr.open_dataset(
                 f"/{self.dist_directory}/{self.climate}_{self.variable_translate().lower()}_{self.mask_str}_dldata_traindist.nc")
            if self.validation:
                data=xr.open_dataset(
            f"/{self.dist_directory}/{self.climate}_{self.variable_translate().lower()}_{self.mask_str}_dldata_traindist_valid.nc")
        if self.unbalanced:
            if not self.validation:
                data=xr.open_dataset(
f"/{self.dist_directory}/{self.climate}_{self.variable_translate().lower()}_{self.mask_str}_dldata_traindist_unbalanced.nc")       
            if self.validation:
                data=xr.open_dataset(
f"/{self.dist_directory}/{self.climate}_{self.variable_translate().lower()}_{self.mask_str}_dldata_traindist_unbalanced_valid.nc")
        self.variable_mean=data.train_mean.values[self.convert_string_height()[0][0]]
        self.variable_std=data.train_std.values[self.convert_string_height()[0][0]]

    def extract_eu_mean_and_std(self):
        
        """Open the file containing mean and std information for u winds (earth relative).
        
        """
        if not self.unbalanced:
            if not self.validation:
                data=xr.open_dataset(f"/{self.dist_directory}/{self.climate}_eu_{self.mask_str}_dldata_traindist.nc")
            if self.validation:
                data=xr.open_dataset(f"/{self.dist_directory}/{self.climate}_eu_{self.mask_str}_dldata_traindist_valid.nc")
        if self.unbalanced:
            if not self.validation:
                data=xr.open_dataset(f"/{self.dist_directory}/{self.climate}_eu_{self.mask_str}_dldata_traindist_unbalanced.nc")
            if self.validation:
                data=xr.open_dataset(
                    f"/{self.dist_directory}/{self.climate}_eu_{self.mask_str}_dldata_traindist_unbalanced_valid.nc")
        self.eu_mean=data.train_mean.values[self.convert_string_height()[0][0]]
        self.eu_std=data.train_std.values[self.convert_string_height()[0][0]]

    def extract_ev_mean_and_std(self):
        
        """Open the file containing mean and std information for v winds (earth relative).
        
        """
        if not self.unbalanced:
            if not self.validation:
                data=xr.open_dataset(f"/{self.dist_directory}/{self.climate}_ev_{self.mask_str}_dldata_traindist.nc")
            if self.validation:
                data=xr.open_dataset(f"/{self.dist_directory}/{self.climate}_ev_{self.mask_str}_dldata_traindist_valid.nc")
        if self.unbalanced:
            if not self.validation:
                data=xr.open_dataset(f"/{self.dist_directory}/{self.climate}_ev_{self.mask_str}_dldata_traindist_unbalanced.nc")
            if self.validation:
                data=xr.open_dataset(
                    f"/{self.dist_directory}/{self.climate}_ev_{self.mask_str}_dldata_traindist_unbalanced_valid.nc")
        self.ev_mean=data.train_mean.values[self.convert_string_height()[0][0]]
        self.ev_std=data.train_std.values[self.convert_string_height()[0][0]]

    def extract_uh03_mean_and_std(self):
        
        """Open the file containing mean and std information for UH (0-3 km AGL).
        
        """
        if not self.unbalanced:
            if not self.validation:
                data=xr.open_dataset(f"/{self.dist_directory}/{self.climate}_uh03_{self.mask_str}_dldata_traindist.nc")
            if self.validation:
                data=xr.open_dataset(f"/{self.dist_directory}/{self.climate}_uh03_{self.mask_str}_dldata_traindist_valid.nc")
        if self.unbalanced:
            if not self.validation:
                data=xr.open_dataset(f"/{self.dist_directory}/{self.climate}_uh03_{self.mask_str}_dldata_traindist_unbalanced.nc")
            if self.validation:
                data=xr.open_dataset(
                    f"/{self.dist_directory}/{self.climate}_uh03_{self.mask_str}_dldata_traindist_unbalanced_valid.nc")
        self.uh03_mean=data['train_mean'].values[0]
        self.uh03_std=data['train_std'].values[0]

    def extract_uh25_mean_and_std(self):
        
        """Open the file containing mean and std information for UH (2-5 km AGL).
        
        """
        if not self.unbalanced:
            if not self.validation:
                data=xr.open_dataset(f"/{self.dist_directory}/{self.climate}_uh25_{self.mask_str}_dldata_traindist.nc")
            if self.validation:
                data=xr.open_dataset(f"/{self.dist_directory}/{self.climate}_uh25_{self.mask_str}_dldata_traindist_valid.nc")
        if self.unbalanced:
            if not self.validation:
                data=xr.open_dataset(f"/{self.dist_directory}/{self.climate}_uh25_{self.mask_str}_dldata_traindist_unbalanced.nc")
            if self.validation:
                data=xr.open_dataset(
                    f"/{self.dist_directory}/{self.climate}_uh25_{self.mask_str}_dldata_traindist_unbalanced_valid.nc")
        self.uh25_mean=data['train_mean'].values[0]
        self.uh25_std=data['train_std'].values[0]

    def extract_dbz_mean_and_std(self):
        
        """Open the file containing mean and std information for dBZ (simulated reflectivity).
        
        """
        if not self.unbalanced:
            if not self.validation:
                data=xr.open_dataset(f"/{self.dist_directory}/{self.climate}_dbz_{self.mask_str}_dldata_traindist.nc")
            if self.validation:
                data=xr.open_dataset(f"/{self.dist_directory}/{self.climate}_dbz_{self.mask_str}_dldata_traindist_valid.nc")
        if self.unbalanced:
            if not self.validation:
                data=xr.open_dataset(f"/{self.dist_directory}/{self.climate}_dbz_{self.mask_str}_dldata_traindist_unbalanced.nc")
            if self.validation:
                data=xr.open_dataset(
                    f"/{self.dist_directory}/{self.climate}_dbz_{self.mask_str}_dldata_traindist_unbalanced_valid.nc")
        self.dbz_mean=data['train_mean'].values[0]
        self.dbz_std=data['train_std'].values[0]

    def extract_model(self):
        
        """Load the keras model from h5 data set.
        
        """
        loaded_model=load_model(f'{self.model_directory}/model_{self.model_num}_current.h5')
        print(loaded_model.summary())
        return loaded_model

    def extract_variable_and_dbz(self):
        
        """Open the file containing the test data.
        
        """
        if not self.outliers:
            ds = xr.open_dataset(f'{self.comp_directory}/composite_results_{self.mask_str}_model{self.model_num}_{self.method}{self.random_choice}.nc')
        if self.outliers:
            ds = xr.open_dataset(f'{self.comp_directory}/composite_outresults_{self.mask_str}_model{self.model_num}_{self.method}{self.random_choice}.nc')
        return ds

    def extract_variable_index(self, data):
    
        """Find the variable index from the respective test data set.
        
        Args:
            data (xarray dataset): Dataset opened with ``extract_variable_and_dbz()``.
        
        """
        return np.where(data.coords['features'].values==self.variable)[0][0]

    def extract_dbz_index(self, data):
    
        """Find the ``DBZ`` index from the respective test data set.
        
        Args:
            data: Dataset opened with ``extract_variable_and_dbz()``.
        
        """
        return np.where(data.coords['features'].values=='DBZ')[0][0]

    def extract_uh03_index(self, data):
    
        """Find the ``UH03`` index from the respective test data set.
        
        Args:
            data: Dataset opened with ``extract_variable_and_dbz()``.
        
        """
        return np.where(data.coords['features'].values=='UH03')[0][0]

    def extract_uh25_index(self, data):
    
        """Find the ``UH25`` index from the respective test data set.
        
        Args:
            data: Dataset opened with ``extract_variable_and_dbz()``.
        
        """
        return np.where(data.coords['features'].values=='UH25')[0][0]

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
            test_data (numpy array): The test data to use for saliency map generation.
        
        """
        levels=[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]
        cmap = colortables.get_colortable('NWSReflectivity')
        return xr.plot.contourf(test_data[composite_group][input_index, :, :, test.extract_dbz_index(test_data)] * test.dbz_std + test.dbz_mean, 
                                cmap=cmap, levels=levels)

    def preview_uh25(self, composite_group, input_index, test_data):
        
        """Preview the testing data ``UH 2-5 km`` values to help choose the example for ``saliency_preview``.
        
        Args:
            composite_group (str): The subset of the test data based on prediction outcome. Choices include true positive ``tp``, 
                                   true positive > 99% probability ``tp_99``, false positive ``fp``, false positive > 99% probability 
                                   ``fp_99``, false negative ``fn``, false negative < 1% probability ``fn_01``, true negative ``tn``, 
                                   true negative < 1% probability ``tn_01``.
            input_index (int): The example's index to preview.
            test_data (numpy array): The test data to use for saliency map generation.
        
        """
        cmap = plt.cm.get_cmap("Reds")
        print(np.nanmax(test_data[composite_group][input_index, :, :, test.extract_uh25_index(test_data)] * test.uh25_std + test.uh25_mean))
        return xr.plot.pcolormesh(
            test_data[composite_group][input_index, :, :, test.extract_uh25_index(test_data)] * test.uh25_std + test.uh25_mean, 
                                       cmap=cmap, vmin=-75, vmax=75)

    def grab_dbz(self, composite_group, input_index, test_data):
        
        """Grab the testing data ``DBZ`` values to help choose the example for ``saliency_preview``.
        
        Args:
            composite_group (str): The subset of the test data based on prediction outcome. Choices include true positive ``tp``, 
                                   true positive > 99% probability ``tp_99``, false positive ``fp``, false positive > 99% probability 
                                   ``fp_99``, false negative ``fn``, false negative < 1% probability ``fn_01``, true negative ``tn``, 
                                   true negative < 1% probability ``tn_01``.
            input_index (int): The example's index to preview.
            test_data (numpy array): The test data to use for saliency map generation.
        
        """
        return test_data[composite_group][input_index, :, :, test.extract_dbz_index(test_data)] * test.dbz_std + test.dbz_mean

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
        
        """
        testdata=test_data[composite_group]
        
        fig, axes=plt.subplots(4, 8, figsize=(16, 8), sharex=True, sharey=True)
        plt.subplots_adjust(0.02, 0.02, 0.96, 0.94, wspace=0,hspace=0)
        
        for conv_filter, ax in enumerate(axes.ravel()):
            print(conv_filter)
            out_diff=K.abs(dl_model.layers[-4].output[0, conv_filter] - 1)     #dense layer that was added
            grad=K.gradients(out_diff, [dl_model.input])[0]
            grad/=K.maximum(K.std(grad), K.epsilon())
            iterate=K.function([dl_model.input, K.learning_phase()], [out_diff, grad])
            input_img_data_neuron_grad=np.zeros((1, 32, 32, 20))
            input_img_data_neuron=np.copy(testdata[input_index:input_index+1,:,:,:-6])
            out_loss, out_grad=iterate([input_img_data_neuron, 1])

            #DBZ
            levels=[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]
            cmap_dbz = colortables.get_colortable('NWSReflectivity')
            ax.contourf(test_data[composite_group][input_index, :, :, self.extract_dbz_index(testdata)] * self.dbz_std + self.dbz_mean, 
                        cmap=cmap_dbz, levels=levels, alpha=0.2)

            ax.contour(gaussian_filter(-out_grad[0, :, :, self.extract_variable_index(testdata)], 1), 
                       [-3, -2, -1, 1, 2, 3], vmin=-3, vmax=3, cmap="seismic", linewidths=3.0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.text(16, 16, conv_filter, fontsize=14)
        plt.suptitle("Final Convolution Filter Saliency Maps", fontsize=14, y=0.98)
        plt.show()
        
    def preview_inputgrad(self, composite_group, input_index, dl_model, test_data):
        
        """Preview the deep learning model input using input x gradient maps.
        
        Args:
            composite_group (str): The subset of the test data based on prediction outcome. Choices include true positive ``tp``, 
                                   true positive > 99% probability ``tp_99``, false positive ``fp``, false positive > 99% probability 
                                   ``fp_99``, false negative ``fn``, false negative < 1% probability ``fn_01``, true negative ``tn``, 
                                   true negative < 1% probability ``tn_01``.
            input_index (int): The example's index to to preview.
            dl_model (Keras saved model): The DL model to preview. Layers and activations will be extracted from loaded model.
            test_data (numpy array): The test data to use for saliency map generation.
        
        """
        testdata=test_data[composite_group]
        
        fig, axes=plt.subplots(4, 8, figsize=(16, 8), sharex=True, sharey=True)
        plt.subplots_adjust(0.02, 0.02, 0.96, 0.94, wspace=0,hspace=0)
        
        for conv_filter, ax in enumerate(axes.ravel()):
            print(conv_filter)
            out_diff=K.abs(dl_model.layers[-4].output[0, conv_filter] - 1)     #dense layer that was added
            grad=K.gradients(out_diff, [dl_model.input])[0]
            grad/=K.maximum(K.std(grad), K.epsilon())
            iterate=K.function([dl_model.input, K.learning_phase()], [out_diff, grad])
            input_img_data_neuron_grad=np.zeros((1, 32, 32, 20))
            input_img_data_neuron=np.copy(testdata[input_index:input_index+1,:,:,:-6])
            out_loss, out_grad=iterate([input_img_data_neuron, 1])

            #DBZ
            levels=[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]
            cmap_dbz = colortables.get_colortable('NWSReflectivity')
            ax.contourf(test_data[composite_group][input_index, :, :, self.extract_dbz_index(testdata)] * self.dbz_std + self.dbz_mean, 
                        cmap=cmap_dbz, levels=levels, alpha=0.2)

            ax.contour(gaussian_filter(
                test_data[composite_group][input_index,:,:,self.extract_variable_index(testdata)]*(-out_grad[0,:,:,self.extract_variable_index(testdata)]),1), 
                       [-3, -2, -1, 1, 2, 3], vmin=-3, vmax=3, cmap="seismic", linewidths=3.0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.text(16, 16, conv_filter, fontsize=14)
        plt.suptitle("Final Convolution Filter Input x Gradient Maps", fontsize=14, y=0.98)
        plt.show()

    def save_saliency_maps(self, composite_group, input_index, dl_model, test_data):
        
        """Save the features using chosen indices to generate final images using the next module.
        
        Args:
            composite_group (str): The subset of the test data based on prediction outcome. Choices include true positive ``tp``, 
                                   true positive > 99% probability ``tp_99``, false positive ``fp``, false positive > 99% probability 
                                   ``fp_99``, false negative ``fn``, false negative < 1% probability ``fn_01``, true negative ``tn``, 
                                   true negative < 1% probability ``tn_01``.
            input_index (int): Index of sample to generate saliency maps for.
            dl_model (Keras saved model): The DL model to preview. Layers and activations will be extracted from loaded model.
            test_data (numpy array): The test data to use for saliency map generation.
        
        """
        testdata=test_data[composite_group]
        for_contours={}
        for conv_filter in range(0,32):
            out_diff=K.abs(dl_model.layers[-4].output[0, conv_filter] - 1)     #dense layer that was added
            grad=K.gradients(out_diff, [dl_model.input])[0]
            grad/=K.maximum(K.std(grad), K.epsilon())
            iterate=K.function([dl_model.input, K.learning_phase()], [out_diff, grad])
            input_img_data_neuron_grad=np.zeros((1, 32, 32, 20))
            input_img_data_neuron=np.copy(testdata[input_index:input_index+1,:,:,:-6])
            out_loss, out_grad=iterate([input_img_data_neuron, 1])        
            for_contours[conv_filter]=gaussian_filter(-out_grad[0, :, :, self.extract_variable_index(testdata)], 1)
        array=[p[1] for p in for_contours.items()]
        thecontours=np.asarray(array)
        thedbz=test_data[composite_group][input_index, :, :, self.extract_dbz_index(testdata)] * self.dbz_std + self.dbz_mean
        theeu=test_data[composite_group][input_index, :, :, self.extract_EU_index(testdata)] * self.eu_std + self.eu_mean
        theev=test_data[composite_group][input_index, :, :, self.extract_EV_index(testdata)] * self.ev_std + self.ev_mean
        theuh25=test_data[composite_group][input_index, :, :, self.extract_uh25_index(testdata)] * self.uh25_std + self.uh25_mean
        theuh03=test_data[composite_group][input_index, :, :, self.extract_uh03_index(testdata)] * self.uh03_std + self.uh03_mean
        data=xr.Dataset({
            'saliency_maps':(['a','x','y'], thecontours),
            'dbz':(['x','y'], thedbz),
            'eu':(['x','y'], theeu),
            'ev':(['x','y'], theev),
            'uh25':(['x','y'], theuh25),
            'uh03':(['x','y'], theuh03),
            })
        data.to_netcdf(
            f"{self.comp_directory}/saliency_{self.mask_str}_model{self.model_num}_{self.method}{self.random_choice}_{composite_group}_{str(input_index)}_{self.variable}.nc")
        return

    def auto_saliency(self, data):
        
        """Preview the saved saliency maps.
        
        Args:
            data (xarray dataset): Saliency map opened netCDF file. 
        
        """
        sm_data=data.saliency_maps
        fig, axes=plt.subplots(4, 8, figsize=(16, 8), sharex=True, sharey=True)
        plt.subplots_adjust(0.02, 0.02, 0.96, 0.94, wspace=0,hspace=0)
        for conv_filter, ax in enumerate(axes.ravel()):
            levels=[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]
            cmap_dbz = colortables.get_colortable('NWSReflectivity')
            ax.contourf(data.dbz,
                        cmap=cmap_dbz, levels=levels, alpha=0.2)
            ax.contour(sm_data[conv_filter, :, :],
                       [-3, -2, -1, 1, 2, 3], vmin=-3, vmax=3, cmap="seismic", linewidths=3.0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.text(16, 16, conv_filter, fontsize=14)
        plt.suptitle("Final Convolution Filter Saliency Maps", fontsize=14, y=0.98)
        plt.show()
