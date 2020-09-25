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


class StudyVisualizer:
    
    """Class instantiation of StudyVisualizer:
    
    Here we load the variable data for visualization of the study data and the deep learning model.
    
    Attributes:
        climate (str): Whether analyzing ``current`` or ``future`` climate simulation.
        method (str): Method for parsing data. Options include ``random``, ``month``, ``season``, ``year``.
        variable1 (str): Variable name for visualization. Options include: 
                        ``EU1``, ``EU3``, ``EU5``, ``EU7``, 
                        ``EV1``, ``EV3``, ``EV5``, ``EV7``, 
                        ``TK1``, ``TK3``, ``TK5``, ``TK7``, 
                        ``QVAPOR1``, ``QVAPOR3``, ``QVAPOR5``, ``QVAPOR7``,
                        ``W1``, ``W3``, ``W5``, ``W7``,
                        ``P1``, ``P3``, ``P5``, ``P7``,
                        ``WMAX``, ``DBZ``,``CTT``,``UH25``, and``UH03``.
        dist_directory (str): The directory path to where the files are located from.
        model_directory (str): Directory where the deep learning model is saved.
        model_num (str): The number of the model as it was saved.
        comp_directory (str): Directory where files and figures will be saved.
        variable2 (str): The second variable for analysis. Options same as ``variable1``. Defaults to ``None``.
        mask (boolean): Whether to train using the masked data or the non-masked data. Defaults to ``False``.
        random_choice (int): The integer the respective ``random`` method file was saved as. Defaults to ``None``.
        month_choice (int): Month for analysis. Defaults to ``None``.
        season_choice (str): Three-month season string, if ``method==season`` (e.g., 'DJF'). Defaults to ``None``.
        year_choice (int): Year for analysis. Defaults to ``None``.
        outliers (boolean): Whether evaluating outlier storms. Defaults to ``True``.
        
    Raises:
        Exceptions: Checks whether correct values were input for ``climate`` and ``method``.
        
    """

    def __init__(self, climate, method, variable1, dist_directory, model_directory, model_num, comp_directory, 
                 variable2=None, mask=False, random_choice=None, outliers=False):
        
        if climate!='current' and climate!='future':
            raise Exception("Please enter ``current`` or ``future`` as string for climate period selection.")
        else:
            self.climate=climate
        if method!='random' and method!='month' and method!='season' and method!='year':
            raise Exception("Please enter ``random``, ``month``, ``season``, or ``year`` as method.")
        else:
            self.method=method
        self.variable1=variable1
        self.variable2=variable2
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
        self.outliers=outliers
    
    def variable_translate(self, variable):
        
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
               'MASK':'MASK',
              }
        try:
            out=var[variable]
            return out
        except:
            raise ValueError("Please enter ``TK``, ``EV``, ``EU``, ``QVAPOR``, ``PRESS``, ``W_vert``, ``UH25``, ``UH03``, ``MAXW``, ``CTT``, or ``DBZ`` as variable with height AGL appended (1, 3, 5, or 7).")

    def convert_string_height(self, variable):
        
        """Convert the string variable name's height to integer for indexing mean and standard deviation data.
        
        """
        if variable!='DBZ' and variable!='CTT' and variable!='UH25' and variable!='UH03' and variable!='MASK':
            the_hgt=int(variable[-1])
            heights=np.array([1,3,5,7])
            the_indx=np.where(heights==the_hgt)
            return the_indx

    def extract_variable_mean_and_std(self):
        
        """Open the file containing mean and std information for the variable.
        
        """
        data=xr.open_dataset(f"/{self.dist_directory}/{self.climate}_{self.variable_translate(self.variable1).lower()}_{self.mask_str}_dldata_traindist.nc")
        if data.features.size>1:
            self.variable1_mean=data.train_mean.values[self.convert_string_height(self.variable1)[0][0]]
            self.variable1_std=data.train_std.values[self.convert_string_height(self.variable1)[0][0]]
        if data.features.size==1:
            self.variable1_mean=data.train_mean.values[0]
            self.variable1_std=data.train_std.values[0]
        data=xr.open_dataset(f"/{self.dist_directory}/{self.climate}_{self.variable_translate(self.variable2).lower()}_{self.mask_str}_dldata_traindist.nc")
        if data.features.size>1:
            self.variable2_mean=data.train_mean.values[self.convert_string_height(self.variable2)[0][0]]
            self.variable2_std=data.train_std.values[self.convert_string_height(self.variable2)[0][0]]
        if data.features.size==1:
            self.variable2_mean=data.train_mean.values[0]
            self.variable2_std=data.train_std.values[0]            

    def extract_eu_mean_and_std(self):
        
        """Open the file containing mean and std information for the variable.
        
        """
        data=xr.open_dataset(f"/{self.dist_directory}/{self.climate}_eu_{self.mask_str}_dldata_traindist.nc")
        self.eu_mean=data.train_mean.values[self.convert_string_height(self.variable1)[0][0]]
        self.eu_std=data.train_std.values[self.convert_string_height(self.variable1)[0][0]]

    def extract_ev_mean_and_std(self):
        
        """Open the file containing mean and std information for the variable.
        
        """
        data=xr.open_dataset(f"/{self.dist_directory}/{self.climate}_ev_{self.mask_str}_dldata_traindist.nc")
        self.ev_mean=data.train_mean.values[self.convert_string_height(self.variable1)[0][0]]
        self.ev_std=data.train_std.values[self.convert_string_height(self.variable1)[0][0]]

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
        if not self.outliers:
            return xr.open_dataset(
                f'{self.comp_directory}/composite_results_{self.mask_str}_model{self.model_num}_{self.method}{self.random_choice}.nc')
        if self.outliers:
            return xr.open_dataset(
                f'{self.comp_directory}/composite_outresults_{self.mask_str}_model{self.model_num}_{self.method}{self.random_choice}.nc')

    def extract_variable1_index(self, data):
    
        """Find the variable index from the respective test data set.
        
        Args:
            data: Dataset opened with ``extract_variable_and_dbz()``.
        
        """
        return np.where(data.coords['features'].values==self.variable1)[0][0]

    def extract_variable2_index(self, data):
    
        """Find the variable index from the respective test data set.
        
        Args:
            data: Dataset opened with ``extract_variable_and_dbz()``.
        
        """
        return np.where(data.coords['features'].values==self.variable2)[0][0]

    def extract_dbz_index(self, data):
    
        """Find the ``DBZ`` index from the respective test data set.
        
        Args:
            data: Dataset opened with ``extract_variable_and_dbz()``.
        
        """
        return np.where(data.coords['features'].values=='DBZ')[0][0]

    def extract_mask_index(self, data):
    
        """Find the ``DBZ`` index from the respective test data set.
        
        Args:
            data: Dataset opened with ``extract_variable_and_dbz()``.
        
        """
        return np.where(data.coords['features'].values=='MASK')[0][0]

    def extract_EV_index(self, data):
    
        """Find the ``EV`` (v-wind) index from the respective test data set for the corresponding variable height.
        
        Args:
            data: Dataset opened with ``extract_variable_and_dbz()``.
        
        """
        return np.where(data.coords['features'].values=='EV'+self.variable1[-1])[0][0], np.where(data.coords['features'].values=='EV'+self.variable2[-1])[0][0]

    def extract_EU_index(self, data):
    
        """Find the ``EU`` (u-wind) index from the respective test data set for the corresponding variable height.
        
        Args:
            data: Dataset opened with ``extract_variable_and_dbz()``.
        
        """
        return np.where(data.coords['features'].values=='EU'+self.variable1[-1])[0][0], np.where(data.coords['features'].values=='EU'+self.variable2[-1])[0][0]

    def extract_QVAPOR_index(self, data):
    
        """Find the ``QVAPOR`` (water vapor mixing ratio) index from the respective test data set for the corresponding variable height.
        
        Args:
            data: Dataset opened with ``extract_variable_and_dbz()``.
        
        """
        return np.where(data.coords['features'].values=='QVAPOR'+self.variable1[-1])[0][0], np.where(data.coords['features'].values=='QVAPOR'+self.variable2[-1])[0][0]

    def extract_TK_index(self, data):
    
        """Find the ``TK`` (temperature) index from the respective test data set for the corresponding variable height.
        
        Args:
            data: Dataset opened with ``extract_variable_and_dbz()``.
        
        """
        return np.where(data.coords['features'].values=='TK'+self.variable1[-1])[0][0], np.where(data.coords['features'].values=='TK'+self.variable2[-1])[0][0]

    def extract_P_index(self, data):
    
        """Find the ``P`` (pressure) index from the respective test data set for the corresponding variable height.
        
        Args:
            data: Dataset opened with ``extract_variable_and_dbz()``.
        
        """
        return np.where(data.coords['features'].values=='P'+self.variable1[-1])[0][0], np.where(data.coords['features'].values=='P'+self.variable2[-1])[0][0]

    def grab_value_of_storm(self, data, group_choice):
        
        """Grab the maximum variable data within the storm object.
        
        Args:
            data (xarray dataset): The test data being analyzed.
            group_choice (str): The prediction choice from the 2x2 contingency matrix.
            
        Returns:
            max_variable1, max_variable2 (xarray data arrays): The maximum values of the respective variable within the storm patch.
        
        """
        mask_data=data.sel(features='MASK')
        if self.variable1!='CTT':
            max_variable1=data[group_choice][:,:,:,self.extract_variable1_index(data)].where(mask_data[group_choice]).max(axis=(1,2), skipna=True)
        if self.variable1=='CTT':
            max_variable1=data[group_choice][:,:,:,self.extract_variable1_index(data)].where(mask_data[group_choice]).min(axis=(1,2), skipna=True)
        if self.variable2!='CTT':
            max_variable2=data[group_choice][:,:,:,self.extract_variable2_index(data)].where(mask_data[group_choice]).max(axis=(1,2), skipna=True)
        if self.variable2=='CTT':
            max_variable2=data[group_choice][:,:,:,self.extract_variable2_index(data)].where(mask_data[group_choice]).min(axis=(1,2), skipna=True)
        max_variable1=max_variable1 * self.variable1_std + self.variable1_mean
        max_variable2=max_variable2 * self.variable2_std + self.variable2_mean
        return max_variable1, max_variable2

    def plot_two_var_scatter(self, data, composite_list, composite_colors, marker_sizes, markers, fillstyles):
        
        """Plot two variables on a scatter plot for data exploration.
        
        Args:
            data (xarray data array): Data for visualization.
            composite_list (str): The 2x2 contingency table option for visualizing (e.g., ``tn``, ``tp``, ``fn``, and ``fp``.)
            composite_colors (str): Corresponding colors for plotting the scatters.
            marker_sizes (int): Size of the marker.
            markers (str): Type of marker for the plot (e.g., .,,,o,v,^,<,>,*,h,H,+,x,X,D,d,|,_).
            fillstyles (str): Whether to fill or not fill the facecolor. Options include ``none`` and ``full``.
        
        """
        if len(composite_list)!=len(composite_colors):
            raise Exception("List of composite type and colors must be equal.")
        for comp_type, comp_color, mark_size, mark, fillsty in zip(composite_list, composite_colors, marker_sizes, markers, fillstyles):
            plt.scatter(self.grab_value_of_storm(data, group_choice=comp_type)[0], self.grab_value_of_storm(data, group_choice=comp_type)[1], 
                        c=comp_color, s=mark_size, marker=mark, facecolors=fillsty)
        return plt.show()

    def compute_wind_shear(self, current_data, future_data, comp_str):
        
        """Computation of wind shear.
        
        """
        data=xr.open_dataset(f"/glade/scratch/molina/DL_proj/current_conus_fields/dl_preprocess/current_eu_{self.mask_str}_dldata_traindist.nc")
        current_eu1_mean=data.train_mean.values[0]
        current_eu1_std=data.train_std.values[0]
        current_eu2_mean=data.train_mean.values[0]
        current_eu2_std=data.train_std.values[0]
        data=xr.open_dataset(f"/glade/scratch/molina/DL_proj/current_conus_fields/dl_preprocess/current_ev_{self.mask_str}_dldata_traindist.nc")
        current_ev1_mean=data.train_mean.values[0]
        current_ev1_std=data.train_std.values[0]
        current_ev2_mean=data.train_mean.values[0]
        current_ev2_std=data.train_std.values[0]
        data=xr.open_dataset(f"/glade/scratch/molina/DL_proj/future_conus_fields/dl_preprocess/future_eu_{self.mask_str}_dldata_traindist.nc")
        future_eu1_mean=data.train_mean.values[0]
        future_eu1_std=data.train_std.values[0]
        future_eu2_mean=data.train_mean.values[0]
        future_eu2_std=data.train_std.values[0]
        data=xr.open_dataset(f"/glade/scratch/molina/DL_proj/future_conus_fields/dl_preprocess/future_ev_{self.mask_str}_dldata_traindist.nc")
        future_ev1_mean=data.train_mean.values[0]
        future_ev1_std=data.train_std.values[0]
        future_ev2_mean=data.train_mean.values[0]
        future_ev2_std=data.train_std.values[0]
        
        current_mask_data=current_data.sel(features='MASK')
        current_u_wind1=current_data.sel(features='EU1')[comp_str].where(current_mask_data[comp_str]).max(axis=(1,2), skipna=True)
        current_u_wind2=current_data.sel(features='EU5')[comp_str].where(current_mask_data[comp_str]).max(axis=(1,2), skipna=True)
        current_v_wind1=current_data.sel(features='EV1')[comp_str].where(current_mask_data[comp_str]).max(axis=(1,2), skipna=True)
        current_v_wind2=current_data.sel(features='EV5')[comp_str].where(current_mask_data[comp_str]).max(axis=(1,2), skipna=True)
        current_u_total = (current_u_wind2 * current_eu2_std + current_eu2_mean) - (current_u_wind1  * current_eu1_std + current_eu1_mean)
        current_v_total = (current_v_wind2 * current_ev2_std + current_ev2_mean) - (current_v_wind1  * current_ev1_std + current_ev1_mean)
        current_wind = np.sqrt(current_u_total**2 + current_v_total**2)
        
        future_mask_data=future_data.sel(features='MASK')
        future_u_wind1=future_data.sel(features='EU1')[comp_str].where(future_mask_data[comp_str]).max(axis=(1,2), skipna=True)
        future_u_wind2=future_data.sel(features='EU5')[comp_str].where(future_mask_data[comp_str]).max(axis=(1,2), skipna=True)
        future_v_wind1=future_data.sel(features='EV1')[comp_str].where(future_mask_data[comp_str]).max(axis=(1,2), skipna=True)
        future_v_wind2=future_data.sel(features='EV5')[comp_str].where(future_mask_data[comp_str]).max(axis=(1,2), skipna=True)
        future_u_total = (future_u_wind2 * future_eu2_std + future_eu2_mean) - (future_u_wind1  * future_eu1_std + future_eu1_mean)
        future_v_total = (future_v_wind2 * future_ev2_std + future_ev2_mean) - (future_v_wind1  * future_ev1_std + future_ev1_mean)
        future_wind = np.sqrt(future_u_total**2 + future_v_total**2)
        return current_wind, future_wind

def plot_current_and_future(current_data, future_data, markersize, marker, facecolor, color=['b','r'], xlabel=None, ylabel=None):
    
    """Plot current and future variable data for a select composite type.
    
    Args: 
        current_data (xarray dataset): Current climate data. (self.grab_value_of_storm(data, group_choice=comp_type))
        future_data (xarray dataset): Future climate data. (self.grab_value_of_storm(future_data, group_choice=comp_type))
        composite_type (str): The 2x2 contingency table option for visualizing (e.g., ``tn``, ``tp``, ``fn``, and ``fp``.)
        color (str): List of two corresponding colors for plotting the scatters. Defaults to ``b`` and ``r``.
        marker_sizes (int): Size of the marker.
        markers (str): Type of marker for the plot (e.g., .,,,o,v,^,<,>,*,h,H,+,x,X,D,d,|,_).
        fillstyles (str): Whether to fill or not fill the facecolor. Options include ``none`` and ``full``.        
    
    """
    plt.scatter(current_data[0], current_data[1], c=color[0], s=markersize, marker=marker, facecolors=facecolor)
    plt.scatter(future_data[0], future_data[1], c=color[1], s=markersize, marker=marker, facecolors=facecolor, alpha=0.5)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    return plt.show()
