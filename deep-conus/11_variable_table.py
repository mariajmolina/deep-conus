import xarray as xr
import numpy as np
import pandas as pd
from scipy import stats


class VariableTable:
    
    """Class instantiation of VariableTable:
    
    Here we create the content in table (storm object median and standard deviation data).
    
    Attributes:
        climate (str): Whether to interpolate variable in the ``current`` or ``future`` climate simulation.
        variables (str): Numpy array of variable name strings. Options include ``EU``, ``EV``, ``TK``, ``QVAPOR``, ``WMAX``, 
                         ``W_vert``,``PRESS``,``DBZ``,``CTT``,``UH25``, ``MASK``, and``UH03``.
        var_directory (str): Directory where the test subset variable data is saved.
        out_directory (str): Directory where the test subset outlier variable data is saved.
        mask (boolean): Whether to train using the masked data or the non-masked data. Defaults to ``False``.
        random_choice (int): The integer the respective ``random`` method file was saved as. Defaults to ``None``.
        boot_num (int): Total sample size of bootstrap analysis. Defaults to ``1,000``.
        boot_min (float): Minimum percentile for the bootstrap confidence intervals. Defaults to ``2.5``.
        boot_max (float): Maximum percentile for the bootstrap confidence intervals. Defaults to ``97.5``.
        
    Raises:
        Exceptions: Checks whether correct values were input for ``climate``.
        
    """
    
    def __init__(self, climate, variables, var_directory, eval_directory, model_num, mask=False, 
                 unbalanced=True, validation=False, random_choice=None, outliers=False,
                 boot_num=1000, boot_min=2.5, boot_max=97.5):
        
        if climate!='current' and climate!='future':
            raise Exception("Please enter ``current`` or ``future`` as string for climate period selection.")
        else:
            self.climate=climate
        self.variables=variables
        self.var_directory=var_directory
        self.eval_directory=eval_directory
        self.model_num=model_num
        self.mask=mask
        if not self.mask:
            self.mask_str='nomask'
        if self.mask:
            self.mask_str='mask'
        self.unbalanced=unbalanced
        self.validation=validation
        self.random_choice=random_choice
        self.outliers=outliers
        self.boot_num = boot_num
        self.boot_min = boot_min
        self.boot_max = boot_max

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
               'MASK':'MASK',
              }
        try:
            out=var[variable]
            return out
        except:
            raise ValueError("Please enter ``TK``, ``EV``, ``EU``, ``QVAPOR``, ``PRESS``, ``W_vert``, ``UH25``, ``MASK``, ``UH03``, ``MAXW``, ``CTT``, or ``DBZ`` as variable.")
        
    def intro_sequence_evaluation(self):
        
        """
        Helper function when running bootstrap or permutation feature importance in job script, to prevent memory issues.
        Follow up with solo_pfi or solo_bootstrap.
        
        """
        if not self.outliers:
            data = xr.open_dataset(f'{self.eval_directory}/testdata_{self.mask_str}_model{self.model_num}_random{self.random_choice}.nc')
            testdata = data.X_test.astype('float16').values
            testlabels = data.X_test_label.values
            data = None
        if self.outliers:
            testdata, testlabels = self.load_qv_files()
        return testdata, testlabels
    
    def load_qv_files(self, model_num=25, upper_perc=99):
        
        """Load and concatenate the testdata and testlabels.
        
        """
        if not self.unbalanced:
            data1=xr.open_dataset(f'{self.eval_directory}/outliers_testdata_{self.mask_str}_model{model_num}_random1_{upper_perc}.nc')
            data2=xr.open_dataset(f'{self.eval_directory}/outliers_testdata_{self.mask_str}_model{model_num}_random2_{upper_perc}.nc')
            data3=xr.open_dataset(f'{self.eval_directory}/outliers_testdata_{self.mask_str}_model{model_num}_random3_{upper_perc}.nc')
            data4=xr.open_dataset(f'{self.eval_directory}/outliers_testdata_{self.mask_str}_model{model_num}_random4_{upper_perc}.nc')
            data5=xr.open_dataset(f'{self.eval_directory}/outliers_testdata_{self.mask_str}_model{model_num}_random5_{upper_perc}.nc')
        if self.unbalanced:
            data1=xr.open_dataset(f'{self.eval_directory}/outliers_testdata_{self.mask_str}_model{model_num}_random1_{upper_perc}_unbalanced.nc')
            data2=xr.open_dataset(f'{self.eval_directory}/outliers_testdata_{self.mask_str}_model{model_num}_random2_{upper_perc}_unbalanced.nc')
            data3=xr.open_dataset(f'{self.eval_directory}/outliers_testdata_{self.mask_str}_model{model_num}_random3_{upper_perc}_unbalanced.nc')
            data4=xr.open_dataset(f'{self.eval_directory}/outliers_testdata_{self.mask_str}_model{model_num}_random4_{upper_perc}_unbalanced.nc')
            data5=xr.open_dataset(f'{self.eval_directory}/outliers_testdata_{self.mask_str}_model{model_num}_random5_{upper_perc}_unbalanced.nc')
        testdata=xr.concat([data1.testdata,data2.testdata,data3.testdata,data4.testdata,data5.testdata],dim='b').values
        labels=xr.concat([data1.testlabels,data2.testlabels,data3.testlabels,data4.testlabels,data5.testlabels],dim='b').values
        return testdata, labels

    def extract_variable_mean_and_std(self):
        
        """Open the file containing mean and std information for the variable.
        
        """
        the_data={}
        for var in self.variables:
            if not self.unbalanced:
                the_data[var]=xr.open_dataset(
                    f"/{self.var_directory}/{self.climate}_{self.variable_translate(var).lower()}_{self.mask_str}_dldata_traindist.nc")
            if self.unbalanced:
                the_data[var]=xr.open_dataset(
                    f"/{self.var_directory}/{self.climate}_{self.variable_translate(var).lower()}_{self.mask_str}_dldata_traindist_unbalanced.nc")
        return the_data

    def table_output_median(self, testdata, datavars):

        """Output paper's table 2 content, which is the variable means of the storm objects (with masks applied).
        
        Args:
            testdata (numpy array): Test data.
            datavars: Output from ``extract_variable_mean_and_std()``.
            
        """
        medians = []
        # print('TK')
        medians.append(np.nanmedian(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,0]).data)*
                       datavars['TK'].train_std[0].values+datavars['TK'].train_mean[0].values)
        medians.append(np.nanmedian(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,1]).data)*
                       datavars['TK'].train_std[1].values+datavars['TK'].train_mean[1].values)
        medians.append(np.nanmedian(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,2]).data)*
                       datavars['TK'].train_std[2].values+datavars['TK'].train_mean[2].values)
        medians.append(np.nanmedian(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,3]).data)*
                       datavars['TK'].train_std[3].values+datavars['TK'].train_mean[3].values)
        # print('EV')
        medians.append(np.nanmedian(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,4]).data)*
                       datavars['EV'].train_std[0].values+datavars['EV'].train_mean[0].values)
        medians.append(np.nanmedian(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,5]).data)*
                       datavars['EV'].train_std[1].values+datavars['EV'].train_mean[1].values)
        medians.append(np.nanmedian(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,6]).data)*
                       datavars['EV'].train_std[2].values+datavars['EV'].train_mean[2].values)
        medians.append(np.nanmedian(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,7]).data)*
                       datavars['EV'].train_std[3].values+datavars['EV'].train_mean[3].values)
        # print('EU')
        medians.append(np.nanmedian(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,8]).data)*
                       datavars['EU'].train_std[0].values+datavars['EU'].train_mean[0].values)
        medians.append(np.nanmedian(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,9]).data)*
                       datavars['EU'].train_std[1].values+datavars['EU'].train_mean[1].values)
        medians.append(np.nanmedian(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,10]).data)*
                       datavars['EU'].train_std[2].values+datavars['EU'].train_mean[2].values)
        medians.append(np.nanmedian(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,11]).data)*
                       datavars['EU'].train_std[3].values+datavars['EU'].train_mean[3].values)
        # print('QVAPOR')
        medians.append((np.nanmedian(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,12]).data)*
                       datavars['QVAPOR'].train_std[0].values+datavars['QVAPOR'].train_mean[0].values)*1000)
        medians.append((np.nanmedian(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,13]).data)*
                       datavars['QVAPOR'].train_std[1].values+datavars['QVAPOR'].train_mean[1].values)*1000)
        medians.append((np.nanmedian(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,14]).data)*
                       datavars['QVAPOR'].train_std[2].values+datavars['QVAPOR'].train_mean[2].values)*1000)
        medians.append((np.nanmedian(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,15]).data)*
                       datavars['QVAPOR'].train_std[3].values+datavars['QVAPOR'].train_mean[3].values)*1000)
        # print('PRESS')
        medians.append(np.nanmedian(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,16]).data)*
                       datavars['PRESS'].train_std[0].values+datavars['PRESS'].train_mean[0].values)
        medians.append(np.nanmedian(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,17]).data)*
                       datavars['PRESS'].train_std[1].values+datavars['PRESS'].train_mean[1].values)
        medians.append(np.nanmedian(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,18]).data)*
                       datavars['PRESS'].train_std[2].values+datavars['PRESS'].train_mean[2].values)
        medians.append(np.nanmedian(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,19]).data)*
                       datavars['PRESS'].train_std[3].values+datavars['PRESS'].train_mean[3].values)
        return medians
    
    def bootstrap_median_percentiles(self, data, var_str, var_indx, hgt_indx, datavars):

        """Bootstrapped percentiles for the variable median (with masks applied).
        
        Args:
            data (numpy array): Test data.
            var_str (str): Variable name. E.g., ``TK``.
            var_indx: Index of the variable among all variables.
            hgt_indx: Index of the height among various heights.
            datavars: Output from ``extract_variable_mean_and_std()``.
            
        """
        median_values = []
        themask = data[:,:,:,-1]
        vardata = data[:,:,:,var_indx]
        maskedarray = np.ma.masked_where(themask==0, vardata)
        for i in range(self.boot_num):
            np.random.seed(seed=i)
            indx = np.random.choice(np.arange(0,len(data[:,0,0,0]),1), size=len(data[:,0,0,0]))
            median_val = (np.nanmedian(maskedarray[indx,:,:].data) * datavars[var_str].train_std[hgt_indx].values) +\
                          datavars[var_str].train_mean[hgt_indx].values
            median_values.append(median_val)
        return np.nanpercentile(median_values, q=[self.boot_min, self.boot_max])
