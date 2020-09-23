import xarray as xr
import numpy as np
import pandas as pd
from scipy import stats


class VariableTable:
    
    """Class instantiation of VariableTable:
    
    Here we create the content in table (storm object median and standard deviation data).
    
    Attributes:
        climate (str): Whether to interpolate variable in the ``current`` or ``future`` climate simulation.
        method (str): Method for parsing data. Options include ``random``, ``month``, ``season``, ``year``.
        variables (str): Numpy array of variable name strings. Options include ``EU``, ``EV``, ``TK``, ``QVAPOR``, ``WMAX``, 
                         ``W_vert``,``PRESS``,``DBZ``,``CTT``,``UH25``, ``MASK``, and``UH03``.
        var_directory (str): Directory where the test subset variable data is saved.
        mask (boolean): Whether to train using the masked data or the non-masked data. Defaults to ``False``.
        random_choice (int): The integer the respective ``random`` method file was saved as. Defaults to ``None``.
        month_choice (int): Month for analysis. Defaults to ``None``.
        season_choice (str): Three-month season string, if ``method==season`` (e.g., 'DJF'). Defaults to ``None``.
        year_choice (int): Year for analysis. Defaults to ``None``.
        boot_num (int): Total sample size of bootstrap analysis. Defaults to ``1,000``.
        boot_min (float): Minimum percentile for the bootstrap confidence intervals. Defaults to ``2.5``.
        boot_max (float): Maximum percentile for the bootstrap confidence intervals. Defaults to ``97.5``.
        
    Raises:
        Exceptions: Checks whether correct values were input for ``climate`` and ``method``.
        
    Todo:
        * Add loading and handling of test subsets that were created using the ``month``, ``season``, and ``year`` methods.
        
    """
    
    def __init__(self, climate, method, variables, var_directory, mask=False, 
                 random_choice=None, month_choice=None, season_choice=None, year_choice=None,
                 boot_num=1000, boot_min=2.5, boot_max=97.5):
        
        if climate!='current' and climate!='future':
            raise Exception("Please enter ``current`` or ``future`` as string for climate period selection.")
        else:
            self.climate=climate
        if method!='random' and method!='month' and method!='season' and method!='year':
            raise Exception("Please enter ``random``, ``month``, ``season``, or ``year`` as method.")
        else:
            self.method=method
        self.variables=variables
        self.var_directory=var_directory
        self.mask=mask
        if not self.mask:
            self.mask_str='nomask'
        if self.mask:
            self.mask_str='mask'
        self.random_choice=random_choice 
        self.month_choice=month_choice 
        self.season_choice=season_choice 
        self.year_choice=year_choice
        self.boot_num = boot_num
        self.boot_min = boot_min
        self.boot_max = boot_max

    def month_translate(self):
        
        """Convert integer month to string month.
        
        Returns:
            out (str): Input month as string.
            
        Raises:
            ValueError: If the month is not within the study's range (Dec-May).
            
        """
        var={12:'dec',
             1:'jan',
             2:'feb',
             3:'mar',
             4:'apr',
             5:'may'}
        try:
            out=var[self.month_choice]
            return out
        except:
            raise ValueError("Please enter month integer from Dec-May.")

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

    def add_dbz(self):
        
        """Function that adds ``DBZ`` variable to last dim of test data array.
        
        """
        self.variables=np.append(self.variables, 'DBZ')

    def add_uh25(self):
        
        """Function that adds ``UH25`` variable to last dim of test data array.
        
        """
        self.variables=np.append(self.variables, 'UH25')

    def add_uh03(self):
        
        """Function that adds ``UH03`` variable to last dim of test data array.
        
        """
        self.variables=np.append(self.variables, 'UH03')

    def add_wmax(self):
        
        """Function that adds ``WMAX`` variable to last dim of test data array.
        
        """
        self.variables=np.append(self.variables, 'WMAX')

    def add_ctt(self):
        
        """Function that adds ``CTT`` variable to last dim of test data array.
        
        """
        self.variables=np.append(self.variables, 'CTT')

    def add_mask(self):
        
        """Function that adds ``MASK`` variable to last dim of test data array.
        
        """
        self.variables=np.append(self.variables, 'MASK')

    def open_test_files(self):

        """Open the subset test data files.
        
        Returns:
            the_data: Dictionary of opened Xarray data arrays containing selected variable training data.
            
        """
        self.add_dbz()
        self.add_uh25()
        self.add_uh03()
        self.add_wmax()
        self.add_ctt()
        self.add_mask()
        the_data={}
        if self.method=='random':
            for var in self.variables:
                the_data[var]=xr.open_dataset(
                    f'/{self.var_directory}/{self.climate}_{self.variable_translate(var).lower()}_{self.mask_str}_{self.method}_test{self.random_choice}.nc')
        if self.method=='month':
            for var in self.variables:
                var   ##TODO 
        if self.method=='season':
            for var in self.variables:
                var   ##TODO 
        if self.method=='year':
            for var in self.variables:
                var   ##TODO 
        return the_data

    def assemble_and_concat(self, **kwargs):
        
        """Eagerly load the testing labels and data.
        
        Args:
            **kwargs: Dictionary containing opened variable testing data, which was opened with ``self.open_test_files()``.
            
        Returns:
            testdata, label: Eagerly loaded test data and labels as a numpy array.
        
        """
        thedata={}
        for key, value in kwargs.items():
            thedata[key]=value.X_test.values
            label=value.X_test_label.values
        if len(kwargs) > 1:
            testdata=np.concatenate(list(thedata.values()), axis=3)
        if len(kwargs)==1:
            testdata=np.squeeze(np.asarray(list(thedata.values())))
        thedata=None
        return testdata, label

    def remove_nans(self, testdata, label):
        
        """Assemble the variables and remove any ``nan`` values from the data. 
        Assigns new attributes to class, including ``test_data`` and ``test_labels``, which are the test data and labels 
        for evaluating deep learning model skill.
        
        Args:
            testdata (numpy array): Test data.
            label (numpy array): Label data.
        
        """
        data=xr.Dataset({
                    'X_test':(['b','x','y','features'], testdata),
                    'X_test_label':(['b'], label),
                    },
                    ).dropna(dim='b')
        test_data=data.X_test.values
        self.test_labels=data.X_test_label.values
        return test_data

    def extract_variable_mean_and_std(self):
        
        """Open the file containing mean and std information for the variable.
        
        """
        the_data={}
        for var in self.variables:
            the_data[var]=xr.open_dataset(
                f"/{self.var_directory}/{self.climate}_{self.variable_translate(var).lower()}_{self.mask_str}_dldata_traindist.nc")
        return the_data

    def table_output_mean(self, testdata, datavars):
        
        """Output paper's table content, which is the variable means of the storm objects (with masks applied).
        
        Args:
            testdata (numpy array): Test data.
            datavars: Output from ``extract_variable_mean_and_std()``.

        """
        print('TK')
        print(np.nanmean(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,0]))*
                    datavars['TK'].train_std[0].values+datavars['TK'].train_mean[0].values)
        print(np.nanmean(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,1]))*
                    datavars['TK'].train_std[1].values+datavars['TK'].train_mean[1].values)
        print(np.nanmean(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,2]))*
                    datavars['TK'].train_std[2].values+datavars['TK'].train_mean[2].values)
        print(np.nanmean(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,3]))*
                    datavars['TK'].train_std[3].values+datavars['TK'].train_mean[3].values)
        print('EV')
        print(np.nanmean(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,4]))*
                    datavars['EV'].train_std[0].values+datavars['EV'].train_mean[0].values)
        print(np.nanmean(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,5]))*
                    datavars['EV'].train_std[1].values+datavars['EV'].train_mean[1].values)
        print(np.nanmean(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,6]))*
                    datavars['EV'].train_std[2].values+datavars['EV'].train_mean[2].values)
        print(np.nanmean(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,7]))*
                    datavars['EV'].train_std[3].values+datavars['EV'].train_mean[3].values)
        print('EU')
        print(np.nanmean(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,8]))*
                    datavars['EU'].train_std[0].values+datavars['EU'].train_mean[0].values)
        print(np.nanmean(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,9]))*
                    datavars['EU'].train_std[1].values+datavars['EU'].train_mean[1].values)
        print(np.nanmean(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,10]))*
                    datavars['EU'].train_std[2].values+datavars['EU'].train_mean[2].values)
        print(np.nanmean(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,11]))*
                    datavars['EU'].train_std[3].values+datavars['EU'].train_mean[3].values)
        print('QVAPOR')
        print(np.nanmean(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,12]))*
                    datavars['QVAPOR'].train_std[0].values+datavars['QVAPOR'].train_mean[0].values)
        print(np.nanmean(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,13]))*
                    datavars['QVAPOR'].train_std[1].values+datavars['QVAPOR'].train_mean[1].values)
        print(np.nanmean(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,14]))*
                    datavars['QVAPOR'].train_std[2].values+datavars['QVAPOR'].train_mean[2].values)
        print(np.nanmean(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,15]))*
                    datavars['QVAPOR'].train_std[3].values+datavars['QVAPOR'].train_mean[3].values)
        print('PRESS')
        print(np.nanmean(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,16]))*
                    datavars['PRESS'].train_std[0].values+datavars['PRESS'].train_mean[0].values)
        print(np.nanmean(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,17]))*
                    datavars['PRESS'].train_std[1].values+datavars['PRESS'].train_mean[1].values)
        print(np.nanmean(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,18]))*
                    datavars['PRESS'].train_std[2].values+datavars['PRESS'].train_mean[2].values)
        print(np.nanmean(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,19]))*
                    datavars['PRESS'].train_std[3].values+datavars['PRESS'].train_mean[3].values)
        print('DBZ')
        print(np.nanmean(np.nanmax(np.nanmax(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,20]), axis=1), axis=1).data)*
                    datavars['DBZ'].train_std[0].values+datavars['DBZ'].train_mean[0].values)
        print('UH25')
        print(np.nanmean(np.nanmax(np.nanmax(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,21]), axis=1), axis=1).data)*
                    datavars['UH25'].train_std[0].values+datavars['UH25'].train_mean[0].values)
        print('UH03')
        print(np.nanmean(np.nanmax(np.nanmax(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,22]), axis=1), axis=1).data)*
                    datavars['UH03'].train_std[0].values+datavars['UH03'].train_mean[0].values)
        print('WMAX')
        print(np.nanmean(np.nanmax(np.nanmax(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,23]), axis=1), axis=1).data)*
                    datavars['WMAX'].train_std[0].values+datavars['WMAX'].train_mean[0].values)
        print('CTT')
        print(np.nanmean(np.nanmin(np.nanmin(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,24]), axis=1), axis=1).data)*
                    datavars['CTT'].train_std[0].values+datavars['CTT'].train_mean[0].values)
        
    def table_output_quartiles(self, testdata, datavars):

        """Output paper's table 2 content, which is the variable means of the storm objects (with masks applied).
        
        Args:
            testdata (numpy array): Test data.
            datavars: Output from ``extract_variable_mean_and_std()``.
            
        """
        print('TK')
        print(np.nanpercentile(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,0]), q=[25.,75.])*
                    datavars['TK'].train_std[0].values+datavars['TK'].train_mean[0].values)
        print(np.nanpercentile(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,1]), q=[25.,75.])*
                    datavars['TK'].train_std[1].values+datavars['TK'].train_mean[1].values)
        print(np.nanpercentile(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,2]), q=[25.,75.])*
                    datavars['TK'].train_std[2].values+datavars['TK'].train_mean[2].values)
        print(np.nanpercentile(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,3]), q=[25.,75.])*
                    datavars['TK'].train_std[3].values+datavars['TK'].train_mean[3].values)
        print('EV')
        print(np.nanpercentile(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,4]), q=[25.,75.])*
                    datavars['EV'].train_std[0].values+datavars['EV'].train_mean[0].values)
        print(np.nanpercentile(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,5]), q=[25.,75.])*
                    datavars['EV'].train_std[1].values+datavars['EV'].train_mean[1].values)
        print(np.nanpercentile(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,6]), q=[25.,75.])*
                    datavars['EV'].train_std[2].values+datavars['EV'].train_mean[2].values)
        print(np.nanpercentile(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,7]), q=[25.,75.])*
                    datavars['EV'].train_std[3].values+datavars['EV'].train_mean[3].values)
        print('EU')
        print(np.nanpercentile(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,8]), q=[25.,75.])*
                    datavars['EU'].train_std[0].values+datavars['EU'].train_mean[0].values)
        print(np.nanpercentile(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,9]), q=[25.,75.])*
                    datavars['EU'].train_std[1].values+datavars['EU'].train_mean[1].values)
        print(np.nanpercentile(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,10]), q=[25.,75.])*
                    datavars['EU'].train_std[2].values+datavars['EU'].train_mean[2].values)
        print(np.nanpercentile(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,11]), q=[25.,75.])*
                    datavars['EU'].train_std[3].values+datavars['EU'].train_mean[3].values)
        print('QVAPOR')
        print(np.nanpercentile(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,12]), q=[25.,75.])*
                    datavars['QVAPOR'].train_std[0].values+datavars['QVAPOR'].train_mean[0].values)
        print(np.nanpercentile(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,13]), q=[25.,75.])*
                    datavars['QVAPOR'].train_std[1].values+datavars['QVAPOR'].train_mean[1].values)
        print(np.nanpercentile(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,14]), q=[25.,75.])*
                    datavars['QVAPOR'].train_std[2].values+datavars['QVAPOR'].train_mean[2].values)
        print(np.nanpercentile(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,15]), q=[25.,75.])*
                    datavars['QVAPOR'].train_std[3].values+datavars['QVAPOR'].train_mean[3].values)
        print('PRESS')
        print(np.nanpercentile(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,16]), q=[25.,75.])*
                    datavars['PRESS'].train_std[0].values+datavars['PRESS'].train_mean[0].values)
        print(np.nanpercentile(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,17]), q=[25.,75.])*
                    datavars['PRESS'].train_std[1].values+datavars['PRESS'].train_mean[1].values)
        print(np.nanpercentile(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,18]), q=[25.,75.])*
                    datavars['PRESS'].train_std[2].values+datavars['PRESS'].train_mean[2].values)
        print(np.nanpercentile(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,19]), q=[25.,75.])*
                    datavars['PRESS'].train_std[3].values+datavars['PRESS'].train_mean[3].values)
        print('DBZ')
        print(np.nanpercentile(np.nanmax(np.nanmax(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,20]), axis=1), 
                                         axis=1).data, q=[25.,75.])*
                    datavars['DBZ'].train_std[0].values+datavars['DBZ'].train_mean[0].values)
        print('UH25')
        print(np.nanpercentile(np.nanmax(np.nanmax(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,21]), axis=1), 
                                         axis=1).data, q=[25.,75.])*
                    datavars['UH25'].train_std[0].values+datavars['UH25'].train_mean[0].values)
        print('UH03')
        print(np.nanpercentile(np.nanmax(np.nanmax(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,22]), axis=1), 
                                         axis=1).data, q=[25.,75.])*
                    datavars['UH03'].train_std[0].values+datavars['UH03'].train_mean[0].values)
        print('WMAX')
        print(np.nanpercentile(np.nanmax(np.nanmax(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,23]), axis=1), 
                                         axis=1).data, q=[25.,75.])*
                    datavars['WMAX'].train_std[0].values+datavars['WMAX'].train_mean[0].values)
        print('CTT')
        print(np.nanpercentile(np.nanmin(np.nanmin(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,24]), axis=1), 
                                         axis=1).data, q=[25.,75.])*
                    datavars['CTT'].train_std[0].values+datavars['CTT'].train_mean[0].values)

    def table_output_median(self, testdata, datavars):

        """Output paper's table 2 content, which is the variable means of the storm objects (with masks applied).
        
        Args:
            testdata (numpy array): Test data.
            datavars: Output from ``extract_variable_mean_and_std()``.
            
        """
        print('TK')
        print(np.nanmedian(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,0]).data)*
                    datavars['TK'].train_std[0].values+datavars['TK'].train_mean[0].values)
        print(np.nanmedian(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,1]).data)*
                    datavars['TK'].train_std[1].values+datavars['TK'].train_mean[1].values)
        print(np.nanmedian(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,2]).data)*
                    datavars['TK'].train_std[2].values+datavars['TK'].train_mean[2].values)
        print(np.nanmedian(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,3]).data)*
                    datavars['TK'].train_std[3].values+datavars['TK'].train_mean[3].values)
        print('EV')
        print(np.nanmedian(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,4]).data)*
                    datavars['EV'].train_std[0].values+datavars['EV'].train_mean[0].values)
        print(np.nanmedian(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,5]).data)*
                    datavars['EV'].train_std[1].values+datavars['EV'].train_mean[1].values)
        print(np.nanmedian(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,6]).data)*
                    datavars['EV'].train_std[2].values+datavars['EV'].train_mean[2].values)
        print(np.nanmedian(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,7]).data)*
                    datavars['EV'].train_std[3].values+datavars['EV'].train_mean[3].values)
        print('EU')
        print(np.nanmedian(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,8]).data)*
                    datavars['EU'].train_std[0].values+datavars['EU'].train_mean[0].values)
        print(np.nanmedian(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,9]).data)*
                    datavars['EU'].train_std[1].values+datavars['EU'].train_mean[1].values)
        print(np.nanmedian(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,10]).data)*
                    datavars['EU'].train_std[2].values+datavars['EU'].train_mean[2].values)
        print(np.nanmedian(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,11]).data)*
                    datavars['EU'].train_std[3].values+datavars['EU'].train_mean[3].values)
        print('QVAPOR')
        print(np.nanmedian(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,12]).data)*
                    datavars['QVAPOR'].train_std[0].values+datavars['QVAPOR'].train_mean[0].values)
        print(np.nanmedian(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,13]).data)*
                    datavars['QVAPOR'].train_std[1].values+datavars['QVAPOR'].train_mean[1].values)
        print(np.nanmedian(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,14]).data)*
                    datavars['QVAPOR'].train_std[2].values+datavars['QVAPOR'].train_mean[2].values)
        print(np.nanmedian(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,15]).data)*
                    datavars['QVAPOR'].train_std[3].values+datavars['QVAPOR'].train_mean[3].values)
        print('PRESS')
        print(np.nanmedian(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,16]).data)*
                    datavars['PRESS'].train_std[0].values+datavars['PRESS'].train_mean[0].values)
        print(np.nanmedian(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,17]).data)*
                    datavars['PRESS'].train_std[1].values+datavars['PRESS'].train_mean[1].values)
        print(np.nanmedian(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,18]).data)*
                    datavars['PRESS'].train_std[2].values+datavars['PRESS'].train_mean[2].values)
        print(np.nanmedian(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,19]).data)*
                    datavars['PRESS'].train_std[3].values+datavars['PRESS'].train_mean[3].values)
        print('DBZ')
        print(np.nanmedian(
            np.nanmax(np.nanmax(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,20]), axis=1), axis=1).data)*
                    datavars['DBZ'].train_std[0].values+datavars['DBZ'].train_mean[0].values)
        print('UH25')
        print(np.nanmedian(
            np.nanmax(np.nanmax(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,21]), axis=1), axis=1).data)*
                    datavars['UH25'].train_std[0].values+datavars['UH25'].train_mean[0].values)
        print('UH03')
        print(np.nanmedian(
            np.nanmax(np.nanmax(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,22]), axis=1), axis=1).data)*
                    datavars['UH03'].train_std[0].values+datavars['UH03'].train_mean[0].values)
        print('WMAX')
        print(np.nanmedian(
            np.nanmax(np.nanmax(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,23]), axis=1), axis=1).data)*
                    datavars['WMAX'].train_std[0].values+datavars['WMAX'].train_mean[0].values)
        print('CTT')
        print(np.nanmedian(
            np.nanmin(np.nanmin(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,24]), axis=1), axis=1).data)*
                    datavars['CTT'].train_std[0].values+datavars['CTT'].train_mean[0].values)

    def table_output_mode(self, testdata, datavars):

        """Output paper's table 2 content, which is the variable means of the storm objects (with masks applied).
        
        Args:
            testdata (numpy array): Test data.
            datavars: Output from ``extract_variable_mean_and_std()``.
        
        """
        print('TK')
        print(stats.mode(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,0]), nan_policy='omit')*
                    datavars['TK'].train_std[0].values+datavars['TK'].train_mean[0].values)
        print(stats.mode(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,1]), nan_policy='omit')*
                    datavars['TK'].train_std[1].values+datavars['TK'].train_mean[1].values)
        print(stats.mode(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,2]), nan_policy='omit')*
                    datavars['TK'].train_std[2].values+datavars['TK'].train_mean[2].values)
        print(stats.mode(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,3]), nan_policy='omit')*
                    datavars['TK'].train_std[3].values+datavars['TK'].train_mean[3].values)
        print('EV')
        print(stats.mode(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,4]), nan_policy='omit')*
                    datavars['EV'].train_std[0].values+datavars['EV'].train_mean[0].values)
        print(stats.mode(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,5]), nan_policy='omit')*
                    datavars['EV'].train_std[1].values+datavars['EV'].train_mean[1].values)
        print(stats.mode(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,6]), nan_policy='omit')*
                    datavars['EV'].train_std[2].values+datavars['EV'].train_mean[2].values)
        print(stats.mode(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,7]), nan_policy='omit')*
                    datavars['EV'].train_std[3].values+datavars['EV'].train_mean[3].values)
        print('EU')
        print(stats.mode(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,8]), nan_policy='omit')*
                    datavars['EU'].train_std[0].values+datavars['EU'].train_mean[0].values)
        print(stats.mode(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,9]), nan_policy='omit')*
                    datavars['EU'].train_std[1].values+datavars['EU'].train_mean[1].values)
        print(stats.mode(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,10]), nan_policy='omit')*
                    datavars['EU'].train_std[2].values+datavars['EU'].train_mean[2].values)
        print(stats.mode(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,11]), nan_policy='omit')*
                    datavars['EU'].train_std[3].values+datavars['EU'].train_mean[3].values)
        print('QVAPOR')
        print(stats.mode(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,12]), nan_policy='omit')*
                    datavars['QVAPOR'].train_std[0].values+datavars['QVAPOR'].train_mean[0].values)
        print(stats.mode(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,13]), nan_policy='omit')*
                    datavars['QVAPOR'].train_std[1].values+datavars['QVAPOR'].train_mean[1].values)
        print(stats.mode(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,14]), nan_policy='omit')*
                    datavars['QVAPOR'].train_std[2].values+datavars['QVAPOR'].train_mean[2].values)
        print(stats.mode(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,15]), nan_policy='omit')*
                    datavars['QVAPOR'].train_std[3].values+datavars['QVAPOR'].train_mean[3].values)
        print('PRESS')
        print(stats.mode(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,16]), nan_policy='omit')*
                    datavars['PRESS'].train_std[0].values+datavars['PRESS'].train_mean[0].values)
        print(stats.mode(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,17]), nan_policy='omit')*
                    datavars['PRESS'].train_std[1].values+datavars['PRESS'].train_mean[1].values)
        print(stats.mode(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,18]), nan_policy='omit')*
                    datavars['PRESS'].train_std[2].values+datavars['PRESS'].train_mean[2].values)
        print(stats.mode(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,19]), nan_policy='omit')*
                    datavars['PRESS'].train_std[3].values+datavars['PRESS'].train_mean[3].values)
        print('DBZ')
        print(stats.mode(
            np.nanmax(np.nanmax(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,20]), axis=1), axis=1).data, nan_policy='omit')*
                    datavars['DBZ'].train_std[0].values+datavars['DBZ'].train_mean[0].values)
        print('UH25')
        print(stats.mode(
            np.nanmax(np.nanmax(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,21]), axis=1), axis=1).data, nan_policy='omit')*
                    datavars['UH25'].train_std[0].values+datavars['UH25'].train_mean[0].values)
        print('UH03')
        print(stats.mode(
            np.nanmax(np.nanmax(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,22]), axis=1), axis=1).data, nan_policy='omit')*
                    datavars['UH03'].train_std[0].values+datavars['UH03'].train_mean[0].values)
        print('WMAX')
        print(stats.mode(
            np.nanmax(np.nanmax(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,23]), axis=1), axis=1).data, nan_policy='omit')*
                    datavars['WMAX'].train_std[0].values+datavars['WMAX'].train_mean[0].values)
        print('CTT')
        print(stats.mode(
            np.nanmin(np.nanmin(np.ma.masked_where(testdata[:,:,:,-1]==0,testdata[:,:,:,24]), axis=1), axis=1).data, nan_policy='omit')*
                    datavars['CTT'].train_std[0].values+datavars['CTT'].train_mean[0].values)

    def table_output_std(self, testdata, datavars):
        
        """Output paper's table 2 content, which is the variable standard deviations of the storm objects (with masks applied).
        
        Args:
            testdata (numpy array): Test data.
            datavars: Output from ``extract_variable_mean_and_std()``.
        
        """
        print('TK')
        print(np.nanstd(np.ma.masked_where(testdata[:,:,:,-1]==0,
                    testdata[:,:,:,0]*datavars['TK'].train_std[0].values+datavars['TK'].train_mean[0].values)))
        print(np.nanstd(np.ma.masked_where(testdata[:,:,:,-1]==0,
                    testdata[:,:,:,1]*datavars['TK'].train_std[1].values+datavars['TK'].train_mean[1].values)))
        print(np.nanstd(np.ma.masked_where(testdata[:,:,:,-1]==0,
                    testdata[:,:,:,2]*datavars['TK'].train_std[2].values+datavars['TK'].train_mean[2].values)))
        print(np.nanstd(np.ma.masked_where(testdata[:,:,:,-1]==0,
                    testdata[:,:,:,3]*datavars['TK'].train_std[3].values+datavars['TK'].train_mean[3].values)))
        print('EV')
        print(np.nanstd(np.ma.masked_where(testdata[:,:,:,-1]==0,
                    testdata[:,:,:,4]*datavars['EV'].train_std[0].values+datavars['EV'].train_mean[0].values)))
        print(np.nanstd(np.ma.masked_where(testdata[:,:,:,-1]==0,
                    testdata[:,:,:,5]*datavars['EV'].train_std[1].values+datavars['EV'].train_mean[1].values)))
        print(np.nanstd(np.ma.masked_where(testdata[:,:,:,-1]==0,
                    testdata[:,:,:,6]*datavars['EV'].train_std[2].values+datavars['EV'].train_mean[2].values)))
        print(np.nanstd(np.ma.masked_where(testdata[:,:,:,-1]==0,
                    testdata[:,:,:,7]*datavars['EV'].train_std[3].values+datavars['EV'].train_mean[3].values)))
        print('EU')
        print(np.nanstd(np.ma.masked_where(testdata[:,:,:,-1]==0,
                    testdata[:,:,:,8]*datavars['EU'].train_std[0].values+datavars['EU'].train_mean[0].values)))
        print(np.nanstd(np.ma.masked_where(testdata[:,:,:,-1]==0,
                    testdata[:,:,:,9]*datavars['EU'].train_std[1].values+datavars['EU'].train_mean[1].values)))
        print(np.nanstd(np.ma.masked_where(testdata[:,:,:,-1]==0,
                    testdata[:,:,:,10]*datavars['EU'].train_std[2].values+datavars['EU'].train_mean[2].values)))
        print(np.nanstd(np.ma.masked_where(testdata[:,:,:,-1]==0,
                    testdata[:,:,:,11]*datavars['EU'].train_std[3].values+datavars['EU'].train_mean[3].values)))
        print('QVAPOR')
        print(np.nanstd(np.ma.masked_where(testdata[:,:,:,-1]==0,
                    testdata[:,:,:,12]*datavars['QVAPOR'].train_std[0].values+datavars['QVAPOR'].train_mean[0].values)))
        print(np.nanstd(np.ma.masked_where(testdata[:,:,:,-1]==0,
                    testdata[:,:,:,13]*datavars['QVAPOR'].train_std[1].values+datavars['QVAPOR'].train_mean[1].values)))
        print(np.nanstd(np.ma.masked_where(testdata[:,:,:,-1]==0,
                    testdata[:,:,:,14]*datavars['QVAPOR'].train_std[2].values+datavars['QVAPOR'].train_mean[2].values)))
        print(np.nanstd(np.ma.masked_where(testdata[:,:,:,-1]==0,
                    testdata[:,:,:,15]*datavars['QVAPOR'].train_std[3].values+datavars['QVAPOR'].train_mean[3].values)))
        print('PRESS')
        print(np.nanstd(np.ma.masked_where(testdata[:,:,:,-1]==0,
                    testdata[:,:,:,16]*datavars['PRESS'].train_std[0].values+datavars['PRESS'].train_mean[0].values)))
        print(np.nanstd(np.ma.masked_where(testdata[:,:,:,-1]==0,
                    testdata[:,:,:,17]*datavars['PRESS'].train_std[1].values+datavars['PRESS'].train_mean[1].values)))
        print(np.nanstd(np.ma.masked_where(testdata[:,:,:,-1]==0,
                    testdata[:,:,:,18]*datavars['PRESS'].train_std[2].values+datavars['PRESS'].train_mean[2].values)))
        print(np.nanstd(np.ma.masked_where(testdata[:,:,:,-1]==0,
                    testdata[:,:,:,19]*datavars['PRESS'].train_std[3].values+datavars['PRESS'].train_mean[3].values)))
        print('DBZ')
        print(np.nanstd(np.ma.masked_where(testdata[:,:,:,-1]==0,
                    testdata[:,:,:,20]*datavars['DBZ'].train_std[0].values+datavars['DBZ'].train_mean[0].values)))
        print('UH25')
        print(np.nanstd(np.ma.masked_where(testdata[:,:,:,-1]==0,
                    testdata[:,:,:,21]*datavars['UH25'].train_std[0].values+datavars['UH25'].train_mean[0].values)))
        print('UH03')
        print(np.nanstd(np.ma.masked_where(testdata[:,:,:,-1]==0,
                    testdata[:,:,:,22]*datavars['UH03'].train_std[0].values+datavars['UH03'].train_mean[0].values)))
        print('WMAX')
        print(np.nanstd(np.ma.masked_where(testdata[:,:,:,-1]==0,
                    testdata[:,:,:,23]*datavars['WMAX'].train_std[0].values+datavars['WMAX'].train_mean[0].values)))
        print('CTT')
        print(np.nanstd(np.ma.masked_where(testdata[:,:,:,-1]==0,
                    testdata[:,:,:,24]*datavars['CTT'].train_std[0].values+datavars['CTT'].train_mean[0].values)))

    def bootstrap_mean_percentiles(self, data, var_str, var_indx, hgt_indx, datavars):

        """Bootstrapped percentiles for the variable mean (with masks applied).
        
        Args:
            data (numpy array): Test data.
            var_str (str): Variable name. E.g., ``TK``.
            var_indx: Index of the variable among all variables.
            hgt_indx: Index of the height among various heights.
            datavars: Output from ``extract_variable_mean_and_std()``.
            
        """
        mean_values = []
        themask = data[:,:,:,-1]
        vardata = data[:,:,:,var_indx]
        maskedarray = np.ma.masked_where(themask==0, vardata)
        for i in range(self.boot_num):
            np.random.seed(seed=i)
            indx = np.random.choice(np.arange(0,len(data[:,0,0,0]),1), size=len(data[:,0,0,0]))
            mean_val = (np.nanmean(maskedarray[indx,:,:]) * datavars[var_str].train_std[hgt_indx].values) +\
                        datavars[var_str].train_mean[hgt_indx].values
            mean_values.append(mean_val)
        return np.nanpercentile(mean_values, q=[self.boot_min, self.boot_max])
    
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

    def bootstrap_max_percentiles(self, data, var_str, var_indx, hgt_indx, datavars):

        """Bootstrapped percentiles for the variable max (with masks applied).
        
        Args:
            data (numpy array): Test data.
            var_str (str): Variable name. E.g., ``TK``.
            var_indx: Index of the variable among all variables.
            hgt_indx: Index of the height among various heights.
            datavars: Output from ``extract_variable_mean_and_std()``.

        """
        max_values = []
        themask = data[:,:,:,-1]
        vardata = data[:,:,:,var_indx]
        maskedarray = np.ma.masked_where(themask==0, vardata)
        for i in range(self.boot_num):
            np.random.seed(seed=i)
            indx = np.random.choice(np.arange(0,len(data[:,0,0,0]),1), size=len(data[:,0,0,0]))
            tempmax = maskedarray[indx,:,:] * datavars[var_str].train_std[hgt_indx].values +\
                      datavars[var_str].train_mean[hgt_indx].values
            max_val = np.nanmax(tempmax)
            max_values.append(max_val)
        return np.nanpercentile(max_values, q=[self.boot_min, self.boot_max])

    def bootstrap_min_percentiles(self, data, var_str, var_indx, hgt_indx, datavars):

        """Bootstrapped percentiles for the variable min (with masks applied).
        
        Args:
            data (numpy array): Test data.
            var_str (str): Variable name. E.g., ``TK``.
            var_indx: Index of the variable among all variables.
            hgt_indx: Index of the height among various heights.
            datavars: Output from ``extract_variable_mean_and_std()``.

        """
        mean_values = []
        themask = data[:,:,:,-1]
        vardata = data[:,:,:,var_indx]
        maskedarray = np.ma.masked_where(themask==0, vardata)
        for i in range(self.boot_num):
            np.random.seed(seed=i)
            indx = np.random.choice(np.arange(0,len(data[:,0,0,0]),1), size=len(data[:,0,0,0]))
            mean_val = (np.nanmin(maskedarray[indx,:,:]) * datavars[var_str].train_std[hgt_indx].values) +\
                        datavars[var_str].train_mean[hgt_indx].values
            mean_values.append(mean_val)
        return np.nanpercentile(mean_values, q=[self.boot_min, self.boot_max])
