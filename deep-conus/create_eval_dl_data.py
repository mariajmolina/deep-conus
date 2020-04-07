import xarray as xr
import numpy as np
import pandas as pd


class CreateEvaluationData:
    
    """Class instantiation of CreateEvaluationData:
    
    Here we create subsets of test data for evaluating the deep learning model.
    
    Attributes:
        climate (str): Whether to interpolate variable in the ``current`` or ``future`` climate simulation.
        method (str): Method for parsing data. Options include ``random``, ``month``, ``season``, ``year``.
        variables (str): Numpy array of variable name strings. Options include ``EU``, ``EV``, ``TK``, ``QVAPOR``, ``WMAX``, 
                         ``W_vert``,``PRESS``,``DBZ``,``CTT``,``UH25``, and``UH03``.
        directory (str): Directory where the deep learning files are saved and where these test data subsets will be saved.
        mask (boolean): Whether to train using the masked data or the non-masked data. Defaults to False.
        season_choice (str): Three-month season string, if ``method==season`` (e.g., 'DJF'). Defaults to ``None``.
        
    Raises:
        Exceptions: Checks whether correct values were input for ``climate``, ``method``, and ``project_code``.
        
    Todo:
        * Add method for ``month``, ``season``, and ``year``.
    
    """
    
    def __init__(self, climate, method, variables, directory, mask=False, season_choice=None):
        
        if climate!='current' and climate!='future':
            raise Exception("Please enter current or future as string for climate period selection.")
        if climate=='current' or climate=='future':
            self.climate=climate
            
        if method!='random' and method!='month' and method!='season' and method!='year':
            raise Exception("Please enter ``random``, ``month``, ``season``, or ``year`` as method.")
        if method=='random' or method=='month' or method=='season' or method=='year':
            self.method=method
            
        self.variables=variables
        self.directory=directory
        
        self.mask=mask
        if not self.mask:
            self.mask_str='nomask'
        if self.mask:
            self.mask_str='mask'
        
        
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

            
    def open_files(self):

        """Open the testing data files and apply the respective parsing method to create the test subset.
            
        """
        all_data={}
        
        for var in self.variables:
            print(f"Opening {var}...")
            data=xr.open_mfdataset(f'/{self.directory}/{self.climate}_{self.variable_translate(var).lower()}_{self.mask_str}_dldata_traintest.nc',
                                   parallel=True, combine='by_coords')
            
            if self.method=='random':
                self.random_method(data, var)
                
            if self.method=='month':
                self.month_method(data, var)
                
            if self.method=='season':
                self.season_method(data, var)
                
            if self.method=='year':
                self.year_method(data, var)
                
        return
    
    
    def random_method(self, data, variable):
        
        """Randomizes and parses the test data into 6 groups, saving each individually to avoid memory issues during evaluation.
        
        Args:
            data (Xarray data array): Test data for respective variable.
            variable (str): The variable being processed and saved.
        
        """
        for num, (i, j) in enumerate(zip([0,100000,200000,300000,400000,500000], [100000,200000,300000,400000,500000,600000])):
            np.random.seed(0)
            select_data=np.random.permutation(data.coords['b'].shape[0])[i:j]
            print(f"Opening {num+1}...")
            data_assemble = xr.Dataset({'X_test':(['b','x','y','features'], data.X_test.transpose('b','x','y','features')[select_data,:,:,:]),
                                        'X_test_label':(['b'], data.X_test_label[select_data])})
            print(f"Saving {num+1}...")
            data_assemble.to_netcdf(f'/{self.directory}/{self.climate}_{self.variable_translate(variable).lower()}_{self.mask_str}_{self.method}_test{num+1}.nc')
        data_assemble=data_assemble.close()
        data=data.close()
        return
    
    
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
            
    
    def month_method(self, data, variable):
        
        """Parses the test data into monthly groups, saving each individually to avoid memory issues during evaluation.
        
        Args:
            data (Xarray data array): Test data for respective variable.
            variable (str): The variable being processed and saved.
        
        """
        return
    
    
    def season_method(self, data, variable):
        
        """Parses the test data into seasonal groups, saving each individually to avoid memory issues during evaluation.
        
        Args:
            data (Xarray data array): Test data for respective variable.
            variable (str): The variable being processed and saved.
        
        """
        return
    
    
    def year_method(self, data, variable):
        
        """Parses the test data into yearly groups, saving each individually to avoid memory issues during evaluation.
        
        Args:
            data (Xarray data array): Test data for respective variable.
            variable (str): The variable being processed and saved.
        
        """
        return
    
