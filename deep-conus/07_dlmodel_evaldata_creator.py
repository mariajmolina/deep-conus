import xarray as xr
import numpy as np
import pandas as pd

class CreateEvaluationData:
    
    """Class instantiation of CreateEvaluationData:
    
    Here we create subsets of test data for evaluating the deep learning model. 
    Data subsets will be saved as a file per variable.
    
    Attributes:
        climate (str): Whether to interpolate variable in the ``current`` or ``future`` climate simulation.
        variables (str): Numpy array of variable name strings. Options include ``EU``, ``EV``, ``TK``, ``QVAPOR``, ``WMAX``, 
                         ``W_vert``,``PRESS``,``DBZ``,``CTT``,``UH25``, and``UH03``.
        directory (str): Directory where the deep learning files are saved and where these test data subsets will be saved.
        unbalanced (boolean): Whether training data will be artificially balanced (``False``) or left unbalanced (``True``). Defaults to ``False``.
        kfold_total
        kfold_indx
        use_kfold
        
    Raises:
        Exceptions: Checks whether correct values were input for ``climate``, ``method``, and ``project_code``.
    
    """
    def __init__(self, climate, variables, directory, unbalanced=False, kfold_total=5, kfold_indx=None, use_kfold=False):
        
        # sanity check
        if climate!='current' and climate!='future':
            raise Exception("Please enter current or future as string for climate period selection.")
        else:
            self.climate=climate
            
        # class attributes
        self.variables=variables
        self.directory=directory
        self.unbalanced=unbalanced
        
        # string help
        self.method='random'
        self.mask=False
        self.mask_str='nomask'
        
        # for k-fold cross validation
        self.use_kfold=use_kfold
        if self.use_kfold:
            self.kfold_total=kfold_total
            self.kfold_indx=kfold_indx

    def add_dbz(self):
        
        """Function that adds ``DBZ`` variable to variable list if not already contained.
        
        """
        if not np.isin('DBZ', self.variables):
            self.variables=np.append(self.variables, 'DBZ')

    def add_uh25(self):
        
        """Function that adds ``UH25`` variable to variable list if not already contained.
        
        """
        if not np.isin('UH25', self.variables):
            self.variables=np.append(self.variables, 'UH25')

    def add_uh03(self):
        
        """Function that adds ``UH03`` variable to variable list if not already contained.
        
        """
        if not np.isin('UH03', self.variables):
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
        
        """Function that adds ``UH03`` variable to variable list if not already contained.
        
        """
        if not np.isin('MASK', self.variables):
            self.variables=np.append(self.variables, 'MASK')

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
            raise ValueError("Please enter ``TK``, ``EV``, ``EU``, ``QVAPOR``, ``PRESS``, ``W_vert``, ``UH25``, ``UH03``, ``MAXW``, ``CTT``, ``MASK``, or ``DBZ`` as variable.")

    def open_files_and_run_method(self):

        """Open the testing data files and apply the respective parsing method to create the test subset.
            
        """
        # run methods
        self.add_dbz()
        self.add_uh25()
        self.add_uh03()
        self.add_wmax()
        self.add_ctt()
        self.add_mask()
        
        for var in self.variables:
            
            print(f"Opening {var}...")
            
            if not self.unbalanced:
                    
                data=xr.open_mfdataset(
                    f'/{self.directory}/{self.climate}_{self.variable_translate(var).lower()}_{self.mask_str}_dldata_traintest.nc',
                    parallel=True, combine='by_coords')
                    
            if self.unbalanced:
                
                if not self.use_kfold:
                    
                    data=xr.open_mfdataset(
                        f'/{self.directory}/{self.climate}_{self.variable_translate(var).lower()}_{self.mask_str}_dldata_traintest_unbalanced.nc',
                        parallel=True, combine='by_coords')
                    
                if self.use_kfold:
                    
                    data=xr.open_mfdataset(
                        f'/{self.directory}/{self.climate}_{self.variable_translate(var).lower()}_{self.mask_str}_dldata_traintest_unbalanced_k{self.kfold_indx}.nc',
                        parallel=True, combine='by_coords')
                    
            if self.method=='random':
                
                self.random_method(data, var)
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
            
            if len(data.X_test.dims)==4:
                
                data_assemble=xr.Dataset({'X_test':(['b','x','y','features'], data.X_test.transpose('b','x','y','features')[select_data,:,:,:]),
                                          'X_test_label':(['b'], data.X_test_label[select_data])})
                
            if len(data.X_test.dims)==3:
                
                data_assemble=xr.Dataset({'X_test':(['b','x','y','features'], data.X_test.transpose('b','x','y')[select_data,:,:].expand_dims('features',axis=3)),
                                          'X_test_label':(['b'], data.X_test_label[select_data])})
                
            print(f"Saving {num+1}...")
            
            if not self.unbalanced:
                
                data_assemble.to_netcdf(
                    f'/{self.directory}/{self.climate}_{self.variable_translate(variable).lower()}_{self.mask_str}_{self.method}_test{num+1}.nc')
                    
            if self.unbalanced:
                
                if not self.use_kfold:
                
                    data_assemble.to_netcdf(
                        f'/{self.directory}/{self.climate}_{self.variable_translate(variable).lower()}_{self.mask_str}_{self.method}_test{num+1}_unbalanced.nc')
                    
                if self.use_kfold:
                    
                    data_assemble.to_netcdf(
                        f'/{self.directory}/{self.climate}_{self.variable_translate(variable).lower()}_{self.mask_str}_{self.method}_test{num+1}_unbalanced_k{self.kfold_indx}.nc')
                                        
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
