#####################################################################################
#####################################################################################
#
# Author: Maria J. Molina
# National Center for Atmospheric Research
#
#####################################################################################
#####################################################################################


#----------------------------------------------------------

import xarray as xr
import numpy as np
from ncar_jobqueue import NCARCluster
from dask.distributed import Client

#----------------------------------------------------------


class split_and_standardize:
    
    """Class instantiation of split_and_standardize:

    Here we will be preprocessing data for deep learning model training.
    This module includes methods for training and testing data splits and standardization.

    Attributes:
        climate (str): The climate period to derive deep learning data for; ``current`` or ``future``.
        variable (str): Variable to run script the for, which can include ``TK``, ``EV``, ``EU``, ``QVAPOR``, 
                        ``PRESS``, ``W_vert``, ``WMAX``, ``DBZ``, ``CTT``, ``UH25``, or ``UH03``.
        percent_split (float): Percentage of total data to assign as training data. The remaining data will be 
                                assigned as testing data. For example, 0.6 is 60% training data, 40% testing data.
        working_directory (str): The directory path to where the produced files will be saved and worked from.
        threshold1 (int): The threshold for used for the chosen classification method (e.g., 75 UH25).
        mask (boolean): Whether the threshold was applied within the storm patch mask or not. Defaults to ``False``.
            
    Raises:
        Exceptions: Checks whether correct values were input for climate, variable, and percent_split.
            
    Todo:
        * Add second threshold option once it is added as to ``preprocess_dl_data.py``.
        
    """
    
    def __init__(self, climate, variable, percent_split, working_directory, threshold1, mask=False):

        if climate!='current' and climate!='future':
            raise Exception("Please enter current or future for climate option.")
        if climate=='current' or climate=='future':
            self.climate=climate
            
        if variable!='TK' and variable!='EV' and variable!='EU' and variable!='QVAPOR' and variable!='PRESS' and variable!='W_vert' \
        and variable!='WMAX' and variable!='DBZ' and variable!='CTT' and variable!='UH25' and variable!='UH03':
            raise Exception("Please enter TK, EV``, EU, QVAPOR, PRESS, W_vert, UH25, UH03, MAXW, CTT, DBZ as variable.")
            
        if variable=='TK' or variable=='EV' or variable=='EU' or variable=='QVAPOR' or variable=='PRESS' or variable=='W_vert' or \
        variable=='WMAX' or variable=='DBZ' or variable=='CTT' or variable=='UH25' or variable=='UH03':
            self.variable=variable
            
            if self.variable=="TK":
                self.choice_var1="temp_sev_1"
                self.choice_var3="temp_sev_3"
                self.choice_var5="temp_sev_5"
                self.choice_var7="temp_sev_7"
                self.attrs_array=np.array(["tk_1km", "tk_3km", "tk_5km", "tk_7km"])
                self.single=False

            if self.variable=="EV":
                self.choice_var1="evwd_sev_1"
                self.choice_var3="evwd_sev_3"
                self.choice_var5="evwd_sev_5"
                self.choice_var7="evwd_sev_7"
                self.attrs_array=np.array(["ev_1km", "ev_3km", "ev_5km", "ev_7km"])
                self.single=False

            if self.variable=="EU":
                self.choice_var1="euwd_sev_1"
                self.choice_var3="euwd_sev_3"
                self.choice_var5="euwd_sev_5"
                self.choice_var7="euwd_sev_7"
                self.attrs_array=np.array(["eu_1km", "eu_3km", "eu_5km", "eu_7km"])
                self.single=False

            if self.variable=="QVAPOR":
                self.choice_var1="qvap_sev_1"
                self.choice_var3="qvap_sev_3"
                self.choice_var5="qvap_sev_5"
                self.choice_var7="qvap_sev_7"
                self.attrs_array=np.array(["qv_1km", "qv_3km", "qv_5km", "qv_7km"])
                self.single=False

            if self.variable=="PRESS":
                self.choice_var1="pres_sev_1"
                self.choice_var3="pres_sev_3"
                self.choice_var5="pres_sev_5"
                self.choice_var7="pres_sev_7"
                self.attrs_array=np.array(["pr_1km", "pr_3km", "pr_5km", "pr_7km"])
                self.single=False

            if self.variable=="W_vert":
                self.choice_var1="wwnd_sev_1"
                self.choice_var3="wwnd_sev_3"
                self.choice_var5="wwnd_sev_5"
                self.choice_var7="wwnd_sev_7"
                self.attrs_array=np.array(["ww_1km", "ww_3km", "ww_5km", "ww_7km"])
                self.single=False

            if self.variable=="WMAX":
                self.choice_var1="maxw_sev_1"
                self.attrs_array=np.array(["maxw"]) 
                self.single=True
                
            if self.variable=="DBZ":
                self.choice_var1="dbzs_sev_1"
                self.attrs_array=np.array(["dbzs"])
                self.single=True
                
            if self.variable=="CTT":
                self.choice_var1="ctts_sev_1"
                self.attrs_array=np.array(["ctts"]) 
                self.single=True
                
            if self.variable=="UH25":
                self.choice_var1="uh25_sev_1"
                self.attrs_array=np.array(["uh25"]) 
                self.single=True
                
            if self.variable=="UH03":
                self.choice_var1="uh03_sev_1"
                self.attrs_array=np.array(["uh03"])    
                self.single=True
            
        if percent_split>=1:
            raise Exception("Percent split should be a float less than 1.")
        if percent_split<1:
            self.percent_split=percent_split
            
        self.working_directory=working_directory
        self.threshold1=threshold1
        
        self.mask=mask
        if not self.mask:
            self.mask_str='nomask'
        if self.mask:
            self.mask_str='mask'
    
    

    def variable_translate(self):
        
        """Variable name for the respective filenames.
        
        Returns:
            variable (str): The variable string used to save files.
            
        Raises:
            ValueError: Input variable must be from available list.
        
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
            out=var[self.variable]
            return out
        except:
            raise ValueError("Please enter TK, EU, EV, QVAPOR, P, W_vert, or WMAX as variable.")
        
    
    
    def open_above_threshold(self):
        
        """Open and concat files for the six months of analysis (threshold exceedance).
        
        Returns:
            data (Xarray dataset): Concatenated six months of data.
        
        """
        data_dec=xr.open_mfdataset(f"/{self.working_directory}/{self.climate}_uh{self.threshold1}_{self.mask_str}_12.nc", 
                                           parallel=True, combine='by_coords')
        data_jan=xr.open_mfdataset(f"/{self.working_directory}/{self.climate}_uh{self.threshold1}_{self.mask_str}_01.nc",
                                           parallel=True, combine='by_coords')
        data_feb=xr.open_mfdataset(f"/{self.working_directory}/{self.climate}_uh{self.threshold1}_{self.mask_str}_02.nc", 
                                           parallel=True, combine='by_coords')
        data_mar=xr.open_mfdataset(f"/{self.working_directory}/{self.climate}_uh{self.threshold1}_{self.mask_str}_03.nc", 
                                           parallel=True, combine='by_coords')
        data_apr=xr.open_mfdataset(f"/{self.working_directory}/{self.climate}_uh{self.threshold1}_{self.mask_str}_04.nc", 
                                           parallel=True, combine='by_coords')
        data_may=xr.open_mfdataset(f"/{self.working_directory}/{self.climate}_uh{self.threshold1}_{self.mask_str}_05.nc", 
                                           parallel=True, combine='by_coords')
        data=xr.concat([data_dec, data_jan, data_feb, data_mar, data_apr, data_may], dim='patch')
        data_dec=data_dec.close()
        data_jan=data_jan.close()
        data_feb=data_feb.close()
        data_mar=data_mar.close()
        data_apr=data_apr.close()
        data_may=data_may.close()
        return data
    
    
        
    def open_below_threshold(self):
        
        """Open and concat files for six months of analysis (threshold non-exceedance).
        
        Returns:
            data (Xarray dataset): Concatenated six months of data.
            
        """        
        data_dec=xr.open_mfdataset(f"/{self.working_directory}/{self.climate}_nonuh{self.threshold1}_{self.mask_str}_12.nc",
                                          parallel=True, combine='by_coords')
        data_jan=xr.open_mfdataset(f"/{self.working_directory}/{self.climate}_nonuh{self.threshold1}_{self.mask_str}_01.nc",
                                           parallel=True, combine='by_coords')
        data_feb=xr.open_mfdataset(f"/{self.working_directory}/{self.climate}_nonuh{self.threshold1}_{self.mask_str}_02.nc",
                                           parallel=True, combine='by_coords')
        data_mar=xr.open_mfdataset(f"/{self.working_directory}/{self.climate}_nonuh{self.threshold1}_{self.mask_str}_03.nc",
                                           parallel=True, combine='by_coords')
        data_apr=xr.open_mfdataset(f"/{self.working_directory}/{self.climate}_nonuh{self.threshold1}_{self.mask_str}_04.nc",
                                           parallel=True, combine='by_coords')
        data_may=xr.open_mfdataset(f"/{self.working_directory}/{self.climate}_nonuh{self.threshold1}_{self.mask_str}_05.nc",
                                           parallel=True, combine='by_coords')
        data=xr.concat([data_dec, data_jan, data_feb, data_mar, data_apr, data_may], dim='patch')
        data_dec=data_dec.close()
        data_jan=data_jan.close()
        data_feb=data_feb.close()
        data_mar=data_mar.close()
        data_apr=data_apr.close()
        data_may=data_may.close()
        return data


        
    def grab_variables(self, data):
        
        """Eagerly load variable data. This function converts dask arrays into numpy arrays.
        
        Args:
            data (Xarray dataset): The original Xarray dataset containing dask arrays.
            
        Returns:
            data_1, data_2, data_3, data_4 or data_1 (numpy array(s)): Input data as numpy arrays.
            
        """
        if not self.single:
            data_1=data[self.choice_var1].values
            data_2=data[self.choice_var3].values
            data_3=data[self.choice_var5].values
            data_4=data[self.choice_var7].values
            return data_1, data_2, data_3, data_4
        if self.single:
            data_1=data[self.choice_var1].values
            return data_1
        
        
        
    def create_traintest_data(self, data_b, data_a, return_label=False):
        
        """This function performs balancing of above and below threshold data for training and testing data. Data is permuted 
        before being assigned to training and testing groups. 
        The training group sample size is computed using the assigned percentage (``self.percent_split``) from the above threshold population. 
        Then, the testing group sample size is computed using the leftover percentage (e.g., 1-``self.percent_split``) from a population 
        with a similar ratio of above and below threshold storm patches (e.g., ~5% above threshold to 95% below threshold). This is done
        artificially balance the ratio of threshold exceeding storms to that of non-exceeding storms, to ensure that the training data set 
        contains sufficient examples of above threshold storm patches, given that they are rare events. The testing data set is left with 
        a population of storms that resembles the original data's population.
        
        Args:
            data_b (numpy array): Concatenated six months of data exceeding the threshold.
            data_a (numpy array): Concatenated six months of data below the threshold.
            return_label (boolean): Whether to return the label data or not. Defaults to ``False``.
            
        Returns:
            train_data, test_data or train_data, test_data, train_label, test_label (numpy arrays): The training and testing data, and if
            return_label=``True``, the training and testing data labels for supervised learning.
            
        """
        #train above
        np.random.seed(0)
        select_data=np.random.permutation(data_a.shape[0])[:int(data_a.shape[0]*self.percent_split)]
        train_above=data_a[select_data]
        #train below
        np.random.seed(0)
        select_data=np.random.permutation(data_b.shape[0])[:int(data_a.shape[0]*self.percent_split)]
        train_below=data_b[select_data]
        #test above
        np.random.seed(0)
        select_data=np.random.permutation(data_a.shape[0])[int(data_a.shape[0]*self.percent_split):]
        test_above=data_a[select_data]
        #test below
        np.random.seed(0)
        #slicing to get respective ratio of above to below UH data patches
        select_data=np.random.permutation(
            data_b.shape[0])[int(data_a.shape[0] * self.percent_split):int((((data_a.shape[0] * (1-self.percent_split)) * data_b.shape[0]) \
                                                                            / data_a.shape[0]) + (data_a.shape[0] * (1 - self.percent_split)))]
        test_below=data_b[select_data]

        #create the label data
        train_above_label=np.ones(train_above.shape[0])
        train_below_label=np.zeros(train_below.shape[0])
        test_above_label=np.ones(test_above.shape[0])
        test_below_label=np.zeros(test_below.shape[0])

        #merge above and below data in prep to shuffle/permute
        train_data=np.vstack([train_above, train_below])
        if return_label:
            train_label=np.hstack([train_above_label, train_below_label])
        test_data=np.vstack([test_above, test_below])
        if return_label:
            test_label=np.hstack([test_above_label, test_below_label])

        #finally, permute the data that has been merged and properly balanced
        np.random.seed(10)
        train_data=np.random.permutation(train_data)
        np.random.seed(10)
        test_data=np.random.permutation(test_data)
        
        if not return_label:
            return train_data, test_data  
        if return_label:
            np.random.seed(10)
            train_label=np.random.permutation(train_label)
            np.random.seed(10)
            test_label=np.random.permutation(test_label)    
            return train_data, test_data, train_label, test_label


    
    def minmax_scale_apply(self, data):
        
        """Min-max standardization of the training data.
        
        Args:
            data (numpy array): Input data to standardize.
            
        Returns:
            data (numpy array): Input data standardized.
        
        """
        from sklearn.preprocessing import MinMaxScaler
        scaler=MinMaxScaler(feature_range=(0, 1))
        scaler.fit(data)
        return scaler.transform(data)
    
    
    
    def minmax_scale_apply_test(self, train, test):
        
        """Min-max standardization of the testing data using training data minimum and maximum values.
        
        Args:
            train (numpy array): Input training data for min-max values.
            test (numpy array): Input testing data to standardize.
            
        Returns:
            test (numpy array): Input testing data standardized.
        
        """
        from sklearn.preprocessing import MinMaxScaler
        scaler=MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train)
        return scaler.transform(test)

    

    def standardize_scale_apply(self, data):
        
        """Z-score standardization of the training data.
        
        Args:
            data (numpy array): Input data to standardize.
            
        Returns:
            data (numpy array): Input data standardized.
            
        """
        return np.divide((data - np.nanmean(data)), np.nanstd(data))



    def standardize_scale_apply_test(self, train, test):
        
        """Z-score standardization of the test data using the training data mean and standard deviation.
        
        Args:
            train (numpy array): Input training data for Z-score distribution values (mean and standard deviation).
            test (numpy array): Input testing data to standardize.
            
        Returns:
            test (numpy array): Input testing data standardized.
            
        """
        return np.divide((test - np.nanmean(train)), np.nanstd(train))

    
        
    def split_data_to_traintest(self, below1=None, below2=None, below3=None, below4=None, above1=None, above2=None, above3=None, above4=None):
        
        """Function that applies ``create_traintest_data()`` to various variables.
        
        Args:
            below1 (numpy array): Storm patch data that does not exceed the threshold. If there are multiple ``below`` arrays, they arranged 
                                  from low-to-high heights above ground level. Defaults to ``None``.
            below2 (numpy array): Storm patch data that does not exceed the threshold. Defaults to ``None``.
            below3 (numpy array): Storm patch data that does not exceed the threshold. Defaults to ``None``.
            below4 (numpy array): Storm patch data that does not exceed the threshold. Defaults to ``None``.
            above1 (numpy array): Storm patch data that exceeds the threshold. If there are multiple ``above`` arrays, they arranged from
                                  low-to-high heights above ground level. Defaults to ``None``.
            above2 (numpy array): Storm patch data that exceeds the threshold. Defaults to ``None``.
            above3 (numpy array): Storm patch data that exceeds the threshold. Defaults to ``None``.
            above4 (numpy array): Storm patch data that exceeds the threshold. Defaults to ``None``.
            
        Returns:
            train1, train2, train3, train4, train_label, test1, test2, test3, test4, test_label or train1, test1, train_label, test_label (numpy arrays):
            The input data variables assembled into training and testing datasets. The corresponding labels are also output. Number of output arrays
            depends on the variable type (e.g., if interpolated across various heights or if a single variable).
        
        """
        if not self.single:
            train1, test1, train_label, test_label=self.create_traintest_data(below1, above1, return_label=True)
            train2, test2=self.create_traintest_data(below2, above2, return_label=False)
            train3, test3=self.create_traintest_data(below3, above3, return_label=False)
            train4, test4=self.create_traintest_data(below4, above4, return_label=False)
            return train1, train2, train3, train4, train_label, test1, test2, test3, test4, test_label
        if self.single:
            train1, test1, train_label, test_label=self.create_traintest_data(below1, above1, return_label=True)
            return train1, test1, train_label, test_label
        


    def standardize_training(self, func, data1, data2=None, data3=None, data4=None):
        
        """Function to standardize the training data.
        
        Args:
            func (class method): The choice of standardization.
            data1 (numpy array): Data to be standardized.
            data2 (numpy array): Data to be standardized. Defaults to ``None``.
            data3 (numpy array): Data to be standardized. Defaults to ``None``.
            data4 (numpy array): Data to be standardized. Defaults to ``None``.
            
        Returns:
            data_scaled1, data_scaled2, data_scaled3, data_scaled4 or data_scaled1 (numpy array(s)): The training data standardized.
            
        """
        data_scaled1=func(data1)
        if not self.single:
            data_scaled2=func(data2)
            data_scaled3=func(data3)
            data_scaled4=func(data4)
            return data_scaled1, data_scaled2, data_scaled3, data_scaled4
        if self.single:
            return data_scaled1
        
        
    
    def standardize_testing(self, func, train1=None, train2=None, train3=None, train4=None, 
                                        test1=None, test2=None, test3=None, test4=None):
        
        """Function to standardize the testing data.
        
        Args:
            func (class method): The choice of standardization.
            train1 (numpy array): Training data for standardization of testing data.
            train2 (numpy array): Training data for standardization of testing data. Defaults to ``None``.
            train3 (numpy array): Training data for standardization of testing data. Defaults to ``None``.
            train4 (numpy array): Training data for standardization of testing data. Defaults to ``None``.
            test1 (numpy array): Testing data for standardization.
            test2 (numpy array): Testing data for standardization. Defaults to ``None``.
            test3 (numpy array): Testing data for standardization. Defaults to ``None``.
            test4 (numpy array): Testing data for standardization. Defaults to ``None``.
            
        Returns:
            data1, data2, data3, data4 or data1 (numpy array(s)): The testing data standardized.
            
        """
        data1=func(train1, test1)
        if not self.single:
            data2=func(train2, test2)
            data3=func(train3, test3)
            data4=func(train4, test4)
            return data1, data2, data3, data4
        if self.single:
            return data1



    def stack_the_data(self, data1, data2, data3, data4):
        
        """Stack the numpy arrays before assembling final xarray netcdf file for saving.
        
        Args:
            data1 (numpy array): Data to be stacked. Arrange from lowest (``data1``) to highest (``data2``) vertical heights.
            data2 (numpy array): Data to be stacked.
            data3 (numpy array): Data to be stacked.
            data4 (numpy array): Data to be stacked.
            
        Returns:
            totaldata (numpy array): Stacked data variables.
            
        """
        if not self.single:
            totaldata=np.stack([data1, data2, data3, data4])
            return totaldata



    def save_data(self, train_data, train_label, test_data, test_label):
        
        """Creates and saves the file that contains the training and testing standardized data for deep learning model training and 
        evaluation. The file contains data for one variable for ease of use later and storage space considerations.
        
        Args:
            train_data (numpy array): The training data.
            train_label (numpy array): The training data labels.
            test_data (numpy array): The testing data.
            test_label (numpy array): The testing data labels.
            
        """
        if not self.single: 
            data_assemble=xr.Dataset({
                'X_train':(['features','a','x','y'], train_data),
                'X_train_label':(['a'], train_label),
                'X_test':(['features','b','x','y'], test_data),
                'X_test_label':(['b'], test_label),
                },
                 coords=
                {'feature':(['features'],self.attrs_array),
                })
        if self.single:
            data_assemble=xr.Dataset({
                'X_train':(['a','x','y'], train_data),
                'X_train_label':(['a'], train_label),
                'X_test':(['b','x','y'], test_data),
                'X_test_label':(['b'], test_label),
                },
                 coords=
                {'feature':(['features'],self.attrs_array),
                })
        data_assemble.to_netcdf(f"/{self.working_directory}/{self.climate}_{self.variable_translate().lower()}_{self.mask_str}_dldata_traintest.nc")
        print(f"File saved ({self.climate}, {self.variable_translate().lower()}, {self.mask_str}).")


    
    def run_sequence(self):
        
        """Function that runs through the sequence of steps in data preprocessing for deep learning model training and testing data creation.
        
        """
        print("Opening files...")
        data_above=self.open_above_threshold()
        data_below=self.open_below_threshold()

        if not self.single:
        
            print("Grabbing variables...")
            above_1, above_2, above_3, above_4=self.grab_variables(data_above)
            below_1, below_2, below_3, below_4=self.grab_variables(data_below)
            
            data_above=data_above.close()
            data_below=data_below.close()
            
            print("Splitting data...")
            train1, train2, train3, train4, train_label, test1, test2, test3, test4, test_label=self.split_data_to_traintest(
                below_1, below_2, below_3, below_4, above_1, above_2, above_3, above_4)
            
            above_1=None; above_2=None; above_3=None; above_4=None
            below_1=None; below_2=None; below_3=None; below_4=None

            print("Standardizing testing...")
            test1, test2, test3, test4=self.standardize_testing(
                self.standardize_scale_apply_test, train1, train2, train3, train4, test1, test2, test3, test4)
            
            print("Standardizing training...")
            train1, train2, train3, train4=self.standardize_training(self.standardize_scale_apply, train1, train2, train3, train4)
            
            print("Stacking files...")
            Xtrain=self.stack_the_data(train1, train2, train3, train4)
            Xtest=self.stack_the_data(test1, test2, test3, test4)

            train1=None; train2=None; train3=None; train4=None
            test1=None;  test2=None;  test3=None;  test4=None
            
        if self.single:
            
            print("Grabbing variables...")
            above_1=self.grab_variables(data_above)
            below_1=self.grab_variables(data_below)   
            
            data_above=data_above.close()
            data_below=data_below.close()
            
            print("Splitting data...")
            train1, test1, train_label, test_label=self.split_data_to_traintest(below1=below_1, above1=above_1)
            
            above_1=None; below_1=None

            print("Standardizing testing...")
            test1=self.standardize_testing(self.standardize_scale_apply_test, train1=train1, test1=test1)
            
            print("Standardizing training...")
            train1=self.standardize_training(self.standardize_scale_apply, data1=train1)
            
            print("Stacking files...")
            Xtrain=train1
            Xtest=test1

            train1=None; test1=None
        
        print("Saving file...")
        self.save_data(Xtrain, train_label, Xtest, test_label) 
        
        Xtrain=None; Xtest=None; train_label=None; test_label=None
        
