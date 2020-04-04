
#####################################################################################
#####################################################################################
#
# Author: Maria J. Molina
# National Center for Atmospheric Research
#
# Script to split data into training and testing sets, and standardize, for deep learning model training. 
#
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
    
    
    def __init__(self, climate, variable, percent_split, working_directory, threshold1, mask = False):
        
        
        """

        Instantiation of split_and_standardize:

        Here we will be preprocessing data for deep learning model training.
        This module includes methods for training and testing data splits and standardization.

        PARAMETERS
        ----------
        climate: climate period to derive deep learning data for (str; current or future)
        variable: variable to run script for (str), including TK, EV, EU, QVAPOR, PRESS, W_vert, WMAX, DBZ, CTT, UH25, and UH03.
        percent_split: percent for training data, remaining will be assigned for testing (float) E.g., 0.6 is 60% for training, 40% for testing.
        working_directory: path to directory where DL preprocessing files will be saved and worked from (str)
        threshold1: threshold for method (int)
        mask: whether the threshold will be applied within the storm patch mask or not (boolean; default False)
        
        """
        
        if climate != 'current' and climate != 'future':
            raise Exception("Please enter current or future for climate option.")
        if climate == 'current' or climate == 'future':
            self.climate = climate
            
        if percent_split>=1:
            raise Exception("Percent split should be a float less than 1.")
        if percent_split<1:
            self.percent_split = percent_split
            
        self.working_directory = working_directory
        self.threshold1 = threshold1
        
        self.mask = mask
        if not self.mask:
            self.mask_str = 'nomask'
        if self.mask:
            self.mask_str = 'mask'

        if variable!='TK' and variable!='EV' and variable!='EU' and variable!='QVAPOR' and variable!='PRESS' and variable!='W_vert' \
        and variable!='WMAX' and variable!='DBZ' and variable!='CTT' and variable!='UH25' and variable!='UH03':
            raise Exception("Please enter TK, EV``, EU, QVAPOR, PRESS, W_vert, UH25, UH03, MAXW, CTT, DBZ as variable.")
            
        if variable=='TK' or variable=='EV' or variable=='EU' or variable=='QVAPOR' or variable=='PRESS' or variable=='W_vert' or \
        variable=='WMAX' or variable=='DBZ' or variable=='CTT' or variable=='UH25' or variable=='UH03':
            self.variable = variable
            
            if self.variable == "TK":
                self.choice_var1 = "temp_sev_1"
                self.choice_var3 = "temp_sev_3"
                self.choice_var5 = "temp_sev_5"
                self.choice_var7 = "temp_sev_7"
                self.attrs_array = np.array(["tk_1km", "tk_3km", "tk_5km", "tk_7km"])
                self.single = False

            if self.variable == "EV":
                self.choice_var1 = "evwd_sev_1"
                self.choice_var3 = "evwd_sev_3"
                self.choice_var5 = "evwd_sev_5"
                self.choice_var7 = "evwd_sev_7"
                self.attrs_array = np.array(["ev_1km", "ev_3km", "ev_5km", "ev_7km"])
                self.single = False

            if self.variable == "EU":
                self.choice_var1 = "euwd_sev_1"
                self.choice_var3 = "euwd_sev_3"
                self.choice_var5 = "euwd_sev_5"
                self.choice_var7 = "euwd_sev_7"
                self.attrs_array = np.array(["eu_1km", "eu_3km", "eu_5km", "eu_7km"])
                self.single = False

            if self.variable == "QVAPOR":
                self.choice_var1 = "qvap_sev_1"
                self.choice_var3 = "qvap_sev_3"
                self.choice_var5 = "qvap_sev_5"
                self.choice_var7 = "qvap_sev_7"
                self.attrs_array = np.array(["qv_1km", "qv_3km", "qv_5km", "qv_7km"])
                self.single = False

            if self.variable == "PRESS":
                self.choice_var1 = "pres_sev_1"
                self.choice_var3 = "pres_sev_3"
                self.choice_var5 = "pres_sev_5"
                self.choice_var7 = "pres_sev_7"
                self.attrs_array = np.array(["pr_1km", "pr_3km", "pr_5km", "pr_7km"])
                self.single = False

            if self.variable == "W_vert":
                self.choice_var1 = "wwnd_sev_1"
                self.choice_var3 = "wwnd_sev_3"
                self.choice_var5 = "wwnd_sev_5"
                self.choice_var7 = "wwnd_sev_7"
                self.attrs_array = np.array(["ww_1km", "ww_3km", "ww_5km", "ww_7km"])
                self.single = False

            if self.variable == "WMAX":
                self.choice_var1 = "maxw_sev_1"
                self.attrs_array = np.array(["maxw"]) 
                self.single = True
                
            if self.variable == "DBZ":
                self.choice_var1 = "dbzs_sev_1"
                self.attrs_array = np.array(["dbzs"])
                self.single = True
                
            if self.variable == "CTT":
                self.choice_var1 = "ctts_sev_1"
                self.attrs_array = np.array(["ctts"]) 
                self.single = True
                
            if self.variable == "UH25":
                self.choice_var1 = "uh25_sev_1"
                self.attrs_array = np.array(["uh25"]) 
                self.single = True
                
            if self.variable == "UH03":
                self.choice_var1 = "uh03_sev_1"
                self.attrs_array = np.array(["uh03"])    
                self.single = True
    
    

    def variable_translate(self):
        
        """
            Variable name for the respective filenames.
        """
        
        var = {
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
            out = var[self.variable]
            return out
        except:
            raise ValueError("Please enter TK, EU, EV, QVAPOR, P, W_vert, or WMAX as variable.")
        
    
    
    def open_above_threshold(self):
        """
            Open and concat files for six months of analysis (threshold exceedance).
        """
        
        data_dec = xr.open_mfdataset(f"/{self.working_directory}/{self.climate}_uh{self.threshold1}_{self.mask_str}_12.nc", 
                                           parallel=True, combine='by_coords')
        data_jan = xr.open_mfdataset(f"/{self.working_directory}/{self.climate}_uh{self.threshold1}_{self.mask_str}_01.nc",
                                           parallel=True, combine='by_coords')
        data_feb = xr.open_mfdataset(f"/{self.working_directory}/{self.climate}_uh{self.threshold1}_{self.mask_str}_02.nc", 
                                           parallel=True, combine='by_coords')
        data_mar = xr.open_mfdataset(f"/{self.working_directory}/{self.climate}_uh{self.threshold1}_{self.mask_str}_03.nc", 
                                           parallel=True, combine='by_coords')
        data_apr = xr.open_mfdataset(f"/{self.working_directory}/{self.climate}_uh{self.threshold1}_{self.mask_str}_04.nc", 
                                           parallel=True, combine='by_coords')
        data_may = xr.open_mfdataset(f"/{self.working_directory}/{self.climate}_uh{self.threshold1}_{self.mask_str}_05.nc", 
                                           parallel=True, combine='by_coords')
        data = xr.concat([data_dec, data_jan, data_feb, data_mar, data_apr, data_may], dim='patch')
        
        data_dec = data_dec.close()
        data_jan = data_jan.close()
        data_feb = data_feb.close()
        data_mar = data_mar.close()
        data_apr = data_apr.close()
        data_may = data_may.close()
        
        return data
    
    
        
    def open_below_threshold(self):
        """
            Open and concat files for six months of analysis (threshold non-exceedance).
        """        
        
        data_dec = xr.open_mfdataset(f"/{self.working_directory}/{self.climate}_nonuh{self.threshold1}_{self.mask_str}_12.nc",
                                          parallel=True, combine='by_coords')
        data_jan = xr.open_mfdataset(f"/{self.working_directory}/{self.climate}_nonuh{self.threshold1}_{self.mask_str}_01.nc",
                                           parallel=True, combine='by_coords')
        data_feb = xr.open_mfdataset(f"/{self.working_directory}/{self.climate}_nonuh{self.threshold1}_{self.mask_str}_02.nc",
                                           parallel=True, combine='by_coords')
        data_mar = xr.open_mfdataset(f"/{self.working_directory}/{self.climate}_nonuh{self.threshold1}_{self.mask_str}_03.nc",
                                           parallel=True, combine='by_coords')
        data_apr = xr.open_mfdataset(f"/{self.working_directory}/{self.climate}_nonuh{self.threshold1}_{self.mask_str}_04.nc",
                                           parallel=True, combine='by_coords')
        data_may = xr.open_mfdataset(f"/{self.working_directory}/{self.climate}_nonuh{self.threshold1}_{self.mask_str}_05.nc",
                                           parallel=True, combine='by_coords')
        
        data = xr.concat([data_dec, data_jan, data_feb, data_mar, data_apr, data_may], dim='patch')
        
        data_dec = data_dec.close()
        data_jan = data_jan.close()
        data_feb = data_feb.close()
        data_mar = data_mar.close()
        data_apr = data_apr.close()
        data_may = data_may.close()
        
        return data


        
    def create_traintest_data(self, data_b, data_a, split_perc, return_label=False):
        """
            Balancing of above and below threshold data for training data, spitting out remainder for testing
            Permute and slice the below threshold data to equal the above threshold data shape.
        """
        
        #train above
        np.random.seed(0)
        select_data = np.random.permutation(data_a.shape[0])[:int(data_a.shape[0]*split_perc)]
        train_above = data_a[select_data]

        #train below
        np.random.seed(0)
        select_data = np.random.permutation(data_b.shape[0])[:int(data_a.shape[0]*split_perc)]
        train_below = data_b[select_data]

        #test above
        np.random.seed(0)
        select_data = np.random.permutation(data_a.shape[0])[int(data_a.shape[0]*split_perc):]
        test_above = data_a[select_data]

        #test below
        np.random.seed(0)
        #slicing to get respective ratio of above to below UH data patches
        select_data = np.random.permutation(
            data_b.shape[0])[int(data_a.shape[0]*split_perc):int((((data_a.shape[0]*(1-split_perc))*data_b.shape[0])/data_a.shape[0])+(data_a.shape[0]*(1-split_perc)))]
        test_below = data_b[select_data]

        #create the label data
        train_above_label = np.ones(train_above.shape[0])
        train_below_label = np.zeros(train_below.shape[0])
        test_above_label = np.ones(test_above.shape[0])
        test_below_label = np.zeros(test_below.shape[0])

        #merge above and below data in prep to shuffle/permute
        train_data = np.vstack([train_above, train_below])
        if return_label:
            train_label = np.hstack([train_above_label, train_below_label])
        test_data = np.vstack([test_above, test_below])
        if return_label:
            test_label = np.hstack([test_above_label, test_below_label])

        #finally, permute the data that has been merged and properly balanced
        np.random.seed(10)
        train_data = np.random.permutation(train_data)
        np.random.seed(10)
        test_data = np.random.permutation(test_data)
        
        if not return_label:
            return train_data, test_data  
        
        if return_label:
            np.random.seed(10)
            train_label = np.random.permutation(train_label)
            np.random.seed(10)
            test_label = np.random.permutation(test_label)    
            return train_data, test_data, train_label, test_label


    
    def minmax_scale_apply(self, data):
        """
            Min-max standardization of the training data.
        """
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(data)
        return scaler.transform(data)
    
    
    
    def minmax_scale_apply_test(self, train, test):
        """
            Min-max standardization of the testing data using training data min/max.
        """
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train)
        return scaler.transform(test)

    

    def standardize_scale_apply(self, data):
        """
            Z-score standardization of the training data.
        """
        return np.divide((data - np.nanmean(data)), np.nanstd(data))



    def standardize_scale_apply_test(self, train, test):
        """
            Z-score standardization of the test data using the training data mean and standard deviation.
        """
        return np.divide((test - np.nanmean(train)), np.nanstd(train))

    
    
    def grab_variables(self, data):
        
        """
            Extract variables from main data set.
        """

        if not self.single:
            
            data_1 = data[self.choice_var1].values
            data_2 = data[self.choice_var3].values
            data_3 = data[self.choice_var5].values
            data_4 = data[self.choice_var7].values
            
            return data_1, data_2, data_3, data_4
            
        if self.single:
            
            data_1 = data[self.choice_var1].values
            
            return data_1


        
    def split_data_to_traintest(self, below1=None, below2=None, below3=None, below4=None, above1=None, above2=None, above3=None, above4=None):
        
        if not self.single:
            
            train1, test1, train_label, test_label = self.create_traintest_data(below1, above1, split_perc=self.percent_split, return_label=True)
            train2, test2 = self.create_traintest_data(below2, above2, split_perc=self.percent_split, return_label=False)
            train3, test3 = self.create_traintest_data(below3, above3, split_perc=self.percent_split, return_label=False)
            train4, test4 = self.create_traintest_data(below4, above4, split_perc=self.percent_split, return_label=False)
            
            return train1, train2, train3, train4, train_label, test1, test2, test3, test4, test_label
        
        if self.single:
            
            train1, test1, train_label, test_label = self.create_traintest_data(below1, above1, split_perc=self.percent_split, return_label=True)
            
            return train1, test1, train_label, test_label
        


    def standardize_training(self, func, data1, data2=None, data3=None, data4=None):
        """
            Function to standardize the training data.
            func: choice of standardization method.
        """
        
        data_scaled1 = func(data1)

        if not self.single:
            data_scaled2 = func(data2)
            data_scaled3 = func(data3)
            data_scaled4 = func(data4)
            return data_scaled1, data_scaled2, data_scaled3, data_scaled4
        
        if self.single:
            return data_scaled1
        
        
    
    def standardize_testing(self, func, train1=None, train2=None, train3=None, train4=None, 
                                        test1=None, test2=None, test3=None, test4=None):
        """
            Function to standardize the testing data.
            func: choice of standardization method.
        """
        
        data1 = func(train1, test1)
        
        if not self.single:
            data2 = func(train2, test2)
            data3 = func(train3, test3)
            data4 = func(train4, test4)
            return data1, data2, data3, data4

        if self.single:
            return data1



    def stack_the_data(self, data1, data2, data3, data4):
        
        """
            Stack the numpy arrays before assembling final xarray netcdf file for saving.
        """

        if not self.single:
            totaldata = np.stack([data1, data2, data3, data4])
            return totaldata



    def save_data(self, train_data, train_label, test_data, test_label):
        
        """
            Create and save file that contains split standardized data for deep learning model training.
            This contains data for one variable for ease of use later and storage space considerations.
        """
        
        if not self.single:
            
            data_assemble = xr.Dataset({
                'X_train':(['features','a','x','y'], train_data),
                'X_train_label':(['a'], train_label),
                'X_test':(['features','b','x','y'], test_data),
                'X_test_label':(['b'], test_label),
                },
                 coords=
                {'feature':(['features'],self.attrs_array),
                })
            
        if self.single:
            
            data_assemble = xr.Dataset({
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
        return


    
    def run_sequence(self):
        """
            Function to run through full set of steps in data preprocessing for DL model training and testing.
        """
        
        print("Opening files...")
        data_above = self.open_above_threshold()
        data_below = self.open_below_threshold()
        
        
        if not self.single:
        
            print("Grabbing variables...")
            above_1, above_2, above_3, above_4 = self.grab_variables(data_above)
            below_1, below_2, below_3, below_4 = self.grab_variables(data_below)
            
            data_above = data_above.close()
            data_below = data_below.close()
            
            print("Splitting data...")
            train1, train2, train3, train4, train_label, test1, test2, test3, test4, test_label = self.split_data_to_traintest(
                below_1, below_2, below_3, below_4, above_1, above_2, above_3, above_4)
            
            above_1=None; above_2=None; above_3=None; above_4=None
            below_1=None; below_2=None; below_3=None; below_4=None

            print("Standardizing testing...")
            test1, test2, test3, test4 = self.standardize_testing(
                self.standardize_scale_apply_test, train1, train2, train3, train4, test1, test2, test3, test4)
            
            print("Standardizing training...")
            train1, train2, train3, train4 = self.standardize_training(self.standardize_scale_apply, train1, train2, train3, train4)
            
            print("Stacking files...")
            Xtrain = self.stack_the_data(train1, train2, train3, train4)
            Xtest = self.stack_the_data(test1, test2, test3, test4)

            train1=None; train2=None; train3=None; train4=None
            test1=None;  test2=None;  test3=None;  test4=None
            
            
        if self.single:
            
            print("Grabbing variables...")
            above_1 = self.grab_variables(data_above)
            below_1 = self.grab_variables(data_below)   
            
            data_above = data_above.close()
            data_below = data_below.close()
            
            print("Splitting data...")
            train1, test1, train_label, test_label = self.split_data_to_traintest(below1=below_1, above1=above_1)
            
            above_1=None; below_1=None

            print("Standardizing testing...")
            test1 = self.standardize_testing(self.standardize_scale_apply_test, train1=train1, test1=test1)
            
            print("Standardizing training...")
            train1 = self.standardize_training(self.standardize_scale_apply, data1=train1)
            
            print("Stacking files...")
            Xtrain = train1
            Xtest = test1

            train1=None; test1=None
        
        
        print("Saving file...")
        self.save_data(Xtrain, train_label, Xtest, test_label) 
        
        Xtrain=None; Xtest=None; train_label=None; test_label=None
        return
        
