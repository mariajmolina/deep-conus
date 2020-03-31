
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
    
    
    def __init__(self, climate, variable, percent_split, project_code, working_directory, threshold1, 
                 mask = False,
                 activate_dask=True, cluster_min=10, cluster_max=40):
        
        
        """

        Instantiation of split_and_standardize:

        Here we will be preprocessing data for deep learning model training.
        This module includes methods for training and testing data splits and standardization.

        PARAMETERS
        ----------
        climate: climate period to derive deep learning data for (str; current or future)
        variable: variable to run script for (str), including TK, EV, EU, QVAPOR, PRESS, W_vert, main (UH2-5, UH0-3, MAXW, CTT, DBZ).
        percent_split: percent for training data, remaining will be assigned for testing (float) E.g., 0.6 is 60% for training, 40% for testing.
        project_code: code to charge for the launch of dask workers (str)
        working_directory: path to directory where DL preprocessing files will be saved and worked from (str)
        threshold1: threshold for method (int)
        mask: whether the threshold will be applied within the storm patch mask or not (boolean; default False)
        activate_dask: whether to initiate dask workers (boolean; default True)
        cluster_min: the minimum number of nodes (with 36 CPUs) to initiate for adaptive dask job (str; default 10 [set for interp])
        cluster_max: the maximum number of nodes (with 36 CPUs) to initiate for adaptive dask job (str; default 40 [set for interp])
        
        """
        
        if climate != 'current' and climate != 'future':
            raise Exception("Please enter current or future for climate option.")
        if climate == 'current' or climate == 'future':
            self.climate = climate
            
        if percent_split>=1:
            raise Exception("Percent split should be a float less than 1.")
        if percent_split<1:
            self.percent_split = percent_split
            
        self.project_code = project_code
        self.working_directory = working_directory
        self.threshold1 = threshold1
        self.cluster_min = cluster_min
        self.cluster_max = cluster_max
        
        self.activate_dask = activate_dask
        
        self.mask = mask
        if not self.mask:
            self.mask_str = 'nomask'
        if self.mask:
            self.mask_str = 'mask'

        if variable!='TK' and variable!='EV' and variable!='EU' and variable!='QVAPOR' and variable!='PRESS' and variable!='W_vert' and variable!='main':
            raise exception("Please enter ``TK``, ``EV``, ``EU``, ``QVAPOR``, ``PRESS``, ``W_vert``, ``main`` (UH2-5, UH0-3, MAXW, CTT, DBZ) as variable.")
        if variable=='TK' or variable=='EV' or variable=='EU' or variable=='QVAPOR' or variable=='PRESS' or variable=='W_vert' or variable=='main':
            self.variable = variable
            
            if self.variable == "TK":
                self.choice_var1 = "temp_sev_1"
                self.choice_var3 = "temp_sev_3"
                self.choice_var5 = "temp_sev_5"
                self.choice_var7 = "temp_sev_7"
                self.attrs_array = np.array(["tk_1km", "tk_3km", "tk_5km", "tk_7km"])

            if self.variable == "EV":
                self.choice_var1 = "evwd_sev_1"
                self.choice_var3 = "evwd_sev_3"
                self.choice_var5 = "evwd_sev_5"
                self.choice_var7 = "evwd_sev_7"
                self.attrs_array = np.array(["ev_1km", "ev_3km", "ev_5km", "ev_7km"])

            if self.variable == "EU":
                self.choice_var1 = "euwd_sev_1"
                self.choice_var3 = "euwd_sev_3"
                self.choice_var5 = "euwd_sev_5"
                self.choice_var7 = "euwd_sev_7"
                self.attrs_array = np.array(["eu_1km", "eu_3km", "eu_5km", "eu_7km"])

            if self.variable == "QVAPOR":
                self.choice_var1 = "qvap_sev_1"
                self.choice_var3 = "qvap_sev_3"
                self.choice_var5 = "qvap_sev_5"
                self.choice_var7 = "qvap_sev_7"
                self.attrs_array = np.array(["qv_1km", "qv_3km", "qv_5km", "qv_7km"])

            if self.variable == "PRESS":
                self.choice_var1 = "pres_sev_1"
                self.choice_var3 = "pres_sev_3"
                self.choice_var5 = "pres_sev_5"
                self.choice_var7 = "pres_sev_7"
                self.attrs_array = np.array(["pr_1km", "pr_3km", "pr_5km", "pr_7km"])

            if self.variable == "W_vert":
                self.choice_var1 = "wwnd_sev_1"
                self.choice_var3 = "wwnd_sev_3"
                self.choice_var5 = "wwnd_sev_5"
                self.choice_var7 = "wwnd_sev_7"
                self.attrs_array = np.array(["ww_1km", "ww_3km", "ww_5km", "ww_7km"])

            if self.variable == "main":
                self.choice_maxw = "maxw_sev_1"
                self.choice_dbzs = "dbzs_sev_1"
                self.choice_ctts = "ctts_sev_1"
                self.choice_uh25 = "uh25_sev_1"
                self.choice_uh03 = "uh03_sev_1"
                self.attrs_array = np.array(["maxw", "dbzs", "ctts", "uh25", "uh03"])    
    
    
    
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
    
    
    
    def activate_workers(self):
        #start dask workers
        cluster = NCARCluster(memory="109GB", cores=36, project=self.project_code)
        cluster.adapt(minimum=self.cluster_min, maximum=self.cluster_max, wait_count=60)
        cluster
        #print scripts
        print(cluster.job_script())
        #start client
        client = Client(cluster)
        client

        
        
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
               'PRESS':'P'
              }
        
        try:
            out = var[self.variable]
            return out
        except:
            raise ValueError("Please enter TK, EU, EV, QVAPOR, P, W_vert, or WMAX as variable.")
        
    
    
    def grab_variables(self, data):
        
        """
            Extract variables from main data set.
        """

        if self.variable != 'main':
            
            data_1 = data[self.choice_var1].values
            data_2 = data[self.choice_var3].values
            data_3 = data[self.choice_var5].values
            data_4 = data[self.choice_var7].values
            
            return data_1, data_2, data_3, data_4
            
        if self.variable == 'main':
            
            data_1 = data[self.choice_maxw].values
            data_2 = data[self.choice_dbzs].values
            data_3 = data[self.choice_ctts].values
            data_4 = data[self.choice_uh25].values
            data_5 = data[self.choice_uh03].values
            
            return data_1, data_2, data_3, data_4, data_5


        
    def split_data_to_traintest(self, below1, below2, below3, below4, above1, above2, above3, above4, below5=None, above5=None):
        
        train1, test1, train_label, test_label = self.create_traintest_data(below1, above1, split_perc=self.percent_split, return_label=True)
        train2, test2 = self.create_traintest_data(below2, above2, split_perc=self.percent_split, return_label=False)
        train3, test3 = self.create_traintest_data(below3, above3, split_perc=self.percent_split, return_label=False)
        train4, test4 = self.create_traintest_data(below4, above4, split_perc=self.percent_split, return_label=False)

        if self.variable != 'main':
            return train1, train2, train3, train4, train_label, test1, test2, test3, test4, test_label
        
        if self.variable == 'main':
            train5, test5 = self.create_traintest_data(below5, above5, split_perc=self.percent_split, return_label=False)
            return train1, train2, train3, train4, train5, train_label, test1, test2, test3, test4, test5, test_label
        


    def standardize_training(self, func, data1, data2, data3, data4, data5=None):
        """
            Function to standardize the training data.
            func: choice of standardization method.
        """
        
        data_scaled1 = func(data1)
        data_scaled2 = func(data2)
        data_scaled3 = func(data3)
        data_scaled4 = func(data4)

        if self.variable != 'main':
            return data_scaled1, data_scaled2, data_scaled3, data_scaled4
        
        if self.variable == 'main':
            data_scaled5 = func(data5)
            return data_scaled1, data_scaled2, data_scaled3, data_scaled4, data_scale5
        
        
    
    def standardize_testing(self, func, train1, train2, train3, train4, 
                            test1, test2, test3, test4, train5=None, test5=None):
        """
            Function to standardize the testing data.
            func: choice of standardization method.
        """
        
        data1 = func(train1, test1)
        data2 = func(train2, test2)
        data3 = func(train3, test3)
        data4 = func(train4, test4)
        
        if self.variable != 'main':
            return data1, data2, data3, data4

        if self.variable == 'main':
            data5 = func(train5, test5)
            return data1, data2, data3, data4, data5



    def stack_the_data(self, data1, data2, data3, data4, data5=None):
        
        """
            Stack the numpy arrays before assembling final xarray netcdf file for saving.
        """

        if self.variable != 'main':
            totaldata = np.stack([data1, data2, data3, data4])
            
        if self.variable == 'main':
            totaldata = np.stack([data1, data2, data3, data4, data5])
            
        return totaldata



    def save_data(self, train_data, train_label, test_data, test_label):
        
        """
            Create and save file that contains split standardized data for deep learning model training.
            This contains data for one variable for ease of use later and storage space considerations.
        """
        
        data_assemble = xr.Dataset({
            'X_train':(['a','x','y','features'], train_data.reshape(train_data.shape[1],32,32,4)),
            'X_train_label':(['a'], train_label),
            'X_test':(['b','x','y','features'], test_data.reshape(test_data.shape[1],32,32,4)),
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
        
        print("Activating workers...")
        if self.activate_dask:
            self.activate_workers()
            
        print("Opening files...")
        data_above = self.open_below_threshold()
        data_below = self.open_below_threshold()
        
        
        if self.variable != 'main':
        
            print("Grabbing variables...")
            above_1, above_2, above_3, above_4 = self.grab_variables(data_above)
            below_1, below_2, below_3, below_4 = self.grab_variables(data_below)
            
            print("Splitting data...")
            train1, train2, train3, train4, train_label, test1, test2, test3, test4, test_label = self.split_data_to_traintest(
                below_1, below_2, below_3, below_4, above_1, above_2, above_3, above_4, below5=None, above5=None)
            
            above_1=None; above_2=None; above_3=None; above_4=None
            below_1=None; below_2=None; below_3=None; below_4=None

            print("Standardizing testing...")
            test1, test2, test3, test4 = self.standardize_testing(
                self.standardize_scale_apply_test, train1, train2, train3, train4, test1, test2, test3, test4, train5=None, test5=None)
            
            print("Standardizing training...")
            train1, train2, train3, train4 = self.standardize_training(
                self.standardize_scale_apply, train1, train2, train3, train4, data5=None)
            
            print("Stacking files...")
            Xtrain = self.stack_the_data(train1, train2, train3, train4, data5=None)
            Xtest = self.stack_the_data(test1, test2, test3, test4, data5=None)

            train1=None; train2=None; train3=None; train4=None
            test1=None; test2=None; test3=None; test4=None
            
            
        if self.variable == 'main':
            
            print("Grabbing variables...")
            above_1, above_2, above_3, above_4, above_5 = self.grab_variables(data_above)
            below_1, below_2, below_3, below_4, below_5 = self.grab_variables(data_below)      
            
            print("Splitting data...")
            train1, train2, train3, train4, train5, train_label, test1, test2, test3, test4, test5, test_label = self.split_data_to_traintest(
                below_1, below_2, below_3, below_4, above_1, above_2, above_3, above_4, below5=below_5, above5=above_5)
            
            above_1=None; above_2=None; above_3=None; above_4=None; above_5=None
            below_1=None; below_2=None; below_3=None; below_4=None; below_5=None

            print("Standardizing testing...")
            test1, test2, test3, test4, test5 = self.standardize_testing(
                self.standardize_scale_apply_test, train1, train2, train3, train4, test1, test2, test3, test4, train5=train5, test5=test5)
            
            print("Standardizing training...")
            train1, train2, train3, train4, train5 = self.standardize_training(
                self.standardize_scale_apply, train1, train2, train3, train4, data5=train5)
            
            print("Stacking files...")
            Xtrain = self.stack_the_data(train1, train2, train3, train4, data5=train5)
            Xtest = self.stack_the_data(test1, test2, test3, test4, data5=test5)
            
            
        print("Saving file...")
        self.save_data(Xtrain, train_label, Xtest, test_label)        
        return
        
