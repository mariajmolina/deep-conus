
#####################################################################################
#####################################################################################
#
# Author: Maria J. Molina
# National Center for Atmospheric Research
#
# Script to preprocess data for deep learning model training. 
#
#
#####################################################################################
#####################################################################################


#-----------------------------

import numpy as np
import xarray as xr
import pandas as pd
import multiprocessing as mp

#-----------------------------


class preprocess_data:


    def __init__(self, working_directory, stormpatch_path, climate, method, threshold1, mask=False, threshold2=None, num_cpus=36):


        """

        Instantiation of preprocess_data:

        Here we will be preprocessing data for deep learning model training.

        PARAMETERS
        ----------
        working_directory: path to directory where DL preprocessing files will be saved and worked from (str)
        destination_path: where storm patch files were saved (str)
        climate: climate period to derive deep learning data for (str; current or future)
        method: variable choice(s) for preprocessing data, includes UHsingle, UHdouble, and GRP (str)
        threshold1: threshold for method (int)
        mask: whether the threshold will be applied within the storm patch mask or not (boolean; default False)
        threshold2: second threshold for UHdual method (int; default None)
        num_cpus: number of CPUs for to use in a node for parallelizing extractions (int; default 36)
        
        """

        self.working_directory = working_directory
        self.stormpatch_path = stormpatch_path

        if climate != 'current' and climate != 'future':
            raise Exception("Please enter current or future for climate option.")
        if climate == 'current' or climate == 'future':
            self.climate = climate
            
        if method!='UHsingle' and method!='UHdual' and method!='GRP':
            raise Exception("Incorrect method. Please enter UHsingle, UHdouble, or GRP.")
        if method=='UHsingle' or method=='UHdual' or method=='GRP':
            self.method = method
            self.threshold1 = threshold1
            if self.method == 'UHdual':
                if self.threshold2:
                    self.threshold2 = threshold2
                if not self.threshold2: 
                    raise Exception("Please enter a threshold for UH 0-3 km (dual UH method).")
                    
        self.mask = mask
        if not self.mask:
            self.mask_str = 'nomask'
        if self.mask:
            self.mask_str = 'mask'

        self.num_cpus = num_cpus



    def generate_time_full(self):
        """
            Creation of full time period that will be looped through for extracting storm patch information for classes.
        """
        return pd.date_range('2000-10-01','2013-09-30',freq='MS')[(pd.date_range('2000-10-01','2013-09-30',freq='MS').month==12)|
                                                                  (pd.date_range('2000-10-01','2013-09-30',freq='MS').month==1)|
                                                                  (pd.date_range('2000-10-01','2013-09-30',freq='MS').month==2)|
                                                                  (pd.date_range('2000-10-01','2013-09-30',freq='MS').month==3)|
                                                                  (pd.date_range('2000-10-01','2013-09-30',freq='MS').month==4)|
                                                                  (pd.date_range('2000-10-01','2013-09-30',freq='MS').month==5)]
    

        
    def generate_time_month(self, month_int):
        """
            Creation of time array that will be looped through for extracting storm patch information for classes.
            month_int: month being computed (int)        
        """
        return pd.date_range('2000-10-01','2013-09-30',freq='MS')[(pd.date_range('2000-10-01','2013-09-30',freq='MS').month == month_int)]



    def create_data_indices(self, time):
        """
            Split data into categories and save first intermediary files. 
            Here we create the indices of the classes in the data for later use.
        """
        
        if self.method == 'UHsingle':
            
            if not self.mask:
                data = xr.open_mfdataset(f"/{self.stormpatch_path}/{self.climate}_SP3hourly_{time.strftime('%Y%m')}*.nc", combine='by_coords')
                data_assemble = xr.Dataset({'grid':(['x'], np.argwhere(data.uh25_grid.values.max(axis=(1,2)) > self.threshold1)[:,0])})
                data_assemble.to_netcdf(f"/{self.working_directory}/{self.climate}_indx{self.threshold1}_{self.mask_str}_{time.strftime('%Y')}{time.strftime('%m')}.nc")
                return
            
            if self.mask:
                data = xr.open_mfdataset(f"/{self.stormpatch_path}/{self.climate}_SP3hourly_{time.strftime('%Y%m')}*.nc", combine='by_coords')
                data_assemble = xr.Dataset({'grid':(['x'], np.argwhere(data.uh25_grid.where(data.mask).max(axis=(1,2), skipna=True).values > self.threshold1)[:,0])})
                data_assemble.to_netcdf(f"/{self.working_directory}/{self.climate}_indx{self.threshold1}_{self.mask_str}_{time.strftime('%Y')}{time.strftime('%m')}.nc")
                return
                
        if self.method == 'UHdual':
           
            if not self.mask:
                ###
                return
            
            if self.mask:
                ###
                return

            
            
    def apply_exceed_mask(self, data_var, data_mask, level):
        """
            Function to retain the patches that exceed threshold.
        """        
        return data_var.var_grid.sel(levels=level)[data_mask.grid.values,:,:]
    
    
    
    def apply_notexceed_mask(self, data_var, data_mask, level):
        """
            Function to remove the patches that do not exceed threshold.
        """
        return np.delete(data_var.var_grid.sel(levels=level).values, data_mask.grid.values, axis=0)
    
    
    
    def flatten_list(self, array):
        """
            Function to flatten the created list of xarray data arrays.
        """
        return [j for i in array for j in i.values]
    
    
    
    def flatten_arraylist(self, array):
        """
            Function to flatten the created list of numpy arrays.
        """
        return [j for i in array for j in i]
    
    
    
    def parallelizing_funcs(self):
        """
            Activate the multiprocessing function to parallelize the functions.
        """
        
        print(f"Starting jobs...")
        
        timearray = self.generate_time_full()
        
        pool1 = mp.Pool(self.num_cpus)
        for time in timearray:
            print(f"Extracting {time.strftime('%Y-%m')} indices...")
            pool1.apply_async(self.create_data_indices, args=([time]))
            
        pool1.close()
        pool1.join()
        
        select_months = np.array([12,1,2,3,4,5])
        
        pool2 = mp.Pool(self.num_cpus)
        for mo in select_months:
            print(f"Creating {self.month_translate(mo)} patches of threshold exceedances...")
            pool2.apply_async(self.create_files_exceed_threshold, args=([mo]))
            print(f"Creating {self.month_translate(mo)} patches of threshold non-exceedances...")
            pool2.apply_async(self.create_files_notexceed_threshold, args=([mo]))
            
        pool2.close()
        pool2.join()        
        
        print(f"Completed the jobs.")
        return
        
        
        
    def month_translate(self, num):
        """
            Convert integer month to string month.
        """
        
        var = {12:'December',
               1:'January',
               2:'February',
               3:'March',
               4:'April',
               5:'May'}
        try:
            out = var[num]
            return out
        except:
            raise ValueError("Please enter month integer from Dec-May.")
        
            

    def create_files_exceed_threshold(self, month_int):
        """
        Create the files containing chosen environment patches for storms that exceed threshold.
        Data files being open contain the storm patches. These are not the full WRF domain.
        """

        time_temp = self.generate_time_month(month_int)

        data_temp_sev_1 = []; data_temp_sev_3 = []; data_temp_sev_5 = []; data_temp_sev_7 = []; data_evwd_sev_1 = []; data_evwd_sev_3 = []
        data_euwd_sev_1 = []; data_euwd_sev_3 = []; data_euwd_sev_5 = []; data_euwd_sev_7 = []; data_evwd_sev_5 = []; data_evwd_sev_7 = []
        data_qvap_sev_1 = []; data_qvap_sev_3 = []; data_qvap_sev_5 = []; data_qvap_sev_7 = []; data_dbzs_sev_1 = []; data_maxw_sev_1 = []
        data_pres_sev_1 = []; data_pres_sev_3 = []; data_pres_sev_5 = []; data_pres_sev_7 = []; data_ctts_sev_1 = []
        data_wwnd_sev_1 = []; data_wwnd_sev_3 = []; data_wwnd_sev_5 = []; data_wwnd_sev_7 = []; data_uh25_sev_1 = []; data_uh03_sev_1 = []

        for time in time_temp:

            print(f"opening files for {time.strftime('%Y')}{time.strftime('%m')}")
            
            data_mask = xr.open_mfdataset(
                f"/{self.working_directory}/{self.climate}_indx{self.threshold1}_{self.mask_str}_{time.strftime('%Y')}{time.strftime('%m')}.nc",    
                combine='by_coords')
            
            data_temp = xr.open_mfdataset(f"/{self.stormpatch_path}/{self.climate}_SP3hourly_tk_{time.strftime('%Y%m')}*.nc",     combine='by_coords')
            data_evwd = xr.open_mfdataset(f"/{self.stormpatch_path}/{self.climate}_SP3hourly_ev_{time.strftime('%Y%m')}*.nc",     combine='by_coords')
            data_euwd = xr.open_mfdataset(f"/{self.stormpatch_path}/{self.climate}_SP3hourly_eu_{time.strftime('%Y%m')}*.nc",     combine='by_coords')
            data_qvap = xr.open_mfdataset(f"/{self.stormpatch_path}/{self.climate}_SP3hourly_qvapor_{time.strftime('%Y%m')}*.nc", combine='by_coords')
            data_pres = xr.open_mfdataset(f"/{self.stormpatch_path}/{self.climate}_SP3hourly_p_{time.strftime('%Y%m')}*.nc",      combine='by_coords')
            data_wwnd = xr.open_mfdataset(f"/{self.stormpatch_path}/{self.climate}_SP3hourly_w_{time.strftime('%Y%m')}*.nc",      combine='by_coords')
            data_maxw = xr.open_mfdataset(f"/{self.stormpatch_path}/{self.climate}_SP3hourly_maxw_{time.strftime('%Y%m')}*.nc",   combine='by_coords')
            data_gen  = xr.open_mfdataset(f"/{self.stormpatch_path}/{self.climate}_SP3hourly_{time.strftime('%Y%m')}*.nc",        combine='by_coords')

            data_temp_sev_1.append(self.apply_exceed_mask(data_temp, data_mask, 0))
            data_temp_sev_3.append(self.apply_exceed_mask(data_temp, data_mask, 1))
            data_temp_sev_5.append(self.apply_exceed_mask(data_temp, data_mask, 2))
            data_temp_sev_7.append(self.apply_exceed_mask(data_temp, data_mask, 3))

            data_evwd_sev_1.append(self.apply_exceed_mask(data_evwd, data_mask, 0))
            data_evwd_sev_3.append(self.apply_exceed_mask(data_evwd, data_mask, 1))
            data_evwd_sev_5.append(self.apply_exceed_mask(data_evwd, data_mask, 2))
            data_evwd_sev_7.append(self.apply_exceed_mask(data_evwd, data_mask, 3))

            data_euwd_sev_1.append(self.apply_exceed_mask(data_euwd, data_mask, 0))
            data_euwd_sev_3.append(self.apply_exceed_mask(data_euwd, data_mask, 1))
            data_euwd_sev_5.append(self.apply_exceed_mask(data_euwd, data_mask, 2))
            data_euwd_sev_7.append(self.apply_exceed_mask(data_euwd, data_mask, 3))

            data_qvap_sev_1.append(self.apply_exceed_mask(data_qvap, data_mask, 0))
            data_qvap_sev_3.append(self.apply_exceed_mask(data_qvap, data_mask, 1))
            data_qvap_sev_5.append(self.apply_exceed_mask(data_qvap, data_mask, 2))
            data_qvap_sev_7.append(self.apply_exceed_mask(data_qvap, data_mask, 3))

            data_pres_sev_1.append(self.apply_exceed_mask(data_pres, data_mask, 0))
            data_pres_sev_3.append(self.apply_exceed_mask(data_pres, data_mask, 1))
            data_pres_sev_5.append(self.apply_exceed_mask(data_pres, data_mask, 2))
            data_pres_sev_7.append(self.apply_exceed_mask(data_pres, data_mask, 3))
            
            data_wwnd_sev_1.append(self.apply_exceed_mask(data_wwnd, data_mask, 0))
            data_wwnd_sev_3.append(self.apply_exceed_mask(data_wwnd, data_mask, 1))
            data_wwnd_sev_5.append(self.apply_exceed_mask(data_wwnd, data_mask, 2))
            data_wwnd_sev_7.append(self.apply_exceed_mask(data_wwnd, data_mask, 3))

            data_maxw_sev_1.append(data_maxw.var_grid[data_mask.grid.values,:,:])
            data_dbzs_sev_1.append(data_gen.dbz_grid[data_mask.grid.values,:,:])
            data_ctts_sev_1.append(data_gen.ctt_grid[data_mask.grid.values,:,:])
            data_uh25_sev_1.append(data_gen.uh25_grid[data_mask.grid.values,:,:])
            data_uh03_sev_1.append(data_gen.uh03_grid[data_mask.grid.values,:,:])

        data_temp_sev_1_patches = self.flatten_list(data_temp_sev_1)
        data_temp_sev_3_patches = self.flatten_list(data_temp_sev_3)
        data_temp_sev_5_patches = self.flatten_list(data_temp_sev_5)
        data_temp_sev_7_patches = self.flatten_list(data_temp_sev_7)

        data_evwd_sev_1_patches = self.flatten_list(data_evwd_sev_1)
        data_evwd_sev_3_patches = self.flatten_list(data_evwd_sev_3)
        data_evwd_sev_5_patches = self.flatten_list(data_evwd_sev_5)
        data_evwd_sev_7_patches = self.flatten_list(data_evwd_sev_7)

        data_euwd_sev_1_patches = self.flatten_list(data_euwd_sev_1)
        data_euwd_sev_3_patches = self.flatten_list(data_euwd_sev_3)
        data_euwd_sev_5_patches = self.flatten_list(data_euwd_sev_5)
        data_euwd_sev_7_patches = self.flatten_list(data_euwd_sev_7)

        data_qvap_sev_1_patches = self.flatten_list(data_qvap_sev_1)
        data_qvap_sev_3_patches = self.flatten_list(data_qvap_sev_3)
        data_qvap_sev_5_patches = self.flatten_list(data_qvap_sev_5)
        data_qvap_sev_7_patches = self.flatten_list(data_qvap_sev_7)

        data_pres_sev_1_patches = self.flatten_list(data_pres_sev_1)
        data_pres_sev_3_patches = self.flatten_list(data_pres_sev_3)
        data_pres_sev_5_patches = self.flatten_list(data_pres_sev_5)
        data_pres_sev_7_patches = self.flatten_list(data_pres_sev_7)
        
        data_wwnd_sev_1_patches = self.flatten_list(data_wwnd_sev_1)
        data_wwnd_sev_3_patches = self.flatten_list(data_wwnd_sev_3)
        data_wwnd_sev_5_patches = self.flatten_list(data_wwnd_sev_5)
        data_wwnd_sev_7_patches = self.flatten_list(data_wwnd_sev_7)

        data_maxw_sev_1_patches = self.flatten_list(data_maxw_sev_1)
        data_dbzs_sev_1_patches = self.flatten_list(data_dbzs_sev_1)
        data_ctts_sev_1_patches = self.flatten_list(data_ctts_sev_1)
        data_uh25_sev_1_patches = self.flatten_list(data_uh25_sev_1)
        data_uh03_sev_1_patches = self.flatten_list(data_uh03_sev_1)
        
        data_assemble = xr.Dataset({
                       'temp_sev_1':(['patch','y','x'], np.array(data_temp_sev_1_patches)), 'temp_sev_3':(['patch','y','x'], np.array(data_temp_sev_3_patches)),
                       'temp_sev_5':(['patch','y','x'], np.array(data_temp_sev_5_patches)), 'temp_sev_7':(['patch','y','x'], np.array(data_temp_sev_7_patches)),

                       'evwd_sev_1':(['patch','y','x'], np.array(data_evwd_sev_1_patches)), 'evwd_sev_3':(['patch','y','x'], np.array(data_evwd_sev_3_patches)),
                       'evwd_sev_5':(['patch','y','x'], np.array(data_evwd_sev_5_patches)), 'evwd_sev_7':(['patch','y','x'], np.array(data_evwd_sev_7_patches)),

                       'euwd_sev_1':(['patch','y','x'], np.array(data_euwd_sev_1_patches)), 'euwd_sev_3':(['patch','y','x'], np.array(data_euwd_sev_3_patches)),
                       'euwd_sev_5':(['patch','y','x'], np.array(data_euwd_sev_5_patches)), 'euwd_sev_7':(['patch','y','x'], np.array(data_euwd_sev_7_patches)),
                      
                       'qvap_sev_1':(['patch','y','x'], np.array(data_qvap_sev_1_patches)), 'qvap_sev_3':(['patch','y','x'], np.array(data_qvap_sev_3_patches)),
                       'qvap_sev_5':(['patch','y','x'], np.array(data_qvap_sev_5_patches)), 'qvap_sev_7':(['patch','y','x'], np.array(data_qvap_sev_7_patches)),
                      
                       'pres_sev_1':(['patch','y','x'], np.array(data_pres_sev_1_patches)), 'pres_sev_3':(['patch','y','x'], np.array(data_pres_sev_3_patches)),
                       'pres_sev_5':(['patch','y','x'], np.array(data_pres_sev_5_patches)), 'pres_sev_7':(['patch','y','x'], np.array(data_pres_sev_7_patches)),
            
                       'wwnd_sev_1':(['patch','y','x'], np.array(data_wwnd_sev_1_patches)), 'wwnd_sev_3':(['patch','y','x'], np.array(data_wwnd_sev_3_patches)),
                       'wwnd_sev_5':(['patch','y','x'], np.array(data_wwnd_sev_5_patches)), 'wwnd_sev_7':(['patch','y','x'], np.array(data_wwnd_sev_7_patches)),

                       'maxw_sev_1':(['patch','y','x'], np.array(data_maxw_sev_1_patches)), 'dbzs_sev_1':(['patch','y','x'], np.array(data_dbzs_sev_1_patches)),
                       'ctts_sev_1':(['patch','y','x'], np.array(data_ctts_sev_1_patches)), 'uh25_sev_1':(['patch','y','x'], np.array(data_uh25_sev_1_patches)),
                       'uh03_sev_1':(['patch','y','x'], np.array(data_uh03_sev_1_patches)), })

        print(f"Exceedances for {time.strftime('%m')} complete...")
        data_assemble.to_netcdf(f"/{self.working_directory}/{self.climate}_uh{self.threshold1}_{self.mask_str}_{time.strftime('%m')}.nc")



    def create_files_notexceed_threshold(self, month_int):
        """
        Create the files containing chosen environment patches for storms that exceed threshold.
        Data files being open contain the storm patches. These are not the full WRF domain.
        """

        time_temp = self.generate_time_month(month_int)

        data_temp_sev_1 = []; data_temp_sev_3 = []; data_temp_sev_5 = []; data_temp_sev_7 = []; data_evwd_sev_1 = []; data_evwd_sev_3 = []
        data_euwd_sev_1 = []; data_euwd_sev_3 = []; data_euwd_sev_5 = []; data_euwd_sev_7 = []; data_evwd_sev_5 = []; data_evwd_sev_7 = []
        data_qvap_sev_1 = []; data_qvap_sev_3 = []; data_qvap_sev_5 = []; data_qvap_sev_7 = []; data_dbzs_sev_1 = []; data_maxw_sev_1 = []
        data_pres_sev_1 = []; data_pres_sev_3 = []; data_pres_sev_5 = []; data_pres_sev_7 = []; data_ctts_sev_1 = []
        data_wwnd_sev_1 = []; data_wwnd_sev_3 = []; data_wwnd_sev_5 = []; data_wwnd_sev_7 = []; data_uh25_sev_1 = []; data_uh03_sev_1 = []

        for time in time_temp:

            print(f"opening files for {time.strftime('%Y')}{time.strftime('%m')}")
            
            data_mask = xr.open_mfdataset(
                f"/{self.working_directory}/{self.climate}_indx{self.threshold1}_{self.mask_str}_{time.strftime('%Y')}{time.strftime('%m')}.nc",    
                combine='by_coords')
            
            data_temp = xr.open_mfdataset(f"/{self.stormpatch_path}/{self.climate}_SP3hourly_tk_{time.strftime('%Y%m')}*.nc",     combine='by_coords')
            data_evwd = xr.open_mfdataset(f"/{self.stormpatch_path}/{self.climate}_SP3hourly_ev_{time.strftime('%Y%m')}*.nc",     combine='by_coords')
            data_euwd = xr.open_mfdataset(f"/{self.stormpatch_path}/{self.climate}_SP3hourly_eu_{time.strftime('%Y%m')}*.nc",     combine='by_coords')
            data_qvap = xr.open_mfdataset(f"/{self.stormpatch_path}/{self.climate}_SP3hourly_qvapor_{time.strftime('%Y%m')}*.nc", combine='by_coords')
            data_pres = xr.open_mfdataset(f"/{self.stormpatch_path}/{self.climate}_SP3hourly_p_{time.strftime('%Y%m')}*.nc",      combine='by_coords')
            data_wwnd = xr.open_mfdataset(f"/{self.stormpatch_path}/{self.climate}_SP3hourly_w_{time.strftime('%Y%m')}*.nc",      combine='by_coords')
            data_maxw = xr.open_mfdataset(f"/{self.stormpatch_path}/{self.climate}_SP3hourly_maxw_{time.strftime('%Y%m')}*.nc",   combine='by_coords')
            data_gen  = xr.open_mfdataset(f"/{self.stormpatch_path}/{self.climate}_SP3hourly_{time.strftime('%Y%m')}*.nc",        combine='by_coords')
            
            data_temp_sev_1.append(self.apply_notexceed_mask(data_temp, data_mask, 0))
            data_temp_sev_3.append(self.apply_notexceed_mask(data_temp, data_mask, 1))
            data_temp_sev_5.append(self.apply_notexceed_mask(data_temp, data_mask, 2))
            data_temp_sev_7.append(self.apply_notexceed_mask(data_temp, data_mask, 3))

            data_evwd_sev_1.append(self.apply_notexceed_mask(data_evwd, data_mask, 0))
            data_evwd_sev_3.append(self.apply_notexceed_mask(data_evwd, data_mask, 1))
            data_evwd_sev_5.append(self.apply_notexceed_mask(data_evwd, data_mask, 2))
            data_evwd_sev_7.append(self.apply_notexceed_mask(data_evwd, data_mask, 3))

            data_euwd_sev_1.append(self.apply_notexceed_mask(data_euwd, data_mask, 0))
            data_euwd_sev_3.append(self.apply_notexceed_mask(data_euwd, data_mask, 1))
            data_euwd_sev_5.append(self.apply_notexceed_mask(data_euwd, data_mask, 2))
            data_euwd_sev_7.append(self.apply_notexceed_mask(data_euwd, data_mask, 3))

            data_qvap_sev_1.append(self.apply_notexceed_mask(data_qvap, data_mask, 0))
            data_qvap_sev_3.append(self.apply_notexceed_mask(data_qvap, data_mask, 1))
            data_qvap_sev_5.append(self.apply_notexceed_mask(data_qvap, data_mask, 2))
            data_qvap_sev_7.append(self.apply_notexceed_mask(data_qvap, data_mask, 3))

            data_pres_sev_1.append(self.apply_notexceed_mask(data_pres, data_mask, 0))
            data_pres_sev_3.append(self.apply_notexceed_mask(data_pres, data_mask, 1))
            data_pres_sev_5.append(self.apply_notexceed_mask(data_pres, data_mask, 2))
            data_pres_sev_7.append(self.apply_notexceed_mask(data_pres, data_mask, 3))
            
            data_wwnd_sev_1.append(self.apply_notexceed_mask(data_wwnd, data_mask, 0))
            data_wwnd_sev_3.append(self.apply_notexceed_mask(data_wwnd, data_mask, 1))
            data_wwnd_sev_5.append(self.apply_notexceed_mask(data_wwnd, data_mask, 2))
            data_wwnd_sev_7.append(self.apply_notexceed_mask(data_wwnd, data_mask, 3))
                                  
            data_maxw_sev_1.append(np.delete(data_maxw.var_grid.values, data_mask.grid.values, axis=0))
            data_dbzs_sev_1.append(np.delete(data_gen.dbz_grid.values,  data_mask.grid.values, axis=0))
            data_ctts_sev_1.append(np.delete(data_gen.ctt_grid.values,  data_mask.grid.values, axis=0))
            data_uh25_sev_1.append(np.delete(data_gen.uh25_grid.values, data_mask.grid.values, axis=0))
            data_uh03_sev_1.append(np.delete(data_gen.uh03_grid.values, data_mask.grid.values, axis=0))

        data_temp_sev_1_patches = self.flatten_arraylist(data_temp_sev_1)
        data_temp_sev_3_patches = self.flatten_arraylist(data_temp_sev_3)
        data_temp_sev_5_patches = self.flatten_arraylist(data_temp_sev_5)
        data_temp_sev_7_patches = self.flatten_arraylist(data_temp_sev_7)

        data_evwd_sev_1_patches = self.flatten_arraylist(data_evwd_sev_1)
        data_evwd_sev_3_patches = self.flatten_arraylist(data_evwd_sev_3)
        data_evwd_sev_5_patches = self.flatten_arraylist(data_evwd_sev_5)
        data_evwd_sev_7_patches = self.flatten_arraylist(data_evwd_sev_7)

        data_euwd_sev_1_patches = self.flatten_arraylist(data_euwd_sev_1)
        data_euwd_sev_3_patches = self.flatten_arraylist(data_euwd_sev_3)
        data_euwd_sev_5_patches = self.flatten_arraylist(data_euwd_sev_5)
        data_euwd_sev_7_patches = self.flatten_arraylist(data_euwd_sev_7)

        data_qvap_sev_1_patches = self.flatten_arraylist(data_qvap_sev_1)
        data_qvap_sev_3_patches = self.flatten_arraylist(data_qvap_sev_3)
        data_qvap_sev_5_patches = self.flatten_arraylist(data_qvap_sev_5)
        data_qvap_sev_7_patches = self.flatten_arraylist(data_qvap_sev_7)

        data_pres_sev_1_patches = self.flatten_arraylist(data_pres_sev_1)
        data_pres_sev_3_patches = self.flatten_arraylist(data_pres_sev_3)
        data_pres_sev_5_patches = self.flatten_arraylist(data_pres_sev_5)
        data_pres_sev_7_patches = self.flatten_arraylist(data_pres_sev_7)
        
        data_wwnd_sev_1_patches = self.flatten_arraylist(data_wwnd_sev_1)
        data_wwnd_sev_3_patches = self.flatten_arraylist(data_wwnd_sev_3)
        data_wwnd_sev_5_patches = self.flatten_arraylist(data_wwnd_sev_5)
        data_wwnd_sev_7_patches = self.flatten_arraylist(data_wwnd_sev_7)

        data_maxw_sev_1_patches = self.flatten_arraylist(data_maxw_sev_1)
        data_dbzs_sev_1_patches = self.flatten_arraylist(data_dbzs_sev_1)
        data_ctts_sev_1_patches = self.flatten_arraylist(data_ctts_sev_1)
        data_uh25_sev_1_patches = self.flatten_arraylist(data_uh25_sev_1)
        data_uh03_sev_1_patches = self.flatten_arraylist(data_uh03_sev_1)
        
        data_assemble = xr.Dataset({
                       'temp_sev_1':(['patch','y','x'], np.array(data_temp_sev_1_patches)), 'temp_sev_3':(['patch','y','x'], np.array(data_temp_sev_3_patches)),
                       'temp_sev_5':(['patch','y','x'], np.array(data_temp_sev_5_patches)), 'temp_sev_7':(['patch','y','x'], np.array(data_temp_sev_7_patches)),

                       'evwd_sev_1':(['patch','y','x'], np.array(data_evwd_sev_1_patches)), 'evwd_sev_3':(['patch','y','x'], np.array(data_evwd_sev_3_patches)),
                       'evwd_sev_5':(['patch','y','x'], np.array(data_evwd_sev_5_patches)), 'evwd_sev_7':(['patch','y','x'], np.array(data_evwd_sev_7_patches)),

                       'euwd_sev_1':(['patch','y','x'], np.array(data_euwd_sev_1_patches)), 'euwd_sev_3':(['patch','y','x'], np.array(data_euwd_sev_3_patches)),
                       'euwd_sev_5':(['patch','y','x'], np.array(data_euwd_sev_5_patches)), 'euwd_sev_7':(['patch','y','x'], np.array(data_euwd_sev_7_patches)),
                      
                       'qvap_sev_1':(['patch','y','x'], np.array(data_qvap_sev_1_patches)), 'qvap_sev_3':(['patch','y','x'], np.array(data_qvap_sev_3_patches)),
                       'qvap_sev_5':(['patch','y','x'], np.array(data_qvap_sev_5_patches)), 'qvap_sev_7':(['patch','y','x'], np.array(data_qvap_sev_7_patches)),
                      
                       'pres_sev_1':(['patch','y','x'], np.array(data_pres_sev_1_patches)), 'pres_sev_3':(['patch','y','x'], np.array(data_pres_sev_3_patches)),
                       'pres_sev_5':(['patch','y','x'], np.array(data_pres_sev_5_patches)), 'pres_sev_7':(['patch','y','x'], np.array(data_pres_sev_7_patches)),
            
                       'wwnd_sev_1':(['patch','y','x'], np.array(data_wwnd_sev_1_patches)), 'wwnd_sev_3':(['patch','y','x'], np.array(data_wwnd_sev_3_patches)),
                       'wwnd_sev_5':(['patch','y','x'], np.array(data_wwnd_sev_5_patches)), 'wwnd_sev_7':(['patch','y','x'], np.array(data_wwnd_sev_7_patches)),

                       'maxw_sev_1':(['patch','y','x'], np.array(data_maxw_sev_1_patches)), 'dbzs_sev_1':(['patch','y','x'], np.array(data_dbzs_sev_1_patches)),
                       'ctts_sev_1':(['patch','y','x'], np.array(data_ctts_sev_1_patches)), 'uh25_sev_1':(['patch','y','x'], np.array(data_uh25_sev_1_patches)),
                       'uh03_sev_1':(['patch','y','x'], np.array(data_uh03_sev_1_patches)), })

        data_assemble.to_netcdf(f"/{self.working_directory}/{self.climate}_nonuh{self.threshold1}_{self.mask_str}_{time.strftime('%m')}.nc")
        
        print(f"Non exceedances for {time.strftime('%m')} complete...")
        return


