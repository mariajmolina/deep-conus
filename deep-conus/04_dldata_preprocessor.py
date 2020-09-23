import numpy as np
import xarray as xr
import pandas as pd
import multiprocessing as mp


class PreprocessData:

    """Class instantiation of PreprocessData:
    
    Here we will be preprocessing data for deep learning model training.

    Attributes:
        working_directory (str): The path to the directory where the deep learning preprocessing files will be saved and worked from. 
        stormpatch_path (str): Where the storm patch files were saved.
        climate (str): The climate period to derive deep learning data for. Options are ``current`` or ``future``.
        method (str): Variable choice(s) for preprocessing data, which include ``UHsingle`` and ``UHdouble``.
        threshold1 (int): The threshold to use for the chosen method. This value will delineate some form of ``severe`` and ``non-severe`` storm patches.
        threshold2 (int): The second threshold for ``UHdual`` method. Defaults to ``None``.
        mask (boolean): Whether the threshold will be applied within the storm patch mask or within the full storm patch. Defaults to ``False``.
        num_cpus (int): Number of CPUs to use in a node for parallelizing extractions. Defaults to 36 (Cheyenne compute nodes contain 36).
        
    Raises:
        Exceptions: Checks whether correct values were input for ``climate`` and ``method``.
        
    Todo:
        * Add ``UHdual`` method functionality.
        
    """
        
    def __init__(self, working_directory, stormpatch_path, climate, method, threshold1, threshold2=None, mask=False, num_cpus=36):

        self.working_directory=working_directory
        self.stormpatch_path=stormpatch_path
        if climate!='current' and climate!='future':
            raise Exception("Please enter current or future for climate option.")
        else:
            self.climate=climate
        if method!='UHsingle' and method!='UHdual':
            raise Exception("Incorrect method. Please enter UHsingle or UHdouble.")
        else:
            self.method=method
            self.threshold1=threshold1
            if self.method=='UHdual':
                if self.threshold2:
                    self.threshold2=threshold2
                if not self.threshold2: 
                    raise Exception("Please enter a threshold for UH 0-3 km (dual UH method).")
        self.mask=mask
        if not self.mask:
            self.mask_str='nomask'
        if self.mask:
            self.mask_str='mask'
        self.num_cpus=num_cpus

    def generate_time_full(self):
        
        """Creation of the full time period that will be looped through for extracting storm patch information.
        Only considering December-May months due to warm season bias over the central CONUS. The CONUS1 simulations
        were run for 2000-2013.
        
        Returns:
            Pandas date range (DatetimeIndex).
        
        """
        return pd.date_range('2000-10-01','2013-09-30',freq='MS')[(pd.date_range('2000-10-01','2013-09-30',freq='MS').month==12)|
                                                                  (pd.date_range('2000-10-01','2013-09-30',freq='MS').month==1)|
                                                                  (pd.date_range('2000-10-01','2013-09-30',freq='MS').month==2)|
                                                                  (pd.date_range('2000-10-01','2013-09-30',freq='MS').month==3)|
                                                                  (pd.date_range('2000-10-01','2013-09-30',freq='MS').month==4)|
                                                                  (pd.date_range('2000-10-01','2013-09-30',freq='MS').month==5)]

    def create_data_indices(self, time):
        
        """Split the loaded data into categories based on the method chosen and save the first intermediary files. Here we create 
        the indices of the storm patches that satisfy method criteria for later use.
        
        Args:
            time (DatetimeIndex): Time object from pandas date range.
        
        """
        if self.method=='UHsingle':
            if not self.mask:
                data=xr.open_mfdataset(f"/{self.stormpatch_path}/{self.climate}_SP3hourly_{time.strftime('%Y%m')}*.nc", combine='by_coords')
                data_assemble=xr.Dataset({'grid':(['x'], np.argwhere(data.uh25_grid.values.max(axis=(1,2)) > self.threshold1)[:,0])})
                data_assemble.to_netcdf(f"/{self.working_directory}/{self.climate}_indx{self.threshold1}_{self.mask_str}_{time.strftime('%Y')}{time.strftime('%m')}.nc")
            if self.mask:
                data=xr.open_mfdataset(f"/{self.stormpatch_path}/{self.climate}_SP3hourly_{time.strftime('%Y%m')}*.nc", combine='by_coords')
                data_assemble=xr.Dataset({'grid':(['x'], np.argwhere(data.uh25_grid.where(data.mask).max(axis=(1,2), skipna=True).values > self.threshold1)[:,0])})
                data_assemble.to_netcdf(f"/{self.working_directory}/{self.climate}_indx{self.threshold1}_{self.mask_str}_{time.strftime('%Y')}{time.strftime('%m')}.nc")
                
        if self.method=='UHdual':
            if not self.mask:
                ### To do
                return
            if self.mask:
                ### To do
                return

    def parallelizing_indxs(self):
        
        """Activate the multiprocessing function to parallelize the functions.
        
        """
        print(f"Starting jobs...")
        timearray=self.generate_time_full()
        pool1=mp.Pool(self.num_cpus)
        for time in timearray:
            print(f"Extracting {time.strftime('%Y-%m')} indices...")
            pool1.apply_async(self.create_data_indices, args=([time]))
        pool1.close()
        pool1.join()
        print(f"Completed the jobs.")
            
    def generate_time_month(self, month_int):
        
        """Creation of the time array that will be looped through for extracting storm patch information.
        
        Args:
            month_int (int): The month being used for the time array (2000-2013 years).
            
        Returns:
            Pandas date range (DatetimeIndex) for the respective month.
            
        """
        return pd.date_range('2000-10-01','2013-09-30',freq='MS')[(pd.date_range('2000-10-01','2013-09-30',freq='MS').month==month_int)]
            
    def apply_exceed_mask(self, data_var, data_mask, level):
        
        """Function to retain the patches that exceeded the threshold.
        
        Args:
            data_var (Xarray data array): The variable's data.
            data_mask (Xarray data array): The storm patch mask.
            level (int): The dataset level coordinate. This could be 0, 1, 2, or 3.
            
        Returns:
            Xarray data array of the variable for the storm patches that exceeded the method's threshold.
        
        """        
        return data_var.var_grid.sel(levels=level)[data_mask.grid.values,:,:]
    
    def apply_notexceed_mask(self, data_var, data_mask, level):

        """Function to retain the patches that did not exceed the threshold.
        
        Args:
            data_var (Xarray data array): The variable's data.
            data_mask (Xarray data array): The storm patch mask.
            level (int): The dataset level coordinate. This could be 0, 1, 2, or 3.
            
        Returns:
            Numpy array of the variable for the storm patches that did not exceed the method's threshold.
        
        """    
        return np.delete(data_var.var_grid.sel(levels=level).values, data_mask.grid.values, axis=0)

    def flatten_list(self, array):
        
        """Function to flatten the created list of Xarray data arrays.
        
        Args:
            array (list): The list of Xarray data arrays.
            
        Returns:
            Flattened list of Xarray data arrays.
        
        """
        return [j for i in array for j in i.values]

    def flatten_arraylist(self, array):
        
        """Function to flatten the created list of numpy arrays.
        
        Args:
            array (list): The list of numpy arrays.
            
        Returns:
            Flattened list of numpy arrays.
        
        """
        return [j for i in array for j in i]

    def month_translate(self, num):
        
        """Convert integer month to string month.
        
        Args:
            num (int): Input month.
            
        Returns:
            out (str): Input month as string.
            
        Raises:
            ValueError: If the month is not within the study's range (Dec-May).
            
        """
        var={12:'December',
               1:'January',
               2:'February',
               3:'March',
               4:'April',
               5:'May'}
        try:
            out=var[num]
            return out
        except:
            raise ValueError("Please enter month integer from Dec-May.")

    def run_months(self, months=np.array([12,1,2,3,4,5]), uh=True, nouh=True):
        
        """Function to automate and parallelize the creation of the exceedance/nonexceedance files.
        
        Args:
            months (int array): Months to iterate through.
            uh (boolean): Whether to compute analysis for threshold exceedances. Defaults to ``True``.
            nouh(boolean): Whether to compute analysis for threshold non-exceedances. Defaults to ``True``.
        
        """ 
        pool2=mp.Pool(self.num_cpus)
        for mo in months:
            if uh:
                print(f"Creating {self.month_translate(mo)} patches of threshold exceedances...")
                pool2.apply_async(self.create_files_exceed_threshold, args=([mo]))
            if nouh:
                print(f"Creating {self.month_translate(mo)} patches of threshold non-exceedances...")
                pool2.apply_async(self.create_files_notexceed_threshold, args=([mo]))
        pool2.close()
        pool2.join()
        print(f"Completed the jobs.")

    def create_files_exceed_threshold(self, month_int):
        
        """Create and save files containing the environment patches for storms that exceeded the threshold.
        Data files being opened contain the storm patches, not the full CONUS WRF domain.
        
        Args:
            month_int (int): Month for analysis.
        
        """
        time_temp=self.generate_time_month(month_int)
        
        data_temp_sev_1=[]; data_temp_sev_3=[]; data_temp_sev_5=[]; data_temp_sev_7=[]; data_evwd_sev_1=[]; data_evwd_sev_3=[]
        data_euwd_sev_1=[]; data_euwd_sev_3=[]; data_euwd_sev_5=[]; data_euwd_sev_7=[]; data_evwd_sev_5=[]; data_evwd_sev_7=[]
        data_qvap_sev_1=[]; data_qvap_sev_3=[]; data_qvap_sev_5=[]; data_qvap_sev_7=[]; data_dbzs_sev_1=[]; data_maxw_sev_1=[]
        data_pres_sev_1=[]; data_pres_sev_3=[]; data_pres_sev_5=[]; data_pres_sev_7=[]; data_ctts_sev_1=[]; data_mask_sev_1=[]
        data_wwnd_sev_1=[]; data_wwnd_sev_3=[]; data_wwnd_sev_5=[]; data_wwnd_sev_7=[]; data_uh25_sev_1=[]; data_uh03_sev_1=[]

        for time in time_temp:
            print(f"opening files for {time.strftime('%Y')}{time.strftime('%m')}")
            data_mask=xr.open_mfdataset(
                f"/{self.working_directory}/{self.climate}_indx{self.threshold1}_{self.mask_str}_{time.strftime('%Y')}{time.strftime('%m')}.nc",    
                combine='by_coords')
            
            data_temp=xr.open_mfdataset(f"/{self.stormpatch_path}/{self.climate}_SP3hourly_tk_{time.strftime('%Y%m')}*.nc",     combine='by_coords')
            data_evwd=xr.open_mfdataset(f"/{self.stormpatch_path}/{self.climate}_SP3hourly_ev_{time.strftime('%Y%m')}*.nc",     combine='by_coords')
            data_euwd=xr.open_mfdataset(f"/{self.stormpatch_path}/{self.climate}_SP3hourly_eu_{time.strftime('%Y%m')}*.nc",     combine='by_coords')
            data_qvap=xr.open_mfdataset(f"/{self.stormpatch_path}/{self.climate}_SP3hourly_qvapor_{time.strftime('%Y%m')}*.nc", combine='by_coords')
            data_pres=xr.open_mfdataset(f"/{self.stormpatch_path}/{self.climate}_SP3hourly_p_{time.strftime('%Y%m')}*.nc",      combine='by_coords')
            data_wwnd=xr.open_mfdataset(f"/{self.stormpatch_path}/{self.climate}_SP3hourly_w_{time.strftime('%Y%m')}*.nc",      combine='by_coords')
            data_maxw=xr.open_mfdataset(f"/{self.stormpatch_path}/{self.climate}_SP3hourly_maxw_{time.strftime('%Y%m')}*.nc",   combine='by_coords')
            data_gen =xr.open_mfdataset(f"/{self.stormpatch_path}/{self.climate}_SP3hourly_{time.strftime('%Y%m')}*.nc",        combine='by_coords')

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
            data_mask_sev_1.append(data_gen.mask[data_mask.grid.values,:,:])

        data_temp_sev_1_patches=self.flatten_list(data_temp_sev_1)
        data_temp_sev_3_patches=self.flatten_list(data_temp_sev_3)
        data_temp_sev_5_patches=self.flatten_list(data_temp_sev_5)
        data_temp_sev_7_patches=self.flatten_list(data_temp_sev_7)

        data_evwd_sev_1_patches=self.flatten_list(data_evwd_sev_1)
        data_evwd_sev_3_patches=self.flatten_list(data_evwd_sev_3)
        data_evwd_sev_5_patches=self.flatten_list(data_evwd_sev_5)
        data_evwd_sev_7_patches=self.flatten_list(data_evwd_sev_7)

        data_euwd_sev_1_patches=self.flatten_list(data_euwd_sev_1)
        data_euwd_sev_3_patches=self.flatten_list(data_euwd_sev_3)
        data_euwd_sev_5_patches=self.flatten_list(data_euwd_sev_5)
        data_euwd_sev_7_patches=self.flatten_list(data_euwd_sev_7)

        data_qvap_sev_1_patches=self.flatten_list(data_qvap_sev_1)
        data_qvap_sev_3_patches=self.flatten_list(data_qvap_sev_3)
        data_qvap_sev_5_patches=self.flatten_list(data_qvap_sev_5)
        data_qvap_sev_7_patches=self.flatten_list(data_qvap_sev_7)

        data_pres_sev_1_patches=self.flatten_list(data_pres_sev_1)
        data_pres_sev_3_patches=self.flatten_list(data_pres_sev_3)
        data_pres_sev_5_patches=self.flatten_list(data_pres_sev_5)
        data_pres_sev_7_patches=self.flatten_list(data_pres_sev_7)
        
        data_wwnd_sev_1_patches=self.flatten_list(data_wwnd_sev_1)
        data_wwnd_sev_3_patches=self.flatten_list(data_wwnd_sev_3)
        data_wwnd_sev_5_patches=self.flatten_list(data_wwnd_sev_5)
        data_wwnd_sev_7_patches=self.flatten_list(data_wwnd_sev_7)

        data_maxw_sev_1_patches=self.flatten_list(data_maxw_sev_1)
        data_dbzs_sev_1_patches=self.flatten_list(data_dbzs_sev_1)
        data_ctts_sev_1_patches=self.flatten_list(data_ctts_sev_1)
        data_uh25_sev_1_patches=self.flatten_list(data_uh25_sev_1)
        data_uh03_sev_1_patches=self.flatten_list(data_uh03_sev_1)
        data_mask_sev_1_patches=self.flatten_list(data_mask_sev_1)
        
        data_assemble=xr.Dataset({
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
                       'uh03_sev_1':(['patch','y','x'], np.array(data_uh03_sev_1_patches)), 'mask_sev_1':(['patch','y','x'], np.array(data_mask_sev_1_patches))})

        data_assemble.to_netcdf(f"/{self.working_directory}/{self.climate}_uh{self.threshold1}_{self.mask_str}_{time.strftime('%m')}.nc")
        print(f"Exceedances for {time.strftime('%m')} complete...")

    def create_files_notexceed_threshold(self, month_int):

        """Create files containing environment patches for storms that did not exceed the threshold.
        Data files being opened contain the storm patches, not the full CONUS WRF domain.
        
        Args:
            month_int (int): Month for analysis.
        
        """
        time_temp=self.generate_time_month(month_int)

        data_temp_sev_1=[]; data_temp_sev_3=[]; data_temp_sev_5=[]; data_temp_sev_7=[]; data_evwd_sev_1=[]; data_evwd_sev_3=[]
        data_euwd_sev_1=[]; data_euwd_sev_3=[]; data_euwd_sev_5=[]; data_euwd_sev_7=[]; data_evwd_sev_5=[]; data_evwd_sev_7=[]
        data_qvap_sev_1=[]; data_qvap_sev_3=[]; data_qvap_sev_5=[]; data_qvap_sev_7=[]; data_dbzs_sev_1=[]; data_maxw_sev_1=[]
        data_pres_sev_1=[]; data_pres_sev_3=[]; data_pres_sev_5=[]; data_pres_sev_7=[]; data_ctts_sev_1=[]; data_mask_sev_1=[]
        data_wwnd_sev_1=[]; data_wwnd_sev_3=[]; data_wwnd_sev_5=[]; data_wwnd_sev_7=[]; data_uh25_sev_1=[]; data_uh03_sev_1=[]

        for time in time_temp:
            print(f"opening files for {time.strftime('%Y')}{time.strftime('%m')}")
            data_mask=xr.open_mfdataset(
                f"/{self.working_directory}/{self.climate}_indx{self.threshold1}_{self.mask_str}_{time.strftime('%Y')}{time.strftime('%m')}.nc",    
                combine='by_coords')
            
            data_temp=xr.open_mfdataset(f"/{self.stormpatch_path}/{self.climate}_SP3hourly_tk_{time.strftime('%Y%m')}*.nc",     combine='by_coords')
            data_evwd=xr.open_mfdataset(f"/{self.stormpatch_path}/{self.climate}_SP3hourly_ev_{time.strftime('%Y%m')}*.nc",     combine='by_coords')
            data_euwd=xr.open_mfdataset(f"/{self.stormpatch_path}/{self.climate}_SP3hourly_eu_{time.strftime('%Y%m')}*.nc",     combine='by_coords')
            data_qvap=xr.open_mfdataset(f"/{self.stormpatch_path}/{self.climate}_SP3hourly_qvapor_{time.strftime('%Y%m')}*.nc", combine='by_coords')
            data_pres=xr.open_mfdataset(f"/{self.stormpatch_path}/{self.climate}_SP3hourly_p_{time.strftime('%Y%m')}*.nc",      combine='by_coords')
            data_wwnd=xr.open_mfdataset(f"/{self.stormpatch_path}/{self.climate}_SP3hourly_w_{time.strftime('%Y%m')}*.nc",      combine='by_coords')
            data_maxw=xr.open_mfdataset(f"/{self.stormpatch_path}/{self.climate}_SP3hourly_maxw_{time.strftime('%Y%m')}*.nc",   combine='by_coords')
            data_gen =xr.open_mfdataset(f"/{self.stormpatch_path}/{self.climate}_SP3hourly_{time.strftime('%Y%m')}*.nc",        combine='by_coords')
            
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
            data_mask_sev_1.append(np.delete(data_gen.mask.values, data_mask.grid.values, axis=0))

        data_temp_sev_1_patches=self.flatten_arraylist(data_temp_sev_1)
        data_temp_sev_3_patches=self.flatten_arraylist(data_temp_sev_3)
        data_temp_sev_5_patches=self.flatten_arraylist(data_temp_sev_5)
        data_temp_sev_7_patches=self.flatten_arraylist(data_temp_sev_7)

        data_evwd_sev_1_patches=self.flatten_arraylist(data_evwd_sev_1)
        data_evwd_sev_3_patches=self.flatten_arraylist(data_evwd_sev_3)
        data_evwd_sev_5_patches=self.flatten_arraylist(data_evwd_sev_5)
        data_evwd_sev_7_patches=self.flatten_arraylist(data_evwd_sev_7)

        data_euwd_sev_1_patches=self.flatten_arraylist(data_euwd_sev_1)
        data_euwd_sev_3_patches=self.flatten_arraylist(data_euwd_sev_3)
        data_euwd_sev_5_patches=self.flatten_arraylist(data_euwd_sev_5)
        data_euwd_sev_7_patches=self.flatten_arraylist(data_euwd_sev_7)

        data_qvap_sev_1_patches=self.flatten_arraylist(data_qvap_sev_1)
        data_qvap_sev_3_patches=self.flatten_arraylist(data_qvap_sev_3)
        data_qvap_sev_5_patches=self.flatten_arraylist(data_qvap_sev_5)
        data_qvap_sev_7_patches=self.flatten_arraylist(data_qvap_sev_7)

        data_pres_sev_1_patches=self.flatten_arraylist(data_pres_sev_1)
        data_pres_sev_3_patches=self.flatten_arraylist(data_pres_sev_3)
        data_pres_sev_5_patches=self.flatten_arraylist(data_pres_sev_5)
        data_pres_sev_7_patches=self.flatten_arraylist(data_pres_sev_7)
        
        data_wwnd_sev_1_patches=self.flatten_arraylist(data_wwnd_sev_1)
        data_wwnd_sev_3_patches=self.flatten_arraylist(data_wwnd_sev_3)
        data_wwnd_sev_5_patches=self.flatten_arraylist(data_wwnd_sev_5)
        data_wwnd_sev_7_patches=self.flatten_arraylist(data_wwnd_sev_7)

        data_maxw_sev_1_patches=self.flatten_arraylist(data_maxw_sev_1)
        data_dbzs_sev_1_patches=self.flatten_arraylist(data_dbzs_sev_1)
        data_ctts_sev_1_patches=self.flatten_arraylist(data_ctts_sev_1)
        data_uh25_sev_1_patches=self.flatten_arraylist(data_uh25_sev_1)
        data_uh03_sev_1_patches=self.flatten_arraylist(data_uh03_sev_1)
        data_mask_sev_1_patches=self.flatten_arraylist(data_mask_sev_1)
        
        data_assemble=xr.Dataset({
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
                       'uh03_sev_1':(['patch','y','x'], np.array(data_uh03_sev_1_patches)), 'mask_sev_1':(['patch','y','x'], np.array(data_mask_sev_1_patches))})

        data_assemble.to_netcdf(f"/{self.working_directory}/{self.climate}_nonuh{self.threshold1}_{self.mask_str}_{time.strftime('%m')}.nc")
        print(f"Non exceedances for {time.strftime('%m')} complete...")
