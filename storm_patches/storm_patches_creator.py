###########################################################################
###########################################################################
#
# Maria J. Molina
# National Center for Atmospheric Research
#
###########################################################################
###########################################################################


#-------------------------------------

import numpy as np
import xarray as xr
import pandas as pd
from datetime import timedelta
import multiprocessing as mp

from hagelslag.hagelslag.processing.tracker import label_storm_objects, extract_storm_patches

#-------------------------------------


class storm_patch_creator:
    
    
    def __init__(self, date1, date2, climate, destination_path, 
                 min_dbz=20, max_dbz=40, patch_radius=16, method='ws',
                 dbz_path = '/gpfs/fs1/collections/rda/data/ds612.0/', 
                 num_cpus=36):
        
        """
        Class instantiation of storm_patch_creator class.
        
        Here we will create the storm patches that will be used for training deep learning models.
        
        Parameters
        ----------
        date1: start date for storm patch extraction (str; e.g., 2012-10-01)
        date2: end date for storm patch extraction (str; e.g., 2012-10-01)
        climate: whether working on the current or future climate (str)
        destination_path: where storm patch files will be saved (str)
        min_dbz: the minimum dbz threshold for the mask using storm object identification method (str; default 20)
        max_dbz: the maximum dbz threshold for the mask using storm object identification method (str; default 40)
        patch_radius: number of grid points from center of mass to extract (str; default 16)
        method: "ew" for enhanced watershed, "ws" for regular watershed, and "hyst" for hysteresis (str; default watershed)
        dbz_path: directory where the radar reflectivity data is contained (str)
        num_cpus: number of CPUs for to use in a node for parallelizing extractions (int; default 36)
        
        """
        
        self.date1 = date1
        self.date2 = date2

        if climate != 'current' and climate != 'future':
            raise Exception("Please enter current of future climate as choice analysis.")
        if climate == 'current' or climate == 'future':
            self.climate = climate

        if self.climate == 'current':
            self.filename = 'CTRL'
        if self.climate == 'future':
            self.filename = 'PGW'

        self.destination_path = destination_path

        self.min_dbz = min_dbz
        self.max_dbz = max_dbz
        self.patch_radius = patch_radius
        
        if method != 'ws' and method != 'ew' and method != 'hyst':
            raise Exception("Please enter ws, ew, or hyst method for identifying storm object.")
        if method == 'ws' or method == 'ew' or method == 'hyst':
            self.method = method
        
        self.dbz_path = dbz_path
        self.num_cpus = num_cpus
        

        
    def generate_timestring(self):
        
        """
            Function to generate the timestring for looping.
            Recommend cycling through one year for optimal parallel job distribution on a Cheyenne node.
        """
        
        return pd.date_range(self.date1, self.date2, freq='D')

    
    
    def total_pixels(self):
        
        """
            Function to help compute the total number of grid boxes (or pixels) in a storm patch.
        """
        return (self.patch_radius*2)*(self.patch_radius*2)
    
    

    def time_slice_help(self, month):
        
        """
            Function to help slice reflectivity files that were saved in three month intervals.
        """
        
        if month == 1 or month == 2 or month == 3:
            mon_1 = '01'; mon_2 = '03'
        if month == 4 or month == 5 or month == 6:
            mon_1 = '04'; mon_2 = '06'
        if month == 7 or month == 8 or month == 9:
            mon_1 = '07'; mon_2 = '09'
        if month == 10 or month == 11 or month == 12:
            mon_1 = '10'; mon_2 = '12'
        return mon_1, mon_2
    

    def create_patches_hourly(self):
        
        """
            Function to create the hourly storm patches using dbz. 
            
            Here, we activate the multiprocessing function, and parallelize the creation of these storm patches for efficiency.
            This is done because it is not necessary to communicate between processes, we will be saving each file independent of ongoing processes.
        """
    
    
        times_thisfile = self.generate_timestring()
    
        ###########################################################################
        
        #default values for the WRF CONUS1 dataset below, 
        #including the full time range for indexing simulations and slicing the CONUS spatially for efficiency 
        #(we don't care about storms over water
        
        total_times = pd.date_range('2000-10-01','2013-09-30 23:00:00',freq='H')
        total_times_indexes = np.arange(0,total_times.shape[0],1)
        the1=135; the2=650; the3=500; the4=1200
        
        ###########################################################################

        #start processes in one core.
        pool = mp.Pool(self.num_cpus)

        results = []

        def collect_result(result):
            global results
            results.append(result)

        for num, thedate in enumerate(times_thisfile):

            mon_1, mon_2 = self.time_slice_help(thedate.month)

            data_path = f'/{self.dbz_path}/{self.filename}radrefl/REFLC/wrf2d_d01_{self.filename}_REFLC_10CM_'+str(thedate.year)+mon_1+'-'+str(thedate.year)+mon_2+'.nc'
            data = xr.open_dataset(data_path)

            print(num, f"start {times_thisfile[num].strftime('%Y%m%d')}")

            data_refl = data.REFLC_10CM.sel(Time=slice(times_thisfile[num],times_thisfile[num]+timedelta(hours=23)))
            data_reflec = data_refl.values[:,the1:the2,the3:the4]
            data_latitu = data.XLAT.values[the1:the2,the3:the4]
            data_longit = data.XLONG.values[the1:the2,the3:the4]

            thetimes = total_times_indexes[np.where(total_times==pd.to_datetime(data_refl.Time.values[0]))[0][0]:
                                           np.where(total_times==pd.to_datetime(data_refl.Time.values[-1])+timedelta(hours=1))[0][0]]

            #double checking that all times contained in original data (wrf) are enclosed within the search time string for that date
            thetimes = np.array([thetimes[i] for i in data_refl.Time.dt.hour])

            if len(thetimes) == 0:
                raise Exception(f"Why is there no time for {times_thisfile[num].strftime('%Y-%m-%d')}")

            pool.apply_async(self.parallelizing_the_func, args=(num, data_reflec, data_latitu, data_longit, thetimes, times_thisfile), callback=collect_result)

        pool.close()
        pool.join()  # block at this line until all processes are done
        
        print("completed")


    
    def parallelizing_the_func(self, num, data, lats, lons, thetimes, times_thisfile):

        """
            Function to run script that finds storm patches in WRF CONUS1 dataset.
            Saves output to Xarray netCDF files with metadata.

        """

        thelabels = label_storm_objects(data, min_intensity=self.min_dbz, max_intensity=self.max_dbz, 
                                        min_area=1, max_area=100, max_range=1, increment=1, gaussian_sd=0)

        print(num, "postlabel")


        storm_objs = extract_storm_patches(label_grid=thelabels, data=data, x_grid=lons, y_grid=lats,
                                           times=thetimes, dx=1, dt=1, patch_radius=self.patch_radius)

        print(num, f"done {times_thisfile[num].strftime('%Y-%m-%d')}")

        data_assemble = xr.Dataset({

             'grid':(['starttime','y','x'],
                 np.array([other.timesteps[0] for obj in storm_objs for other in obj if other.timesteps[0].shape[0]*other.timesteps[0].shape[1] == self.total_pixels()])),
             'mask':(['starttime','y','x'],
                 np.array([other.masks[0] for obj in storm_objs for other in obj if other.timesteps[0].shape[0]*other.timesteps[0].shape[1] == self.total_pixels()])),
             'row_indices':(['starttime','y','x'],
                 np.array([other.i[0] for obj in storm_objs for other in obj if other.timesteps[0].shape[0]*other.timesteps[0].shape[1] == self.total_pixels()])),
             'col_indices':(['starttime','y','x'],
                 np.array([other.j[0] for obj in storm_objs for other in obj if other.timesteps[0].shape[0]*other.timesteps[0].shape[1] == self.total_pixels()])),
             'lats':(['starttime','y','x'],
                 np.array([other.y[0] for obj in storm_objs for other in obj if other.timesteps[0].shape[0]*other.timesteps[0].shape[1] == self.total_pixels()])),
             'lons':(['starttime','y','x'],
                 np.array([other.x[0] for obj in storm_objs for other in obj if other.timesteps[0].shape[0]*other.timesteps[0].shape[1] == self.total_pixels()])),
            },

             coords=
            {'starttime':(['starttime'],
                        np.array([other.start_time for obj in storm_objs for other in obj if other.timesteps[0].shape[0]*other.timesteps[0].shape[1] == self.total_pixels()])),
             'endtime':
                        np.array([other.end_time for obj in storm_objs for other in obj if other.timesteps[0].shape[0]*other.timesteps[0].shape[1] == self.total_pixels()]),
             'x_speed':(['starttime'],
                        np.array([other.u[0] for obj in storm_objs for other in obj if other.timesteps[0].shape[0]*other.timesteps[0].shape[1] == self.total_pixels()])),
             'y_speed':(['starttime'],
                        np.array([other.v[0] for obj in storm_objs for other in obj if other.timesteps[0].shape[0]*other.timesteps[0].shape[1] == self.total_pixels()]))
            })


        data_assemble.to_netcdf(f"/{self.destination_path}/{self.climate}_conus1_{times_thisfile[num].strftime('%Y%m%d')}.nc")

        return(num)
    
    


        