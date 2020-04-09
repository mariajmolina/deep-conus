import numpy as np
import xarray as xr
import pandas as pd
from datetime import timedelta
import calendar

import multiprocessing as mp

import cartopy.io.shapereader as shpreader
import shapely.geometry as sgeom
from shapely.ops import unary_union
from shapely.prepared import prep

from hagelslag.hagelslag.processing.tracker import label_storm_objects, extract_storm_patches


class StormPatchCreator:
    
    """Class instantiation of StormPatchCreator:
        
    Here we will create the storm patches that will be used for training deep learning models.
        
    Attributes:
        date1 (str): Start date for storm patch extraction (e.g., ``2012-10-01``).
        date2 (str): End date for storm patch extraction (e.g., ``2012-10-01``).
        climate (str): Whether working on the ``current`` or ``future`` climate simulations.
        destination_path (str): Directory to save the storm patch files.
        min_dbz (int): The minimum dbz threshold to use for storm object identification method. Defaults to 20.
        max_dbz (int): The maximum dbz threshold to use for storm object identification method. Defaults to 40.
        patch_radius (int): The number of grid points to extract from the center of mass. Defaults to 16 for a 32x32 pixel storm patch.
        method (str): Method for identifying storm ojects. Options include ``ew`` for enhanced watershed, ``ws`` for regular watershed, 
                      and ``hyst`` for hysteresis. Defaults to ``ws``.
        dbz_path (str): Directory where the radar reflectivity data is contained.
        uh25_path (str): Directory where the updraft helicity (2-5-km) data is contained. Defaults to None.
        uh03_path (str): Directory where the updraft helicity (0-3-km) data is contained. Defaults to None.
        ctt_path (str): Directory where the cloud top temperature data is contained. Defaults to None.
        variable (str): Variable to extract for storm patches. Options include ``TK``, ``EU``, ``EV``, ``QVAPOR``, ``PRESS``, ``W_vert``, 
                        or ``WMAX``. Defaults to None.
        variable_path (str): Path to where the variable files are located. Defaults to None.
        num_cpus (int): Number of CPUs for to use in a node for parallelizing extractions. Defaults to 36 (Cheyenne compute nodes contain 36).
        
    Raises:
        Exceptions: Checks whether correct values were input for ``climate`` and ``method``.
        
    """
    
    def __init__(self, date1, date2, climate, destination_path, 
                 min_dbz=20, max_dbz=40, patch_radius=16, method='ws',
                 dbz_path='/gpfs/fs1/collections/rda/data/ds612.0/',
                 uh25_path=None, uh03_path=None, ctt_path=None, 
                 variable=None, variable_path=None,
                 num_cpus=36):
        
        self.date1=date1
        self.date2=date2

        if climate!='current' and climate!='future':
            raise Exception("Please enter current of future climate as choice analysis.")
        else:
            self.climate=climate

        if self.climate=='current':
            self.filename='CTRL'
        if self.climate=='future':
            self.filename='PGW'

        self.destination_path=destination_path
        self.min_dbz=min_dbz
        self.max_dbz=max_dbz
        self.patch_radius=patch_radius
        
        if method!='ws' and method!='ew' and method!='hyst':
            raise Exception("Please enter ws, ew, or hyst method for identifying storm object.")
        else:
            self.method=method
        
        self.dbz_path=dbz_path
        self.uh25_path=uh25_path
        self.uh03_path=uh03_path
        self.ctt_path=ctt_path
        self.variable=variable
        self.variable_path=variable_path
        self.num_cpus=num_cpus
        
        
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
               'PRESS':'P'
              }
        try:
            out=var[self.variable]
            return out
        except:
            raise ValueError("Please enter ``TK``, ``EU``, ``EV``, ``QVAPOR``, ``PRESS``, ``W_vert``, or ``WMAX`` as variable.")

        
    def generate_timestring(self):
        
        """Function to generate the timestrings for looping and computing variable over the chosen climate period. 
        Recommend cycling through one year for optimal parallel job distribution on a Cheyenne node.
            
        Returns:
            Pandas date range (DatetimeIndex).
        
        """
        return pd.date_range(self.date1, self.date2, freq='D')

    
    def total_pixels(self):
        
        """Function to help compute the total number of grid boxes (or pixels) in a storm patch.
        
        Returns:
            Total number of pixels in complete storm patch (e.g., 32 x 32 = 128).
            
        """
        return (self.patch_radius*2)*(self.patch_radius*2)
    

    def time_slice_help(self, month):
        
        """Function to help slice reflectivity files that were saved in three-month intervals.
        
        Args:
            month (int): Month.
            
        Returns:
            mon_1 (str): Lower bound month as two-digit string for opening three-month file.
            mon_2 (str): Upper bound month as two-digit string for opening three-month file.
            
        """
        if month==1 or month==2 or month==3:
            mon_1='01'; mon_2='03'
        if month==4 or month==5 or month==6:
            mon_1='04'; mon_2='06'
        if month==7 or month==8 or month==9:
            mon_1='07'; mon_2='09'
        if month==10 or month==11 or month==12:
            mon_1='10'; mon_2='12'
        return mon_1, mon_2
    
    
    def prep_land(self):
        
        """Function to generate landmass that will be used for identifying storms that occur over land and over water.
        
        Returns:
            Shapely PreparedGeometry object.
        
        """
        land_shp_fname=shpreader.natural_earth(resolution='50m', category='physical', name='land')
        land_geom=unary_union(list(shpreader.Reader(land_shp_fname).geometries()))
        return prep(land_geom) 
    
    
    def is_land(self, land, x, y):
        
        """Function to check if grid points are over land or not with previously generated landmass (prep_land).
        
        Args:
            land (PreparedGeometry object): Previously generated Shapely object.
            x, y (float): Longitude and latitude coordinate.
        
        Returns:
            Whether the latitude and longitude values correspond to over land or not (boolean).
            
        """
        return land.contains(sgeom.Point(x, y))
    

    def parallelizing_hourly_func(self):
        
        """Function to create the hourly storm patches using ``dbz``. Here, we activate the multiprocessing function, 
        and parallelize the creation of these storm patches for efficiency. This is done because it is not necessary 
        to communicate between processes, we will be saving each file independent of ongoing processes.
        
        """
        times_thisfile=self.generate_timestring()
    
        #default values for the WRF CONUS1 dataset below, 
        #including the full time range for indexing simulations and slicing the CONUS spatially for efficiency 
        #(we don't care about storms over water
        total_times=pd.date_range('2000-10-01','2013-10-01 00:00:00',freq='H')
        total_times_indexes=np.arange(0,total_times.shape[0],1)
        the1=135; the2=650; the3=500; the4=1200

        pool=mp.Pool(self.num_cpus)
        for num, thedate in enumerate(times_thisfile):
            mon_1, mon_2=self.time_slice_help(thedate.month)
            data_path=f'/{self.dbz_path}/{self.filename}radrefl/REFLC/wrf2d_d01_{self.filename}_REFLC_10CM_'+str(thedate.year)+mon_1+'-'+str(thedate.year)+mon_2+'.nc'
            data=xr.open_dataset(data_path)
            print(num, f"start {times_thisfile[num].strftime('%Y%m%d')}")
            data_refl=data.REFLC_10CM.sel(Time=slice(times_thisfile[num],times_thisfile[num]+timedelta(hours=23)))
            data_reflec=data_refl.values[:,the1:the2,the3:the4]
            data_latitu=data.XLAT.values[the1:the2,the3:the4]
            data_longit=data.XLONG.values[the1:the2,the3:the4]
            thetimes=total_times_indexes[np.where(total_times==pd.to_datetime(data_refl.Time.values[0]))[0][0]:
                                           np.where(total_times==pd.to_datetime(data_refl.Time.values[-1])+timedelta(hours=1))[0][0]]
            thetimes=np.array([thetimes[i] for i in data_refl.Time.dt.hour])
            if len(thetimes)==0:
                raise Exception(f"Why is there no time for {times_thisfile[num].strftime('%Y-%m-%d')}")
            pool.apply_async(self.create_patches_hourly, args=(num, data_reflec, data_latitu, data_longit, thetimes, times_thisfile))
        pool.close()
        pool.join()
        print("completed")

    
    def create_patches_hourly(self, num, data, lats, lons, thetimes, times_thisfile):

        """Function to find storm patches in WRF CONUS1 dataset. Saves output to Xarray dataset with metadata.
        
        Args:
            num (int): Number of job assignment (enumerated loop).
            data (numpy array): Dbz data to use for storm patch extraction.
            lats (numpy array): Latitudes of dbz data being used for storm patch extraction.
            lons (numpy array): Longitudes of dbz data being used for storm patch extraction.
            thetimes (numpy array): Time indices of the full time period of the climate simulations (2000-2013).
            times_thisfile (DatetimeIndex): Pandas date range.
            
        Returns:
            num (int): Number of job assignment (enumerated loop).
        
        """
        thelabels=label_storm_objects(data, method=self.method, min_intensity=self.min_dbz, max_intensity=self.max_dbz, 
                                        min_area=1, max_area=100, max_range=1, increment=1, gaussian_sd=0)
        print(num, "Postlabel")
        storm_objs=extract_storm_patches(label_grid=thelabels, data=data, x_grid=lons, y_grid=lats,
                                           times=thetimes, dx=1, dt=1, patch_radius=self.patch_radius)
        print(num, f"Done {times_thisfile[num].strftime('%Y-%m-%d')}")
        data_assemble=xr.Dataset({
             'grid':(['starttime','y','x'],
                 np.array([other.timesteps[0] for obj in storm_objs for other in obj if other.timesteps[0].shape[0]*other.timesteps[0].shape[1]==self.total_pixels()])),
             'mask':(['starttime','y','x'],
                 np.array([other.masks[0] for obj in storm_objs for other in obj if other.timesteps[0].shape[0]*other.timesteps[0].shape[1]==self.total_pixels()])),
             'row_indices':(['starttime','y','x'],
                 np.array([other.i[0] for obj in storm_objs for other in obj if other.timesteps[0].shape[0]*other.timesteps[0].shape[1]==self.total_pixels()])),
             'col_indices':(['starttime','y','x'],
                 np.array([other.j[0] for obj in storm_objs for other in obj if other.timesteps[0].shape[0]*other.timesteps[0].shape[1]==self.total_pixels()])),
             'lats':(['starttime','y','x'],
                 np.array([other.y[0] for obj in storm_objs for other in obj if other.timesteps[0].shape[0]*other.timesteps[0].shape[1]==self.total_pixels()])),
             'lons':(['starttime','y','x'],
                 np.array([other.x[0] for obj in storm_objs for other in obj if other.timesteps[0].shape[0]*other.timesteps[0].shape[1]==self.total_pixels()])),
            },
             coords=
            {'starttime':(['starttime'],
                        np.array([other.start_time for obj in storm_objs for other in obj if other.timesteps[0].shape[0]*other.timesteps[0].shape[1]==self.total_pixels()])),
             'endtime':
                        np.array([other.end_time for obj in storm_objs for other in obj if other.timesteps[0].shape[0]*other.timesteps[0].shape[1]==self.total_pixels()]),
             'x_speed':(['starttime'],
                        np.array([other.u[0] for obj in storm_objs for other in obj if other.timesteps[0].shape[0]*other.timesteps[0].shape[1]==self.total_pixels()])),
             'y_speed':(['starttime'],
                        np.array([other.v[0] for obj in storm_objs for other in obj if other.timesteps[0].shape[0]*other.timesteps[0].shape[1]==self.total_pixels()]))
            })
        data_assemble.to_netcdf(f"/{self.destination_path}/{self.climate}_SPhourly_{times_thisfile[num].strftime('%Y%m%d')}.nc")
        return(num)
    
    
    def create_patches_3H(self, datetime_value):

        """Function to extract data that corresponds to previously extracted storm patches with ``create_patches_hourly()``. These 
        data are only available in 3-hour intervals (e.g., UH).
            
        Args:
            datetime_value (DatetimeIndex): One date.
            
        """
        land=self.prep_land()
        
        #default values for the WRF CONUS1 dataset below, 
        #including the full time range for indexing simulations and slicing the CONUS spatially for efficiency 
        #(we don't care about storms over water
        total_times=pd.date_range('2000-10-01','2013-10-01 00:00:00',freq='H')
        total_times_indexes=np.arange(0,total_times.shape[0],1)
        the1=135; the2=650; the3=500; the4=1200
        #creation of blank lists for variable saving
        UH25_to_return=[]; UH03_to_return=[]; CT_to_return=[]; DZ_to_return=[]; MASK_to_return=[]
        ROW_to_return=[]; COL_to_return=[]; LAT_to_return=[]; LON_to_return=[]
        STM_to_return=[]; ETM_to_return=[]; XSPD_to_return=[]; YSPD_to_return=[]        
        
        print(f"doing {self.climate}: {datetime_value.strftime('%Y%m%d')}")
        #open the original hourly storm patches that were created and saved
        file_storm=f"/{self.destination_path}/{self.climate}_SPhourly_{datetime_value.strftime('%Y%m%d')}.nc"
        data_storm=xr.open_dataset(file_storm)
        #create boolean list of the values that are 3 hourly for variable extraction
        check_3H_datastorm=np.isin(data_storm.starttime.values,total_times_indexes[::3])
        
        #opening the variable data sets (UH and CTT)
        if not self.uh03_path or not self.uh25_path or not self.ctt_path:
            raise Exception("Please enter the paths to UH and CTT data.")
        file_uh25=f"/{self.uh25_path}/wrf2d_UH_{datetime_value.strftime('%Y%m')}.nc"
        file_uh03=f"/{self.uh03_path}/wrf2d_UH_{datetime_value.strftime('%Y%m')}.nc"
        file_ctt =f"/{self.ctt_path}/wrf2d_CTT_{datetime_value.strftime('%Y%m')}.nc"
        try:
            data_uh25=xr.open_dataset(file_uh25)
            data_uh03=xr.open_dataset(file_uh03)
            data_ctt=xr.open_dataset(file_ctt)
        except IOError:
            #very few days do not contain identified storms, avoid job kill with exception here.
            print(f"not found {datetime_value.strftime('%Y%m%d')}")
            return

        uh_time_equiv_storm_patch=total_times_indexes[
                    np.isin(total_times, pd.date_range(datetime_value.strftime('%Y-%m')+'-01',
                    pd.to_datetime(datetime_value.strftime('%Y-%m')+'-'+str(calendar.monthrange(datetime_value.year, 
                                                                                                datetime_value.month)[1]))+timedelta(hours=23), freq='3H'))]

        for storm_patch_file_idx, (storm_time, storm_time_indx) in enumerate(zip(check_3H_datastorm, data_storm.starttime.values)):
            if storm_time:
                if self.is_land(land,
                                data_storm.lons[storm_patch_file_idx,
                                                np.where(data_storm.grid[storm_patch_file_idx,:,:]==data_storm.grid[storm_patch_file_idx,:,:].max())[0][0],
                                                np.where(data_storm.grid[storm_patch_file_idx,:,:]==data_storm.grid[storm_patch_file_idx,:,:].max())[1][0]], 
                                data_storm.lats[storm_patch_file_idx,
                                                np.where(data_storm.grid[storm_patch_file_idx,:,:]==data_storm.grid[storm_patch_file_idx,:,:].max())[0][0],
                                                np.where(data_storm.grid[storm_patch_file_idx,:,:]==data_storm.grid[storm_patch_file_idx,:,:].max())[1][0]]):

                    temp_uh25=data_uh25.uh[np.where(uh_time_equiv_storm_patch==storm_time_indx)[0],the1:the2,the3:the4].values[0,:,:]
                    temp_uh03=data_uh03.uh[np.where(uh_time_equiv_storm_patch==storm_time_indx)[0],the1:the2,the3:the4].values[0,:,:]
                    temp_ct  =data_ctt.ctt[np.where(uh_time_equiv_storm_patch==storm_time_indx)[0],the1:the2,the3:the4].values[0,:,:]
                    
                    UH25_to_return.append(temp_uh25[data_storm.row_indices[storm_patch_file_idx].values, data_storm.col_indices[storm_patch_file_idx].values])
                    UH03_to_return.append(temp_uh03[data_storm.row_indices[storm_patch_file_idx].values, data_storm.col_indices[storm_patch_file_idx].values])
                    CT_to_return.append(temp_ct[data_storm.row_indices[storm_patch_file_idx].values,     data_storm.col_indices[storm_patch_file_idx].values])
                    DZ_to_return.append(data_storm.grid[storm_patch_file_idx,:,:].values)
                    MASK_to_return.append(data_storm.mask[storm_patch_file_idx,:,:].values)
                    ROW_to_return.append(data_storm.row_indices[storm_patch_file_idx,:,:].values)
                    COL_to_return.append(data_storm.col_indices[storm_patch_file_idx,:,:].values)
                    LAT_to_return.append(data_storm.lats[storm_patch_file_idx,:,:].values)
                    LON_to_return.append(data_storm.lons[storm_patch_file_idx,:,:].values)
                    STM_to_return.append(data_storm.coords['starttime'][storm_patch_file_idx].values)
                    ETM_to_return.append(data_storm.coords['endtime'][storm_patch_file_idx].values)
                    XSPD_to_return.append(data_storm.coords['x_speed'][storm_patch_file_idx].values)
                    YSPD_to_return.append(data_storm.coords['y_speed'][storm_patch_file_idx].values)

        data_assemble=xr.Dataset({
             'uh25_grid':    (['starttime','y','x'],np.array([obj for obj in UH25_to_return])),
             'uh03_grid':    (['starttime','y','x'],np.array([obj for obj in UH03_to_return])),
             'ctt_grid':   (['starttime','y','x'],np.array([obj for obj in CT_to_return])),
             'dbz_grid':   (['starttime','y','x'],np.array([obj for obj in DZ_to_return])),
             'mask':       (['starttime','y','x'],np.array([obj for obj in MASK_to_return])),
             'row_indices':(['starttime','y','x'],np.array([obj for obj in ROW_to_return])),
             'col_indices':(['starttime','y','x'],np.array([obj for obj in COL_to_return])),
             'lats':       (['starttime','y','x'],np.array([obj for obj in LAT_to_return])),
             'lons':       (['starttime','y','x'],np.array([obj for obj in LON_to_return]))
            },
             coords=
            {'starttime':(['starttime'],np.array([float(obj) for obj in STM_to_return])),
             'endtime':  (['starttime'],np.array([float(obj) for obj in ETM_to_return])),
             'x_speed':  (['starttime'],np.array([int(obj) for obj in XSPD_to_return])),
             'y_speed':  (['starttime'],np.array([int(obj) for obj in YSPD_to_return]))
            })
        data_assemble.to_netcdf(f"/{self.destination_path}/{self.climate}_SP3hourly_{datetime_value.strftime('%Y%m%d')}.nc")
        print(f"completed {datetime_value.strftime('%Y%m%d')}")
        return
        
        
    def parallelizing_3hourly_func(self):
            
        """Activate the multiprocessing function and parallelize the creation of 3-hourly storm patches for efficiency.
            
        """
        times_thisfile=self.generate_timestring()
        pool=mp.Pool(self.num_cpus)
        for time_for_file in times_thisfile:
            print(f"start {time_for_file.strftime('%Y%m%d')}")
            pool.apply_async(self.create_patches_3H, args=([time_for_file]))
        pool.close()
        pool.join()
        print("completed")

        
    def create_patches_variable(self, datetime_value):

        """Function to extract variable data that corresponds to previously extracted 3-hour storm patches with ``create_patches_3H()``.
        
        Args:
            datetime_value (DatetimeIndex): One date.
            
        """
        land=self.prep_land()
        
        #default values for the WRF CONUS1 dataset below, 
        #including the full time range for indexing simulations and slicing the CONUS spatially for efficiency 
        #(we don't care about storms over water
        total_times=pd.date_range('2000-10-01','2013-10-01 00:00:00',freq='H')
        total_times_indexes=np.arange(0,total_times.shape[0],1)
        the1=135; the2=650; the3=500; the4=1200
        
        #creation of blank lists for variable saving
        var_to_return=[]; DZ_to_return=[]; MASK_to_return=[]
        ROW_to_return=[]; COL_to_return=[]; LAT_to_return=[]; LON_to_return=[]
        STM_to_return=[]; ETM_to_return=[]; XSPD_to_return=[]; YSPD_to_return=[]
        
        print(f"doing {self.climate}, {self.variable}: {datetime_value.strftime('%Y%m%d')}")
        file_storm=f"/{self.destination_path}/{self.climate}_SPhourly_{datetime_value.strftime('%Y%m%d')}.nc"
        data_storm=xr.open_dataset(file_storm)
        #opening the variable data set
        if self.variable!='WMAX':
            file_var=f"/{self.variable_path}/{self.variable}/wrf2d_interp_{self.variable_translate()}_{datetime_value.strftime('%Y%m')}.nc"
        if self.variable=='WMAX':
            file_var=f"/{self.variable_path}/{self.variable}/wrf2d_max_{self.variable_translate()}_{datetime_value.strftime('%Y%m')}.nc"
        try:
            data_var=xr.open_dataset(file_var)
        except IOError:
            #very few days do not contain identified storms, avoid job kill with exception here.
            print(f"not found {datetime_value.strftime('%Y%m%d')}")
            return
        
        check_3H_datastorm=np.isin(data_storm.starttime.values,total_times_indexes[::3])
        var_time_equiv_storm_patch=total_times_indexes[np.isin(total_times, 
                  pd.date_range(datetime_value.strftime('%Y-%m')+'-01',
                  pd.to_datetime(datetime_value.strftime('%Y-%m')+'-'+str(calendar.monthrange(datetime_value.year, datetime_value.month)[1]))+timedelta(hours=23),
                  freq='3H'))]

        for storm_patch_file_idx, (storm_time, storm_time_indx) in enumerate(zip(check_3H_datastorm, data_storm.starttime.values)):
            if storm_time:
                if self.is_land(land,
                                data_storm.lons[storm_patch_file_idx,
                                                np.where(data_storm.grid[storm_patch_file_idx,:,:]==data_storm.grid[storm_patch_file_idx,:,:].max())[0][0],
                                                np.where(data_storm.grid[storm_patch_file_idx,:,:]==data_storm.grid[storm_patch_file_idx,:,:].max())[1][0]], 
                                data_storm.lats[storm_patch_file_idx,
                                                np.where(data_storm.grid[storm_patch_file_idx,:,:]==data_storm.grid[storm_patch_file_idx,:,:].max())[0][0],
                                                np.where(data_storm.grid[storm_patch_file_idx,:,:]==data_storm.grid[storm_patch_file_idx,:,:].max())[1][0]]):

                    if self.variable!='WMAX':
                        temp_var=data_var.levels[np.where(var_time_equiv_storm_patch==storm_time_indx)[0],:,the1:the2,the3:the4].values[0,:,:,:]
                        var_to_return.append(temp_var[:, data_storm.row_indices[storm_patch_file_idx].values, data_storm.col_indices[storm_patch_file_idx].values])
                    if self.variable=='WMAX':
                        temp_var=data_var.max_in_vert[np.where(var_time_equiv_storm_patch==storm_time_indx)[0],the1:the2,the3:the4].values[0,:,:]
                        var_to_return.append(temp_var[data_storm.row_indices[storm_patch_file_idx].values, data_storm.col_indices[storm_patch_file_idx].values])
                    
                    DZ_to_return.append(data_storm.grid[storm_patch_file_idx,:,:].values)
                    MASK_to_return.append(data_storm.mask[storm_patch_file_idx,:,:].values)
                    ROW_to_return.append(data_storm.row_indices[storm_patch_file_idx,:,:].values)
                    COL_to_return.append(data_storm.col_indices[storm_patch_file_idx,:,:].values)
                    LAT_to_return.append(data_storm.lats[storm_patch_file_idx,:,:].values)
                    LON_to_return.append(data_storm.lons[storm_patch_file_idx,:,:].values)
                    STM_to_return.append(data_storm.coords['starttime'][storm_patch_file_idx].values)
                    ETM_to_return.append(data_storm.coords['endtime'][storm_patch_file_idx].values)
                    XSPD_to_return.append(data_storm.coords['x_speed'][storm_patch_file_idx].values)
                    YSPD_to_return.append(data_storm.coords['y_speed'][storm_patch_file_idx].values)

        if self.variable!='WMAX':
            data_assemble=xr.Dataset({
                 'var_grid':(['starttime','levels','y','x'],np.array([obj for obj in var_to_return])),
                 'dbz_grid':(['starttime','y','x'],np.array([obj for obj in DZ_to_return])),
                 'mask':(['starttime','y','x'],np.array([obj for obj in MASK_to_return])),
                 'row_indices':(['starttime','y','x'],np.array([obj for obj in ROW_to_return])),
                 'col_indices':(['starttime','y','x'],np.array([obj for obj in COL_to_return])),
                 'lats':(['starttime','y','x'],np.array([obj for obj in LAT_to_return])),
                 'lons':(['starttime','y','x'],np.array([obj for obj in LON_to_return]))
                },
                 coords=
                {'starttime':(['starttime'],np.array([float(obj) for obj in STM_to_return])),
                 'endtime':(['starttime'],np.array([float(obj) for obj in ETM_to_return])),
                 'x_speed':(['starttime'],np.array([int(obj) for obj in XSPD_to_return])),
                 'y_speed':(['starttime'],np.array([int(obj) for obj in YSPD_to_return])),
                 'levels':(['levels'],np.array([0,1,2,3]))
                })
            
        if self.variable=='WMAX':
            data_assemble=xr.Dataset({
                 'var_grid':(['starttime','y','x'],np.array([obj for obj in var_to_return])),
                 'dbz_grid':(['starttime','y','x'],np.array([obj for obj in DZ_to_return])),
                 'mask':(['starttime','y','x'],np.array([obj for obj in MASK_to_return])),
                 'row_indices':(['starttime','y','x'],np.array([obj for obj in ROW_to_return])),
                 'col_indices':(['starttime','y','x'],np.array([obj for obj in COL_to_return])),
                 'lats':(['starttime','y','x'],np.array([obj for obj in LAT_to_return])),
                 'lons':(['starttime','y','x'],np.array([obj for obj in LON_to_return]))
                },
                 coords=
                {'starttime':(['starttime'],np.array([float(obj) for obj in STM_to_return])),
                 'endtime':(['starttime'],np.array([float(obj) for obj in ETM_to_return])),
                 'x_speed':(['starttime'],np.array([int(obj) for obj in XSPD_to_return])),
                 'y_speed':(['starttime'],np.array([int(obj) for obj in YSPD_to_return])),
                })            

        data_assemble.to_netcdf(f"/{self.destination_path}/{self.climate}_SP3hourly_{self.variable_translate().lower()}_{datetime_value.strftime('%Y%m%d')}.nc")
        print(f"completed {datetime_value.strftime('%Y%m%d')}")
        return

    
    def parallelizing_3Hvariable_func(self):
            
        """Activate the multiprocessing function, and parallelize the creation of 3-hourly variable extractions.

        """
        if not self.variable_path:
            raise Exception(f"Please enter the directory path for {self.variable}.")
        _=self.variable_translate()
        times_thisfile=self.generate_timestring()
        pool=mp.Pool(self.num_cpus)
        for time_for_file in times_thisfile:
            print(f"start {time_for_file.strftime('%Y%m%d')}")
            pool.apply_async(self.create_patches_variable, args=([time_for_file]))
        pool.close()
        pool.join()
        print("completed")
        
        