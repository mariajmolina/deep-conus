###################################################
###################################################
##
##  Author: Maria J. Molina
##  National Center for Atmospheric Research
##
###################################################
###################################################


#Script to interpolate specific variables onto a fixed height above ground level (AGL). 
#CONUS1 WRF 4-km


#--------------------------------------------------

import xarray as xr
import numpy as np
from ncar_jobqueue import NCARCluster
from dask.distributed import Client

#--------------------------------------------------



class interpolate_variable:


    def __init__(self, climate, variable, month_start, month_end, year_start, year_end):

        """

	Instantiation of interpolate_variable:

        Here we will be interpolating the respective variable onto 1, 3, 5, and 7 km above ground level (AGL).

        climate: 'current' or 'future' (str)
        variable: 'TK', QVAPOR, EU, EV, P, QGRAUP (str)
        month: start and end month for the respective interpolation operation (int)
        year: start year of analysis (int)

        """

        self.climate = climate

        if self.climate == 'current':
            self.folder = 'CTRL3D'
            self.filename = 'CTRL'
        if self.climate == 'future':
            self.folder = 'PGW3D'
            self.filename = 'PGW'

        self.variable = variable
        self.month1 = month_start
        self.month2 = month_end
        self.year1 = year_start

        if year_start == year_end:
            self.year2 = year_end+1
        if year_start != year_end:
            self.year2 = year_end

    

    def open_files(self, year, month):
    
        """
        Helper function to open data sets for interpolation calculation for current climate.
          Inputs:
             year: year in the loop (str; 4-digit)
             month: month in the loop (str; 2-digit)
             variable: TK, QVAPOR, EU, EV, P, QGRAUP (str) 
          Outputs:
             data_AGL: above ground level heights [m]
             data_var: the variable data
        """                        

        #geopotential height (m)
        data_zstag = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/{self.folder}/{year}/wrf3d_d01_{self.filename}_Z_{year}{month}*.nc', 
                                   combine='by_coords', parallel=True, chunks={'Time':1}).Z
        data_zstag = 0.5*(data_zstag[:,0:50,:,:]+data_zstag[:,1:51,:,:])

        #terrain (m)
        data_mapfc = xr.open_dataset('/gpfs/fs1/collections/rda/data/ds612.0/INVARIANT/RALconus4km_wrf_constants.nc').HGT.sel(Time='2000-10-01')  
        data_AGL = data_zstag - data_mapfc

        if self.variable == 'W':
            #z-wind (m/s)
            data_var = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/{self.folder}/{year}/wrf3d_d01_{self.filename}_W_{year}{month}*.nc',
                                       combine='by_coords', parallel=True, chunks={'Time':1}).W
            data_var = 0.5*(data_var[:,0:50,:,:]+data_var[:,1:51,:,:])

        if self.variable == 'TK':
            #temperature (kelvin)
            data_var = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/{self.folder}/{year}/wrf3d_d01_{self.filename}_TK_{year}{month}*.nc',
                                       combine='by_coords', parallel=True, chunks={'Time':1}).TK
        if self.variable == 'QVAPOR':
            #water vapor mixing ratio (kg/kg)
            data_var = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/{self.folder}/{year}/wrf3d_d01_{self.filename}_QVAPOR_{year}{month}*.nc',
                                       combine='by_coords', parallel=True, chunks={'Time':1}).QVAPOR      
        if self.variable == 'EU':
            #earth rotated u-winds (m/s)
            data_var = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/{self.folder}/{year}/wrf3d_d01_{self.filename}_EU_{year}{month}*.nc', 
                                       combine='by_coords', parallel=True, chunks={'Time':1}).EU
        if self.variable == 'EV':
            #earth rotated v-winds (m/s)
            data_var = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/{self.folder}/{year}/wrf3d_d01_{self.filename}_EV_{year}{month}*.nc', 
                                       combine='by_coords', parallel=True, chunks={'Time':1}).EV
        if self.variable == 'P':
            #pres_hpa: full pressure (perturb and base state) #convert from Pa to hPa.
            data_var = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/{self.folder}/{year}/wrf3d_d01_{self.filename}_P_{year}{month}*.nc',
                                       combine='by_coords', parallel=True, chunks={'Time':1}).P*0.01 
        if self.variable == 'QGRAUP':
            #Graupel mixing ratio (kg/kg)
            data_var = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/{self.folder}/{year}/wrf3d_d01_{self.filename}_QGRAUP_{year}{month}.nc', 
                                       combine='by_coords', parallel=True, chunks={'Time':1}).QGRAUP

        return data_AGL, data_var



    def activate_workers(self):
        #start dask workers
        cluster = NCARCluster(memory="109GB", cores=36, project="P54048000")
        cluster.adapt(minimum=10, maximum=40, wait_count=60)
        cluster
        #print scripts
        print(cluster.job_script())
        #start client
        client = Client(cluster)
        client



    def generate_timestrings(self):
        #temporal arrays
        formatter = "{:02d}".format
        months = np.array(list(map(formatter, np.arange(self.month1, self.month2, 1))))
        years  = np.array(list(map(formatter, np.arange(self.year1, self.year2, 1))))
        return years, months



    def create_the_interp_files(self):

        """
            Automate the work pipeline
        """

        self.activate_workers()
        yrs, mos = self.generate_timestrings()

        for yr in yrs:
            for mo in mos:
                print(f"opening {yr} {mo} files for {self.variable}")
                data_AGL, data_var = self.open_files(yr, mo)
                print(f"generating u_func")

                if self.variable != 'W':
                    result_ufunc = apply_wrf_interp(data_AGL, data_var)
                if self.variable == 'W':
                    result_ufunc = apply_wrf_interp_W(data_AGL, data_var)

                print(f"starting interp for {yr} {mo}")
                r = result_ufunc.compute(retries=10)
                print(f"Saving file")
                r.to_dataset(name='levels').to_netcdf(f"/glade/scratch/molina/DL_proj/{self.climate}_conus_fields/wrf2d_interp_{self.variable}_{yr}{mo}.nc")
                r = r.close()
                data_AGL = data_AGL.close()
                data_var = data_var.close()
                print(f"woohoo! {yr} {mo} complete")



def wrf_interp(data_AGL, data_var):

    """
    Function to compute interpolation using wrf-python.
    """

    #Imports needed for job workers.
    import os
    os.environ["PROJ_LIB"] = "/glade/work/molina/miniconda3/envs/python-tutorial/share/proj/"
    import wrf

    return (wrf.interplevel(data_var.squeeze(),
                            data_AGL.squeeze(),
                            [1000,3000,5000,7000]).expand_dims("Time"))


def apply_wrf_interp(data_AGL, data_var):

    """
    Generate Xarray ufunc to parallelize the wrf-python interpolation computation.
    """

    return xr.apply_ufunc(wrf_interp, data_AGL, data_var,
                          dask='parallelized',
                          output_dtypes=[float],
                          input_core_dims=[['bottom_top_stag','south_north','west_east'],
                                           ['bottom_top','south_north','west_east']],
                          output_sizes=dict(level=4, south_north=1015, west_east=1359),
                          output_core_dims=[['level','south_north','west_east']])


def apply_wrf_interp_W(data_AGL, data_var):

    """
    Generate Xarray ufunc to parallelize the wrf-python interpolation computation for W, 
    which has staggered dims.
    """

    return xr.apply_ufunc(wrf_interp, data_AGL, data_var,
                          dask='parallelized',
                          output_dtypes=[float],
                          input_core_dims=[['bottom_top_stag','south_north','west_east'],
                                           ['bottom_top_stag','south_north','west_east']],
                          output_sizes=dict(level=4, south_north=1015, west_east=1359),
                          output_core_dims=[['level','south_north','west_east']])



