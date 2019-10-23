###################################################
###################################################
##
##Author: Maria J. Molina
##National Center for Atmospheric Research
##
###################################################
###################################################


#Script to interpolate specific variable onto a height above ground level (AGL). 
#current climate WRF 4-km

#--------------------------------------------------

import xarray as xr
import numpy as np

#--------------------------------------------------
            

def open_files(year, month, variable):
    
    """
    Helper function to open data sets for interpolation calculation.
    
    Inputs:
    year: year in the loop (str; 4-digit)
    month: month in the loop (str; 2-digit)
    variable: TK, QVAPOR, EU, EV, P, QGRAUP (str)
    
    Outputs:
    data_AGL: above ground level heights [m]
    data_var: the variable data
    """                        
    
    #geopotential height (m)
    data_zstag = xr.open_mfdataset(f'/glade/scratch/molina/WRF_CONUS1_derived/current/wrf3d_d01_CTRL_Zdestag_{year}{month}*.nc', 
                                   combine='by_coords', parallel=True, chunks={'Time':1}).Z
    #terrain (m)
    data_mapfc = xr.open_dataset('/gpfs/fs1/collections/rda/data/ds612.0/INVARIANT/RALconus4km_wrf_constants.nc').HGT.sel(Time='2000-10-01')  

    #convert geopotential heights in m to AGL in m
    data_AGL = data_zstag - data_mapfc
    
    if variable == 'TK':
        #temperature (kelvin)
        data_var = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/CTRL3D/{year}/wrf3d_d01_CTRL_TK_{year}{month}*.nc',
                                       combine='by_coords', parallel=True, chunks={'Time':1}).TK
    if variable == 'QVAPOR':
        #water vapor mixing ratio (kg/kg)
        data_var = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/CTRL3D/{year}/wrf3d_d01_CTRL_QVAPOR_{year}{month}*.nc',
                                       combine='by_coords', parallel=True, chunks={'Time':1}).QVAPOR      
    if variable == 'EU':
        #earth rotated u-winds (m/s)
        data_var = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/CTRL3D/{year}/wrf3d_d01_CTRL_EU_{year}{month}*.nc', 
                                       combine='by_coords', parallel=True, chunks={'Time':1}).EU
    if variable == 'EV':
        #earth rotated v-winds (m/s)
        data_var = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/CTRL3D/{year}/wrf3d_d01_CTRL_EV_{year}{month}*.nc', 
                                       combine='by_coords', parallel=True, chunks={'Time':1}).EV
    if variable == 'P':
        #pres_hpa: full pressure (perturb and base state) #convert from Pa to hPa.
        data_var = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/CTRL3D/{year}/wrf3d_d01_CTRL_P_{year}{month}*.nc',
                                       combine='by_coords', parallel=True, chunks={'Time':1}).P*0.01 
    if variable == 'QGRAUP':
        #Graupel mixing ratio (kg/kg)
        data_var = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/CTRL3D/{year}/wrf2d_d01_CTRL_QGRAUP_{year}{month}.nc', 
                                       combine='by_coords', parallel=True, chunks={'Time':1}).QGRAUP

    return data_AGL, data_var


def wrf_interp(data_var, data_AGL):
    
    """
    Function to compute interpolation using wrf-python.
    """ 
 
    #Imports needed for job workers.
    import os
    os.environ["PROJ_LIB"] = "/glade/work/abanihi/softwares/miniconda3/envs/python-tutorial/share/proj"
    import wrf
    
    return (wrf.interplevel(data_var.squeeze(),
                            data_AGL.squeeze(), 
                            [1000,3000,5000,7000]).expand_dims("Time"))


def apply_wrf_interp(data_var, data_AGL):
    
    """
    Generate Xarray ufunc to parallelize the wrf-python interpolation computation.
    """

    return xr.apply_ufunc(wrf_interp, data_var, data_AGL,
                          dask='parallelized', 
                          output_dtypes=[float],
                          input_core_dims=[['bottom_top','south_north','west_east'],
                                           ['bottom_top','south_north','west_east']],
                          output_sizes=dict(level=4,south_north=1015, west_east=1359),
                          output_core_dims=[['level','south_north','west_east']])





#--------------------------------------------------

if __name__== "__main__":

    from ncar_jobqueue import NCARCluster
    from dask.distributed import Client

    #start dask workers
    cluster = NCARCluster(memory="109GB", cores=36)
    cluster.adapt(minimum=10, maximum=40, wait_count=60)
    cluster
    #print scripts
    print(cluster.job_script())
    #start client
    client = Client(cluster)
    client
    
    #temporal arrays
    formatter = "{:02d}".format
    months = np.array(list(map(formatter, np.arange(2,13,1))))
    years = np.array(list(map(formatter, np.arange(2001,2002,1))))

    #variable options: TK, QVAPOR, EU, EV, P, QGRAUP
    variable = 'TK'

    for year in years:
        for month in months:

            print(f"opening {year} {month} files")
            data_AGL, data_var = open_files(year, month, variable)

            print(f"generating u_func")
            result_ufunc = apply_wrf_interp(data_AGL, data_var)

            print(f"starting interp for {year} {month}")
            r = result_ufunc.compute(retries=10)

            print(f"Saving file")
            r.to_dataset(name='levels').to_netcdf(f"/glade/scratch/molina/WRF_CONUS1_derived/current/wrf2d_interp_{variable}_{year}{month}.nc")

            r = r.close()
            data_AGL = data_AGL.close()
            data_var = data_var.close()

            print(f"woohoo! {year} {month} complete")

#--------------------------------------------------
