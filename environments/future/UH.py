###################################################
###################################################
##
##Author: Maria J. Molina
##National Center for Atmospheric Research
##
###################################################
###################################################


#Script to compute updraft helicity (2-5-km) from 
#future climate WRF simulations (CONUS1, 4-km).


#--------------------------------------------------

import xarray as xr
import numpy as np
from ncar_jobqueue import NCARCluster
from dask.distributed import Client

#--------------------------------------------------


def open_files(year, month):
    
    """
    Helper function to open data sets.

    Inputs:
    year: year in the loop (str; 4-digit)
    month: month in the loop (str; 2-digit)

    Outputs:
    data_zstag: geopotential heights (3d)
    data_wstag: z-wind component (3d)
    data_ustag: x-component of the wind (3d)
    data_vstag: y-component of the wind (3d)
    """

    data_zstag = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/PGW3D/{year}/wrf3d_d01_PGW_Z_{year}{month}*.nc', 
                                   combine='by_coords', parallel=True, chunks={'Time':1}).Z
    data_wstag = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/PGW3D/{year}/wrf3d_d01_PGW_W_{year}{month}*.nc', 
                                   combine='by_coords', parallel=True, chunks={'Time':1}).W
    data_ustag = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/PGW3D/{year}/wrf3d_d01_PGW_EU_{year}{month}*.nc', 
                                   combine='by_coords', parallel=True, chunks={'Time':1}).EU
    data_vstag = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/PGW3D/{year}/wrf3d_d01_PGW_EV_{year}{month}*.nc', 
                                   combine='by_coords', parallel=True, chunks={'Time':1}).EV

    return data_zstag, data_wstag, data_ustag, data_vstag
    
    
def wrf_UH(z, mapfc, u, v, w, dx=4000.0, dy=4000.0, bottom=2000.0, top=5000.0):
    
    #Imports needed for job workers.
    import os
    os.environ["PROJ_LIB"] = "/glade/work/abanihi/softwares/miniconda3/envs/python-tutorial/share/proj"
    import wrf
    
    """
    Function to compute updraft helicity using wrf-python.

    Inputs:
    z (zstag): geopotential heights (3d)
    mapfc: wrf proj map factor (2d)
    u (ustag): x-component of the wind (3d)
    v (vstag): y-component of the wind (3d)
    w (wstag): z-wind component (3d)
    dx and dy: horizontal resolutions (float)
    bottom and top: lower and upper vertical delimiter for UH computation (float)

    Outputs:
    Updraft Helicity values across spatial domain (2d) and with time aggregation encoded for ufunc.
    """ 
    
    return (wrf.udhel( z.squeeze(), mapfc, u.squeeze(), v.squeeze(), w.squeeze(), \
                      dx=dx, dy=dy, bottom=bottom, top=top, meta=True).expand_dims("Time"))


def apply_wrf_UH(data_zstag, data_mapfc, data_ustag, data_vstag, data_wstag):
    
    """
    Generate Xarray ufunc to parallelize the wrf-python UH computation.
    """

    return xr.apply_ufunc(wrf_UH, data_zstag, data_mapfc, data_ustag, data_vstag, data_wstag,
                          dask='parallelized', output_dtypes=[float],
                          input_core_dims=[['bottom_top_stag','south_north','west_east'],
                                           ['south_north','west_east'],
                                           ['bottom_top','south_north','west_east'],
                                           ['bottom_top','south_north','west_east'],
                                           ['bottom_top_stag','south_north','west_east']],
                          output_sizes=dict(south_north=1015,west_east=1359),
                          output_core_dims=[['south_north','west_east']])
                          
def iterate_compute(mapfc):
    """
    Function to loop the computation of UH.

    Inputs:
    years: 1d array of years to be looped through
    mapfc: wrf proj map factor (2d)
    directory: where to save the UH file (string)
    """

    import numpy as np
    
    #temporal arrays
    formatter = "{:02d}".format
    months = np.array(list(map(formatter, np.arange(1,13,1))))
    years = np.array(list(map(formatter, np.arange(2010,2014,1))))

    #loop through years and months
    for year in years:
        for month in months:
            print(f"opening {year} {month}...")
            data_zstag, data_wstag, data_ustag, data_vstag = open_files(year=year, month=month)
            result_ufunc = apply_wrf_UH(data_zstag, data_mapfc, data_ustag, data_vstag, data_wstag)
            
            print(f"starting UH for {year} {month}")
            r = result_ufunc.compute(retries=10)
            r.to_dataset(name='uh').to_netcdf(f"/glade/scratch/molina/WRF_CONUS1_derived/future/wrf2d_UH_{year}{month}.nc")
           
            print(f"woohoo! {year} {month} complete")
    return 
    

def main():
    cluster = NCARCluster(memory="100GB", cores=36)
    cluster.adapt(minimum=5, maximum=50, wait_count=60)
    cluster
    print(cluster.job_script())
    client = Client(cluster)
    client
    data_mapfc = xr.open_dataset('/gpfs/fs1/collections/rda/data/ds612.0/INVARIANT/RALconus4km_wrf_constants.nc')\
                  .MAPFAC_M.sel(Time='2000-10-01').values
    iterate_compute(data_mapfc)
    

#--------------------------------------------------
    
if __name__== "__main__":
    main()

#--------------------------------------------------
    
