

import xarray as xr
import numpy as np
from ncar_jobqueue import NCARCluster
from dask.distributed import Client

cluster = NCARCluster(memory="109GB", cores=36)
cluster.adapt(minimum=5, maximum=20, wait_count=60)
cluster

print(cluster.job_script())

client = Client(cluster)
client

def open_files(year, month):
    
    """
    Helper function to open data sets for SRH calculation.

    Parameters
    ----------
    Inputs:
    year: year in the loop (str; 4-digit)
    month: month in the loop (str; 2-digit)

    Outputs:
    data_ustag: x-component of the wind (3d, in m/s)
    data_vstag: y-component of the wind (3d, in m/s)
    data_zstag: geopotential height (3d, in meters)
    """

    import wrf

    print("opening files for SRH now")
    data_ustag = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/CTRL3D/{year}/wrf3d_d01_CTRL_EU_{year}{month}*.nc', 
                                   combine='by_coords', parallel=True, chunks={'Time':1}).EU

    data_vstag = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/CTRL3D/{year}/wrf3d_d01_CTRL_EV_{year}{month}*.nc', 
                                   combine='by_coords', parallel=True, chunks={'Time':1}).EV

    data_zstag = xr.open_mfdataset(f"/glade/scratch/molina/WRF_CONUS1_derived/current/wrf3d_d01_CTRL_Zdestag_{year}{month}*.nc", 
                                   combine='by_coords', parallel=True, chunks={'Time':1}).Z

    data_mapfc = xr.open_dataset('/gpfs/fs1/collections/rda/data/ds612.0/INVARIANT/RALconus4km_wrf_constants.nc').HGT.sel(Time='2000-10-01')
    
    return data_ustag, data_vstag, data_zstag, data_mapfc
    
    
def wrf_srh(data_ustag, data_vstag, data_zstag, data_mapfc, top=3000.0, lats=None, meta=True):
    
    #Imports needed for job workers.
    import os
    os.environ["PROJ_LIB"] = "/glade/work/abanihi/softwares/miniconda3/envs/python-tutorial/share/proj"
    import wrf
    
    """
    Function to compute storm relative helicity (SRH) using wrf-python.
    
    Parameters
    ----------
    Inputs:
    top: The height of the layer below which helicity is calculated (meters above ground level). (float)
    
    Outputs:
    Updraft Helicity values across spatial domain (2d) and with time aggregation encoded for ufunc.
    """ 
    
    return (wrf.srhel(data_ustag.squeeze(), data_vstag.squeeze(), data_zstag.squeeze(), 
                      data_mapfc, top=3000.0, lats=None, meta=True).expand_dims("Time"))
                      
                    
def apply_wrf_srh(data_ustag, data_vstag, data_zstag, data_mapfc):
    
    """
    Generate Xarray ufunc to parallelize the wrf-python SRH computation.
    """

    return xr.apply_ufunc(wrf_srh, data_ustag, data_vstag, data_zstag, data_mapfc,
                          dask='parallelized', output_dtypes=[float],
                          input_core_dims=[['bottom_top','south_north','west_east'],
                                           ['bottom_top','south_north','west_east'],
                                           ['bottom_top','south_north','west_east'],
                                           ['south_north','west_east']],
                          output_sizes=dict(south_north=1015,west_east=1359),
                          output_core_dims=[['south_north','west_east']])
                          
                          
#temporal arrays
formatter = "{:02d}".format
months = np.array(list(map(formatter, np.arange(1,13,1))))
years = np.array(list(map(formatter, np.arange(2001,2013,1))))

#loop through years and months
for year in years:
    for month in months:
        
        data_ustag, data_vstag, data_zstag, data_mapfc = open_files(year=year, month=month)

        print(f"generating u_func")
        result_ufunc = apply_wrf_srh(data_ustag, data_vstag, data_zstag, data_mapfc)

        print(f"starting SRH for {year} {month}")
        r = result_ufunc.compute(retries=10)

        print(f"Saving file")
        #save file to current climate folder directory.
        r.to_dataset(name='srh03').to_netcdf(f"/glade/scratch/molina/WRF_CONUS1_derived/current/wrf2d_03SRH_{year}{month}.nc")
        
        r = r.close()
        data_ustag = data_ustag.close()
        data_vstag = data_vstag.close()
        data_zstag = data_zstag.close()
        data_mapfc = data_mapfc.close()

        print(f"woohoo! {year} {month} complete")
