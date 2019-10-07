
#Author: Maria J. Molina
#National Center for Atmospheric Research

#Script to compute CAPE, CIN, LCL, and LFC from WRF simulation data (4-km, CONUS1).

import xarray as xr
import numpy as np
from ncar_jobqueue import NCARCluster
from dask.distributed import Client

cluster = NCARCluster(memory="109GB", cores=36)
cluster.adapt(minimum=12, maximum=45, wait_count=60)
cluster

print(cluster.job_script())

client = Client(cluster)
client


def open_files(year, month):
    
    """
    Helper function to open data sets for CAPE2D, etc. calculation.

    Inputs:
    year: year in the loop (str; 4-digit)
    month: month in the loop (str; 2-digit)

    Outputs:
    data_pstag: total pressure (P0+PB) (3d)
    data_tstag: temperature (3d)
    data_qstag: water vapor mixing ratio (3d)
    data_zstag: geopotential height (3d)
    data_sstag: surface pressure (2d)
    """
    
    #pres_hpa: full pressure (perturb and base state) #convert from Pa to hPa.
    data_pstag = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/CTRL3D/{year}/wrf3d_d01_CTRL_P_{year}{month}*.nc',
                                   combine='by_coords', parallel=True, chunks={'Time':1}).P*0.01 
    #temperature (kelvin)
    data_tstag = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/CTRL3D/{year}/wrf3d_d01_CTRL_TK_{year}{month}*.nc', 
                                   combine='by_coords', parallel=True, chunks={'Time':1}).TK
    #water vapor mixing ratio (kg/kg)
    data_qstag = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/CTRL3D/{year}/wrf3d_d01_CTRL_QVAPOR_{year}{month}*.nc', 
                                   combine='by_coords', parallel=True, chunks={'Time':1}).QVAPOR
    #geopotential height (m)
    data_zstag = xr.open_mfdataset(f'/glade/scratch/molina/WRF_CONUS1_derived/current/wrf3d_d01_CTRL_Zdestag_{year}{month}*.nc', 
                                   combine='by_coords', parallel=True, chunks={'Time':1}).Z
    #terrain (m)
    data_mapfc = xr.open_dataset('/gpfs/fs1/collections/rda/data/ds612.0/INVARIANT/RALconus4km_wrf_constants.nc').HGT.sel(Time='2000-10-01')  
            
    if month == '01' or month == '02' or month == '03':
        month1 = '01'
        month2 = '03'
    if month == '04' or month == '05' or month == '06':
        month1 = '04'
        month2 = '06'
    if month == '07' or month == '08' or month == '09':
        month1 = '07'
        month2 = '09'
    if month == '10' or month == '11' or month == '12':
        month1 = '10'
        month2 = '12'
            
    data_sstag = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/CTRL/{year}/wrf2d_d01_CTRL_PSFC_{year}{month1}-{year}{month2}*.nc', 
                                   combine='by_coords', parallel=True, chunks={'Time':1}).PSFC*0.01 
    data_sstag = data_sstag[(data_sstag.Time.dt.hour==0)|(data_sstag.Time.dt.hour==3)|(data_sstag.Time.dt.hour==6)|
                            (data_sstag.Time.dt.hour==9)|(data_sstag.Time.dt.hour==12)|(data_sstag.Time.dt.hour==15)|
                            (data_sstag.Time.dt.hour==18)|(data_sstag.Time.dt.hour==21)]
    data_sstag = data_sstag[(data_sstag.Time.dt.month==int(month))]
    
    return data_pstag, data_tstag, data_qstag, data_zstag, data_mapfc, data_sstag


def wrf_cape(data_pstag, data_tstag, data_qstag, data_zstag, data_mapfc, data_sstag):
    
    """
    Function to compute cape, cin, lcl, and lfc using wrf-python.

    Outputs:
    Cape... values across spatial domain (2d) and with time aggregation encoded for ufunc.
    """ 
 
    #Imports needed for job workers.
    import os
    os.environ["PROJ_LIB"] = "/glade/work/abanihi/softwares/miniconda3/envs/python-tutorial/share/proj"
    import wrf
    
    return (wrf.cape_2d(pres_hpa = data_pstag.squeeze(), 
                        tkel = data_tstag.squeeze(), 
                        qv = data_qstag.squeeze(), 
                        height = data_zstag.squeeze(),
                        terrain = data_mapfc, 
                        psfc_hpa = data_sstag.squeeze(), 
                        ter_follow = True, meta = True).expand_dims("Time"))


def apply_wrf_cape(data_pstag, data_tstag, data_qstag, data_zstag, data_mapfc, data_sstag):
    
    """
    Generate Xarray ufunc to parallelize the wrf-python cape2d computation.
    """

    return xr.apply_ufunc(wrf_cape, data_pstag, data_tstag, 
                          data_qstag, data_zstag, data_mapfc, 
                          data_sstag,
                          dask='parallelized', 
                          output_dtypes=[float],
                          input_core_dims=[['bottom_top','south_north','west_east'],
                                           ['bottom_top','south_north','west_east'],
                                           ['bottom_top','south_north','west_east'],
                                           ['bottom_top','south_north','west_east'],
                                           ['south_north','west_east'],
                                           ['south_north','west_east']],
                          output_sizes=dict(mcape_mcin_lcl_lfc=4,south_north=1015, west_east=1359),
                          output_core_dims=[['mcape_mcin_lcl_lfc','south_north','west_east']])
                          
                          
#temporal arrays
formatter = "{:02d}".format
months = np.array(list(map(formatter, np.arange(1,13,1))))
years = np.array(list(map(formatter, np.arange(2001,2013,1))))


for year in years:
    for month in months:

        print(f"opening {year} {month} files")
        data_pstag, data_tstag, data_qstag, data_zstag, data_mapfc, data_sstag = open_files(year, month)
        
        print(f"generating u_func")
        result_ufunc = apply_wrf_cape(data_pstag, data_tstag, data_qstag, data_zstag, data_mapfc, data_sstag)

        print(f"starting cape for {year} {month}")
        r = result_ufunc.compute(retries=10)
        
        print(f"Saving file")
        r.to_dataset(name='cape_cin_lcl_lfc').to_netcdf(f"/glade/scratch/molina/WRF_CONUS1_derived/current/wrf2d_CAPE_{year}{month}.nc")
        
        r = r.close()
        data_pstag = data_pstag.close()
        data_tstag = data_tstag.close()
        data_qstag = data_qstag.close()
        data_zstag = data_zstag.close()
        data_mapfc = data_mapfc.close()
        data_sstag = data_sstag.close()
        
        print(f"woohoo! {year} {month} complete")
