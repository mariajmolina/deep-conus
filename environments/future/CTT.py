###################################################
###################################################
##
##Author: Maria J. Molina
##National Center for Atmospheric Research
##
###################################################
###################################################


#Script to compute cloud top temperature from environmental data 
#(WRF CONUS1 4-km simulations) during future climate.


#--------------------------------------------------

import xarray as xr
import numpy as np
from ncar_jobqueue import NCARCluster
from dask.distributed import Client

#--------------------------------------------------


def main():
    #start dask workers
    cluster = NCARCluster(memory="109GB", cores=36)
    cluster.adapt(minimum=14, maximum=45, wait_count=60)
    cluster
    #print scripts
    print(cluster.job_script())
    #start client
    client = Client(cluster)
    client

    #temporal arrays
    formatter = "{:02d}".format
    months = np.array(list(map(formatter, np.arange(1,13,1))))
    years = np.array(list(map(formatter, np.arange(2001,2013,1))))

    for year in years:
        for month in months:

            print(f"opening {year} {month} files")
            data_pstag, data_tstag, data_qstag, data_cloudstag, data_zstag, data_mapfc, data_icestag = open_files(year, month)

            print(f"generating u_func")
            result_ufunc = apply_wrf_cloudtemp(data_pstag, data_tstag, data_qstag, data_cloudstag, data_zstag, data_mapfc, data_icestag)

            print(f"starting ctt for {year} {month}")
            r = result_ufunc.compute(retries=10)

            print(f"Saving file")
            r.to_dataset(name='ctt').to_netcdf(f"/glade/scratch/molina/WRF_CONUS1_derived/future/wrf2d_CTT_{year}{month}.nc")

            r = r.close()
            data_pstag = data_pstag.close()
            data_tstag = data_tstag.close()
            data_qstag = data_qstag.close()
            data_cloudstag = data_cloudstag.close()
            data_zstag = data_zstag.close()
            data_mapfc = data_mapfc.close()
            data_icestag = data_icestag.close()

            print(f"woohoo! {year} {month} complete")
            

def open_files(year, month):
    
    """
    Helper function to open data sets for ctt calculation.
    Inputs:
    year: year in the loop (str; 4-digit)
    month: month in the loop (str; 2-digit)
    Outputs:
    data_pstag: total pressure (P0+PB) (3d)
    data_tstag: temperature (3d)
    data_qstag: water vapor mixing ratio (3d)
    data_cloudstag: cloud water vapor mixing ratio (3d)
    data_zstag: geopotential height (3d)
    data_mapfc: terrain height in [m] (2d)
    data_icestag: ice mixing ratio (3d)
    """
    
    #pres_hpa: full pressure (perturb and base state) #convert from Pa to hPa.
    data_pstag = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/PGW3D/{year}/wrf3d_d01_PGW_P_{year}{month}*.nc',
                                   combine='by_coords', parallel=True, chunks={'Time':1}).P*0.01 
    #temperature (kelvin)
    data_tstag = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/PGW3D/{year}/wrf3d_d01_PGW_TK_{year}{month}*.nc',
                                   combine='by_coords', parallel=True, chunks={'Time':1}).TK
    #water vapor mixing ratio (kg/kg)
    data_qstag = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/PGW3D/{year}/wrf3d_d01_PGW_QVAPOR_{year}{month}*.nc',
                                   combine='by_coords', parallel=True, chunks={'Time':1}).QVAPOR                              
    #qcld: cloud water vapor mixing ratio (kg/kg)
    data_cloudstag = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/PGW3D/{year}/wrf3d_d01_PGW_QCLOUD_{year}{month}.nc', 
                                       combine='by_coords', parallel=True, chunks={'Time':1}).QCLOUD
    #geopotential height (m)
    data_zstag = xr.open_mfdataset(f'/glade/scratch/molina/WRF_CONUS1_derived/future/wrf3d_d01_PGW_Zdestag_{year}{month}*.nc', 
                                   combine='by_coords', parallel=True, chunks={'Time':1}).Z
    #terrain (m)
    data_mapfc = xr.open_dataset('/gpfs/fs1/collections/rda/data/ds612.0/INVARIANT/RALconus4km_wrf_constants.nc').HGT.sel(Time='2000-10-01')  
    #qice: ice mixing ratio (kg/kg)
    data_icestag = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/PGW3D/{year}/wrf3d_d01_PGW_QICE_{year}{month}.nc', 
                                    combine='by_coords', parallel=True, chunks={'Time':1}).QICE
                                     
    return data_pstag, data_tstag, data_qstag, data_cloudstag, data_zstag, data_mapfc, data_icestag


def wrf_cloudtemp(data_pstag, data_tstag, data_qstag, data_cloudstag, data_zstag, data_mapfc, data_icestag):
    
    """
    Function to compute cloud top temp (ctt) using wrf-python.
    """ 
 
    #Imports needed for job workers.
    import os
    os.environ["PROJ_LIB"] = "/glade/work/abanihi/softwares/miniconda3/envs/python-tutorial/share/proj"
    import wrf
    
    return (wrf.ctt(data_pstag.squeeze(), 
                    data_tstag.squeeze(), 
                    data_qstag.squeeze(), 
                    data_cloudstag.squeeze(), 
                    data_zstag.squeeze(), 
                    data_mapfc, 
                    data_icestag.squeeze(), 
                    fill_nocloud=False, missing=9.969209968386869e+36, opt_thresh=1.0, 
                    meta=True, units='degC').expand_dims("Time"))


def apply_wrf_cloudtemp(data_pstag, data_tstag, data_qstag, data_cloudstag, data_zstag, data_mapfc, data_icestag):
    
    """
    Generate Xarray ufunc to parallelize the wrf-python cloud top temp (ctt) computation.
    """

    return xr.apply_ufunc(wrf_cloudtemp, data_pstag, data_tstag, data_qstag, 
                          data_cloudstag, data_zstag, data_mapfc, data_icestag,
                          dask='parallelized', 
                          output_dtypes=[float],
                          input_core_dims=[['bottom_top','south_north','west_east'],
                                           ['bottom_top','south_north','west_east'],
                                           ['bottom_top','south_north','west_east'],
                                           ['bottom_top','south_north','west_east'],
                                           ['bottom_top','south_north','west_east'],
                                           ['south_north','west_east'],
                                           ['bottom_top','south_north','west_east']],
                          output_sizes=dict(south_north=1015, west_east=1359),
                          output_core_dims=[['south_north','west_east']])

        
#--------------------------------------------------

if __name__== "__main__":
    main()

#--------------------------------------------------
