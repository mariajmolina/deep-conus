
#####################################################################################
#####################################################################################
#
# Author: Maria J. Molina
# National Center for Atmospheric Research
#
# Script to compute CAPE (CIN, LCL, and LFC), CTT (cloud top temp) and UH (updraft helicity) 
# from WRF simulation data (4-km, CONUS1).
#
#####################################################################################
#####################################################################################


#------------------------------------------------------

import xarray as xr
import numpy as np
from ncar_jobqueue import NCARCluster
from dask.distributed import Client

#------------------------------------------------------



class compute_variable:


    def __init__(self, climate, variable, month_start, month_end, year_start, year_end, destination,
                 start_dask=True, project_code=None, cluster_min=10, cluster_max=40, 
                 dx=4000.0, dy=4000.0, uh_bottom=2000.0, uh_top=5000.0):

        """

        Instantiation of compute_variable:

        Here we will be computing a variable using state variable data output from CONUS1 simulations.

        PARAMETERS
        ----------
        climate: 'current' or 'future' (str)
        variable: CAPE, CTT, UH (str)
        month: start and end month for the respective interpolation operation (int)
        year: start year of analysis (int)
        destination: directory to save at (str)
        start_dask: whether to launch dask workers or not (boolean)
        project_code: charge code for supercomputer account (str; default None)
        cluster_min: the minimum number of nodes (with 36 CPUs) to initiate for adaptive dask job (str; default 10 [set for interp])
        cluster_max: the maximum number of nodes (with 36 CPUs) to initiate for adaptive dask job (str; default 40 [set for interp])
        dx, dy: domain grid spacing (float; default 4000.0 meters)
        uh_bottom, uh_top: bottom and top vertical levels for updraft helicity calculation (float; default 2000.0 meters and 5000.0 meters)

        """

        if climate!='current' and climate!='future':
            raise Exception("Please enter current or future as string for climate period selection.")
        if climate=='current' or climate=='future':
            self.climate = climate

        if self.climate == 'current':
            self.folder = 'CTRL3D'
            self.filename = 'CTRL'
        if self.climate == 'future':
            self.folder = 'PGW3D'
            self.filename = 'PGW'

        if variable != 'CAPE' and variable != 'CTT' and variable != 'UH':
            raise Exception("Variable not available. Please enter CAPE, CTT, or UH.")
        if variable == 'CAPE' or variable == 'CTT' or variable == 'UH':
            self.variable = variable

        self.month1 = month_start
        self.month2 = month_end
        self.year1 = year_start

        if year_start == year_end:
            self.year2 = year_end+1
        if year_start != year_end:
            self.year2 = year_end

        self.destination = destination

        self.daskstatus = start_dask
        if self.daskstatus:
            if not project_code:
                raise Exception("Must provide project code to launch dask workers.")
            if project_code:
                self.project_code = project_code
                self.cluster_min = cluster_min
                self.cluster_max = cluster_max

        self.dx = dx
        self.dy = dy
        self.uh_bottom = uh_bottom
        self.uh_top = uh_top



    def open_files(self, year, month):
    
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

        #terrain (m)
        data_mapfc = xr.open_dataset('/gpfs/fs1/collections/rda/data/ds612.0/INVARIANT/RALconus4km_wrf_constants.nc').HGT.sel(Time='2000-10-01')
    
        if self.variable == 'CAPE':
            #pres_hpa: full pressure (perturb and base state) #convert from Pa to hPa.
            data_pstag = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/{self.folder}/{year}/wrf3d_d01_{self.filename}_P_{year}{month}*.nc',
                                           combine='by_coords', parallel=True, chunks={'Time':1}).P*0.01 
            #temperature (kelvin)
            data_tstag = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/{self.folder}/{year}/wrf3d_d01_{self.filename}_TK_{year}{month}*.nc', 
                                           combine='by_coords', parallel=True, chunks={'Time':1}).TK
            #water vapor mixing ratio (kg/kg)
            data_qstag = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/{self.folder}/{year}/wrf3d_d01_{self.filename}_QVAPOR_{year}{month}*.nc', 
                                           combine='by_coords', parallel=True, chunks={'Time':1}).QVAPOR
            #geopotential height (m)
            data_zstag = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/{self.folder}/{year}/wrf3d_d01_{self.filename}_Z_{year}{month}*.nc', 
                                           combine='by_coords', parallel=True, chunks={'Time':1}).Z
            data_zstag = 0.5*(data_zstag[:,0:50,:,:]+data_zstag[:,1:51,:,:])    
            
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
            
            data_sstag = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/{self.folder}/{year}/wrf2d_d01_{self.filename}_PSFC_{year}{month1}-{year}{month2}*.nc', 
                                           combine='by_coords', parallel=True, chunks={'Time':1}).PSFC*0.01 
            data_sstag = data_sstag[(data_sstag.Time.dt.hour==0)|(data_sstag.Time.dt.hour==3)|(data_sstag.Time.dt.hour==6)|
                                    (data_sstag.Time.dt.hour==9)|(data_sstag.Time.dt.hour==12)|(data_sstag.Time.dt.hour==15)|
                                    (data_sstag.Time.dt.hour==18)|(data_sstag.Time.dt.hour==21)]
            data_sstag = data_sstag[(data_sstag.Time.dt.month==int(month))]
            return data_pstag, data_tstag, data_qstag, data_zstag, data_mapfc, data_sstag


        if self.variable == 'CTT':
            #pres_hpa: full pressure (perturb and base state) #convert from Pa to hPa.
            data_pstag = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/{self.folder}/{year}/wrf3d_d01_{self.filename}_P_{year}{month}*.nc',
                                           combine='by_coords', parallel=True, chunks={'Time':1}).P*0.01 
            #temperature (kelvin)
            data_tstag = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/{self.folder}/{year}/wrf3d_d01_{self.filename}_TK_{year}{month}*.nc',
                                           combine='by_coords', parallel=True, chunks={'Time':1}).TK
            #water vapor mixing ratio (kg/kg)
            data_qstag = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/{self.folder}/{year}/wrf3d_d01_{self.filename}_QVAPOR_{year}{month}*.nc',
                                           combine='by_coords', parallel=True, chunks={'Time':1}).QVAPOR                              
            #qcld: cloud water vapor mixing ratio (kg/kg)
            data_cloudstag = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/{self.folder}/{year}/wrf3d_d01_{self.filename}_QCLOUD_{year}{month}.nc', 
                                               combine='by_coords', parallel=True, chunks={'Time':1}).QCLOUD
            #geopotential height (m)
            data_zstag = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/{self.folder}/{year}/wrf3d_d01_{self.filename}_Z_{year}{month}*.nc',
                                           combine='by_coords', parallel=True, chunks={'Time':1}).Z
            data_zstag = 0.5*(data_zstag[:,0:50,:,:]+data_zstag[:,1:51,:,:])
            #qice: ice mixing ratio (kg/kg)
            data_icestag = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/{self.folder}/{year}/wrf3d_d01_{self.filename}_QICE_{year}{month}.nc', 
                                             combine='by_coords', parallel=True, chunks={'Time':1}).QICE                         
            return data_pstag, data_tstag, data_qstag, data_cloudstag, data_zstag, data_mapfc, data_icestag


        if self.variable == 'UH':
            #geopotential height (m)
            data_zstag = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/{self.folder}/{year}/wrf3d_d01_{self.filename}_Z_{year}{month}*.nc', 
                                           combine='by_coords', parallel=True, chunks={'Time':1}).Z
            #vertical wind component (m/s)
            data_wstag = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/{self.folder}/{year}/wrf3d_d01_{self.filename}_W_{year}{month}*.nc', 
                                           combine='by_coords', parallel=True, chunks={'Time':1}).W
            #U-wind component (m/s)
            data_ustag = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/{self.folder}/{year}/wrf3d_d01_{self.filename}_EU_{year}{month}*.nc', 
                                           combine='by_coords', parallel=True, chunks={'Time':1}).EU
            #V-wind component (m/s)
            data_vstag = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/{self.folder}/{year}/wrf3d_d01_{self.filename}_EV_{year}{month}*.nc', 
                                           combine='by_coords', parallel=True, chunks={'Time':1}).EV
            return data_zstag, data_wstag, data_ustag, data_vstag, data_mapfc




    def activate_workers(self):
        #start dask workers
        cluster = NCARCluster(memory="109GB", cores=36, project=self.project_code)
        cluster.adapt(minimum=self.cluster_min, maximum=self.cluster_max, wait_count=60)
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




    def create_the_variable_files(self):

        """
            Automate the work pipeline
        """


        if self.daskstatus:
            self.activate_workers()

        yrs, mos = self.generate_timestrings()

        for yr in yrs:
            for mo in mos:

                if self.variable == 'CAPE':
                    print(f"opening {yr} {mo} files for cape")
                    data_pstag, data_tstag, data_qstag, data_zstag, data_mapfc, data_sstag = self.open_files(yr, mo)
                    print(f"generating u_func")
                    result_ufunc = apply_wrf_cape(data_pstag, data_tstag, data_qstag, data_zstag, data_mapfc, data_sstag)
                    print(f"starting cape for {yr} {mo}")
                    r = result_ufunc.compute(retries=10)
                    print(f"Saving file")
                    r.to_dataset(name='cape_cin_lcl_lfc').to_netcdf(f"/{self.destination}/wrf2d_CAPE_{yr}{mo}.nc")
                    r = r.close()
                    data_pstag = data_pstag.close()
                    data_tstag = data_tstag.close()
                    data_qstag = data_qstag.close()
                    data_zstag = data_zstag.close()
                    data_mapfc = data_mapfc.close()
                    data_sstag = data_sstag.close()
                    print(f"woohoo! {yr} {mo} complete for CAPE")

                if self.variable == 'CTT':
                    print(f"opening {yr} {mo} files for ctt")
                    data_pstag, data_tstag, data_qstag, data_cloudstag, data_zstag, data_mapfc, data_icestag = self.open_files(yr, mo) 
                    print(f"generating u_func")
                    result_ufunc = apply_wrf_cloudtemp(data_pstag, data_tstag, data_qstag, data_cloudstag, data_zstag, data_mapfc, data_icestag)
                    print(f"starting ctt for {yr} {mo}")
                    r = result_ufunc.compute(retries=10)
                    print(f"Saving file")
                    r.to_dataset(name='ctt').to_netcdf(f"/{self.destination}/wrf2d_CTT_{yr}{mo}.nc")
                    r = r.close()
                    data_pstag = data_pstag.close()
                    data_tstag = data_tstag.close()
                    data_qstag = data_qstag.close()
                    data_cloudstag = data_cloudstag.close()
                    data_zstag = data_zstag.close()
                    data_mapfc = data_mapfc.close()
                    data_icestag = data_icestag.close()
                    print(f"woohoo! {yr} {mo} complete for CTT")

                if self.variable == 'UH':
                    print(f"opening {yr} {mo}...")
                    data_zstag, data_wstag, data_ustag, data_vstag, data_mapfc = self.open_files(yr, mo)
                    print(f"generating u_func")
                    result_ufunc = apply_wrf_UH(data_zstag, data_mapfc, data_ustag, data_vstag, data_wstag, dx=self.dx, dy=self.dy, bottom=self.uh_bottom, top=self.uh_top)
                    print(f"starting UH for {yr} {mo}")
                    r = result_ufunc.compute(retries=10)
                    print(f"Saving file")
                    r.to_dataset(name='uh').to_netcdf(f"/{self.destination}/wrf2d_UH_{yr}{mo}.nc")
                    r.close()
                    data_zstag = data_zstag.close()
                    data_mapfc = data_mapfc.close()
                    data_ustag = data_ustag.close()
                    data_vstag = data_vstag.close()
                    data_wstag = data_wstag.close()           
                    print(f"woohoo! {yr} {mo} complete")




def wrf_cape(data_pstag, data_tstag, data_qstag, data_zstag, data_mapfc, data_sstag):
    
    """
    Function to compute cape, cin, lcl, and lfc using wrf-python.

    Outputs:
    Cape... values across spatial domain (2d) and with time aggregation encoded for ufunc.
    """ 
 
    #Imports needed for job workers.
    import os
    os.environ["PROJ_LIB"] = "/glade/work/molina/miniconda3/envs/python-tutorial/share/proj/"
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
                          


def wrf_cloudtemp(data_pstag, data_tstag, data_qstag, data_cloudstag, data_zstag, data_mapfc, data_icestag):
    
    """
    Function to ctt using wrf-python.
    Outputs:
    Values across spatial domain (2d) and with time aggregation encoded for ufunc.
    """ 
 
    #Imports needed for job workers.
    import os
    os.environ["PROJ_LIB"] = "/glade/work/molina/miniconda3/envs/python-tutorial/share/proj/"
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
        Generate Xarray ufunc to parallelize the wrf-python ctt computation.
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

                

def wrf_UH(z, mapfc, u, v, w, dx, dy, bottom, top):
    
    #Imports needed for job workers.
    import os
    os.environ["PROJ_LIB"] = "/glade/work/molina/miniconda3/envs/python-tutorial/share/proj/"
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



def apply_wrf_UH(data_zstag, data_mapfc, data_ustag, data_vstag, data_wstag, dx, dy, bottom, top):
    
    """
        Generate Xarray ufunc to parallelize the wrf-python UH computation.
    """

    return xr.apply_ufunc(wrf_UH, data_zstag, data_mapfc, data_ustag, data_vstag, data_wstag, dx, dy, bottom, top,
                          dask='parallelized', output_dtypes=[float],
                          input_core_dims=[['bottom_top_stag','south_north','west_east'],
                                           ['south_north','west_east'],
                                           ['bottom_top','south_north','west_east'],
                                           ['bottom_top','south_north','west_east'],
                                           ['bottom_top_stag','south_north','west_east'],
                                           [None], [None], [None], [None]],
                          output_sizes=dict(south_north=1015,west_east=1359),
                          output_core_dims=[['south_north','west_east']])



