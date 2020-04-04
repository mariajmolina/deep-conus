#####################################################################################
#####################################################################################
#
# Author: Maria J. Molina
# National Center for Atmospheric Research
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

    """Class instantiation of compute_variable:

    Here we will be computing a variable using state variable data output from CONUS1 simulations.

    Attributes:
        climate (str): Whether to interpolate variable in the ``current`` or ``future`` climate simulation.
        variable (str): Variable for analysis. Options include ``CAPE``, ``CTT``, and ``UH``. 
        month_start (int): Start month for the respective variable calculation.
        month_end (int): End month for the respective variable calculation.
        year_start (int): Start year for the respective variable calculation.
        year_end (int): End year for the respective variable calculation.
        destination (str): Directory path of where to save the calculated variable.
        rda_path (str): Directory path of where the data files are saved. Defaults to ``/gpfs/fs1/collections/rda/data/ds612.0/``.
        start_dask (boolean): Whether to launch dask workers or not. Defaults to ``True``.
        project_code (str): The supercomputer account charge code. Defaults to ``None``.
        cluster_min (str): The minimum number of nodes to initiate for adaptive dask job. Defaults to 10. Each node contains 36 CPUs on Cheyenne.
        cluster_max (str): The maximum number of nodes to initiate for adaptive dask job. Defaults to 40.
        dx, dy (float): The domain grid spacing. Defaults to 4000.0 (meters).
        uh_bottom, uh_top (float): Bottom and top vertical levels for updraft helicity calculation. Defaults to 2000.0 and 5000.0 meters.

    """
    

    def __init__(self, climate, variable, month_start, month_end, year_start, year_end, destination,
                 rda_path='/gpfs/fs1/collections/rda/data/ds612.0/',
                 start_dask=True, project_code=None, cluster_min=10, cluster_max=40, 
                 dx=4000.0, dy=4000.0, uh_bottom=2000.0, uh_top=5000.0):

        if climate!='current' and climate!='future':
            raise Exception("Please enter current or future as string for climate period selection.")
        if climate=='current' or climate=='future':
            self.climate=climate

        if self.climate == 'current':
            self.folder='CTRL3D'
            self.filename='CTRL'
        if self.climate == 'future':
            self.folder='PGW3D'
            self.filename='PGW'

        if variable!='CAPE' and variable!='CTT' and variable!='UH':
            raise Exception("Variable not available. Please enter CAPE, CTT, or UH.")
        if variable == 'CAPE' or variable == 'CTT' or variable == 'UH':
            self.variable=variable

        self.month1=month_start
        self.month2=month_end
        self.year1=year_start
        if year_start == year_end:
            self.year2 = year_end + 1
        if year_start!=year_end:
            self.year2=year_end

        self.destination=destination
        self.rda_path=rda_path

        self.daskstatus=start_dask
        if self.daskstatus:
            if not project_code:
                raise Exception("Must provide project code to launch dask workers.")
            if project_code:
                self.project_code=project_code
                self.cluster_min=cluster_min
                self.cluster_max=cluster_max

        self.dx=dx
        self.dy=dy
        self.uh_bottom=uh_bottom
        self.uh_top=uh_top



    def open_files(self, year, month):
    
        """Helper function to open data sets for variable calculations.
        
        Args:
            year (str): Year to open (4-digit).
            month (str): Month to open (2-digit).
            
        Returns:
            data_pstag (Xarray data array): Total pressure (P0+PB) (3d)
            data_tstag (Xarray data array): Temperature (3d)
            data_qstag (Xarray data array): Water vapor mixing ratio (3d)
            data_zstag (Xarray data array): Geopotential height (3d)
            data_mapfc (Xarray data array): Terrain data (2d).
            data_sstag (Xarray data array): Surface pressure (2d).
            data_cloudstag (Xarray data array): Cloud water vapor mixing ratio (3d).
            data_icestag (Xarray data array): Ice mixing ratio (3d).
            data_wstag (Xarray data array): Z-wind component (3d).
            data_ustag (Xarray data array): U-wind component (3d).
            data_vstag (Xarray data array): V-wind component (3d).
             
        """
        #terrain (m)
        data_mapfc=xr.open_dataset(f'{self.rda_path}INVARIANT/RALconus4km_wrf_constants.nc').HGT.sel(Time='2000-10-01')
        if self.variable == 'CAPE':
            #pres_hpa: full pressure (perturb and base state) #convert from Pa to hPa.
            data_pstag=xr.open_mfdataset(f'{self.rda_path}{self.folder}/{year}/wrf3d_d01_{self.filename}_P_{year}{month}*.nc',
                                           combine='by_coords', parallel=True, chunks={'Time':1}).P * 0.01 
            #temperature (kelvin)
            data_tstag=xr.open_mfdataset(f'{self.rda_path}{self.folder}/{year}/wrf3d_d01_{self.filename}_TK_{year}{month}*.nc', 
                                           combine='by_coords', parallel=True, chunks={'Time':1}).TK
            #water vapor mixing ratio (kg/kg)
            data_qstag=xr.open_mfdataset(f'{self.rda_path}{self.folder}/{year}/wrf3d_d01_{self.filename}_QVAPOR_{year}{month}*.nc', 
                                           combine='by_coords', parallel=True, chunks={'Time':1}).QVAPOR
            #geopotential height (m)
            data_zstag=xr.open_mfdataset(f'{self.rda_path}{self.folder}/{year}/wrf3d_d01_{self.filename}_Z_{year}{month}*.nc', 
                                           combine='by_coords', parallel=True, chunks={'Time':1}).Z
            data_zstag = 0.5 * (data_zstag[:,0:50,:,:] + data_zstag[:,1:51,:,:])    
            if month == '01' or month == '02' or month == '03':
                month1='01'
                month2='03'
            if month == '04' or month == '05' or month == '06':
                month1='04'
                month2='06'
            if month == '07' or month == '08' or month == '09':
                month1='07'
                month2='09'
            if month == '10' or month == '11' or month == '12':
                month1='10'
                month2='12'
            data_sstag=xr.open_mfdataset(f'{self.rda_path}{self.folder}/{year}/wrf2d_d01_{self.filename}_PSFC_{year}{month1}-{year}{month2}*.nc', 
                                           combine='by_coords', parallel=True, chunks={'Time':1}).PSFC * 0.01 
            data_sstag=data_sstag[(data_sstag.Time.dt.hour==0)|(data_sstag.Time.dt.hour==3)|(data_sstag.Time.dt.hour==6)|
                                    (data_sstag.Time.dt.hour==9)|(data_sstag.Time.dt.hour==12)|(data_sstag.Time.dt.hour==15)|
                                    (data_sstag.Time.dt.hour==18)|(data_sstag.Time.dt.hour==21)]
            data_sstag=data_sstag[(data_sstag.Time.dt.month==int(month))]
            return data_pstag, data_tstag, data_qstag, data_zstag, data_mapfc, data_sstag

        if self.variable == 'CTT':
            #pres_hpa: full pressure (perturb and base state) #convert from Pa to hPa.
            data_pstag=xr.open_mfdataset(f'{self.rda_path}{self.folder}/{year}/wrf3d_d01_{self.filename}_P_{year}{month}*.nc',
                                           combine='by_coords', parallel=True, chunks={'Time':1}).P*0.01 
            #temperature (kelvin)
            data_tstag=xr.open_mfdataset(f'{self.rda_path}{self.folder}/{year}/wrf3d_d01_{self.filename}_TK_{year}{month}*.nc',
                                           combine='by_coords', parallel=True, chunks={'Time':1}).TK
            #water vapor mixing ratio (kg/kg)
            data_qstag=xr.open_mfdataset(f'{self.rda_path}{self.folder}/{year}/wrf3d_d01_{self.filename}_QVAPOR_{year}{month}*.nc',
                                           combine='by_coords', parallel=True, chunks={'Time':1}).QVAPOR                              
            #qcld: cloud water vapor mixing ratio (kg/kg)
            data_cloudstag=xr.open_mfdataset(f'{self.rda_path}{self.folder}/{year}/wrf3d_d01_{self.filename}_QCLOUD_{year}{month}.nc', 
                                               combine='by_coords', parallel=True, chunks={'Time':1}).QCLOUD
            #geopotential height (m)
            data_zstag=xr.open_mfdataset(f'{self.rda_path}{self.folder}/{year}/wrf3d_d01_{self.filename}_Z_{year}{month}*.nc',
                                           combine='by_coords', parallel=True, chunks={'Time':1}).Z
            data_zstag = 0.5 * (data_zstag[:,0:50,:,:] + data_zstag[:,1:51,:,:])
            #qice: ice mixing ratio (kg/kg)
            data_icestag=xr.open_mfdataset(f'{self.rda_path}{self.folder}/{year}/wrf3d_d01_{self.filename}_QICE_{year}{month}.nc', 
                                             combine='by_coords', parallel=True, chunks={'Time':1}).QICE                         
            return data_pstag, data_tstag, data_qstag, data_cloudstag, data_zstag, data_mapfc, data_icestag

        if self.variable == 'UH':
            #geopotential height (m)
            data_zstag=xr.open_mfdataset(f'{self.rda_path}{self.folder}/{year}/wrf3d_d01_{self.filename}_Z_{year}{month}*.nc', 
                                           combine='by_coords', parallel=True, chunks={'Time':1}).Z
            #vertical wind component (m/s)
            data_wstag=xr.open_mfdataset(f'{self.rda_path}{self.folder}/{year}/wrf3d_d01_{self.filename}_W_{year}{month}*.nc', 
                                           combine='by_coords', parallel=True, chunks={'Time':1}).W
            #U-wind component (m/s)
            data_ustag=xr.open_mfdataset(f'{self.rda_path}{self.folder}/{year}/wrf3d_d01_{self.filename}_EU_{year}{month}*.nc', 
                                           combine='by_coords', parallel=True, chunks={'Time':1}).EU
            #V-wind component (m/s)
            data_vstag=xr.open_mfdataset(f'{self.rda_path}{self.folder}/{year}/wrf3d_d01_{self.filename}_EV_{year}{month}*.nc', 
                                           combine='by_coords', parallel=True, chunks={'Time':1}).EV
            return data_zstag, data_wstag, data_ustag, data_vstag, data_mapfc



    def activate_workers(self):
        
        """Function to activate dask workers.
        
        """
        #start dask workers
        cluster=NCARCluster(memory="109GB", cores=36, project=self.project_code)
        cluster.adapt(minimum=self.cluster_min, maximum=self.cluster_max, wait_count=60)
        cluster
        #print scripts
        print(cluster.job_script())
        #start client
        client=Client(cluster)
        client



    def generate_timestrings(self):
        
        """Function to generate analysis years and months as strings.

        Returns:
            years (str): Array of year strings formatted with four digits.
            months (str): Array of month strings formatted with two digits.
            
        """
        formatter="{:02d}".format
        months=np.array(list(map(formatter, np.arange(self.month1, self.month2, 1))))
        formatter="{:04d}".format
        years =np.array(list(map(formatter, np.arange(self.year1, self.year2, 1))))
        return years, months



    def create_the_variable_files(self):

        """Automate the work pipeline and call methods in sequential order to yield saved files.
        
        """
        if self.daskstatus:
            self.activate_workers()
        yrs, mos=self.generate_timestrings()

        for yr in yrs:
            for mo in mos:

                if self.variable == 'CAPE':
                    print(f"opening {yr} {mo} files for cape")
                    data_pstag, data_tstag, data_qstag, data_zstag, data_mapfc, data_sstag=self.open_files(yr, mo)
                    print(f"generating u_func")
                    result_ufunc=apply_wrf_cape(data_pstag, data_tstag, data_qstag, data_zstag, data_mapfc, data_sstag)
                    print(f"starting cape for {yr} {mo}")
                    r=result_ufunc.compute(retries=10)
                    print(f"Saving file")
                    r.to_dataset(name='cape_cin_lcl_lfc').to_netcdf(f"/{self.destination}/wrf2d_CAPE_{yr}{mo}.nc")
                    r=r.close()
                    data_pstag=data_pstag.close()
                    data_tstag=data_tstag.close()
                    data_qstag=data_qstag.close()
                    data_zstag=data_zstag.close()
                    data_mapfc=data_mapfc.close()
                    data_sstag=data_sstag.close()
                    print(f"woohoo! {yr} {mo} complete for CAPE")

                if self.variable == 'CTT':
                    print(f"opening {yr} {mo} files for ctt")
                    data_pstag, data_tstag, data_qstag, data_cloudstag, data_zstag, data_mapfc, data_icestag=self.open_files(yr, mo) 
                    print(f"generating u_func")
                    result_ufunc=apply_wrf_cloudtemp(data_pstag, data_tstag, data_qstag, data_cloudstag, data_zstag, data_mapfc, data_icestag)
                    print(f"starting ctt for {yr} {mo}")
                    r=result_ufunc.compute(retries=10)
                    print(f"Saving file")
                    r.to_dataset(name='ctt').to_netcdf(f"/{self.destination}/wrf2d_CTT_{yr}{mo}.nc")
                    r=r.close()
                    data_pstag=data_pstag.close()
                    data_tstag=data_tstag.close()
                    data_qstag=data_qstag.close()
                    data_cloudstag=data_cloudstag.close()
                    data_zstag=data_zstag.close()
                    data_mapfc=data_mapfc.close()
                    data_icestag=data_icestag.close()
                    print(f"woohoo! {yr} {mo} complete for CTT")

                if self.variable == 'UH':
                    print(f"opening {yr} {mo}...")
                    data_zstag, data_wstag, data_ustag, data_vstag, data_mapfc=self.open_files(yr, mo)
                    print(f"generating u_func")
                    result_ufunc=apply_wrf_UH(data_zstag, data_mapfc, data_ustag, data_vstag, data_wstag, dx=self.dx, dy=self.dy, bottom=self.uh_bottom, top=self.uh_top)
                    print(f"starting UH for {yr} {mo}")
                    r=result_ufunc.compute(retries=10)
                    print(f"Saving file")
                    r.to_dataset(name='uh').to_netcdf(f"/{self.destination}/wrf2d_UH_{yr}{mo}.nc")
                    r.close()
                    data_zstag=data_zstag.close()
                    data_mapfc=data_mapfc.close()
                    data_ustag=data_ustag.close()
                    data_vstag=data_vstag.close()
                    data_wstag=data_wstag.close()           
                    print(f"woohoo! {yr} {mo} complete")



def wrf_cape(data_pstag, data_tstag, data_qstag, data_zstag, data_mapfc, data_sstag):
    
    """Function to compute ``cape``, ``cin``, ``lcl``, and ``lfc`` using wrf-python.
    
    Args:
        data_pstag (Xarray data array): Total pressure (P0+PB) (3d)
        data_tstag (Xarray data array): Temperature (3d)
        data_qstag (Xarray data array): Water vapor mixing ratio (3d)
        data_zstag (Xarray data array): Geopotential height (3d)
        data_mapfc (Xarray data array): Terrain data (2d).
        data_sstag (Xarray data array): Surface pressure (2d).  
    
    Returns:
        Data variables (``cape``, ``cin``, ``lcl``, and ``lfc``) in Xarray data array.

    """
    import os
    os.environ["PROJ_LIB"]="/glade/work/molina/miniconda3/envs/python-tutorial/share/proj/"
    import wrf
    return (wrf.cape_2d(pres_hpa=data_pstag.squeeze(), 
                        tkel=data_tstag.squeeze(), 
                        qv=data_qstag.squeeze(), 
                        height=data_zstag.squeeze(),
                        terrain=data_mapfc, 
                        psfc_hpa=data_sstag.squeeze(), 
                        ter_follow=True, meta=True).expand_dims("Time"))



def apply_wrf_cape(data_pstag, data_tstag, data_qstag, data_zstag, data_mapfc, data_sstag):
    
    """Generate Xarray ufunc to parallelize the wrf-python ``cape``, ``cin``, ``lcl``, and ``lfc`` computation.
    
    Args:
        data_pstag (Xarray data array): Total pressure (P0+PB) (3d)
        data_tstag (Xarray data array): Temperature (3d)
        data_qstag (Xarray data array): Water vapor mixing ratio (3d)
        data_zstag (Xarray data array): Geopotential height (3d)
        data_mapfc (Xarray data array): Terrain data (2d).
        data_sstag (Xarray data array): Surface pressure (2d).
        
    Returns:
        Function to parallelize data variable computation for ``cape``, ``cin``, ``lcl``, and ``lfc``.
        
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
    
    """Function to compute ``ctt`` using wrf-python.
    
    Args:
        data_pstag (Xarray data array): Total pressure (P0+PB) (3d).
        data_tstag (Xarray data array): Temperature (3d).
        data_qstag (Xarray data array): Water vapor mixing ratio (3d).
        data_cloudstag (Xarray data array): Cloud water vapor mixing ratio (3d).
        data_zstag (Xarray data array): Geopotential height (3d).
        data_mapfc (Xarray data array): Terrain data (2d).    
        data_icestag (Xarray data array): Ice mixing ratio (3d).
            
    Returns:
        Data variable ``ctt`` in Xarray data array.
        
    """ 
    import os
    os.environ["PROJ_LIB"]="/glade/work/molina/miniconda3/envs/python-tutorial/share/proj/"
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
    
    """Generate Xarray ufunc to parallelize the wrf-python ``ctt`` computation.
    
    Args:
        data_pstag (Xarray data array): Total pressure (P0+PB) (3d).
        data_tstag (Xarray data array): Temperature (3d).
        data_qstag (Xarray data array): Water vapor mixing ratio (3d).
        data_cloudstag (Xarray data array): Cloud water vapor mixing ratio (3d).
        data_zstag (Xarray data array): Geopotential height (3d).
        data_mapfc (Xarray data array): Terrain data (2d).    
        data_icestag (Xarray data array): Ice mixing ratio (3d).    
    
    Returns:
        Function to parallelize data variable computation for ``ctt`.
    
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
    
    """Function to compute ``updraft helicity`` using wrf-python.
    
    Args:
        z (Xarray data array): Geopotential height (3d).
        mapfc (Xarray data array): Terrain data (2d).
        u (Xarray data array): U-wind component (3d).
        v (Xarray data array): V-wind component (3d).
        w (Xarray data array): Z-wind component (3d).
        dx, dy (float): The domain grid spacing.
        bottom, top (float): Bottom and top vertical levels for updraft helicity calculation. 
        
    Returns:
        Data variable ``UH`` in Xarray data array.
        
    """
    import os
    os.environ["PROJ_LIB"]="/glade/work/molina/miniconda3/envs/python-tutorial/share/proj/"
    import wrf
    return (wrf.udhel( z.squeeze(), mapfc, u.squeeze(), v.squeeze(), w.squeeze(), \
                      dx=dx, dy=dy, bottom=bottom, top=top, meta=True).expand_dims("Time"))



def apply_wrf_UH(data_zstag, data_mapfc, data_ustag, data_vstag, data_wstag, dx, dy, bottom, top):
    
    """Generate Xarray ufunc to parallelize the wrf-python ``UH`` computation.
        
    Args:
        data_zstag (Xarray data array): Geopotential height (3d).
        data_mapfc (Xarray data array): Terrain data (2d).
        data_ustag (Xarray data array): U-wind component (3d).
        data_vstag (Xarray data array): V-wind component (3d).
        data_wstag (Xarray data array): Z-wind component (3d).
        dx, dy (float): The domain grid spacing.
        bottom, top (float): Bottom and top vertical levels for updraft helicity calculation.      
    
    Returns:
        Function to parallelize data variable computation for ``UH`.

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


