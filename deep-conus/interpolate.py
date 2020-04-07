import xarray as xr
import numpy as np


class InterpolateVariable:

    """Class instantiation of InterpolateVariable:

    Here we will be interpolating the respective variable onto 1, 3, 5, and 7 km above ground level (AGL).

    Attributes:
        climate (str): Whether to interpolate variable in the ``current`` or ``future`` climate simulation.
        variable (str): Variable for analysis. Options include ``TK``, ``QVAPOR``, ``EU``, ``EV``, ``P``, ``QGRAUP``, ``W``, and ``MAXW``.
        month_start (int): Start month for the respective interpolation operation.
        month_end (int): End month for the respective interpolation operation.
        year_start (int): Start year for the respective interpolation operation.
        year_end (int): End year for the respective interpolation operation.
        destination (str): Directory path of where to save the interpolated variable.
        rda_path (str): Directory path of where the data files are saved. Defaults to ``/gpfs/fs1/collections/rda/data/ds612.0/``.
        start_dask (boolean): Whether to launch dask workers or not. Defaults to ``True``.
        project_code (str): The supercomputer account charge code. Defaults to ``None``.
        cluster_min (str): The minimum number of nodes to initiate for adaptive dask job. Defaults to 10. Each node contains 36 CPUs on Cheyenne.
        cluster_max (str): The maximum number of nodes to initiate for adaptive dask job. Defaults to 40.

    """
        
    def __init__(self, climate, variable, month_start, month_end, year_start, year_end, destination, 
                 rda_path='/gpfs/fs1/collections/rda/data/ds612.0/',
                 start_dask=True, project_code=None, cluster_min=10, cluster_max=40):

        if climate!='current' and climate!='future':
            raise Exception("Please enter current or future as string for climate period selection.")
        if climate=='current' or climate=='future':
            self.climate=climate

        if self.climate=='current':
            self.folder='CTRL3D'
            self.filename='CTRL'
        if self.climate=='future':
            self.folder='PGW3D'
            self.filename='PGW'

        if variable!='TK' and variable!='QVAPOR' and variable!='EU' and variable!='EV' and variable!='P' and variable!='QGRAUP' and variable!='W' and variable!='MAXW':
            raise Exception("Variable not available. Please enter TK, QVAPOR, EU, EV, P, QGRAUP, W, or MAXW.")
        if variable=='TK' or variable=='QVAPOR' or variable=='EU' or variable=='EV' or variable=='P' or variable=='QGRAUP' or variable=='W' or variable=='MAXW':
            self.variable=variable

        self.month1=month_start
        self.month2=month_end
        self.year1=year_start

        if year_start==year_end:
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

    
    def open_files(self, year, month):
    
        """Helper function to open data sets for interpolation calculation for current climate.
        
        Args:
            year (str): Year. Should be a 4-digit string.
            month (str): Month. Should be a 2-digit string.
            
          Outputs:
             data_AGL (Xarray dask array): The above ground level heights (in meters). This is not output for variable ``MAXW``.
             data_var (Xarray dask array): The variable data.
             
        """                        
        if self.variable=='MAXW':
            data_var=xr.open_mfdataset(f'{self.rda_path}{self.folder}/{year}/wrf3d_d01_{self.filename}_W_{year}{month}*.nc',
                                       combine='by_coords', parallel=True, chunks={'Time':1}).W
            return data_var

        #geopotential height (m)
        data_zstag=xr.open_mfdataset(f'{self.rda_path}{self.folder}/{year}/wrf3d_d01_{self.filename}_Z_{year}{month}*.nc', 
                                   combine='by_coords', parallel=True, chunks={'Time':1}).Z
        data_zstag = 0.5 * (data_zstag[:,0:50,:,:] + data_zstag[:,1:51,:,:])
        #terrain (m)
        data_mapfc=xr.open_dataset(f'{self.rda_path}INVARIANT/RALconus4km_wrf_constants.nc').HGT.sel(Time='2000-10-01')  
        data_AGL = data_zstag - data_mapfc

        if self.variable=='W':
            #z-wind (m/s)
            data_var=xr.open_mfdataset(f'{self.rda_path}{self.folder}/{year}/wrf3d_d01_{self.filename}_W_{year}{month}*.nc',
                                       combine='by_coords', parallel=True, chunks={'Time':1}).W
            data_var = 0.5 * (data_var[:,0:50,:,:] + data_var[:,1:51,:,:])
        if self.variable=='TK':
            #temperature (kelvin)
            data_var=xr.open_mfdataset(f'{self.rda_path}{self.folder}/{year}/wrf3d_d01_{self.filename}_TK_{year}{month}*.nc',
                                       combine='by_coords', parallel=True, chunks={'Time':1}).TK
        if self.variable=='QVAPOR':
            #water vapor mixing ratio (kg/kg)
            data_var=xr.open_mfdataset(f'{self.rda_path}{self.folder}/{year}/wrf3d_d01_{self.filename}_QVAPOR_{year}{month}*.nc',
                                       combine='by_coords', parallel=True, chunks={'Time':1}).QVAPOR      
        if self.variable=='EU':
            #earth rotated u-winds (m/s)
            data_var=xr.open_mfdataset(f'{self.rda_path}{self.folder}/{year}/wrf3d_d01_{self.filename}_EU_{year}{month}*.nc', 
                                       combine='by_coords', parallel=True, chunks={'Time':1}).EU
        if self.variable=='EV':
            #earth rotated v-winds (m/s)
            data_var=xr.open_mfdataset(f'{self.rda_path}{self.folder}/{year}/wrf3d_d01_{self.filename}_EV_{year}{month}*.nc', 
                                       combine='by_coords', parallel=True, chunks={'Time':1}).EV
        if self.variable=='P':
            #pres_hpa: full pressure (perturb and base state) #convert from Pa to hPa.
            data_var=xr.open_mfdataset(f'{self.rda_path}{self.folder}/{year}/wrf3d_d01_{self.filename}_P_{year}{month}*.nc',
                                       combine='by_coords', parallel=True, chunks={'Time':1}).P*0.01 
        if self.variable=='QGRAUP':
            #Graupel mixing ratio (kg/kg)
            data_var=xr.open_mfdataset(f'{self.rda_path}{self.folder}/{year}/wrf3d_d01_{self.filename}_QGRAUP_{year}{month}.nc', 
                                       combine='by_coords', parallel=True, chunks={'Time':1}).QGRAUP
        return data_AGL, data_var


    def activate_workers(self):
        
        """Function to activate dask workers.
        
        """
        from ncar_jobqueue import NCARCluster
        from dask.distributed import Client
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


    def create_the_interp_files(self):

        """Automate the work pipeline and sequence of functions to be run to create the interpolation files.
        
        """
        if self.daskstatus:
            self.activate_workers()
        yrs, mos=self.generate_timestrings()
        for yr in yrs:
            for mo in mos:
                print(f"opening {yr} {mo} files for {self.variable}")
                data_AGL, data_var=self.open_files(yr, mo)
                print(f"generating u_func")
                if self.variable!='W':
                    result_ufunc=apply_wrf_interp(data_AGL, data_var)
                if self.variable=='W':
                    result_ufunc=apply_wrf_interp_W(data_AGL, data_var)
                print(f"starting interp for {yr} {mo}")
                r=result_ufunc.compute(retries=10)
                print(f"Saving file")
                r.to_dataset(name='levels').to_netcdf(f"/{self.destination}/wrf2d_interp_{self.variable}_{yr}{mo}.nc")
                r=r.close()
                data_AGL=data_AGL.close()
                data_var=data_var.close()
                print(f"woohoo! {yr} {mo} complete")


    def create_the_max_files(self):

        """Automate the work pipeline and sequence of functions to be run to create the maximum value files.
        
        """
        if self.daskstatus:
            self.activate_workers()
        yrs, mos=self.generate_timestrings()
        for yr in yrs:
            for mo in mos:
                print(f"opening {yr} {mo} files for {self.variable}")
                if self.variable!='MAXW':
                    raise Exception("Max variable computation only available for W right now.")
                if self.variable=='MAXW':
                    data_var=self.open_files(yr, mo)
                    print(f"generating u_func")
                    result_ufunc=data_var.max(dim='bottom_top_stag')
                print(f"starting max for {yr} {mo}")
                r=result_ufunc.compute(retries=10)
                print(f"Saving file")
                r.to_dataset(name='max_in_vert').to_netcdf(f"/{self.destination}/wrf2d_max_{self.variable}_{yr}{mo}.nc")
                r=r.close()
                data_var=data_var.close()
                print(f"woohoo! {yr} {mo} complete")


def wrf_interp(data_AGL, data_var):

    """Function to compute interpolation using wrf-python.
    
    Args:
        data_AGL (Xarray dask array): Data of heights above ground level.
        data_var (Xarray dask array): Data of variable for interpolation.
        
    Returns:
        Data interpolated onto four fixed heights (1, 3, 5, and 7 km).
    
    """
    import os
    os.environ["PROJ_LIB"]="/glade/work/molina/miniconda3/envs/python-tutorial/share/proj/"
    import wrf
    return (wrf.interplevel(data_var.squeeze(),
                            data_AGL.squeeze(),
                            [1000,3000,5000,7000]).expand_dims("Time"))


def apply_wrf_interp(data_AGL, data_var):

    """Generate Xarray ufunc to parallelize the wrf-python interpolation computation.
    
    Args:
        data_AGL (Xarray dask array): Data of heights above ground level.
        data_var (Xarray dask array): Data of variable for interpolation.
        
    Returns:
        Function to parallelize data interpolation onto four fixed heights (1, 3, 5, and 7 km) for all but ``W_vert`` variable.
        
    """
    return xr.apply_ufunc(wrf_interp, data_AGL, data_var,
                          dask='parallelized',
                          output_dtypes=[float],
                          input_core_dims=[['bottom_top_stag','south_north','west_east'],
                                           ['bottom_top','south_north','west_east']],
                          output_sizes=dict(level=4, south_north=1015, west_east=1359),
                          output_core_dims=[['level','south_north','west_east']])


def apply_wrf_interp_W(data_AGL, data_var):

    """Generate Xarray ufunc to parallelize the wrf-python interpolation computation.
    
    Args:
        data_AGL (Xarray dask array): Data of heights above ground level.
        data_var (Xarray dask array): Data of variable for interpolation.
        
    Returns:
        Function to parallelize data interpolation onto four fixed heights (1, 3, 5, and 7 km) for ``W_vert`` variable.
        
    """
    return xr.apply_ufunc(wrf_interp, data_AGL, data_var,
                          dask='parallelized',
                          output_dtypes=[float],
                          input_core_dims=[['bottom_top_stag','south_north','west_east'],
                                           ['bottom_top_stag','south_north','west_east']],
                          output_sizes=dict(level=4, south_north=1015, west_east=1359),
                          output_core_dims=[['level','south_north','west_east']])

