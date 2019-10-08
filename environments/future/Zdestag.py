####################################################################################
####################################################################################
##
##Author: Maria J. Molina
##National Center for Atmospheric Research
##
####################################################################################
####################################################################################


#Code for creating Z unstaggered fields from model staggered fields (geopotential height) in serial.


#----------------------------------------------------------------------------------
#imports
import xarray as xr
import wrf
import numpy as np
#----------------------------------------------------------------------------------


#helper function
def num_days_in_mo(year, month):
    """
    Determine the number of days in month.
    
    Inputs: month 2-digit, year 4-digit (str)
    Outputs: 1d array of days in month (str)
    """
    
    from calendar import monthrange
    
    formatter = "{:02d}".format
    
    days = np.array(list(map(formatter, 
                             np.arange(1,monthrange(int(year), 
                                                    int(month))[1]+1,1))))

    return days
    

#main code
def main():

    #temporal arrays
    formatter = "{:02d}".format
    months = np.array(list(map(formatter, np.arange(12,13,1))))
    years = np.array(list(map(formatter, np.arange(2010,2011,1))))
    print(months)
    print(years)

    for year in years:
        for month in months:
            days = num_days_in_mo(year, month)
            for day in days[:]:

                print(f'opening {year}{month}{day}')
                data_zstag = xr.open_mfdataset(f'/gpfs/fs1/collections/rda/data/ds612.0/PGW3D/{year}/wrf3d_d01_PGW_Z_{year}{month}{day}.nc', 
                                                combine='by_coords', parallel=True).Z

                data_zstag2 = wrf.destagger(data_zstag, stagger_dim=1, meta=True)

                data_zstag2.coords['Time'] = data_zstag.coords["Time"]

                data_zstag2.to_dataset(name='Z').to_netcdf(f"/glade/scratch/molina/WRF_CONUS1_derived/future/wrf3d_d01_PGW_Zdestag_{year}{month}{day}.nc")

                print(f'{year}{month}{day} complete!')
                

if __name__== "__main__":
    main()

#----------------------------------------------------------------------------------
