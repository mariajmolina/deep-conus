#Author: Maria J. Molina
#Using Hagelslag code from David John Gagne

#CODE TO EXTRACT STORM PATCHES FROM CONUS1 CURRENT CLIMATE SIMULATIONS.


from scipy.ndimage import find_objects, center_of_mass, gaussian_filter
import numpy as np
import xarray as xr
from skimage.morphology import watershed
from scipy.ndimage import label, find_objects
import pandas as pd
from datetime import timedelta
import multiprocessing as mp


#FOR CURRENT CLIMATE
#dates to loop script through
times_thisfile = pd.date_range('2012-10-01','2013-09-30 23:00:00',freq='D')


##########DONT TOUCH ANYTHING BELOW THIS LINE IN THIS BLOCK#####################

total_times = pd.date_range('2000-10-01','2013-09-30 23:00:00',freq='H')
total_times_indexes = np.arange(0,total_times.shape[0],1)
the1=135
the2=650
the3=500
the4=1200

#################################################################################



def main():

    #start 36 processes in one core.
    pool = mp.Pool(36)

    results = []

    def collect_result(result):
        global results
        results.append(result)

    for num in range(times_thisfile.shape[0]):

        if times_thisfile[num].month == 1 or times_thisfile[num].month == 2 or times_thisfile[num].month == 3:
            mon_1 = '01'
            mon_2 = '03'
        if times_thisfile[num].month == 4 or times_thisfile[num].month == 5 or times_thisfile[num].month == 6:
            mon_1 = '04'
            mon_2 = '06'
        if times_thisfile[num].month == 7 or times_thisfile[num].month == 8 or times_thisfile[num].month == 9:
            mon_1 = '07'
            mon_2 = '09'
        if times_thisfile[num].month == 10 or times_thisfile[num].month == 11 or times_thisfile[num].month == 12:
            mon_1 = '10'
            mon_2 = '12'

        data_path = r'/gpfs/fs1/collections/rda/data/ds612.0/CTRLradrefl/REFLC/wrf2d_d01_CTRL_REFLC_10CM_'+str(times_thisfile[num].strftime('%Y'))+mon_1+'-'+str(times_thisfile[num].strftime('%Y'))+mon_2+'.nc'

        data = xr.open_dataset(data_path)

        print(num, f"start {times_thisfile[num].strftime('%Y%m%d')}")

        data_refl = data.REFLC_10CM.sel(Time=slice(times_thisfile[num],times_thisfile[num]+timedelta(hours=23)))

        data_reflec = data_refl.values[:,the1:the2,the3:the4]
        data_latitu = data.XLAT.values[the1:the2,the3:the4]
        data_longit = data.XLONG.values[the1:the2,the3:the4]

        thetimes = total_times_indexes[np.where(total_times==pd.to_datetime(data_refl.Time.values[0]))[0][0]:
                                       np.where(total_times==pd.to_datetime(data_refl.Time.values[-1]))[0][0]]

        pool.apply_async(parallelizing_the_func, args=(num, data_reflec, data_latitu, data_longit, thetimes), callback=collect_result)




def parallelizing_the_func(num, data_reflec, data_latitu, data_longit, thetimes):

    """
    Function to run script that finds storm patches in WRF CONUS1 dataset.
    Saves output to Xarray netCDF files with metadata.
    
    """

    thelabels = label_storm_objects(data_reflec, 
                                    min_intensity=20, 
                                    max_intensity=40, 
                                    min_area=1, 
                                    max_area=100, 
                                    max_range=1, 
                                    increment=1, 
                                    gaussian_sd=0)
    
    print(num, "postlabel")
    
    
    storm_objs = extract_storm_patches(label_grid = thelabels, 
                                       data = data_reflec, 
                                       x_grid = data_longit, 
                                       y_grid = data_latitu,
                                       times = thetimes,
                                       dx=1, dt=1,
                                       patch_radius=16)
    
    print(num, "done")
    
    data_assemble = xr.Dataset({
        'grid':(['starttime','y','x'],
         np.array([other.timesteps[0] for obj in storm_objs for other in obj if other.timesteps[0].shape[0]*other.timesteps[0].shape[1] == 1024])),
         'mask':(['starttime','y','x'],
         np.array([other.masks[0] for obj in storm_objs for other in obj if other.timesteps[0].shape[0]*other.timesteps[0].shape[1] == 1024])),
         'row_indices':(['starttime','y','x'],
         np.array([other.i[0] for obj in storm_objs for other in obj if other.timesteps[0].shape[0]*other.timesteps[0].shape[1] == 1024])),
         'col_indices':(['starttime','y','x'],
         np.array([other.j[0] for obj in storm_objs for other in obj if other.timesteps[0].shape[0]*other.timesteps[0].shape[1] == 1024])),
         'lats':(['starttime','y','x'],
         np.array([other.y[0] for obj in storm_objs for other in obj if other.timesteps[0].shape[0]*other.timesteps[0].shape[1] == 1024])),
         'lons':(['starttime','y','x'],
         np.array([other.x[0] for obj in storm_objs for other in obj if other.timesteps[0].shape[0]*other.timesteps[0].shape[1] == 1024])),
        },
         coords=
        {'starttime':(['starttime'],
         np.array([other.start_time for obj in storm_objs for other in obj if other.timesteps[0].shape[0]*other.timesteps[0].shape[1] == 1024])),
         'endtime':np.array([other.end_time for obj in storm_objs for other in obj if other.timesteps[0].shape[0]*other.timesteps[0].shape[1] == 1024]),
         'x_speed':(['starttime'],np.array([other.u[0] for obj in storm_objs for other in obj if other.timesteps[0].shape[0]*other.timesteps[0].shape[1] == 1024])),
         'y_speed':(['starttime'],np.array([other.v[0] for obj in storm_objs for other in obj if other.timesteps[0].shape[0]*other.timesteps[0].shape[1] == 1024]))
        })
    
    data_assemble.to_netcdf(f"/glade/scratch/molina/WRF_CONUS1_derived/storm_tracks/current_conus1_{times_thisfile[num].strftime('%Y%m%d')}.nc")
    
    return(num)
    
    
#################################################################################
###################################HAGELSLAG CODE START##########################
#################################################################################


def label_storm_objects(data, min_intensity, max_intensity, min_area=1, max_area=100, max_range=1,
                        increment=1, gaussian_sd=0):
    """
    From a 2D grid or time series of 2D grids, this method labels storm objects with either the Enhanced Watershed,
    Watershed, or Hysteresis methods.
    Args:
        data: the gridded data to be labeled. Should be a 2D numpy array in (y, x) coordinate order or a 3D numpy array
            in (time, y, x) coordinate order
        method: "ew" for Enhanced Watershed, "ws" for regular watershed, and "hyst" for hysteresis
        min_intensity: Minimum intensity threshold for gridpoints contained within any objects
        max_intensity: For watershed, any points above max_intensity are considered as the same value as max intensity.
            For hysteresis, all objects have to contain at least 1 pixel that equals or exceeds this value
        min_area: (default 1) The minimum area of any object in pixels.
        max_area: (default 100) The area threshold in pixels at which the enhanced watershed ends growth. Object area
            may exceed this threshold if the pixels at the last watershed level exceed the object area.
        max_range: Maximum difference in bins for search before growth is stopped.
        increment: Discretization increment for the enhanced watershed
        gaussian_sd: Standard deviation of Gaussian filter applied to data
    Returns:
        label_grid: an ndarray with the same shape as data in which each pixel is labeled with a positive integer value.
    """
    
    labeler = Watershed(min_intensity, max_intensity)
    
    if len(data.shape) == 2:
        if gaussian_sd > 0:
            label_grid = labeler.label(gaussian_filter(data, gaussian_sd))
        else:
            label_grid = labeler.label(data)
        label_grid[data < min_intensity] = 0
        if min_area > 1:
            label_grid = labeler.size_filter(label_grid, min_area)
    
    else:
        label_grid = np.zeros(data.shape, dtype=np.int32)
        
        for t in range(data.shape[0]):
            if gaussian_sd > 0:
                label_grid[t] = labeler.label(gaussian_filter(data[t], gaussian_sd))
            else:
                label_grid[t] = labeler.label(data[t])
                
            label_grid[t][data[t] < min_intensity] = 0
            if min_area > 1:
                label_grid[t] = labeler.size_filter(label_grid[t], min_area)
                
    return label_grid



def extract_storm_patches(label_grid, data, x_grid, y_grid, times, dx=1, dt=1, patch_radius=16):
    """
    After storms are labeled, this method extracts boxes of equal size centered on each storm from the grid and places
    them into STObjects. The STObjects contain intensity, location, and shape information about each storm
    at each timestep.
    Args:
        label_grid: 2D or 3D array output by label_storm_objects.
        data: 2D or 3D array used as input to label_storm_objects.
        x_grid: 2D array of x-coordinate data, preferably on a uniform spatial grid with units of length.
        y_grid: 2D array of y-coordinate data.
        times: List or array of time values, preferably as integers
        dx: grid spacing in same units as x_grid and y_grid.
        dt: period elapsed between times
        patch_radius: Number of grid points from center of mass to extract
    Returns:
        storm_objects: list of lists containing STObjects identified at each time.
    """
    storm_objects = []
    if len(label_grid.shape) == 3:
        ij_grid = np.indices(label_grid.shape[1:])
        for t, time in enumerate(times):
            storm_objects.append([])
            centers = list(center_of_mass(data[t], labels=label_grid[t], index=np.arange(1, label_grid[t].max() + 1)))
            if len(centers) > 0:
                for o, center in enumerate(centers):
                    int_center = np.round(center).astype(int)
                    obj_slice_buff = (slice(int_center[0] - patch_radius, int_center[0] + patch_radius),
                                      slice(int_center[1] - patch_radius, int_center[1] + patch_radius))

                    storm_objects[-1].append(STObject(data[t][obj_slice_buff],
                                                      np.where(label_grid[t][obj_slice_buff] == o + 1, 1, 0),
                                                      x_grid[obj_slice_buff],
                                                      y_grid[obj_slice_buff],
                                                      ij_grid[0][obj_slice_buff],
                                                      ij_grid[1][obj_slice_buff],
                                                      time,
                                                      time,
                                                      dx=dx,
                                                      step=dt))
                    if t > 0:
                        dims = storm_objects[-1][-1].timesteps[0].shape
                        storm_objects[-1][-1].estimate_motion(time, data[t - 1], dims[1], dims[0])

    else:
        ij_grid = np.indices(label_grid.shape)
        storm_objects.append([])
        centers = list(center_of_mass(data, labels=label_grid, index=np.arange(1, label_grid.max() + 1)))

        if len(centers) > 0:
            for o, center in enumerate(centers):
                int_center = np.round(center).astype(int)
                obj_slice_buff = (slice(int_center[0] - patch_radius, int_center[0] + patch_radius),
                                  slice(int_center[1] - patch_radius, int_center[1] + patch_radius))
                
                storm_objects[-1].append(STObject(data[obj_slice_buff],
                                                  np.where(label_grid[obj_slice_buff] == o + 1, 1, 0),
                                                  x_grid[obj_slice_buff],
                                                  y_grid[obj_slice_buff],
                                                  ij_grid[0][obj_slice_buff],
                                                  ij_grid[1][obj_slice_buff],
                                                  times[0],
                                                  times[0],
                                                  dx=dx,
                                                  step=dt))
    return storm_objects


def extract_storm_objects(label_grid, data, x_grid, y_grid, times, dx=1, dt=1, obj_buffer=0):
    """
    After storms are labeled, this method extracts the storm objects from the grid and places them into STObjects.
    The STObjects contain intensity, location, and shape information about each storm at each timestep.
    Args:
        label_grid: 2D or 3D array output by label_storm_objects.
        data: 2D or 3D array used as input to label_storm_objects.
        x_grid: 2D array of x-coordinate data, preferably on a uniform spatial grid with units of length.
        y_grid: 2D array of y-coordinate data.
        times: List or array of time values, preferably as integers
        dx: grid spacing in same units as x_grid and y_grid.
        dt: period elapsed between times
        obj_buffer: number of extra pixels beyond bounding box of object to store in each STObject
    Returns:
        storm_objects: list of lists containing STObjects identified at each time.
    """
    storm_objects = []
    if len(label_grid.shape) == 3:
        ij_grid = np.indices(label_grid.shape[1:])
        for t, time in enumerate(times):
            storm_objects.append([])
            object_slices = list(find_objects(label_grid[t], label_grid[t].max()))
            if len(object_slices) > 0:
                for o, obj_slice in enumerate(object_slices):
                    if obj_buffer > 0:
                        obj_slice_buff = [slice(np.maximum(0, osl.start - obj_buffer),
                                                np.minimum(osl.stop + obj_buffer, label_grid.shape[l + 1]))
                                          for l, osl in enumerate(obj_slice)]
                    else:
                        obj_slice_buff = obj_slice
                    storm_objects[-1].append(STObject(data[t][obj_slice_buff],
                                                      np.where(label_grid[t][obj_slice_buff] == o + 1, 1, 0),
                                                      x_grid[obj_slice_buff],
                                                      y_grid[obj_slice_buff],
                                                      ij_grid[0][obj_slice_buff],
                                                      ij_grid[1][obj_slice_buff],
                                                      time,
                                                      time,
                                                      dx=dx,
                                                      step=dt))
                    if t > 0:
                        dims = storm_objects[-1][-1].timesteps[0].shape
                        storm_objects[-1][-1].estimate_motion(time, data[t - 1], dims[1], dims[0])
    else:
        ij_grid = np.indices(label_grid.shape)
        storm_objects.append([])
        object_slices = list(find_objects(label_grid, label_grid.max()))
        if len(object_slices) > 0:
            for o, obj_slice in enumerate(object_slices):
                if obj_buffer > 0:
                    obj_slice_buff = [slice(np.maximum(0, osl.start - obj_buffer),
                                            np.minimum(osl.stop + obj_buffer, label_grid.shape[l + 1]))
                                      for l, osl in enumerate(obj_slice)]
                else:
                    obj_slice_buff = obj_slice
                storm_objects[-1].append(STObject(data[obj_slice_buff],
                                                  np.where(label_grid[obj_slice_buff] == o + 1, 1, 0),
                                                  x_grid[obj_slice_buff],
                                                  y_grid[obj_slice_buff],
                                                  ij_grid[0][obj_slice_buff],
                                                  ij_grid[1][obj_slice_buff],
                                                  times[0],
                                                  times[0],
                                                  dx=dx,
                                                  step=dt))
    return storm_objects


class Watershed(object):
    """
    This watershed approach performs a standard labeling of intense objects then grows the intense
    objects out to the minimum intensity. It will create separate objects for the area around each
    core in a line of storms, for example.
    Args:
        min_intensity: minimum intensity for the storm field
        core_intensity: the intensity used to determine the initial objects.
    """
    def __init__(self, min_intensity, core_intensity):
        self.min_intensity = min_intensity
        self.core_intensity = core_intensity

    def label(self, data):
        core_labels, n_labels = label(data >= self.core_intensity)
        ws_labels = watershed(data.max() - data, markers=core_labels, mask=data >= self.min_intensity)
        return ws_labels

    @staticmethod
    def size_filter(labeled_grid, min_size):
        """
        Removes labeled objects that are smaller than min_size, and relabels the remaining objects.
        Args:
            labeled_grid: Grid that has been labeled
            min_size: Minimium object size.
        Returns:
            Labeled array with re-numbered objects to account for those that have been removed
        """
        out_grid = np.zeros(labeled_grid.shape, dtype=int)
        slices = find_objects(labeled_grid)
        j = 1
        for i, s in enumerate(slices):
            box = labeled_grid[s]
            size = np.count_nonzero(box == i + 1)
            if size >= min_size and box.shape[0] > 1 and box.shape[1] > 1:
                out_grid[np.where(labeled_grid == i + 1)] = j
                j += 1
        return out_grid
    
    
import numpy as np
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
from skimage.morphology import convex_hull_image
import json
import os


class STObject(object):
    """
    The STObject stores data and location information for objects extracted from the ensemble grids.
    Attributes:
        grid (ndarray): All of the data values. Supports a 2D array of values, a list of 2D arrays, or a 3D array.
        mask (ndarray): Grid of 1's and 0's in which 1's indicate the location of the object.
        x (ndarray): Array of x-coordinate values in meters. Longitudes can also be placed here.
        y (ndarray): Array of y-coordinate values in meters. Latitudes can also be placed here.
        i (ndarray): Array of row indices from the full model domain.
        j (ndarray): Array of column indices from the full model domain.
        start_time: The first time of the object existence.
        end_time: The last time of the object existence.
        step: number of hours between timesteps
        dx: grid spacing
        u: storm motion in x-direction
        v: storm motion in y-direction
    """

    def __init__(self, grid, mask, x, y, i, j, start_time, end_time, step=1, dx=4000, u=None, v=None):
        if hasattr(grid, "shape") and len(grid.shape) == 2:
            self.timesteps = [grid]
            self.masks = [np.array(mask, dtype=int)]
            self.x = [x]
            self.y = [y]
            self.i = [i]
            self.j = [j]
        elif hasattr(grid, "shape") and len(grid.shape) > 2:
            self.timesteps = []
            self.masks = []
            self.x = []
            self.y = []
            self.i = []
            self.j = []
            self.stormid = []
            for l in range(grid.shape[0]):
                self.timesteps.append(grid[l])
                self.masks.append(np.array(mask[l], dtype=int))
                self.x.append(x[l])
                self.y.append(y[l])
                self.i.append(i[l])
                self.j.append(j[l])
        else:
            self.timesteps = grid
            self.masks = mask
            self.x = x
            self.y = y
            self.i = i
            self.j = j
        if u is not None and v is not None:
            self.u = u
            self.v = v
        else:
            self.u = np.zeros(len(self.timesteps))
            self.v = np.zeros(len(self.timesteps))
        self.dx = dx
        self.start_time = start_time
        self.end_time = end_time
        self.step = step
        self.times = np.arange(start_time, end_time + step, step)
        self.attributes = {}
        self.observations = None

    @property
    def __str__(self):
        com_x, com_y = self.center_of_mass(self.start_time)
        data = dict(maxsize=self.max_size(), comx=com_x, comy=com_y, start=self.start_time, end=self.end_time)
        return "ST Object [maxSize=%(maxsize)d,initialCenter=%(comx)0.2f,%(comy)0.2f,duration=%(start)02d-%(end)02d]" %\
               data

    def center_of_mass(self, time):
        """
        Calculate the center of mass at a given timestep.
        Args:
            time: Time at which the center of mass calculation is performed
        Returns:
            The x- and y-coordinates of the center of mass.
        """
        if self.start_time <= time <= self.end_time:
            diff = time - self.start_time
            valid = np.flatnonzero(self.masks[diff] != 0)
            if valid.size > 0:
                com_x = 1.0 / self.timesteps[diff].ravel()[valid].sum() * np.sum(self.timesteps[diff].ravel()[valid] *
                                                                                 self.x[diff].ravel()[valid])
                com_y = 1.0 / self.timesteps[diff].ravel()[valid].sum() * np.sum(self.timesteps[diff].ravel()[valid] *
                                                                                 self.y[diff].ravel()[valid])
            else:
                com_x = np.mean(self.x[diff])
                com_y = np.mean(self.y[diff])
        else:
            com_x = None
            com_y = None
        return com_x, com_y

    def closest_distance(self, time, other_object, other_time):
        """
        The shortest distance between two objects at specified times.
        Args:
            time (int or datetime): Valid time for this STObject
            other_object: Another STObject being compared
            other_time: The time within the other STObject being evaluated.
        Returns:
            Distance in units of the x-y coordinates
        """
        ti = np.where(self.times == time)[0][0]
        oti = np.where(other_object.times == other_time)[0][0]
        xs = self.x[ti].ravel()[self.masks[ti].ravel() == 1]
        xs = xs.reshape(xs.size, 1)
        ys = self.y[ti].ravel()[self.masks[ti].ravel() == 1]
        ys = ys.reshape(ys.size, 1)
        o_xs = other_object.x[oti].ravel()[other_object.masks[oti].ravel() == 1]
        o_xs = o_xs.reshape(1, o_xs.size)
        o_ys = other_object.y[oti].ravel()[other_object.masks[oti].ravel() == 1]
        o_ys = o_ys.reshape(1, o_ys.size)
        distances = (xs - o_xs) ** 2 + (ys - o_ys) ** 2
        return np.sqrt(distances.min())

    def percentile_distance(self, time, other_object, other_time, percentile):
        ti = np.where(self.times == time)[0][0]
        oti = np.where(other_object.times == other_time)[0][0]
        xs = self.x[ti][self.masks[ti] == 1]
        xs = xs.reshape(xs.size, 1)
        ys = self.y[ti][self.masks[ti] == 1]
        ys = ys.reshape(ys.size, 1)
        o_xs = other_object.x[oti][other_object.masks[oti] == 1]
        o_xs = o_xs.reshape(1, o_xs.size)
        o_ys = other_object.y[oti][other_object.masks[oti] == 1]
        o_ys = o_ys.reshape(1, o_ys.size)
        distances = (xs - o_xs) ** 2 + (ys - o_ys) ** 2
        return np.sqrt(np.percentile(distances, percentile))

    def trajectory(self):
        """
        Calculates the center of mass for each time step and outputs an array
        Returns:
        """
        traj = np.zeros((2, self.times.size))
        for t, time in enumerate(self.times):
            traj[:, t] = self.center_of_mass(time)
        return traj

    def get_corner(self, time):
        """
        Gets the corner array indices of the STObject at a given time that corresponds 
        to the upper left corner of the bounding box for the STObject.
        Args:
            time: time at which the corner is being extracted.
        Returns:
              corner index.
        """
        if self.start_time <= time <= self.end_time:
            diff = time - self.start_time
            return self.i[diff][0, 0], self.j[diff][0, 0]
        else:
            return -1, -1

    def size(self, time):
        """
        Gets the size of the object at a given time.
        Args:
            time: Time value being queried.
        Returns:
            size of the object in pixels
        """
        if self.start_time <= time <= self.end_time:
            return self.masks[time - self.start_time].sum()
        else:
            return 0

    def max_size(self):
        """
        Gets the largest size of the object over all timesteps.
        
        Returns:
            Maximum size of the object in pixels
        """
        sizes = np.array([m.sum() for m in self.masks])
        return sizes.max()

    def max_intensity(self, time):
        """
        Calculate the maximum intensity found at a timestep.
        """
        ti = np.where(time == self.times)[0][0]
        return self.timesteps[ti].max()

    def extend(self, step):
        """
        Adds the data from another STObject to this object.
        
        Args:
            step: another STObject being added after the current one in time.
        """
        self.timesteps.extend(step.timesteps)
        self.masks.extend(step.masks)
        self.x.extend(step.x)
        self.y.extend(step.y)
        self.i.extend(step.i)
        self.j.extend(step.j)
        self.end_time = step.end_time
        self.times = np.arange(self.start_time, self.end_time + self.step, self.step)
        self.u = np.concatenate((self.u, step.u))
        self.v = np.concatenate((self.v, step.v))
        for attr in self.attributes.keys():
            if attr in step.attributes.keys():
                self.attributes[attr].extend(step.attributes[attr])

    def boundary_polygon(self, time):
        """
        Get coordinates of object boundary in counter-clockwise order
        """
        ti = np.where(time == self.times)[0][0]
        com_x, com_y = self.center_of_mass(time)
        # If at least one point along perimeter of the mask rectangle is unmasked, find_boundaries() works.
        # But if all perimeter points are masked, find_boundaries() does not find the object.
        # Therefore, pad the mask with zeroes first and run find_boundaries on the padded array.
        padded_mask = np.pad(self.masks[ti], 1, 'constant', constant_values=0)
        chull = convex_hull_image(padded_mask)
        boundary_image = find_boundaries(chull, mode='inner', background=0)
        # Now remove the padding.
        boundary_image = boundary_image[1:-1,1:-1]
        boundary_x = self.x[ti].ravel()[boundary_image.ravel()]
        boundary_y = self.y[ti].ravel()[boundary_image.ravel()]
        r = np.sqrt((boundary_x - com_x) ** 2 + (boundary_y - com_y) ** 2)
        theta = np.arctan2((boundary_y - com_y), (boundary_x - com_x)) * 180.0 / np.pi + 360
        polar_coords = np.array([(r[x], theta[x]) for x in range(r.size)], dtype=[('r', 'f4'), ('theta', 'f4')])
        coord_order = np.argsort(polar_coords, order=['theta', 'r'])
        ordered_coords = np.vstack([boundary_x[coord_order], boundary_y[coord_order]])
        return ordered_coords

    def estimate_motion(self, time, intensity_grid, max_u, max_v):
        """
        Estimate the motion of the object with cross-correlation on the intensity values from the previous time step.
        Args:
            time: time being evaluated.
            intensity_grid: 2D array of intensities used in cross correlation.
            max_u: Maximum x-component of motion. Used to limit search area.
            max_v: Maximum y-component of motion. Used to limit search area
        Returns:
            u, v, and the minimum error.
        """
        ti = np.where(time == self.times)[0][0]
        mask_vals = np.where(self.masks[ti].ravel() == 1)
        i_vals = self.i[ti].ravel()[mask_vals]
        j_vals = self.j[ti].ravel()[mask_vals]
        obj_vals = self.timesteps[ti].ravel()[mask_vals]
        #obj_vals = self.timesteps[ti].stack(z=('south_north','west_east'))[mask_vals]
        u_shifts = np.arange(-max_u, max_u + 1)
        v_shifts = np.arange(-max_v, max_v + 1)
        min_error = 99999999999.0
        best_u = 0
        best_v = 0
        for u in u_shifts:
            j_shift = j_vals - u
            for v in v_shifts:
                i_shift = i_vals - v
                if np.all((0 <= i_shift) & (i_shift < intensity_grid.shape[0]) &
                                  (0 <= j_shift) & (j_shift < intensity_grid.shape[1])):
                    shift_vals = intensity_grid[i_shift, j_shift]
                else:
                    shift_vals = np.zeros(i_shift.shape)
                # This isn't correlation; it is mean absolute error.
                error = np.abs(shift_vals - obj_vals).mean()
                if error < min_error:
                    min_error = error
                    best_u = u * self.dx
                    best_v = v * self.dx
        # 60 seems arbitrarily high
        #if min_error > 60:
        #    best_u = 0
        #    best_v = 0
        self.u[ti] = best_u
        self.v[ti] = best_v
        return best_u, best_v, min_error

    def count_overlap(self, time, other_object, other_time):
        """
        Counts the number of points that overlap between this STObject and another STObject. Used for tracking.
        """
        ti = np.where(time == self.times)[0][0]
        ma = np.where(self.masks[ti].ravel() == 1)
        oti = np.where(other_time == other_object.times)[0]
        obj_coords = np.zeros(self.masks[ti].sum(), dtype=[('x', int), ('y', int)])
        other_obj_coords = np.zeros(other_object.masks[oti].sum(), dtype=[('x', int), ('y', int)])
        obj_coords['x'] = self.i[ti].ravel()[ma]
        obj_coords['y'] = self.j[ti].ravel()[ma]
        other_obj_coords['x'] = other_object.i[oti][other_object.masks[oti] == 1]
        other_obj_coords['y'] = other_object.j[oti][other_object.masks[oti] == 1]
        return float(np.intersect1d(obj_coords,
                                    other_obj_coords).size) / np.maximum(self.masks[ti].sum(),
                                                                         other_object.masks[oti].sum())

    def extract_attribute_grid(self, model_grid, potential=False, future=False):
        """
        Extracts the data from a ModelOutput or ModelGrid object within the bounding box region of the STObject.
        
        Args:
            model_grid: A ModelGrid or ModelOutput Object
            potential: Extracts from the time before instead of the same time as the object
        """

        if potential:
            var_name = model_grid.variable + "-potential"
            timesteps = np.arange(self.start_time - 1, self.end_time)
        elif future:
            var_name = model_grid.variable + "-future"
            timesteps = np.arange(self.start_time + 1, self.end_time + 2)
        else:
            var_name = model_grid.variable
            timesteps = np.arange(self.start_time, self.end_time + 1)
        self.attributes[var_name] = []
        for ti, t in enumerate(timesteps):
            self.attributes[var_name].append(
                model_grid.data[t - model_grid.start_hour, self.i[ti], self.j[ti]])

    def extract_attribute_array(self, data_array, var_name):
        """
        Extracts data from a 2D array that has the same dimensions as the grid used to identify the object.
        Args:
            data_array: 2D numpy array
        """
        if var_name not in self.attributes.keys():
            self.attributes[var_name] = []
        for t in range(self.times.size):
            self.attributes[var_name].append(data_array[self.i[t], self.j[t]])


    def extract_tendency_grid(self, model_grid):
        """
        Extracts the difference in model outputs
        Args:
            model_grid: ModelOutput or ModelGrid object.
        """
        var_name = model_grid.variable + "-tendency"
        self.attributes[var_name] = []
        timesteps = np.arange(self.start_time, self.end_time + 1)
        for ti, t in enumerate(timesteps):
            t_index = t - model_grid.start_hour
            self.attributes[var_name].append(
                model_grid.data[t_index, self.i[ti], self.j[ti]] - model_grid.data[t_index - 1, self.i[ti], self.j[ti]]
                )

    def calc_attribute_statistics(self, statistic_name):
        """
        Calculates summary statistics over the domains of each attribute.
        
        Args:
            statistic_name (string): numpy statistic, such as mean, std, max, min
        Returns:
            dict of statistics from each attribute grid.
        """
        stats = {}
        for var, grids in self.attributes.items():
            if len(grids) > 1:
                stats[var] = getattr(np.array([getattr(np.ma.array(x, mask=self.masks[t] == 0), statistic_name)()
                                               for t, x in enumerate(grids)]), statistic_name)()
            else:
                stats[var] = getattr(np.ma.array(grids[0], mask=self.masks[0] == 0), statistic_name)()
        return stats

    def calc_attribute_statistic(self, attribute, statistic, time):
        """
        Calculate statistics based on the values of an attribute. The following statistics are supported:
        mean, max, min, std, ptp (range), median, skew (mean - median), and percentile_(percentile value).
        Args:
            attribute: Attribute extracted from model grid
            statistic: Name of statistic being used.
            time: timestep of the object being investigated
        Returns:
            The value of the statistic
        """
        ti = np.where(self.times == time)[0][0]
        ma = np.where(self.masks[ti].ravel() == 1)
        if statistic in ['mean', 'max', 'min', 'std', 'ptp']:
            stat_val = getattr(self.attributes[attribute][ti].ravel()[ma], statistic)()
        elif statistic == 'median':
            stat_val = np.median(self.attributes[attribute][ti].ravel()[ma])
        elif statistic == "skew":
            stat_val = np.mean(self.attributes[attribute][ti].ravel()[ma]) - \
                       np.median(self.attributes[attribute][ti].ravel()[ma])
        elif 'percentile' in statistic:
            per = int(statistic.split("_")[1])
            stat_val = np.percentile(self.attributes[attribute][ti].ravel()[ma], per)
        elif 'dt' in statistic:
            stat_name = statistic[:-3]
            if ti == 0:
                stat_val = 0
            else:
                stat_val = self.calc_attribute_statistic(attribute, stat_name, time) \
                    - self.calc_attribute_statistic(attribute, stat_name, time - 1)
        else:
            stat_val = np.nan
        return stat_val

    def calc_timestep_statistic(self, statistic, time):
        """
        Calculate statistics from the primary attribute of the StObject.
        Args:
            statistic: statistic being calculated
            time: Timestep being investigated
        Returns:
            Value of the statistic
        """
        ti = np.where(self.times == time)[0][0]
        ma = np.where(self.masks[ti].ravel() == 1)
        if statistic in ['mean', 'max', 'min', 'std', 'ptp']:
            stat_val = getattr(self.timesteps[ti].ravel()[ma], statistic)()
        elif statistic == 'median':
            stat_val = np.median(self.timesteps[ti].ravel()[ma])
        elif 'percentile' in statistic:
            per = int(statistic.split("_")[1])
            stat_val = np.percentile(self.timesteps[ti].ravel()[ma], per)
        elif 'dt' in statistic:
            stat_name = statistic[:-3]
            if ti == 0:
                stat_val = 0
            else:
                stat_val = self.calc_timestep_statistic(stat_name, time) -\
                    self.calc_timestep_statistic(stat_name, time - 1)
        else:
            stat_val = np.nan
        return stat_val

    def calc_shape_statistics(self, stat_names):
        """
        Calculate shape statistics using regionprops applied to the object mask.
        
        Args:
            stat_names: List of statistics to be extracted from those calculated by regionprops.
        Returns:
            Dictionary of shape statistics
        """
        stats = {}
        try:
            all_props = [regionprops(m) for m in self.masks]
        except TypeError:
            print(self.masks)
            exit()
        for stat in stat_names:
            stats[stat] = np.mean([p[0][stat] for p in all_props])
        return stats

    def calc_shape_step(self, stat_names, time):
        """
        Calculate shape statistics for a single time step
        Args:
            stat_names: List of shape statistics calculated from region props
            time: Time being investigated
        Returns:
            List of shape statistics
        """
        ti = np.where(self.times == time)[0][0]
        props = regionprops(self.masks[ti], self.timesteps[ti])[0]
        shape_stats = []
        for stat_name in stat_names:
            if "moments_hu" in stat_name:
                hu_index = int(stat_name.split("_")[-1])
                hu_name = "_".join(stat_name.split("_")[:-1])
                hu_val = np.log(props[hu_name][hu_index])
                if np.isnan(hu_val):
                    shape_stats.append(0)
                else:
                    shape_stats.append(hu_val)
            else:
                shape_stats.append(props[stat_name])
        return shape_stats

    def to_geojson(self, filename, proj, metadata=None):
        """
        Output the data in the STObject to a geoJSON file.
        Args:
            filename: Name of the file
            proj: PyProj object for converting the x and y coordinates back to latitude and longitue values.
            metadata: Metadata describing the object to be included in the top-level properties.
        """
        if metadata is None:
            metadata = {}
        json_obj = {"type": "FeatureCollection", "features": [], "properties": {}}
        json_obj['properties']['times'] = self.times.tolist()
        json_obj['properties']['dx'] = self.dx
        json_obj['properties']['step'] = self.step
        json_obj['properties']['u'] = self.u.tolist()
        json_obj['properties']['v'] = self.v.tolist()
        for k, v in metadata.items():
            json_obj['properties'][k] = v
        for t, time in enumerate(self.times):
            feature = {"type": "Feature",
                       "geometry": {"type": "Polygon"},
                       "properties": {}}
            boundary_coords = self.boundary_polygon(time)
            lonlat = np.vstack(proj(boundary_coords[0], boundary_coords[1], inverse=True))
            lonlat_list = lonlat.T.tolist()
            if len(lonlat_list) > 0:
                lonlat_list.append(lonlat_list[0])
            feature["geometry"]["coordinates"] = [lonlat_list]
            for attr in ["timesteps", "masks", "x", "y", "i", "j"]:
                feature["properties"][attr] = getattr(self, attr)[t].tolist()
            feature["properties"]["attributes"] = {}
            for attr_name, steps in self.attributes.items():
                feature["properties"]["attributes"][attr_name] = steps[t].tolist()
            json_obj['features'].append(feature)
        file_obj = open(filename, "w")
        json.dump(json_obj, file_obj, indent=1, sort_keys=True)
        file_obj.close()
        return

def read_geojson(filename):
    """
    Reads a geojson file containing an STObject and initializes a new STObject from the information in the file.
    Args:
        filename: Name of the geojson file
    Returns:
        an STObject
    """
    json_file = open(filename)
    data = json.load(json_file)
    json_file.close()
    times = data["properties"]["times"]
    main_data = dict(timesteps=[], masks=[], x=[], y=[], i=[], j=[])
    attribute_data = dict()
    for feature in data["features"]:
        for main_name in main_data.keys():
            main_data[main_name].append(np.array(feature["properties"][main_name]))
        for k, v in feature["properties"]["attributes"].items():
            if k not in attribute_data.keys():
                attribute_data[k] = [np.array(v)]
            else:
                attribute_data[k].append(np.array(v))
    kwargs = {}
    for kw in ["dx", "step", "u", "v"]:
        if kw in data["properties"].keys():
            kwargs[kw] = data["properties"][kw]
    sto = STObject(main_data["timesteps"], main_data["masks"], main_data["x"], main_data["y"],
                   main_data["i"], main_data["j"], times[0], times[-1], **kwargs)
    for k, v in attribute_data.items():
        sto.attributes[k] = v
    return sto
    
    
#################################################################################
################################HAGELSLAG END####################################
#################################################################################



#--------------------------------------------------

if __name__== "__main__":
    main()

#--------------------------------------------------



