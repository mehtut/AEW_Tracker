from __future__ import division  # makes division not round with integers 
import os
import pygrib
from netCDF4 import Dataset
import numpy as np
import xarray as xr
import wrf as wrf
from datetime import datetime, timedelta
from skimage.feature import peak_local_max
#matplotlib.use('pdf')
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import rcParams
import matplotlib.patches as mpatches
#import matplotlib.animation as animation
#from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
import cartopy.feature as cfeature
#import cartopy.feature as cfeature
#import cartopy.io.shapereader as shpreader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy.ndimage import gaussian_filter
import ctypes
import numpy.ctypeslib as ctl

# Pull_data.py contains functions that are used by AEW_Tracks.py. All of the functions in Pull_Data.py aid in the pulling of variables and formatting
# them to be in a common format so AEW_Tracks.py does not need to know what model the data came from.

# This function takes a common_object and assigns the lat, lon, lat/lon indices over Africa/the Atlantic, and dt value based on the data source
# which is the model attribute of the common_object. 
def get_common_track_data(common_object):
	# box of interest over Africa/the Atlantic (values are from Albany)
	north_lat = 30. #35.
	south_lat = 5.
	west_lon = -45.
	east_lon = 25. 

	# lat/lon values to crop data to speed up vorticity calculations
	north_lat_crop = 50. #35.
	south_lat_crop = -20.
	west_lon_crop = -80.
	east_lon_crop = 40.

	if common_object.model == 'WRF':
		dt = 6 # time between files
		# get the latitude and longitude and the north, south, east, and west indices of a rectangle over Africa and the Atlantic 
		file_location = '/global/cscratch1/sd/ebercosh/WRF_TCM/Historical/wrfout_d01_2008-07-01_00_00_00'
		data = Dataset(file_location)
		# get lat and lon values
		# get the latitude and longitude at a single time (since they don't change with time)
		lat = wrf.getvar(data, "lat", meta=False) # ordered lat, lon
		lon = wrf.getvar(data, "lon", meta=False) # ordered lat, lon
		#print(lat)
		#print(lon)
		#print(lat.shape)
		#print(lon.shape)
		# get north, south, east, west indices
		lon_index_west, lat_index_south = wrf.ll_to_xy(data,south_lat,west_lon, meta=False) 
		lon_index_east, lat_index_north = wrf.ll_to_xy(data,north_lat,east_lon, meta=False) 
		#lat_crop = lat.values[lat_index_south:lat_index_north+1,lon_index_west:lon_index_east+1]
		#lon_crop = lon.values[lat_index_south:lat_index_north+1,lon_index_west:lon_index_east+1]
		# the following two lines are to correct for the weird negative indexing that comes back from the wrf.ll_to_xy function
		lon_index_west = lon.shape[1] + lon_index_west
		lon_index_east = lon.shape[1] + lon_index_east

		# the total number of degrees in the longitude dimension
		lon_degrees = 360.
		
	elif common_object.model == 'MERRA2':
		dt = 3 # time between files
		# get the latitude and longitude and the north, south, east, and west indices of a rectangle over Africa and the Atlantic 
		file_location = '/global/cscratch1/sd/ebercosh/MERRA2/U1000_20170701.nc'
		data = xr.open_dataset(file_location)
		# get lat and lon values
		# get the latitude and longitude at a single time (since they don't change with time)
		lat_1d = data.lat.values # ordered lat
		lon_1d = data.lon.values # ordered lon 
		# make the lat and lon arrays from the GCM 2D (ordered lat, lon)
		lon = np.tile(lon_1d, (lat_1d.shape[0],1))
		lat_2d = np.tile(lat_1d, (len(lon_1d),1))
		lat = np.rot90(lat_2d,3)
		# switch lat and lon arrays to float32 instead of float64
		lat = np.float32(lat)
		lon = np.float32(lon)
		# make lat and lon arrays C continguous 
		lat = np.asarray(lat, order='C')
		lon = np.asarray(lon, order='C')

		# get north, south, east, west indices
		lat_index_north = (np.abs(lat_1d - north_lat)).argmin()
		lat_index_south = (np.abs(lat_1d - south_lat)).argmin()
		lon_index_west = (np.abs(lon_1d - west_lon)).argmin()
		lon_index_east = (np.abs(lon_1d - east_lon)).argmin()

	elif common_object.model == 'CAM5':
#		dt = 3 # time between files
		dt = 6 # time between files to compare with WRF
		# get the latitude and longitude and the north, south, east, and west indices of a rectangle over Africa and the Atlantic 
		file_location = '/global/cscratch1/sd/ebercosh/CAM5/Africa/Historical/run3/2006/U_CAM5-1-0.25degree_All-Hist_est1_v3_run3.cam.h4.2006-07-01-00000_AEW.nc'
		data = xr.open_dataset(file_location, decode_times=False)
		# get lat and lon values
		# get the latitude and longitude at a single time (since they don't change with time)
		lat_1d = data.lat.values # ordered lat
		lon_1d = data.lon.values # ordered lon 
		# make the lat and lon arrays from the GCM 2D (ordered lat, lon)
		lon = np.tile(lon_1d, (lat_1d.shape[0],1))
		lat_2d = np.tile(lat_1d, (len(lon_1d),1))
		lat = np.rot90(lat_2d,3)
		# switch lat and lon arrays to float32 instead of float64
		lat = np.float32(lat)
		lon = np.float32(lon)
		# make lat and lon arrays C continguous 
		lat = np.asarray(lat, order='C')
		lon = np.asarray(lon, order='C')

		# get north, south, east, west indices
		lat_index_north = (np.abs(lat_1d - north_lat)).argmin()
		lat_index_south = (np.abs(lat_1d - south_lat)).argmin()
		lon_index_west = (np.abs(lon_1d - west_lon)).argmin()
		lon_index_east = (np.abs(lon_1d - east_lon)).argmin()

		# the total number of degrees in the longitude dimension
		lon_degrees = np.abs(lon[0,0] - lon[0,-1])

	elif common_object.model == 'ERA5':
		# dt for ERA5 is 1 hour (data is hourly), but set dt to whatever the dt is for the dataset to be compared with
		# Eg dt=3 to compare with CAM5 or dt=6 to compare with WRF
		dt = 6 # time between files
		file_location = '/global/cfs/projectdirs/m3522/cmip6/ERA5/e5.oper.an.pl/200509/e5.oper.an.pl.128_131_u.ll025uv.2005090100_2005090123.nc'
		data = xr.open_dataset(file_location)
		# get lat and lon values
		# get the latitude and longitude at a single time (since they don't change with time)
		lat_1d_n_s = data.latitude.values # ordered lat, and going from north to south (so 90, 89, 88, .....-88, -89, -90)
		lon_1d_360 = data.longitude.values # ordered lon and goes from 0-360 degrees
		# make the lat array go from south to north 
		lat_1d = np.flip(lat_1d_n_s)
		# make the longitude go from -180 to 180 degrees
		lon_1d = np.array([x - 180.0 for x in lon_1d_360])

		# get north, south, east, west indices for cropping 
		lat_index_north_crop = (np.abs(lat_1d - north_lat_crop)).argmin()
		lat_index_south_crop = (np.abs(lat_1d - south_lat_crop)).argmin()
		lon_index_west_crop = (np.abs(lon_1d - west_lon_crop)).argmin()
		lon_index_east_crop = (np.abs(lon_1d - east_lon_crop)).argmin()

		# set the lat and lon cropping indices in common_object
		common_object.lat_index_north_crop = lat_index_north_crop
		common_object.lat_index_south_crop = lat_index_south_crop
		common_object.lon_index_east_crop = lon_index_east_crop
		common_object.lon_index_west_crop = lon_index_west_crop

		# crop the lat and lon arrays. We don't need the entire global dataset
		lat_1d_crop = lat_1d[lat_index_south_crop:lat_index_north_crop+1]
		lon_1d_crop = lon_1d[lon_index_west_crop:lon_index_east_crop+1]

		# make the lat and lon arrays from the GCM 2D (ordered lat, lon)
		lon = np.tile(lon_1d_crop, (lat_1d_crop.shape[0],1))
		lat_2d = np.tile(lat_1d_crop, (len(lon_1d_crop),1))
		lat = np.rot90(lat_2d,3)
		# switch lat and lon arrays to float32 instead of float64
		lat = np.float32(lat)
		lon = np.float32(lon)
		# make lat and lon arrays C continguous 
		lat = np.asarray(lat, order='C')
		lon = np.asarray(lon, order='C')

		# get north, south, east, west indices for tracking
		lat_index_north = (np.abs(lat_1d_crop - north_lat)).argmin()
		lat_index_south = (np.abs(lat_1d_crop - south_lat)).argmin()
		lon_index_west = (np.abs(lon_1d_crop - west_lon)).argmin()
		lon_index_east = (np.abs(lon_1d_crop - east_lon)).argmin()

		# the total number of degrees in the longitude dimension
		lon_degrees = np.abs(lon[0,0] - lon[0,-1])

	elif common_object.model == 'ERAI':
		dt = 6 # time between files
		file_location = '/global/cscratch1/sd/ebercosh/Reanalysis/ERA-I/ei.oper.an.pl.regn128uv.2010113000'
		grbs = pygrib.open(file_location)
		grb = grbs.select(name = 'U component of wind')[23]
		# lat and lon are 2D, ordered lat, lon
		lat_2d_n_s,lon_2d_360 = grb.latlons() # the lat goes from north to south (so 90, 89, 88, .....-88, -89, -90), and lon goes from 0-360 degrees
		# make the lat array go from south to north 
		lat = np.flip(lat_2d_n_s,axis=0)
		# make the longitude go from -180 to 180 degrees
		lon = lon_2d_360 - 180.

		# switch lat and lon arrays to float32 instead of float64
		lat = np.float32(lat)
		lon = np.float32(lon)
		# make lat and lon arrays C continguous 
		lat = np.asarray(lat, order='C')
		lon = np.asarray(lon, order='C')

		# get north, south, east, west indices for tracking
		lat_index_north = (np.abs(lat[:,0] - north_lat)).argmin()
		lat_index_south = (np.abs(lat[:,0] - south_lat)).argmin()
		lon_index_west = (np.abs(lon[0,:] - west_lon)).argmin()
		lon_index_east = (np.abs(lon[0,:] - east_lon)).argmin()

	# set dt in the common_object
	common_object.dt = dt
	# set lat and lon in common_object
	common_object.lat = lat # switch from float64 to float32
	common_object.lon = lon # switch from float64 to float32
	# set the lat and lon indices in common_object
	common_object.lat_index_north = lat_index_north
	common_object.lat_index_south = lat_index_south
	common_object.lon_index_east = lon_index_east
	common_object.lon_index_west = lon_index_west
	# set the total number of degrees longitude in common_object
	common_object.total_lon_degrees = lon_degrees
	print(common_object.total_lon_degrees)
	return

# This is a function to get the WRF variables required for tracking
# The function takes the common_object that holds lat/lon information, the scenario type, and the date and time for the desired file
# The function returns u, v, relative vorticity, and curvature vorticity on specific pressure levels
def get_WRF_variables(common_object, scenario_type, date_time): #, lon_index_west, lat_index_south, lon_index_east, lat_index_north):
	# location of WRF file
	file_location = '/global/cscratch1/sd/ebercosh/WRF_TCM/' + scenario_type + '/' + date_time.strftime('%Y') + '/wrfout_d01_crop_'
	# open file
	data = Dataset(file_location + date_time.strftime("%Y-%m-%d_%H_%M_%S") + '.nc')
	# get u, v, and p
	print("Pulling variables...")
	p_3d = wrf.getvar(data, 'pressure') # pressure in hPa
	u_3d = wrf.getvar(data, 'ua') # zonal wind in m/s
	v_3d = wrf.getvar(data, 'va') # meridional wind in m/s

	# get u and v at the pressure levels 850, 700, and 600 hPa
	u_levels = calc_var_pres_levels(p_3d,u_3d)
	v_levels = calc_var_pres_levels(p_3d,v_3d)

	# calculate the relative vorticity
	rel_vort_levels = calc_rel_vort(u_levels.values,v_levels.values,common_object.lat,common_object.lon)
	# calculate the curvature vorticity
	curve_vort_levels = calc_curve_vort(common_object,u_levels.values,v_levels.values,rel_vort_levels)

	return u_levels.values, v_levels.values, rel_vort_levels, curve_vort_levels

# This function interpolates WRF variables to specific pressure levels
# This function takes the pressure and the variable to be interpolated.
# The fucntion returns a three dimensional array ordered lev (pressure), lat, lon
def calc_var_pres_levels(p, var):
	# pressure levels needed
	pressure_levels = [850., 700., 600.]
	# interpolate the variable to the above pressure levels
	# returns an array with the lev dim the length of pressure_levels
	var_levels = wrf.interplevel(var, p, pressure_levels)
	# get rid of any nans
	# linearly interpolate the missing values
#	print("Any nans?")
#	print(np.isnan(var_levels).any())
	mask = np.isnan(var_levels.values)
	var_levels.values[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), var_levels.values[~mask]) 
#	print("Any nans now?")
#	print(np.isnan(var_levels).any())

	return var_levels

# This is a function to get the MERRA2 variables required for tracking
# The function takes the common_object that holds lat/lon information, the scenario type, and the date and time for the desired file
# The function returns u, v, relative vorticity, and curvature vorticity on specific pressure levels
def get_MERRA2_variables(common_object,date_time):
	# location of MERRA-2 files
	u_file_location = '/global/cscratch1/sd/ebercosh/MERRA2/U1000_'
	v_file_location = '/global/cscratch1/sd/ebercosh/MERRA2/V1000_'
	# open files 
	u_data = xr.open_dataset(u_file_location + date_time.strftime("%Y%m%d") + '.nc')
	v_data = xr.open_dataset(v_file_location + date_time.strftime("%Y%m%d") + '.nc')
#	print(u_file_location + date_time.strftime("%Y%m%d") + '.nc')
#	print(u_data)
	# get u and v
	print("Pulling variables...")
	# the MERRA data has 8 times in one file (data is 3 hourly), so pull only the hour that matches the current date_time
	# unfortunately all of Dustin's MERRA2 data has the hours attached to July 1, so need to have a work around using a dictionary
	time_dict = {'00' : 0, '03' : 1, '06' : 2, '09' : 3, '12' : 4, '15' : 5, '18' : 6, '21' : 7}
#	u_3d = u_data.U.sel(time=np.datetime64(date_time)) 
#	v_3d = v_data.V.sel(time=np.datetime64(date_time)) 
	u_3d = u_data.U[time_dict[date_time.strftime("%H")],:,:,:]
	v_3d = v_data.V[time_dict[date_time.strftime("%H")],:,:,:]
#	print(u_3d)
#	print(v_3d)
	# get u and v only on the levels 850, 700, and 600 hPa, which correspond to the levels 63, 56, and 53 in MERRA2 (according to Dustin)
	lev_list = [53, 56, 63]
	u_levels = np.zeros([3,u_3d.shape[1],u_3d.shape[2]])
	v_levels = np.zeros([3,v_3d.shape[1],v_3d.shape[2]])
	for level_index in range(0,3):
		u_levels[level_index,:,:] = u_3d.sel(lev=lev_list[level_index])
		v_levels[level_index,:,:] = v_3d.sel(lev=lev_list[level_index])

	# get rid of any NANs
	if np.isnan(u_levels).any():
		mask_u = np.isnan(u_levels)
		u_levels[mask_u] = np.interp(np.flatnonzero(mask_u), np.flatnonzero(~mask_u), u_levels[~mask_u]) 
	if np.isnan(v_levels).any():
		mask_v = np.isnan(v_levels)
		v_levels[mask_v] = np.interp(np.flatnonzero(mask_v), np.flatnonzero(~mask_v), v_levels[~mask_v]) 

	# calculate the relative vorticity
	rel_vort_levels = calc_rel_vort(u_levels,v_levels,common_object.lat,common_object.lon)
	# calculate the curvature voriticty
	curve_vort_levels = calc_curve_vort(common_object,u_levels,v_levels,rel_vort_levels)

	# switch the arrays to be float32 instead of float64
	u_levels = np.float32(u_levels)
	v_levels = np.float32(v_levels)
	rel_vort_levels = np.float32(rel_vort_levels)
	curve_vort_levels = np.float32(curve_vort_levels)

	# make the arrays C contiguous (will need this later for the wrapped C smoothing function)
	u_levels = np.asarray(u_levels, order='C')
	v_levels = np.asarray(v_levels, order='C')
	rel_vort_levels = np.asarray(rel_vort_levels, order='C')
	curve_vort_levels = np.asarray(curve_vort_levels, order='C')

	return u_levels, v_levels, rel_vort_levels, curve_vort_levels

# This is a function to get the CAM5 variables required for tracking
# The function takes the common_object that holds lat/lon information, the scenario type, and the date and time for the desired file
# The function returns u, v, relative vorticity, and curvature vorticity on specific pressure levels
def get_CAM5_variables(common_object,scenario_type,date_time): 
	# location of CAM5 files
	if scenario_type == 'Historical':
		u_file_location = '/global/cscratch1/sd/ebercosh/CAM5/Africa/' + scenario_type + '/run3/' + date_time.strftime('%Y') + '/U_CAM5-1-0.25degree_All-Hist_est1_v3_run3.cam.h4.'
		v_file_location = '/global/cscratch1/sd/ebercosh/CAM5/Africa/' + scenario_type + '/run3/' + date_time.strftime('%Y') + '/V_CAM5-1-0.25degree_All-Hist_est1_v3_run3.cam.h4.'
	elif scenario_type == 'Plus30':
		u_file_location = '/global/cscratch1/sd/ebercosh/CAM5/Africa/' + scenario_type + '/run3/' + date_time.strftime('%Y') + '/U_fvCAM5_UNHAPPI30_run003.cam.h4.'
		v_file_location = '/global/cscratch1/sd/ebercosh/CAM5/Africa/' + scenario_type + '/run3/' + date_time.strftime('%Y') + '/V_fvCAM5_UNHAPPI30_run003.cam.h4.'
	# open files 
	u_data = xr.open_dataset(u_file_location + date_time.strftime("%Y-%m-%d") + '-00000_AEW.nc', decode_times=False)
	v_data = xr.open_dataset(v_file_location + date_time.strftime("%Y-%m-%d") + '-00000_AEW.nc', decode_times=False)
	# get u and v
	print("Pulling variables...")
	# the CAM5 data has 8 times in one file (data is 3 hourly), so pull only the hour that matches the current date_time
	# unfortunately xarray has trouble decoding the times in the file so use the following dictionary to get the correct time index
	# using the time from the time loop 
	time_dict = {'00' : 0, '03' : 1, '06' : 2, '09' : 3, '12' : 4, '15' : 5, '18' : 6, '21' : 7}
	# some of the CAM5 data is missing the last time in the file (so 7 times in a file instead of 8). Use try and except to catch these 
	# rare cases and then use the previous time step when the last time step is missing. Because this happens so infrequently, using the previous
	# time step does lead to any issues.
	try:
		u_3d = u_data.U[time_dict[date_time.strftime("%H")],:,:,:]
	except IndexError:
		u_3d = u_data.U[time_dict[date_time.strftime("%H")]-1,:,:,:]
	try:
		v_3d = v_data.V[time_dict[date_time.strftime("%H")],:,:,:]
	except IndexError:
		v_3d = v_data.V[time_dict[date_time.strftime("%H")]-1,:,:,:]
	# get u and v only on the levels 850, 700, and 600 hPa
	lev_list = [850, 700, 600]
	u_levels = np.zeros([3,u_3d.shape[1],u_3d.shape[2]])
	v_levels = np.zeros([3,v_3d.shape[1],v_3d.shape[2]])
	for level_index in range(0,3):
		u_levels[level_index,:,:] = u_3d.sel(plev=lev_list[level_index])
		v_levels[level_index,:,:] = v_3d.sel(plev=lev_list[level_index])

	# get rid of any NANs
	if np.isnan(u_levels).any():
		mask_u = np.isnan(u_levels)
		u_levels[mask_u] = np.interp(np.flatnonzero(mask_u), np.flatnonzero(~mask_u), u_levels[~mask_u]) 
	if np.isnan(v_levels).any():
		mask_v = np.isnan(v_levels)
		v_levels[mask_v] = np.interp(np.flatnonzero(mask_v), np.flatnonzero(~mask_v), v_levels[~mask_v]) 

	# calculate the relative vorticity
	rel_vort_levels = calc_rel_vort(u_levels,v_levels,common_object.lat,common_object.lon)
	# calculate the curvature voriticty
	curve_vort_levels = calc_curve_vort(common_object,u_levels,v_levels,rel_vort_levels)

	# switch the arrays to be float32 instead of float64
	u_levels = np.float32(u_levels)
	v_levels = np.float32(v_levels)
	rel_vort_levels = np.float32(rel_vort_levels)
	curve_vort_levels = np.float32(curve_vort_levels)

	# make the arrays C contiguous (will need this later for the wrapped C smoothing function)
	u_levels = np.asarray(u_levels, order='C')
	v_levels = np.asarray(v_levels, order='C')
	rel_vort_levels = np.asarray(rel_vort_levels, order='C')
	curve_vort_levels = np.asarray(curve_vort_levels, order='C')

	return u_levels, v_levels, rel_vort_levels, curve_vort_levels

# This is a function to get the ERA5 variables required for tracking
# The function takes the common_object that holds lat/lon information, the scenario type, and the date and time for the desired file
# The function returns u, v, relative vorticity, and curvature vorticity on specific pressure levels
def get_ERA5_variables(common_object,date_time):
	# location of ERA5 files
	u_file_location = '/global/cfs/projectdirs/m3522/cmip6/ERA5/e5.oper.an.pl/' + date_time.strftime("%Y%m") + '/e5.oper.an.pl.128_131_u.ll025uv.'
	v_file_location = '/global/cfs/projectdirs/m3522/cmip6/ERA5/e5.oper.an.pl/' + date_time.strftime("%Y%m") + '/e5.oper.an.pl.128_132_v.ll025uv.'
	# open files 
	u_data = xr.open_dataset(u_file_location + date_time.strftime("%Y%m%d") + '00_' + date_time.strftime("%Y%m%d") + '23.nc')
	v_data = xr.open_dataset(v_file_location + date_time.strftime("%Y%m%d") + '00_' + date_time.strftime("%Y%m%d") + '23.nc')
	# get u and v
	print("Pulling variables...")
	# ERA5 is hourly. To compare with CAM5, which is 3 hourly, average every three hours together (eg 00, 01, and 02 are averaged)
	# to make the comparison easier. To compare with WRF, which is 6 hourly, average every six hours. The averaging is controlled by
	# what is set at dt in the common_object (eg dt=3 for CAM5 or dt=6 for WRF)
	u_3d = u_data.U[int(date_time.strftime("%H")),:,:,:]
	v_3d = v_data.V[int(date_time.strftime("%H")),:,:,:]
	# get u and v only on the levels 850, 700, and 600 hPa
	lev_list = [850, 700, 600]
	u_levels_360 = np.zeros([3,u_3d.shape[1],u_3d.shape[2]])
	v_levels_360 = np.zeros([3,v_3d.shape[1],v_3d.shape[2]])
	for level_index in range(0,3):
		# the ERA5 data goes from north to south. Use flip to flip it 180 degrees in the latitude dimension so that
		# the array now goes from south to north like the other datasets.
		u_levels_360[level_index,:,:] = np.flip(u_3d.sel(level=lev_list[level_index]),axis=0)
		v_levels_360[level_index,:,:] = np.flip(v_3d.sel(level=lev_list[level_index]),axis=0)
	# need to roll the u and v variables on the longitude axis because the longitudes were changed from 
	# 0-360 to -180 to 180
	u_levels_full = np.roll(u_levels_360, int(u_levels_360.shape[2]/2), axis=2)
	v_levels_full = np.roll(v_levels_360, int(v_levels_360.shape[2]/2), axis=2)

	# Crop the data. This is a global dataset and we don't need to calculate vorticity values over the entire globe, only over the region of interest. 
	# The tracking algorithm only looks over Africa/the Atlantic, so it's unnecessary to have a global dataset.
	u_levels = u_levels_full[:,common_object.lat_index_south_crop:common_object.lat_index_north_crop+1,common_object.lon_index_west_crop:common_object.lon_index_east_crop+1]
	v_levels = v_levels_full[:,common_object.lat_index_south_crop:common_object.lat_index_north_crop+1,common_object.lon_index_west_crop:common_object.lon_index_east_crop+1]

	# get rid of any NANs
	if np.isnan(u_levels).any():
		mask_u = np.isnan(u_levels)
		u_levels[mask_u] = np.interp(np.flatnonzero(mask_u), np.flatnonzero(~mask_u), u_levels[~mask_u]) 
	if np.isnan(v_levels).any():
		mask_v = np.isnan(v_levels)
		v_levels[mask_v] = np.interp(np.flatnonzero(mask_v), np.flatnonzero(~mask_v), v_levels[~mask_v]) 

	# calculate the relative vorticity
	rel_vort_levels = calc_rel_vort(u_levels,v_levels,common_object.lat,common_object.lon)
	# calculate the curvature voriticty
	curve_vort_levels = calc_curve_vort(common_object,u_levels,v_levels,rel_vort_levels)

	# switch the arrays to be float32 instead of float64
	u_levels = np.float32(u_levels)
	v_levels = np.float32(v_levels)
	rel_vort_levels = np.float32(rel_vort_levels)
	curve_vort_levels = np.float32(curve_vort_levels)

	# make the arrays C contiguous (will need this later for the wrapped C smoothing function)
	u_levels = np.asarray(u_levels, order='C')
	v_levels = np.asarray(v_levels, order='C')
	rel_vort_levels = np.asarray(rel_vort_levels, order='C')
	curve_vort_levels = np.asarray(curve_vort_levels, order='C')

	return u_levels, v_levels, rel_vort_levels, curve_vort_levels

# This is a function to get the ERAI variables required for tracking
# The function takes the common_object that holds lat/lon information, the scenario type, and the date and time for the desired file
# The function returns u, v, relative vorticity, and curvature vorticity on specific pressure levels
def get_ERAI_variables(common_object,date_time):
	# location of ERA-Interim data, which is in GRIB format
	file_location = '/global/cscratch1/sd/ebercosh/Reanalysis/ERA-I/ei.oper.an.pl.regn128uv.'
	# open file
	grbs = pygrib.open(file_location + date_time.strftime("%Y%m%d%H"))

	# pressure list
	lev_list = [30, 25, 23] # 30, 25, and 23 correspond to 850, 700, and 600 hPa, respectively

	u_levels_360 = np.zeros([3,grbs.select(name = 'U component of wind')[23].values.shape[0],grbs.select(name = 'U component of wind')[23].values.shape[1]])
	v_levels_360 = np.zeros_like(u_levels_360)
	# get the desired pressure levels
	for level_index in range(0,3):
		# the ERA-Interim data goes from north to south. Use flip to flip it 180 degrees in the latitude dimension so that
		# the array now goes from south to north like the other datasets.
		u_levels_360[level_index,:,:] = np.flip(grbs.select(name = 'U component of wind')[lev_list[level_index]].values, axis=0)
		v_levels_360[level_index,:,:] = np.flip(grbs.select(name = 'V component of wind')[lev_list[level_index]].values, axis=0)

	# need to roll the u and v variables on the longitude axis because the longitudes were changed from 
	# 0-360 to -180 to 180
	u_levels = np.roll(u_levels_360, int(u_levels_360.shape[2]/2), axis=2)
	v_levels = np.roll(v_levels_360, int(v_levels_360.shape[2]/2), axis=2)

	# calculate the relative vorticity
	rel_vort_levels = calc_rel_vort(u_levels,v_levels,common_object.lat,common_object.lon)
	# calculate the curvature voriticty
	curve_vort_levels = calc_curve_vort(common_object,u_levels,v_levels,rel_vort_levels)

	# switch the arrays to be float32 instead of float64
	u_levels = np.float32(u_levels)
	v_levels = np.float32(v_levels)
	rel_vort_levels = np.float32(rel_vort_levels)
	curve_vort_levels = np.float32(curve_vort_levels)

	# make the arrays C contiguous (will need this later for the wrapped C smoothing function)
	u_levels = np.asarray(u_levels, order='C')
	v_levels = np.asarray(v_levels, order='C')
	rel_vort_levels = np.asarray(rel_vort_levels, order='C')
	curve_vort_levels = np.asarray(curve_vort_levels, order='C')

	return u_levels, v_levels, rel_vort_levels, curve_vort_levels

# Calculate and return the relative vorticity. Relative vorticity is defined as
# rel vort = dv/dx - du/dy. This function takes the 3-dimensional variables u and v,
# ordered (lev, lat, lon), and the 2-dimensional variables latitude and longitude, ordered
# (lat, lon), as parameters and returns the relative voriticty.
def calc_rel_vort(u,v,lat,lon):
	# take derivatives of u and v
	dv_dx = x_derivative(v, lat, lon)
	du_dy = y_derivative(u, lat)
	# subtract derivatives to calculate relative vorticity
	rel_vort = dv_dx - du_dy
#	print("rel_vort shape =", rel_vort.shape)
	return rel_vort
	
# This function takes the derivative with respect to x (longitude).
# The function takes a three-dimensional variable ordered lev, lat, lon
# and returns d/dx of the variable.
def x_derivative(variable, lat, lon):
	# subtract neighboring longitude points to get delta lon
	# then switch to radians
	dlon = np.radians(lon[0,2]-lon[0,1])
#	print("dlon =", dlon)
	# allocate space for d/dx array
	d_dx = np.zeros_like(variable)
#	print(d_dx.shape)
	# loop through latitudes
	for nlat in range(0,len(lat[:,0])):
			# calculate dx by multiplying dlon by the radius of the Earth, 6367500 m, and the cos of the lat
			dx = 6367500.0*np.cos(np.radians(lat[nlat,0]))*dlon  # constant at this latitude
			#print dx
			# the variable will have dimensions lev, lat, lon
			grad = np.gradient(variable[:,nlat,:], dx)
			d_dx[:,nlat,:] = grad[1]
#	print(d_dx.shape)
	return d_dx

# This function takes the derivative with respect to y (latitude).
# The function takes a 3 dimensional variable (lev, lat, lon) and
# returns d/dy of the variable.
def y_derivative(variable,lat):
	# subtract neighboring latitude points to get delta lat
	# then switch to radians
	dlat = np.radians(lat[2,0]-lat[1,0])
#	print("dlat =", dlat)
	# calculate dy by multiplying dlat by the radius of the Earth, 6367500 m
	dy = 6367500.0*dlat
#	print("dy =", dy)
	# calculate the d/dy derivative using the gradient function
	# the gradient function will return a list of arrays of the same dimensions as the
	# WRF variable, where each array is a derivative with respect to one of the dimensions
	d_dy = np.gradient(variable, dy)
	#print d_dy.shape
	#print d_dy[1].shape
	# return the second item in the list, which is the d/dy array
	return d_dy[1]

# This function calculates and returns the curvature vorticity.
# The function takes hte common_object for lat/lon information, the zonal wind u, 
# the meridional wind v, and relative vorticity.
def calc_curve_vort(common_object,u,v,rel_vort):
	# calculate dx and dy
	# subtract neighboring latitude points to get delta lat (dlat), then switch to radians
	dlat = np.radians(np.absolute(common_object.lat[2,0]-common_object.lat[1,0]))
	# calculate dy by multiplying dlat by the radius of the Earth, 6367500 m
	dy = np.full((common_object.lat.shape[0],common_object.lat.shape[1]),6367500.0*dlat)
	# subtract neighboring longitude points to get delta lon (dlon), then switch to radians
#	dlon = np.radians(np.absolute(common_object.lon[0,2]-common_object.lon[0,1]))
	# calculate dx by taking the cosine of the lat (in radius) and muliply by dlat times the radius of the Earth, 6367500 m
	dx = np.cos(np.radians(common_object.lat))*(dlat*6367500.0)

	# calculate the magnitude of the wind vector sqrt(u^2+v^2), and then make u and v unit vectors
	vec_mag = np.sqrt(np.square(u) + np.square(v))
	u_unit_vec = u/vec_mag
	v_unit_vec = v/vec_mag

	# calculate the shear vorticity first
	shear_vort = np.empty_like(u)
	# loop through the latitudes and longitudes
	for lon_index in range(0,u.shape[2]): # loop through longitudes
		for lat_index in range(0,u.shape[1]): # loop through latitudes
			# get the previous and next indices for the current lon_index and lat_index
			lon_index_previous = max(lon_index-1,0) # previous is the max of either lon_index-1, or zero (the first index)
			lon_index_next = min(lon_index+1,u.shape[2]-1) # next is the min of either lon_index+1, or u.shape[2]-1 (the last lon index)
			lat_index_previous = max(lat_index-1,0) # previous is the max of either lat_index-1, or zero (the first index)
			lat_index_next = min(lat_index+1,u.shape[1]-1) # next is the min of either lat_index+1, or u.shape[1]-1 (the last lat index)
			# set some counters
			di = 0 
			dj = 0
			# calculate v1, v2, u1, and u2
			v1 = ((u_unit_vec[:,lat_index,lon_index]*u[:,lat_index,lon_index_previous]) + (v_unit_vec[:,lat_index,lon_index]*v[:,lat_index,lon_index_previous]))*v_unit_vec[:,lat_index,lon_index]
			if lon_index_previous != lon_index:  
				di += 1

			v2 = ((u_unit_vec[:,lat_index,lon_index]*u[:,lat_index,lon_index_next]) + (v_unit_vec[:,lat_index,lon_index]*v[:,lat_index,lon_index_next]))*v_unit_vec[:,lat_index,lon_index]
			if lon_index_next != lon_index: 
				di += 1

			u1 = ((u_unit_vec[:,lat_index,lon_index]*u[:,lat_index_previous,lon_index]) + (v_unit_vec[:,lat_index,lon_index]*v[:,lat_index_previous,lon_index]))*u_unit_vec[:,lat_index,lon_index]
			if lat_index_previous != lat_index:  
				dj += 1

			u2 = ((u_unit_vec[:,lat_index,lon_index]*u[:,lat_index_next,lon_index]) + (v_unit_vec[:,lat_index,lon_index]*v[:,lat_index_next,lon_index]))*u_unit_vec[:,lat_index,lon_index]
			if lat_index_next != lat_index:  
				dj += 1
			# fill in the shear vorticity array
			if di > 0 and dj > 0:
				shear_vort[:,lat_index,lon_index] = ((v2 - v1) / (float(di) * dx[lat_index,lon_index])) - ((u2 - u1) / (float(dj)*dy[lat_index,lon_index]))

	# calculate curvature vorticity, where curvature vorticty = relative vorticity - shear vorticity
	curve_vort = rel_vort - shear_vort 

	return curve_vort

# This function is called from AEW_Tracks.py and is what begins the process of acquiring u, v, relative vorticity, and curvature vorticity
# at various pressure levels. The function takes the common_object, the scenario type and date and time and returns the previoiusly mentioned variables.
def get_variables(common_object, scenario_type, date_time):
	if common_object.model == 'WRF':
		u_levels, v_levels, rel_vort_levels, curve_vort_levels = get_WRF_variables(common_object,scenario_type,date_time)
	elif common_object.model == 'MERRA2':
		u_levels, v_levels, rel_vort_levels, curve_vort_levels = get_MERRA2_variables(common_object,date_time)
	elif common_object.model == 'CAM5':
		u_levels, v_levels, rel_vort_levels, curve_vort_levels = get_CAM5_variables(common_object,scenario_type,date_time)
	elif common_object.model == 'ERA5':
		u_levels, v_levels, rel_vort_levels, curve_vort_levels = get_ERA5_variables(common_object,date_time)
	elif common_object.model == 'ERAI':
		u_levels, v_levels, rel_vort_levels, curve_vort_levels = get_ERAI_variables(common_object,date_time)

	return u_levels, v_levels, rel_vort_levels, curve_vort_levels

