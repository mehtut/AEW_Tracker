from __future__ import division  # makes division not round with integers 
import os
import pickle
from netCDF4 import Dataset
import numpy as np
import xarray as xr
import wrf as wrf
import argparse
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
#from Circle_functions import *
from Tracking_functions import *
from Pull_data import *
from scipy.ndimage import gaussian_filter
import ctypes
import numpy.ctypeslib as ctl

# AEW_Tracks.py is the main program for the AEW tracking algorithm. The model type, scenario type, and year are parsed from the command line.
# The timeframe over which AEW tracks are found can be modified in main and is typically set to run from May - October. The output of this 
# program is an AEW track object and a figure of all tracks found over the given time period. To run the program, for example, type:
# python AEW_Tracks.py --model 'WRF' --scenario 'late_century' --year '2010'

# Common info class. Each instance/object of Common_track_data is an object that holds the latitude and longitude arrays,
# the lat and lon indices for the boundaries over Africa/the Atlantic, the time step (dt) for the model data, the min_threshold
# value, the radius, and the model type
class Common_track_data:
	def __init__(self):
		self.model = None
		self.lat = None
		self.lon = None
		self.lat_index_north = None
		self.lat_index_south = None
		self.lon_index_east = None
		self.lon_index_west = None
		self.lat_index_north_crop = None
		self.lat_index_south_crop = None
		self.lon_index_east_crop = None
		self.lon_index_west_crop = None
		self.total_lon_degrees = None
		self.dt = None
		self.min_threshold = None
		self.radius = None
	# function to add the model type to self.model
	def add_model(self, model_type):
		self.model = model_type

# AEW track class. Each instance/object of AEW_track is a an AEW track and has corresponding lat/lon points
# and magnitudes of the vorticity at those points. The lat/lon and magnitudes are stored in lists that are 
# associated with each object. The lat/lon and magnitude lists show the progression of the AEW in time, so the 
# first lat/lon point in the list is where the AEW starts, and then the following lat/lon points are where it 
# travels to over time.
class AEW_track:
	def __init__(self):
		self.latlon_list = [] # creates a new empty list for lat/lon tuples
		self.magnitude_list = [] # creates a new empty list for the magnitude of the vorticity at each lat/lon point
		self.time_list = [] # creates a new empty list for the times for each new location in the track

	def add_latlon(self, latlon):
		self.latlon_list.append(latlon)

	def remove_latlon(self, latlon):
		self.latlon_list.remove(latlon)

	def add_magnitude(self, magnitude):
		self.magnitude_list.append(magnitude)

	def add_time(self, time):
		self.time_list.append(time)


# make figure with connected lat/lon points over map that show the AEW tracks 
def plot_points_map(aew_track_list,scenario_type,model_type):

	# plot information
	fig, ax = plt.subplots()

#	cart_proj = wrf.get_cartopy(u_3d)
	# Set the GeoAxes to the projection used by WRF
	ax = plt.axes(projection=ccrs.Mercator())	
	# plotting options
	ax.coastlines('50m', linewidth=0.8)
	ax.set_extent([-65., 25., -5., 35.], crs=ccrs.PlateCarree())  # 65W, 25E, 5S, 35N; 45W, 25E, 5N, 35N (from Albany get_starting_targets)
	ax.add_feature(cfeature.LAND)
#	ax.set_xlim(wrf.cartopy_xlim(var_cat1))
#	ax.set_ylim(wrf.cartopy_ylim(var_cat1))

	# Add the gridlines
	gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='black', alpha=0.5, linestyle=':')
	gl.xlabels_top = False
	gl.ylabels_right = False
	gl.ylocator = mticker.FixedLocator([-5, 5, 15, 25, 35, 45, 50]) #45, 55]) #40])
	gl.xformatter = LONGITUDE_FORMATTER
	gl.yformatter = LATITUDE_FORMATTER

	colors = plt.cm.rainbow(np.linspace(0, 1, len(aew_track_list)))
	color_index = 0
	for aew_track in aew_track_list:
#		colors = plt.cm.rainbow_r(np.linspace(0, 1, len(aew_track.latlon_list)))
#		color_index = 0
		track_color = colors[color_index]

		# unzip list of lat/lon tuples into a list of two lists. The first list is all of the lats and the
		# second list is all of the lons. Then create separate lists with the lats and the lons used for plotting.
		track_latlons = [list(t) for t in zip(*aew_track.latlon_list)]
		track_lats = track_latlons[0]
		track_lons = track_latlons[1]

		plt.scatter(track_lons,track_lats, color=track_color, linewidth=0.25, marker='o', transform=ccrs.Geodetic())
		plt.plot(track_lons,track_lats, color=track_color, linewidth=1, label=aew_track.time_list[0].strftime('%Y-%m-%d_%H'), transform=ccrs.Geodetic())
		del track_latlons
		del track_lats
		del track_lons

#		for lat_lon_pair in aew_track.latlon_list:
#			plt.scatter(lat_lon_pair[1],lat_lon_pair[0], color=colors[color_index], linewidth=2, marker='o',transform=ccrs.Geodetic())
#			plt.scatter(lat_lon_pair[1],lat_lon_pair[0], color=track_color, linewidth=1, marker='o',transform=ccrs.Geodetic())
#			plt.plot(lat_lon_pair[1],lat_lon_pair[0], color=track_color, linewidth=1, transform=ccrs.Geodetic())
		color_index += 1
#	plt.scatter(next_max_locs_lons, next_max_locs_lats, color='red', linewidth=2, marker='o',transform=ccrs.Geodetic())
#	plt.legend(loc="upper left", ncol = 3, fontsize = 'xx-small') # upper left or best location for legend, 3 columns, xx-small fontsize
#	plt.show()
	fig.savefig(model_type + '_' + scenario_type + '_AEW_Tracks.pdf', bbox_inches='tight')


# This function takes the variable and smooths it using a circular smoothing algorithm written in C for speed.
# The function needs the common_object for the lat/lon information, the variable to be smoothed, and the radius for circular smoothing.
# The return value is a smoothed version of the variable.
def c_smooth(common_object,var,radius):
	# create a copy of the unsmoothed variable called smoothed_var. This will be what gets smoothed and returned.
	smoothed_var = np.copy(var)
	# load in the C program C_circle_functions
	c_circle_avg_m = ctypes.CDLL('/global/homes/e/ebercosh/Python/C_circle_functions.so').circle_avg_m
	# set the types of all of the variables so C understands what's coming into the function (eg an int will be ctypes.c_int)
	c_circle_avg_m.argtypes = [ctypes.c_int,ctypes.c_int,ctypes.c_int,ctl.ndpointer(np.float32,flags='aligned, c_contiguous'),\
		ctl.ndpointer(np.float32,flags='aligned, c_contiguous'),ctl.ndpointer(np.float32,flags='aligned, c_contiguous'),ctypes.c_float,\
		ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float]
	# run the function from C
	c_circle_avg_m(var.shape[0],common_object.lat.shape[0],common_object.lat.shape[1],var,smoothed_var,common_object.lat,radius,\
		common_object.lat_index_north,common_object.lat_index_south,common_object.lon_index_east,common_object.lon_index_west,common_object.total_lon_degrees)

	return smoothed_var


def main():

	# set up argparse, which lets command line entries define variable_type and scenario_type
	parser = argparse.ArgumentParser(description='AEW Tracker.')
	parser.add_argument('--model', dest='model_type', help='Get the model type for the data (WRF, CAM5, ERA5)', required=True)
	parser.add_argument('--scenario', dest='scenario_type', help='Get the scenario type to be used (Historical, late_century, Plus30)', required=True)
	parser.add_argument('--year', dest='year', help='Get the year of interest', required=True)
	args = parser.parse_args()

	# set the model type that is parsed from the command line
	model_type = args.model_type 
	# set the scenario type that is parsed from the command line
	scenario_type = args.scenario_type 
	# set the year that is parsed from the command line
	year = args.year

#	model_type = 'WRF' #'WRF' #'MERRA2' #'CAM5' #'ERA5' #'ERAI'
#	scenario_type = 'late_century'  #'Historical' #'Plus30' # 'late_century'
#	year = 2010

	# set radius in km for smoothing and for finding points that belong to the same track
	# Albany used 500, 400, and 300 km
	if model_type == 'WRF':
		radius_km = 450. # km
	elif model_type == 'ERA5':
		radius_km = 700. # km
	elif model_type == 'CAM5':
		radius_km = 500. # km

	min_threshold = .000002 # default was set at .000002, Brannan and Martin (2019) used .0000015

	# create an object from Common_track_data that will hold the model_type, lat, lon and dt information
	common_object = Common_track_data()
	# assign the model type to the common_object
	common_object.add_model(model_type)
	# assign the min_threshold to the common_object
	common_object.min_threshold = min_threshold
	# assign the radius (in km) to the common_object
	common_object.radius = radius_km
	# get the common track data (like lat and lon) and assign it to the appropriate attributes in common_object
	get_common_track_data(common_object)

	# set time information
	times = np.arange(datetime(int(year),5,1,0), datetime(int(year),11,1,0), timedelta(hours=common_object.dt)).astype(datetime) # May - October (AEW seasn)
#	times = np.arange(datetime(int(year),6,1,0), datetime(int(year),12,1,0), timedelta(hours=common_object.dt)).astype(datetime) # June - November (tropical cyclone season)
#	times = np.arange(datetime(int(year),7,1,0), datetime(int(year),8,1,0), timedelta(hours=common_object.dt)).astype(datetime) # month of July
#	times = np.arange(datetime(int(year),7,1,0), datetime(int(year),7,15,0), timedelta(hours=common_object.dt)).astype(datetime) # first two weeks of July

	# create a working list for AEW tracks
	AEW_tracks_list = []
	finished_AEW_tracks_list = []

	# loop through all times and find AEW tracks
	for time_index in range(0,times.shape[0]):
		print(times[time_index].strftime('%Y-%m-%d_%H'))

		# get variables for each time step
		u_3d, v_3d, rel_vort_3d, curve_vort_3d = get_variables(common_object, scenario_type, times[time_index])

		# smooth the curvature and relative vorticities
		print("Smoothing...")
		curve_vort_smooth = c_smooth(common_object,curve_vort_3d,common_object.radius*1.5)
#		curve_vort_smooth_delta = xr.Dataset({'curve_vort_smooth': (['lev','y','x'], curve_vort_smooth)}) #, coords={'time': time, 'lat': (['y','x'], wrf_lat[0,:,:]), 'lon': (['y','x'], wrf_lon[0,:,:])})
#		curve_vort_smooth_delta.to_netcdf('curve_vort_smooth_C_cam.nc')
		rel_vort_smooth = c_smooth(common_object,rel_vort_3d,common_object.radius*1.5)

		# Find new starting points 
		print("Get starting targets...")
		unique_max_locs = get_starting_targets(common_object,curve_vort_smooth) #,lon_index_west, lat_index_south, lon_index_east, lat_index_north)
#		print(unique_max_locs)

		# Combine potential locations into tracked locations
		print("Get multi positions...")
		alternative_unique_max_locs = get_multi_positions(common_object,curve_vort_smooth,rel_vort_smooth,unique_max_locs)
#		print(alternative_unique_max_locs)
		# Remove duplicate locations
		# The 99999999 is the starting value for unique_loc_number; it just needs to be way bigger than the 
		# possible number of local maxima in weighted_max_indices. This function recursively calls itself until
		# the value for unique_loc_number doesn't decrease anymore.
		# unique_max_locs+alternative_unique_max_locs is joining the two lists (E.g. [a,b,c]+[d,e,f] -> [a,b,c,d,e,f])
		# use the following conditional to catch the case where there is only one location, in which case we don't need to use unique_locations
		if len(unique_max_locs+alternative_unique_max_locs)>1:
			combined_unique_max_locs = unique_locations(unique_max_locs+alternative_unique_max_locs,common_object.radius,99999999)
#			print(combined_unique_max_locs)
		else:
			combined_unique_max_locs = unique_max_locs+alternative_unique_max_locs

		# Compare combined_unique_max_locs at time t with the track object locations at time t.
		# If there are locations that are too close together (so duplicates between the track object and the new 
		# track locations that are in combined_unique_max_locs) average the two lat/lon pairs, then use this new value to
		# replace the old lat/lon value in the existing track object and remove the duplicate lat/lon location from combined_unique_max_locs.
		if AEW_tracks_list: # only enter if the list isn't empty
			for track_object in AEW_tracks_list:
				# get the index of current time from the time list associated with the track object
				current_time_index = track_object.time_list.index(times[time_index])
				# use the time index to get the corresponding track object lat/lon location and then check to make sure that
				# the new track locations aren't duplicates of existing track objects
				unique_track_locations(track_object,combined_unique_max_locs,current_time_index,common_object.radius)
		
		# for the locations in combined_unique_max_locs (assuming it isn't empty), create new
		# AEW track objects. For reach object, add the lat/lon pair from combined_unique_max_locs and 
		# also add the time. Then append the new track object to AEW_tracks_list
		if combined_unique_max_locs: # only enter if the list isn't empty
			for lat_lon_pair in combined_unique_max_locs:
				aew_track = AEW_track()
				aew_track.add_latlon(lat_lon_pair)
				aew_track.add_time(times[time_index])
#				print(aew_track.latlon_list)
				AEW_tracks_list.append(aew_track)
				del aew_track

		# loop through all track objects and assign magnitudes to the new lat/lon pairs that have been added to each track object
		# Then filter out any tracks that don't meet AEW qualifications and advect the tracks that are leftover that didn't get filtered.
		print("Assign magnitudes...")
		print("Filter...")
		print("Advect tracks...")
		# use list(AEW_tracks_list), which is a copy of AEW_tracks_list, since AEW_tracks_list is being modified in the loop and using that in the for statement 
		# causes track objects to be skipped
		for track_object in list(AEW_tracks_list): 
			# assign a magnitude from the vorticity to each lat/lon point
			assign_magnitude(common_object, curve_vort_smooth, rel_vort_smooth, track_object)
			# filter tracks. A track is either removed entirely, or removed as an acitve track and added to a finished track list.
			filter_result = filter_tracks(common_object,track_object)
			if filter_result.reject_track_direction: # or filter_result.reject_track_speed:
				print("in reject track")
				AEW_tracks_list.remove(track_object)
				del filter_result
				continue
			elif filter_result.magnitude_finish_track or filter_result.latitude_finish_track:
				print("in weak magnitude")
				finished_AEW_tracks_list.append(track_object)
				AEW_tracks_list.remove(track_object)
				del filter_result
				continue
			del filter_result
			# advect tracks
			advect_tracks(common_object, u_3d, v_3d, track_object, times, time_index)


		print("Length of AEW_tracks_list =", len(AEW_tracks_list))
		print("Length of finished_AEW_tracks_list =", len(finished_AEW_tracks_list))

	# the final list of AEW track objects. Use set here so the combination of the active AEW_track_list and the finished_AEW_tracks_list
	# doesn't produce any duplicate track objects
	finished_AEW_tracks_list = list(set(AEW_tracks_list + finished_AEW_tracks_list))
	print("Total number of AEW tracks =", len(finished_AEW_tracks_list))

	# More filtering to check for tracks that weren't long enough, didn't start far enough east, didn't go far enough west, or goes too far south.
	# don't include tracks that are less than a day (four time steps)
	for aew_track in list(finished_AEW_tracks_list):
		# check for tracks that haven't lasted long enough. If the track hasn't lasted for two days (which is < 48/dt + 1 time steps ), get rid of it
		if len(aew_track.latlon_list) < ((48/common_object.dt)+1):
			print("not enough times")
#			print(aew_track.latlon_list)
#			print(aew_track.time_list)
			finished_AEW_tracks_list.remove(aew_track)
			continue
		# separate the lat/lon tuples into a list of lat list and lon list (e.g. [[lats],[lons]]) and make this
		# a numpy array so that operations like subtraction can be used on the lats and lon
		track_latlons = np.array([list(t) for t in zip(*aew_track.latlon_list)]) 
		# check to see if the distance along the track is less than 15 degress, if it is, get rid of it
		# To do this, find the differences: the second-last lats minus first-second to last lats and the second-last lon minus first-second to last lon
		# These differences are the deltas in between the latitudes and the longitudes of the track. Then calculate the distance between each lat/lon 
		# point using these deltas and the standart dist = sqrt( (x2-x1)^2 + (y2-y1)^2 ). Then take the sum of these distances. If the sum of the 
		# distances between each lat/lon point is < 10 degrees, remove the track.
		if np.sum(np.sqrt(np.square(track_latlons[0][1:] - track_latlons[0][:-1]) + np.square(track_latlons[1][1:] - track_latlons[1][:-1]))) < 15:
			print("short track")
#			print(aew_track.latlon_list)
#			print(aew_track.time_list)
			finished_AEW_tracks_list.remove(aew_track)
			continue

		# !!!! NOT IN USE - removes too many tracks !!!!
		# AEWs have a period of 3-5 days but some studies show this range can be 2-10 days. Their wavelengths are 2000-4000km. Filter out any tracks that
		# don't follow these AEW characteristics. To be conservative, check to see if a track has traveled 2000km in 8 days. This equates to 250km/day, which is 
		# 2.25 degrees/day (using ~111km per degree). Look at the first and last longitude of the aew_track.latlon_list and calculate the difference. Also 
		# calculate how far the track should have gone in degrees by taking the length of the aew_track.latlon_list and dividing it by 24/common_object.dt to get 
		# the number of days covered by the track (eg length of 16/(24/3)=2 day). Then this is multiplied by 2.25 to see how many degrees the track should have gone 
		# to meet the AEW requirement (eg 16 days * 2.25 = 36 degrees). If the distance in degrees between the first and last longitudes of latlon_list is less than
		# this amount in degrees (eg lon first - lon last < 36 degrees), the track object is removed. 
#		if abs(aew_track.latlon_list[0][1] - aew_track.latlon_list[-1][1]) < (len(aew_track.latlon_list)/(24./common_object.dt))*2.25:
#			print("short track part 2")
#			print(aew_track.latlon_list)
#			print(aew_track.time_list)
#			finished_AEW_tracks_list.remove(aew_track)
#			continue

		# if the smallest (farthest west) longitude value is greater than -20 (20W), which means east of 20W, then remove the track 
		# because it doesn't travel far enough west
#		if np.amin(track_latlons[1]) > -20:
		if aew_track.latlon_list[-1][1] > -20:
			print("not far enough west")
#			print(aew_track.latlon_list)
#			print(aew_track.time_list)
			finished_AEW_tracks_list.remove(aew_track)
			continue

		# if the largest (farthest east) longitude value is less than -5 (5W), which means west of 5W, then remove the track
		# because it doesn't start far enough east
#		if np.amax(track_latlons[1]) < -5:
		if aew_track.latlon_list[0][1] < -5:
			print("doesn't start far enough east")
#			print(aew_track.latlon_list)
#			print(aew_track.time_list)
			finished_AEW_tracks_list.remove(aew_track)
			continue

		# if the largest (farthest north) latitude value is less than 5 (5N), then remove the track because it is too far south
		if np.amax(track_latlons[0]) < 5:
			print("too far south")
#			print(aew_track.latlon_list)
#			print(aew_track.time_list)
			finished_AEW_tracks_list.remove(aew_track)
			continue
		

	print("Total number of AEW tracks =", len(finished_AEW_tracks_list))
	print(radius_km)
	print(model_type)

	# create figure which shows all the tracks
	plot_points_map(finished_AEW_tracks_list,scenario_type,model_type)

	# save tracks to file
	tracks_file = open(model_type + '_' + scenario_type + '_AEW_tracks_' + year + '_May-Oct.obj', 'wb') # 'wb' means write binary, if just 'w' is used, a string is expected
	pickle.dump(finished_AEW_tracks_list, tracks_file)


if __name__ == '__main__':
	main()
