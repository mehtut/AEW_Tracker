# AEW_Tracker
African easterly wave tracking algorithm

AEW Tracker Readme

The African easterly wave (AEW) tracking algorithm has multiple components. The main program that is executed is AEW_Tracks.py. The arguments model type (WRF, CAM5 or ERA5 data), scenario type (Historical, late_century, and Plus30), and the year are parsed from the command line. To run the program, type: python AEW_Tracks.py --model 'WRF' --scenario 'late_century' --year '2010'

The C smoothing program needs to be compiled before running the AEW_Tracks.py. To compile on Cori, type: cc -shared -Wl,-soname,C_circle_functions -o C_circle_functions.so -fPIC C_circle_functions.c

The radius used for smoothing and searching for repeat tracks is set based on model type in AEW_Tracks.py. This can be adjusted if necessary. The minimum threshold to qualify as a potential AEW track point is set in AEW_Tracks.py, and can also be changed. 

To run the tracker, the locations of the data in Pull_data.py need to be verified and then updated if the data locations are different. 
