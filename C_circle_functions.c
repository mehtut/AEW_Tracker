#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define PI 3.14159265

// to compile on Cori use: cc -shared -Wl,-soname,C_circle_functions -o C_circle_functions.so -fPIC C_circle_functions.c

int main(void) {
}

void circle_avg_m(int p_levels, int lat_dim, int lon_dim, float var[p_levels][lat_dim][lon_dim], float smoothed_var[p_levels][lat_dim][lon_dim], 
	float lat[lat_dim][lon_dim], float radius, int lat_index_north, int lat_index_south, int lon_index_east, int lon_index_west, float total_lon_degrees)
{
//	printf("I'm in the function\n");
	float R=6371; // Earth radius in km
	int lat_index, lon_index, radius_index, lon_circle_index, cyclic_lon_index, p_index;  // all the indices for loops
	float cos_lat[lat_dim][lon_dim];
	int radius_gridpts;
	float tempv[p_levels], divider[p_levels];
	float lat1, lat2, angle_rad;
	int lon_gridpts;

//	printf("Calculate cos of lat\n");
	for(lat_index=0; lat_index<lat_dim; lat_index++) {
		for(lon_index=0; lon_index<lon_dim; lon_index++) {
//			printf("In loop\n");
//			printf("%f\n", lat[lat_index][lon_index]);
			cos_lat[lat_index][lon_index] = cos(0.0174533*lat[lat_index][lon_index]); // multiply by pi/180 = 0.0174533 to switch to radians
//			printf("%f\n", cos_lat[lat_index][lon_index]);
		}
	}

	// Get the number of gridpoints equivalent to the radius being used for the smoothing.
	// To convert the smoothing radius in km to gridpoints, multiply by the total number of 
	// longitude gridpoints = var.shape[2] for the whole globe divided by the circumference of the Earth = 2piR.
	// So radius_gridpts = radius*(longitude gridpoints/circumference of Earth)
	// Make radius_gridpts an int so it can be used as a loop index later.
	radius_gridpts = (int)radius*(lon_dim/((total_lon_degrees/360.0)*2*PI*R));
//	printf("I calculated radius_gridpts\n");
//	printf("%i\n", radius_gridpts);

//	printf("%f\n", lat[154][927]);
//	printf("Heading into loop\n");
	// loop over all longitude indices
	for(lon_index=lon_index_west; lon_index<lon_index_east; lon_index++) {  // longitude loop
//		printf("longitude loop\n");
//		printf("%i\n", lon_index);
		// loop over all latitude indices between lat_index_north and lat_index_south
		// do this instead of all the latitudes so that the circles that are drawn
		// do not go past the southern and norther latitude boundaries
		for(lat_index=lat_index_south; lat_index<lat_index_north; lat_index++) {  // latitude loop
//			printf("latitude loop\n");
//			printf("%i\n", lat_index);
			memset(tempv, 0.0, sizeof(tempv));
			memset(divider, 0.0, sizeof(divider));

			// latitude loop through the circle, starting at the bottom and moving to the top
			for(radius_index=-radius_gridpts; radius_index<radius_gridpts; radius_index++) {  // work way up circle
//				printf("radius loop\n");
//				printf("%i\n", radius_index);
				lat1 = lat[lat_index][lon_index];  // centre of circle
				lat2 = lat[lat_index+radius_index][lon_index]; // vertical distance from circle centre
				angle_rad = acos(-((sin(lat1*0.0174533)*sin(lat2*0.0174533))-cos(radius/R))/(cos(lat1*0.0174533)*cos(lat2*0.0174533)));  // haversine trig
//				printf("%f\n", lat1);
//				printf("%f\n", lat2);
//				printf("%f\n", angle_rad);

				// convert angle from radians to degrees and then from degrees to gridpoints
				// divide angle_rad by pi/180 = 0.0174533 to convert radians to degrees
				// multiply by lat.shape[1]/360 which is the lon gridpoints over the total 360 degrees around the globe
				// the conversion to gridpoints comes from (degrees)*(gridpoints/degrees) = gridpoints
				// lon_gridpts defines the longitudinal grid points for each lat
				lon_gridpts = (int)((angle_rad/0.0174533)*(lon_dim/360.0));

				// longitude loop through circle
				for(lon_circle_index=lon_index-lon_gridpts; lon_circle_index<lon_index+lon_gridpts; lon_circle_index++) {  // work across circle
//					printf("longitude circle loop\n");
					// the following conditionals handle the cyclic nature of the data. For example, if the circle spans the 0/360
					// line, the following conditionals will take an out of bounds longitude index and wrap it around to the beginning 
					// of the array (so 1483 would get wrapped to index 0, 1484 would get wrapped to index 1, etc.)
					cyclic_lon_index = lon_circle_index;
					if(cyclic_lon_index<0) { 
						cyclic_lon_index = cyclic_lon_index+lon_dim;
					}
					if(cyclic_lon_index>lon_dim-1) {
						cyclic_lon_index = cyclic_lon_index-lon_dim;
					}

					for(p_index=0; p_index<p_levels; p_index++) {
//						printf("pressure loop\n");
						tempv[p_index] = tempv[p_index] + (cos_lat[lat_index+radius_index][lon_index]*var[p_index][lat_index+radius_index][cyclic_lon_index]);
						divider[p_index] = divider[p_index] + cos_lat[lat_index+radius_index][lon_index];
						smoothed_var[p_index][lat_index][lon_index] =  tempv[p_index]/divider[p_index];
						//exit(0);
					}
				}  
			}
		}
	}

//	printf("Finished loop\n");

}