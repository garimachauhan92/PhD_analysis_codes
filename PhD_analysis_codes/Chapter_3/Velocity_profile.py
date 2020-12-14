import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import os
import math as math
import time
from scipy import integrate
from scipy import interpolate 
import random as random
import sys

# Constants
G = 4.301e-9

v_g = 8


exec(open("/home/garima/pleiades_icrar/Emission_Lines/Final Codes/Reading_data.py").read())

#path = '/home/garima/pleiades_icrar/Outputs/Edge_on_results/Central_galaxies/'


#all_galaxies = np.load(path + 'all_galaxies.npy')

#index_Massive = np.array(index_Massive)

#index_Massive = index_Massive[np.logical_not(np.isnan(all_galaxies))]


#l = int(input('Galaxy number :'))


# Function generating float sequence

def frange(x,y,jump):
	while x<y:
		yield x
		x += jump


def movingaverage(interval, window_size):

	window = np.ones(int(window_size))/float(window_size)
	return np.convolve(interval, window, 'same')



loop_length = len(index_Massive)  # Total Number of galaxies being considered


def velocity_profile(k):     # K is the index of galaxy, sini - sine of angle of inclination [theta dataset]
	
	## Making the Radius Channel - -------------------------------------------------------------------------

	start_all = time.clock() 
	
	xmax = 1
	dx = 0.005
	nx = int(xmax/dx)
	x = np.zeros(shape = nx)


	for i in range(1,nx):
		x[i - 1] = (i - 0.5) * dx


	r_x = (5*R_disk_2[k])*x
	
	R = R_halo[k]
	R_bulge = R_bulge_2[k]

	## Making Velocity Profile------------------------------------------------------------------------


	start_velocity = time.clock()


	V_bulge_sqr = np.zeros(shape = nx)
	numerator_b = np.zeros(shape = nx)
	denominator_b = np.zeros(shape = nx)


	for i in range(1,nx):

		numerator_b[i] = ((c_bulge[index_Massive[k]]*(r_x[i]/R))**2)*(c_bulge[index_Massive[k]])

		if (c_bulge[index_Massive[k]] == 0 or r_x[i]/R == 0):
			
			V_bulge_sqr[i] = 0

		else:
			denominator_b[i] = (1 + (c_bulge[index_Massive[k]]*(r_x[i]/R))**2)**1.5
			V_bulge_sqr[i] = (((G*M_bulge[index_Massive[k]])/R)*(numerator_b[i]/denominator_b[i]))



	V_halo_sqr = np.zeros(shape = nx)
	numerator_h = np.zeros(shape = nx)


	for i in range(1,nx):

		denominator_h = np.log(1 + c_halo[index_Massive[k]]) - ((c_halo[index_Massive[k]]/(1 + c_halo[index_Massive[k]])))


		numerator_h[i] = np.log(1 + c_halo[index_Massive[k]]*(r_x[i]/R)) - ((c_halo[index_Massive[k]]*(r_x[i]/R))/(1 + c_halo[index_Massive[k]]*(r_x[i]/R)))


		if r_x[i]/R == 0:
			V_halo_sqr[i] = 0
		else:	
			V_halo_sqr[i] = (((G*M_halo[index_Massive[k]])/H_Rv[index_Massive[k]])*(numerator_h[i]/((r_x[i]/R)*denominator_h)))

		


	V_disk_sqr = np.zeros(shape= nx)
	numerator_d = np.zeros(shape = nx)
	denominator_d = np.zeros(shape = nx)


	for i in range(1,nx):

		if (c_disk[index_Massive[k]] == 0 or (r_x[i]/R) == 0):
		
			numerator_d[i] = 0
	
		else:	
			numerator_d[i] = c_disk[index_Massive[k]]*4.8*c_disk[index_Massive[k]]*(np.exp((-0.35*c_disk[index_Massive[k]]*(r_x[i]/R)) - (3.5/(c_disk[index_Massive[k]]*(r_x[i]/R)))))
		
		if (c_disk[index_Massive[k]] == 0 or (r_x[i]/R) == 0):
	
			V_disk_sqr[i] = 0
		
		else:
			denominator_d[i] = (c_disk[index_Massive[k]]*(r_x[i]/R)) + (c_disk[index_Massive[k]]*(r_x[i]/R))**(-2) + 2*((c_disk[index_Massive[k]]*(r_x[i]/R)))**(-0.5)

			V_disk_sqr[i] = (((G*M_disk[index_Massive[k]])/R)*(numerator_d[i]/denominator_d[i]))




	V_circ = np.sqrt(V_disk_sqr + V_halo_sqr + V_bulge_sqr)

	V_disk = np.sqrt(V_disk_sqr)
	V_halo = np.sqrt(V_halo_sqr)
	V_bulge = np.sqrt(V_bulge_sqr)



	max_Vdisk = max(np.sqrt(V_disk_sqr))
	max_Vhalo = max(np.sqrt(V_halo_sqr))
	max_Vbulge = max(np.sqrt(V_bulge_sqr))

	r_disk = r_x[V_disk == max_Vdisk]
	r_halo = r_x[V_halo == max_Vhalo]
	r_bulge = r_x[V_bulge == max_Vbulge]

	V_flat = V_circ[r_x > 4*R_disk_2[k]]
	#r_x_flat = r_x[r_x  > 3*R_disk_2[k]]
	r_x_flat = r_x[r_x  > 4*R_disk_2[k]]
	



	if max(r_x) < R_bulge:
		max_Vcirc = max(V_circ[r_x > R_bulge/2])
	else:
		max_Vcirc = max(V_circ[r_x > R_bulge])	

	
	max_Vflat = max(V_flat)
	mean_Vflat = np.mean(V_flat)

	r_circ = r_x[V_circ == max_Vcirc]
	r_flat_max = r_x_flat[V_flat == max_Vflat]


	
	plt.figure(figsize = (12,12))
	plt.plot(r_x[0:nx-1],np.sqrt(V_halo_sqr[0:nx-1]) , '--r', linewidth = 2)
	plt.plot(r_x[0:nx-1], np.sqrt(V_disk_sqr[0:nx-1]),'--b', linewidth = 2)
	plt.plot(r_x[0:nx-1], np.sqrt(V_bulge_sqr[0:nx-1]), '--g', linewidth = 2)
	plt.plot(r_x[0:nx-1], V_circ[0:nx-1], 'k', linewidth=2)
	#plt.plot(r_x_flat, V_flat, linewidth = 4, alpha = 0.5, linestyle = '-', color = 'm')
	#plt.xlim(0, 0.05)
	plt.xlabel('Radius (Mpc)', size = 30)
	plt.ylabel('Circular Velocity (km/s)', size = 30)
	plt.legend(['$V^{Halo}_{Circ}$', '$V^{Disk}_{Circ}$', '$V^{Bulge}_{Circ}$', '$V^{Total}_{Circ}$'], fontsize=20, loc=0, frameon = False)

	# plt.annotate('$V_{disk}$', xy = (r_disk,max_Vdisk), fontsize = 15)
	# plt.annotate('$V_{halo}$', xy = (r_halo,max_Vhalo), fontsize = 15)
	# plt.annotate('$V_{max}$', xy = (r_circ,max_Vcirc), fontsize = 15)
	# #plt.annotate('$V_{flat}$', xy = (r_flat_max,max_Vflat), fontsize = 15)
	# plt.annotate('$V_{bulge}$', xy = (r_bulge,max_Vbulge), fontsize = 15)
	
	plt.xticks(fontsize = 20)
	plt.yticks(fontsize = 20)
	
	plt.title('Velocity Profile', size=30)
	plt.show()

	return V_circ


for l in [10,11,12,13,14,15]:
	V_circ = velocity_profile(l)