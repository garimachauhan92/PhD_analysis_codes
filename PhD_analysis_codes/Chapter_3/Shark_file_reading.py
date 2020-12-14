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
import pandas as pd



###---------------------------------------------------------------------------------------------------------------------------
# Constants
###--------------------------------------------------------------------------------------------------------------------------

G             = 4.301e-9
G_cgs         = 6.674e-8

M_solar_2_g   = 1.99e33
Mpc_2_cm      = 3.086e24
k_boltzmann   = 1.3807e-16

cm_2_mpc      = 3.24e-25
g_2_M_solar   = 0.5e-33

h 			  = 0.73

dt  = np.dtype(int)
df  = np.dtype(float)


#*********************************************************************************************************************************

class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)


def movingaverage(interval, window_size):

	window = np.ones(int(window_size))/float(window_size)
	return np.convolve(interval, window, 'same')

#************************************************************************************************************************************
def flux_danail(M_cold_gas,M_cold_gas_mol, distance , z):

	fHI		=  1.4204	
	mass 	= (M_cold_gas - M_cold_gas_mol)/h*0.74 
	dis_cor	= distance/h
	Luminosity_HI = 6.27e-9*mass
	flux 	=	Luminosity_HI/1.04e-3/(dis_cor)**2/(1+z)/fHI

	return flux



def flux_catinella(M_cold_gas,M_cold_gas_mol, distance , z):

	fHI		=  1.4204	
	mass 	= (M_cold_gas - M_cold_gas_mol)/h*0.74 
	dis_cor	= distance/h
	
	flux 	= mass*(1+z)/dis_cor**2/2.356*10**(-5)


	return flux

#******************************************************************************************************************************************



def line_emission(R_halo,R_disk,R_HI,R_bulge,M_cold_gas,M_stars_disk,M_halo,M_disk,M_bulge,c_halo,c_disk,c_bulge,c_HI, sini):     # K is the index of galaxy, sini - sine of angle of inclination [theta dataset]
	
	## adresses 
	# if M_halo > 10**10:
	# 	v_g = 10
	# else:	
	# 	v_g = 20

	v_g = 10

	## Making the Radius Channel - -------------------------------------------------------------------------

	
	xmax = 1
	dx = 0.004
	nx = int(xmax/dx)
	x = np.zeros(shape = nx)


	for i in range(1,nx + 1):
		x[i - 1] = (i - 0.5) * dx


	r_x = (R_halo/1.67)*x
	
	R = R_halo/1.67

	if R_HI == 0:
		R_HI = 3*R_disk/1.67	

		c_HI = R_halo/R_HI


	if R_disk == 0:
		return  None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

	if R_halo == 0:
		return  None, None, None, None, None, None, None, None, None, None, None, None, None, None, None




	## Making Surface Density Profile-------------------------------------------------------------------

	Gas_galaxy = np.zeros(shape = nx)
	P_star_galaxy = np.zeros(shape = nx)
	P_gas_galaxy = np.zeros(shape = nx)
	v_star_galaxy = np.zeros(shape=nx)
	hstar_galaxy = R_HI/7.3*Mpc_2_cm

	M_cold_gas_cgs = M_cold_gas*M_solar_2_g
	M_stars_disk_cgs = M_stars_disk*M_solar_2_g
	R_HI_cgs = R_HI*Mpc_2_cm

	#M_stars_bulge_cgs = M_stars_bulge*M_solar_2_g
	#R_bulge_cgs = R_bulge*Mpc_2_cm
	
	r_x_cgs = r_x*Mpc_2_cm



	A_cgs = -r_x_cgs/R_HI_cgs

	for i in range(0,nx):

		Gas_galaxy[i] = M_cold_gas_cgs/(2*np.pi*(R_HI_cgs)**2)*np.exp(A_cgs[i])

		
		P_star_galaxy[i] = ((M_stars_disk_cgs)/(2*np.pi*(R_HI_cgs)**2)*np.exp(A_cgs[i]))

		P_gas_galaxy[i] = (M_cold_gas_cgs/(2*np.pi*(R_HI_cgs)**2))*np.exp(A_cgs[i])

		v_star_galaxy[i] = np.sqrt(np.pi*G_cgs*hstar_galaxy*P_star_galaxy[i])


	P_ext_galaxy = np.pi/2*G_cgs*Gas_galaxy
	P_ext_galaxy = P_ext_galaxy*(P_gas_galaxy + (v_g*1e5/v_star_galaxy)*P_star_galaxy)
	P_ext_galaxy = P_ext_galaxy/k_boltzmann

	P_ext = max(P_ext_galaxy)
	
	P_0 = 3.7*10e4


	R_c_galaxy = (P_ext_galaxy/P_0)**0.8

	f_H1_galaxy = 1/(1 + R_c_galaxy)
	f_H2_galaxy = R_c_galaxy/(1 + R_c_galaxy)



	surface_H1_galaxy = f_H1_galaxy*Gas_galaxy
	H1 = np.trapz(r_x_cgs*surface_H1_galaxy, dx=dx*Mpc_2_cm)
	surface_H1_galaxy = surface_H1_galaxy/H1/cm_2_mpc**2

	surface_H2_galaxy = f_H2_galaxy*Gas_galaxy
	H2 = np.trapz(r_x_cgs*surface_H2_galaxy,dx = dx*Mpc_2_cm) 
	surface_H2_galaxy = surface_H2_galaxy/H2/cm_2_mpc**2





	#---------------------------------------------------------------------------------------------------------

	## Making Velocity Profile------------------------------------------------------------------------


	A = r_x/R

	V_bulge_sqr = np.zeros(shape = nx)
	numerator_b = np.zeros(shape = nx)
	denominator_b = np.zeros(shape = nx)


	for i in range(0,nx):

		numerator_b[i] = ((c_bulge*A[i])**2)*(c_bulge)

		if (c_bulge == 0 or r_x[i]/R == 0 or R_bulge == 0):
			
			V_bulge_sqr[i] = 0

		else:
			denominator_b[i] = (1 + (c_bulge*A[i])**2)**1.5
			V_bulge_sqr[i] = (((G*M_bulge)/R)*(numerator_b[i]/denominator_b[i]))



	V_halo_sqr = np.zeros(shape = nx)
	numerator_h = np.zeros(shape = nx)


	for i in range(0,nx):

		denominator_h = np.log(1 + c_halo) - ((c_halo/(1 + c_halo)))


		numerator_h[i] = np.log(1 + c_halo*A[i]) - ((c_halo*A[i])/(1 + c_halo*A[i]))


		if r_x[i]/R == 0:
			V_halo_sqr[i] = 0
		else:	
			V_halo_sqr[i] = (((G*M_halo)/R)*(numerator_h[i]/(A[i]*denominator_h)))

		


	V_disk_sqr = np.zeros(shape= nx)
	numerator_d = np.zeros(shape = nx)
	denominator_d = np.zeros(shape = nx)


	for i in range(0,nx):

		if (c_disk == 0 or A[i] == 0):
		
			numerator_d[i] = 0
	
		else:	
			numerator_d[i] = c_disk + 4.8*c_disk*(np.exp((-0.35*c_disk*A[i]) - (3.5/(c_disk*A[i]))))
		
		if (c_disk == 0 or A[i] == 0):
	
			V_disk_sqr[i] = 0
		
		else:
			denominator_d[i] = (c_disk*A[i]) + (c_disk*A[i])**(-2) + 2*((c_disk*A[i]))**(-0.5)

			V_disk_sqr[i] = (((G*M_disk)/R)*(numerator_d[i]/denominator_d[i]))



	

	V_HI_sqr = np.zeros(shape= nx)
	numerator_d = np.zeros(shape = nx)
	denominator_d = np.zeros(shape = nx)


	for i in range(0,nx):

		if (c_HI == 0 or A[i] == 0):
		
			numerator_d[i] = 0
	
		else:	
			numerator_d[i] = c_HI + 4.8*c_HI*(np.exp((-0.35*c_HI*A[i]) - (3.5/(c_HI*A[i]))))
		
		if (c_HI == 0 or A[i] == 0):
	
			V_HI_sqr[i] = 0
		
		else:
			denominator_d[i] = (c_HI*A[i]) + (c_HI*A[i])**(-2) + 2*((c_HI*A[i]))**(-0.5)

			V_HI_sqr[i] = (((G*M_cold_gas*0.73)/R)*(numerator_d[i]/denominator_d[i]))




	

	V_circ = np.sqrt(V_disk_sqr + V_halo_sqr + V_bulge_sqr + V_HI_sqr)

	

	max_Vcirc = max(V_circ)

	max_Vdisk = max(np.sqrt(V_disk_sqr))
	max_Vhalo = max(np.sqrt(V_halo_sqr))
	max_Vbulge = max(np.sqrt(V_bulge_sqr))
	max_VHI		= max(np.sqrt(V_HI_sqr))

	 
	
	
	
	# if max(r_x) < R_bulge and R_bulge > R_disk:
	# 	max_Vcirc = max(V_circ)
	# elif max(r_x) < R_bulge:
	# 	max_Vcirc = max(V_circ[r_x > R_bulge/2])
	# else:
	# 	max_Vcirc = max(V_circ[r_x > R_bulge])	


	if np.isnan(max_Vcirc):
         
    		return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

		
	if max(V_circ) > 5000:

		return  None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
		#return None, None, None, None

	#-------------------------------------------------------------------------------------------------------------



	################################################################################################################3

	#  STARTING THE CONVULATION #################

	# NOte - Smoothing factor for Random orientation is 20
	#		 Smoothing factor for Edge on is 50





	## Making Velocity Channel----------------------------------------------------------------------------------

	vmax = (max_Vcirc + v_g*2)*1.2
	nv = 300
	dv = vmax/nv
	v_x = np.zeros(shape = nv)

	for i in range(1,nv):
		v_x[i - 1] = (i - 0.5)*dv 

	#------------------------------------------------------------------------------------------------------------

	## Making the smoothing filter-------------------------------------------------------------------------------

	disp = v_g/dv



	nfilter = int(disp)

	filter_x = np.array([n for n in range(-nfilter,nfilter) ])

	filter_final = np.zeros(shape = len(filter_x))

	for j in range(0,len(filter_x)):
		for i in filter_x:

			filter_final[j] = np.exp(-(i**2)/disp**2/2)

	filter_final = filter_final/sum(filter_final)

	#----------------------------------------------------------------------------------------------------------

	## Calculating the flux using the surface density calculated earlier

	s = np.zeros(shape = nv)

	for j in range(1,nx):

		y = v_x/(V_circ[j-1]*sini)

		dy = dv/(V_circ[j-1]*sini)

		f = r_x[j-1]*surface_H1_galaxy[j-1]/np.pi/dv

		for i in range(1,nv):
			ym = y[i - 1] - dy/2
			yp = y[i - 1] + dy/2

			if yp > 1:
				yp = 1

			if ym > 1:
				continue

			s[i - 1] = s[i - 1] + f*(np.arcsin(yp) - np.arcsin(ym)) 

	#s = s/sum(dv*s)
	#print(np.trapz(s, dx = dv))

	#-------------------------------------------------------------------------------------------------------------

	# Smooth lines by velocity dispersion

	s_2 = np.zeros(shape = nv)

	for i in range(1,nv):

		for j in range(0, len(filter_x)):

			yo = i+filter_x[j]

			if (yo < 1):

				yo = 1 - yo

			if (yo <= nv):

				s_2[i -1] = s_2[i - 1] + s[yo - 1]*filter_final[j]


	s_2 = s_2/sum(dv*s_2)/2			




	v_x_n = -v_x

	trial = movingaverage(s_2,50)
	
	final_v_x = np.hstack((v_x_n[0:nv-2], v_x[0:nv-2]))

	final_s = np.hstack((s_2[0:50], trial[50:nv-2]))
	final_s = np.hstack((final_s,s_2[0:50]))
	final_s = np.hstack((final_s, trial[50:nv-2]))


	xs,ys = zip(*sorted(zip(final_v_x, final_s)))



	#------------------------------------------------------------------------------------------------------------

	# Calculating the W50 and W20 and W_peak

	#--------------------------------------------------------------------------------------------------------------

	s_peak = max(trial)

	s_central = trial[0]

	for i in range(nv-1,1,-1):

		if trial[i-1] == s_peak :
			
			W_peak = v_x[i-1]*2
			break

	s_high = i

	starget_50 = 0.5*s_peak

	for i in range(nv-1, 1, -1):

		if trial[i - 1] > starget_50:
			f = (trial[i-1] - starget_50)/(trial[i-1] - trial[i])
			W50 = ((1 - f)*v_x[i-1]+f*v_x[i])*2
			break
	
	s_50 = i



	starget_20 = 0.2*s_peak

	for i in range(nv-1, 1, -1):

		if trial[i-1] > starget_20:
			f = (trial[i-1] - starget_20)/(trial[i-1] - trial[i])
			W20 = ((1 - f)*v_x[i-1]+f*v_x[i])*2
			break
	
	s_20 = i

	
	
	

		

	
	#-------------------------------------------------------------------------------------------------------------
	try:
		return xs, ys, max_Vhalo, max_Vdisk, max_Vcirc, max_Vbulge, max_VHI, W_peak, W50, W20, s_peak, starget_50, starget_20, V_circ, r_x
	except (UnboundLocalError):
		return  None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


def empty_dataset(file_name):
	em_line			=	file_name.create_group("Emission_Line")
	em_line.create_dataset("v_x",(loop_length,596), dtype = 'float32')
	em_line.create_dataset("s_normalized",(loop_length,596), dtype = 'float32')
	em_line.create_dataset("s_peak", data = s_high)
	em_line.create_dataset("s_50",data = s_50)
	em_line.create_dataset("s_20",data = s_20)

	vel_pro			=	file_name.create_group("Velocity_Profile")
	vel_pro.create_dataset("r_x",(loop_length,250), dtype = 'float32')
	vel_pro.create_dataset("V_max_circ_profile",(loop_length,250), dtype = 'float32')
	
	vel_cal			=	file_name.create_group("Velocity_Calculated")
	vel_cal.create_dataset("V_halo", (loop_length,), dtype = 'float32')
	vel_cal.create_dataset("V_disk", (loop_length,), dtype = 'float32')
	vel_cal.create_dataset("V_bulge", (loop_length,), dtype = 'float32')
	vel_cal.create_dataset("V_HI", (loop_length,), dtype = 'float32')
	vel_cal.create_dataset("V_max_circ", (loop_length,), dtype = 'float32')
	vel_cal.create_dataset("W_peak", (loop_length,), dtype = 'float32')
	vel_cal.create_dataset("W_20", (loop_length,), dtype = 'float32')
	vel_cal.create_dataset("W_50",(loop_length,), dtype = 'float32')

	file_name.create_dataset("Theta", data = theta)

	flux_gal		= 	file_name.create_group("flux")
	flux_gal.create_dataset("flux", data = flux_lines)

def other_props_assigning(file_name):


	central = file_name.create_dataset('is_central', data = gal_type)

	G_Radius	=	file_name.create_group('Radius_file')

	G_Radius.create_dataset('R_vir', data = R_S_halo)
	G_Radius.create_dataset('R_disk_star', data = rstar_disk)
	G_Radius.create_dataset('R_bulge_star', data = rstar_bulge)
	G_Radius.create_dataset('R_disk_gas', data = rgas_disk)
	G_Radius.create_dataset('R_bulge_gas', data = rgas_bulge)

	#*************************************************************************************************


	G_Velocity = file_name.create_group('Velocity_File')

	G_Velocity.create_dataset('Virial_velocity', data = vmax_halo)
	G_Velocity.create_dataset('Max_Circular_Velocity', data = vmax_gal)

	#**************************************************************************************************



	#**************************************************************************************************

	G_Mass		=	file_name.create_group('Mass')

	G_Mass.create_dataset('M_gas_disk', data = mgas_disk)
	G_Mass.create_dataset('M_gas_bulge', data = mgas_bulge_file)
	G_Mass.create_dataset('M_Host_Halo', data = mvir_hosthalo)
	G_Mass.create_dataset('M_Sub_Halo', data = mvir_subhalo)
	G_Mass.create_dataset('M_stars_bulge', data = mstars_bulge)
	G_Mass.create_dataset('M_stars_disk', data = mstars_disk)
	G_Mass.create_dataset('M_stars_tot', data = mstars)
	
	#*****************************************************************************************************

	G_cals		= file_name.create_group('All_else')

	G_cals.create_dataset('Ratio_gas_to_stars', data = mass_gas)
	G_cals.create_dataset('log_Mass_stars', data = mass_galaxies)
	G_cals.create_dataset('B_T Ratio', data = B_T)
	G_cals.create_dataset('C_halo', data = c_halo)
	G_cals.create_dataset('C_disk', data = c_disk)
	G_cals.create_dataset('C_bulge', data = c_bulge)
	G_cals.create_dataset('C_HI', data = c_HI)
	G_cals.create_dataset('Distance_Modulus', data = distance)
	G_cals.create_dataset('z_cos', data = zcos)
	G_cals.create_dataset('z_obs', data = zobs)
	#G_cals.create_dataset('Distance', data = Distance[index_z])
	G_cals.create_dataset('dec', data = dec)
	G_cals.create_dataset('ra', data = ra)

#********************************************************************************************************

path_write 			= 	'/mnt/su3ctm/gchauhan/devils_shark/'

##------------------------------------------------------------------------------------



f = h5.File("/mnt/su3ctm/dobreschkow/Stingray/mocksky_DEVILS.hdf5",'r')

snapshot 		= np.array(f['Galaxies/snapshot'], dtype = dt)
subsnapshot		= np.array(f['Galaxies/subsnapshot'], dtype = dt)
id_galaxy_dan 	= np.array(f['Galaxies/id_galaxy_sam'], dtype = dt)
id_halo_dan 	= np.array(f['Galaxies/id_halo_sam'], dtype = dt)
dec 			= np.array(f['Galaxies/Dec'], dtype = df )
ra 				= np.array(f['Galaxies/RA'], dtype = df )
inclination		= np.array(f['Galaxies/inclination'], dtype = df )
mstars 			= np.array(f['Galaxies/mstars'], dtype = df)
mvir_hosthalo	= np.array(f['Galaxies/mvir_hosthalo'], dtype = df)
mvir_subhalo	= np.array(f['Galaxies/mvir_subhalo'], dtype = df)
rgas_disk		= np.array(f['Galaxies/rgas_disk'], dtype = df)
rstar_bulge 	= np.array(f['Galaxies/rstar_bulge'], dtype = df)
rstar_disk 		= np.array(f['Galaxies/rstar_disk'], dtype = df)
gal_type	 	= np.array(f['Galaxies/type'], dtype = dt)
zcos 			= np.array(f['Galaxies/zcos'], dtype = df)
zobs 			= np.array(f['Galaxies/zobs'], dtype = df)
distance 		= np.array(f['Galaxies/dc'], dtype = df)




##----------------------------------------------------------------------------------------------------------------
### From Claudia Files
##_----------------------------------------------------------------------------------------------------------------

c_halo 						= []
matom_bulge 				= []
matom_disk					= []
mgas_bulge					= []
mgas_disk					= []
mstars_bulge 				= []
mstars_disk 				= []
rgas_bulge					= []
vmax_gal					= []
vmax_halo					= []




for k,i, j in zip(snapshot,subsnapshot, id_halo_dan):

	print(k)

	hf = h5.File("/mnt/su3ctm/clagos/SHArk_Out/medi-SURFS/SHArk/%s/%s/galaxies.hdf5" %(k,i), 'r')

	yo = np.array(hf['Galaxies/id_halo'], dtype = dt)
	
	indices = np.where(np.in1d(j, yo))[0]
	#indices = np.where(j == yo)[0]

	# id_halo_sorted = np.argsort(yo)
	# ypos 		   = np.searchsorted(yo[id_halo_sorted],id_halo_dan)
	# indices		   =  id_halo_sorted[ypos]


	c_halo_file					= np.array(hf['Galaxies/cnfw_subhalo'], dtype = dt)
	matom_bulge_file			= np.array(hf['Galaxies/matom_bulge'], dtype = df)
	matom_disk_file				= np.array(hf['Galaxies/matom_disk'], dtype = df)
	mgas_bulge_file				= np.array(hf['Galaxies/mgas_bulge'], dtype = df)
	mgas_disk_file				= np.array(hf['Galaxies/mgas_disk'], dtype = df)
	mstars_bulge_file			= np.array(hf['Galaxies/mstars_bulge'], dtype = df)
	mstars_disk_file			= np.array(hf['Galaxies/mstars_disk'], dtype = df)
	rgas_bulge_file				= np.array(hf['Galaxies/rgas_bulge'], dtype = df)
	vmax_gal_file				= np.array(hf['Galaxies/vmax_subhalo'], dtype = df)
	vmax_halo_file				= np.array(hf['Galaxies/vvir_hosthalo'], dtype = df)

	
	c_halo 						= np.append(c_halo, c_halo_file[indices]) 
	matom_bulge 				= np.append(matom_bulge, matom_bulge_file[indices]) 
	matom_disk					= np.append(matom_disk, matom_disk_file[indices]) 
	mgas_bulge					= np.append(mgas_bulge, mgas_bulge_file[indices]) 
	mgas_disk					= np.append(mgas_disk, mgas_disk_file[indices]) 
	mstars_bulge 				= np.append(mstars_bulge, mstars_bulge_file[indices]) 
	mstars_disk 				= np.append(mstars_disk, mstars_disk_file[indices]) 
	rgas_bulge					= np.append(rgas_bulge, rgas_bulge_file[indices]) 
	vmax_gal					= np.append(vmax_gal, vmax_gal_file[indices]) 
	vmax_halo					= np.append(vmax_halo, vmax_halo_file[indices]) 



R_S_halo			= 	vmax_halo/10/67.77

print(R_S_halo[0:1000])
print(vmax_halo[0:1000])




c_disk 	= np.zeros(len(rstar_disk))

c_bulge = np.zeros(len(rstar_bulge))

c_HI    = np.zeros(len(rgas_disk))

for i in range(len(rgas_disk)):
	if rgas_disk[i] == 0:
		c_HI[i] = 0
	else:
		c_HI[i] 	= 	R_S_halo[i]/(rgas_disk[i]/1.67)


for i in range(len(rstar_disk)):
	if rstar_disk[i] == 0:
		c_disk[i] = 0
	else:
		c_disk[i]	=	R_S_halo[i]/(rstar_disk[i]/1.67)

for i in range(len(rstar_bulge)):
	if rstar_bulge[i] == 0:
		c_bulge[i] = 0
	else:
		c_bulge[i]	=	R_S_halo[i]/(1.7*rstar_bulge[i]/1.67)






M_bulge 			= 	mstars_bulge + mgas_bulge  					# Total mass of the bulge
				
M_disk 				= 	mstars_disk + mgas_disk						# Total mass of the disk	

mass_gas 			= 	mgas_disk/mstars

mass_galaxies 		= 	np.log10(mstars)

B_T 				=	mstars_bulge/mstars

theta 				= np.sin(inclination*0.01745329252)


loop_length 	= len(mstars)

print('loop-length', loop_length)


v_x             = {}
r_x             = {}
s_final         = {}
V_circ_all      = {}



W_peak			= np.zeros(loop_length)
W_50            = np.zeros(loop_length)
W_20            = np.zeros(loop_length)
V_halo          = np.zeros(loop_length)
V_disk          = np.zeros(loop_length)
V_circ          = np.zeros(loop_length) 
V_bulge         = np.zeros(loop_length) 
V_HI 	        = np.zeros(loop_length) 
s_high          = np.zeros(loop_length)
s_50            = np.zeros(loop_length)
s_20            = np.zeros(loop_length)
flux_lines		= np.zeros(loop_length)


empty_HI 		= np.zeros(loop_length)

for i in range(len(mstars)):
	if i%200 == 0: print(i)
	
	v_x[i], s_final[i], V_halo[i], V_disk[i], V_circ[i], V_bulge[i],V_HI[i] ,W_peak[i], W_50[i] , W_20[i], s_high[i], s_50[i], s_20[i], V_circ_all[i], r_x[i] 	=	\
	line_emission(R_S_halo[i],rstar_disk[i],
		rgas_disk[i],rstar_bulge[i],mgas_disk[i],
		mstars_disk[i],mvir_hosthalo[i],M_disk[i],
		M_bulge[i],c_halo[i],c_disk[i]
		,c_bulge[i],c_HI[i],theta[i])

	flux_lines[i]	=	flux_catinella(matom_disk[i] + matom_bulge[i], empty_HI[i],distance[i],zcos[i])



hf = h5.File(path_write + 'Devils_area.h5', 'w')

empty_dataset(hf)


for i in range(loop_length):
	if v_x[i] == None:
		continue
	else:
		hf["Emission_Line/v_x"][i,] = v_x[i]
		hf["Emission_Line/s_normalized"][i,] = s_final[i]
		hf["Velocity_Profile/r_x"][i,] = r_x[i]
		hf["Velocity_Profile/V_max_circ_profile"][i,] = V_circ_all[i]
		
		hf["Velocity_Calculated/V_halo"][i] 		= V_halo[i]
		hf["Velocity_Calculated/V_disk"][i] 		= V_disk[i]
		hf["Velocity_Calculated/V_bulge"][i]		= V_bulge[i]
		hf["Velocity_Calculated/V_max_circ"][i]		= V_circ[i]
		hf["Velocity_Calculated/V_HI"][i]			= V_HI[i]
		hf["Velocity_Calculated/W_peak"][i]			= W_peak[i]
		hf["Velocity_Calculated/W_50"][i]			= W_50[i]
		hf["Velocity_Calculated/W_20"][i]			= W_20[i]



other_props_assigning(hf)
hf.close()








