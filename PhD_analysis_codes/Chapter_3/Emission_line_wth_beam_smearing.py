import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as scipy
import pandas as pd
import csv as csv
import mpl_toolkits.axisartist.floating_axes as floating_axes
import numpy as np
from matplotlib.projections import PolarAxes
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
#import matplotlib.pyplot as plt
import Plotting_emission_line_prop as plotting
import mpl_toolkits.axisartist.angle_helper as angle_helper
from mpl_toolkits.axisartist.grid_finder import (FixedLocator, MaxNLocator,DictFormatter)
from matplotlib.transforms import Affine2D
import random as random

import matplotlib as mpl

figSize = (12,8)
labelFS = 20
tickFS = 20
titleFS = 20
textFS = 20
legendFS = 20
linewidth_plot = 3
markersize_lines_plot = 10

# Adjust axes
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['axes.axisbelow'] = 'line'

# Adjust Fonts
mpl.rcParams['font.family'] = "sans-serif"
mpl.rcParams['text.usetex'] = False
mpl.rcParams['text.latex.unicode'] = True
mpl.rcParams['mathtext.fontset'] = "dejavusans"


mpl.rcParams['axes.titlesize'] = titleFS
mpl.rcParams['axes.labelsize'] = labelFS
mpl.rcParams['xtick.labelsize'] = tickFS
mpl.rcParams['ytick.labelsize'] = tickFS
mpl.rcParams['legend.fontsize'] = legendFS

# Adjust line-widths
mpl.rcParams['lines.linewidth'] = linewidth_plot
mpl.rcParams['lines.markersize'] = markersize_lines_plot

#Adjust Legend
mpl.rcParams['legend.markerscale'] = 1

# Adjust ticks
for a in ['x','y']:
	mpl.rcParams['{0}tick.major.size'.format(a)] = 5.0
	mpl.rcParams['{0}tick.minor.size'.format(a)] = 2.5
	
	mpl.rcParams['{0}tick.major.width'.format(a)] = 1.0
	mpl.rcParams['{0}tick.minor.width'.format(a)] = 1.0
	
	mpl.rcParams['{0}tick.direction'.format(a)] = 'in'
	mpl.rcParams['{0}tick.minor.visible'.format(a)] = True

mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True

# Adjust figure and subplots
mpl.rcParams['figure.figsize'] = figSize
mpl.rcParams['figure.subplot.left'] = 0.1
mpl.rcParams['figure.subplot.right'] = 0.96
mpl.rcParams['figure.subplot.top'] = 0.96
mpl.rcParams['savefig.bbox'] = 'tight'


###--------------------------------------------------------------------------------------------------------------------------------
# Defining data_type and constants
###-------------------------------------------------------------------------------------------------------------------------------

dt  = np.dtype(float)
G   = 4.301e-9
h   = 0.67
M_solar_2_g   = 1.99e33

c = 299792 #km/s

#***********************************************************************************************************************************
# Define lower and upper percentile functions
#***********************************************************************************************************************************

def percentile_lower(arr,q=15.87): 
	arr = arr[~np.isnan(arr)]
	
	if (len(arr) != 0): return (np.percentile(a=arr,q=q))
	else: return None
	
def percentile_upper(arr,q=84.13): 
	arr = arr[~np.isnan(arr)]
	
	if (len(arr) != 0): return (np.percentile(a=arr,q=q))
	else: return None


def percentile_lower_sfr(arr,q=5.87): 
	arr = arr[~np.isnan(arr)]
	
	if (len(arr) != 0): return (np.percentile(a=arr,q=q))
	else: return None
	
def percentile_upper_sfr(arr,q=95.13): 
	arr = arr[~np.isnan(arr)]
	
	if (len(arr) != 0): return (np.percentile(a=arr,q=q))
	else: return None


# Flux Calcuation
def flux_catinella(M_cold_gas,M_cold_gas_mol, distance , z):

	fHI     =  1.4204
	mass    = (M_cold_gas - M_cold_gas_mol)/h*0.74 
	dis_cor = distance/h
	
	flux    = mass*(1+z)/dis_cor**2/2.356*10**(-5)

	
	

	return flux


#***********************************************************************************************************************************
#### Running SHArk Properties
#****************************************************************************************************************************************


######

f = h5.File("/media/garima/Seagate Backup Plus Drive/pleiades_icrar/Shark/Old_ALFALFA_Files/SHArk_ALFALFA/micro_alfalfa_virial_velocity_without_limit.h5", 'r')

ra					=	np.array(f['All_else/ra'], dtype = dt)					# B_T Ratio of centrals
dec					=	np.array(f['All_else/dec'], dtype = dt)					# B_T Ratio of centrals

is_central			=	np.array(f['is_central'], dtype = dt)			
Distance			=	np.array(f['All_else/Distance'], dtype = dt)					# Mag of the distance (needs conversion)
z_obs				=	np.array(f['All_else/z_obs'], dtype = dt)						# Cosmological redshift
z_cos				=	np.array(f['All_else/z_cos'], dtype = dt)						# Cosmological redshift

s_peak				=	np.array(f['Emission_Line/s_peak'], dtype = dt)#*flux_central[k]	# S_peak

M_halo				=	np.array(f['Mass/M_Host_Halo'], dtype = dt)							# Halo Mass
M_cold_gas			=	np.array(f['Mass/M_gas_disk'], dtype = dt) + np.array(f['Mass/M_gas_bulge'], dtype = dt)						# Total cold gas in disk
M_cold_bulge		=	np.array(f['Mass/M_gas_bulge'], dtype = dt)				# Bulge gas molecular

R_disk				=	np.array(f['Radius_file/R_disk_gas'], dtype = dt)					# Total Disk mass
R_halo				=   np.array(f['Radius_file/R_vir'])
R_disk_star			=	np.array(f['Radius_file/R_disk_star'], dtype = dt)					# Total Disk mass
R_bulge_star		=	np.array(f['Radius_file/R_bulge_star'], dtype = dt)					# Stellar mass Bulge


M_Sub_Halo			= 	np.array(f['Mass/M_Sub_Halo'], dtype = dt)
M_stars_bulge		=	np.array(f['Mass/M_stars_bulge'], dtype = dt)					# Stellar mass Disk
M_stars_tot			=	np.array(f['Mass/M_stars_tot'], dtype = dt)					

V_bulge				=	np.array(f['Velocity_Calculated/V_bulge'], dtype = dt)			# Circular Velocity Bulge
V_disk				=	np.array(f['Velocity_Calculated/V_disk'], dtype = dt)			# Circular Velocity Disk
V_HI				=	np.array(f['Velocity_Calculated/V_HI'], dtype = dt)				# Circular Velocity Flat
V_halo				=	np.array(f['Velocity_Calculated/V_halo'], dtype = dt)			# Circular Velocity Halo
V_circ				=	np.array(f['Velocity_Calculated/V_max_circ'], dtype = dt)		# Circular Velocity total
W_20				=	np.array(f['Velocity_Calculated/W_20'], dtype = dt)				# W20
W_50				=	np.array(f['Velocity_Calculated/W_50'], dtype = dt)				# W50
W_peak				=	np.array(f['Velocity_Calculated/W_peak'], dtype = dt)			# Wpeak

theta				=	np.array(f['Theta'], dtype = dt)

V_halo_file			=	np.array(f['Velocity_File/Virial_velocity'], dtype = dt)		# Virial Velocityu


flux_central		=	np.array(f['flux/flux'], dtype = dt)							# Flux

M_atom_disk			=	np.array(f['Mass/M_atom_disk'], dtype = dt)						# Total cold gas in disk
M_atom_bulge		=	np.array(f['Mass/M_atom_bulge'], dtype = dt)						# Total cold gas in disk

s_peak = s_peak*flux_central


M_HI				=    M_atom_disk + M_atom_bulge


B_T = M_stars_bulge/M_stars_tot

f.close()


#****************************************************************************************************************************************


f = h5.File("/media/garima/Seagate Backup Plus Drive/pleiades_icrar/Shark/Old_ALFALFA_Files/SHArk_ALFALFA/medi_alfalfa_virial_velocity_without_limit.h5", 'r')


ra_2					=	np.array(f['All_else/ra'], dtype = dt)					# B_T Ratio of centrals
dec_2					=	np.array(f['All_else/dec'], dtype = dt)					# B_T Ratio of centrals

is_central_2			=	np.array(f['is_central'], dtype = dt)			
Distance_2				=	np.array(f['All_else/Distance'], dtype = dt)					# Mag of the distance (needs conversion)
z_obs_2					=	np.array(f['All_else/z_obs'], dtype = dt)						# Cosmological redshift
z_cos_2					=	np.array(f['All_else/z_cos'], dtype = dt)						# Cosmological redshift

s_peak_2				=	np.array(f['Emission_Line/s_peak'], dtype = dt)#*flux_central[k]	# S_peak

M_halo_2				=	np.array(f['Mass/M_Host_Halo'], dtype = dt)							# Halo Mass
M_cold_gas_2			=	np.array(f['Mass/M_gas_disk'], dtype = dt) + np.array(f['Mass/M_gas_bulge'], dtype = dt)						# Total cold gas in disk
M_cold_bulge_2			=	np.array(f['Mass/M_gas_bulge'], dtype = dt)				# Bulge gas molecular

R_disk_2				=	np.array(f['Radius_file/R_disk_gas'], dtype = dt)					# Total Disk mass
R_halo_2				=   np.array(f['Radius_file/R_vir'])
R_disk_star_2			=	np.array(f['Radius_file/R_disk_star'], dtype = dt)					# Total Disk mass
R_bulge_star_2			=	np.array(f['Radius_file/R_bulge_star'], dtype = dt)					# Stellar mass Bulge


M_Sub_Halo_2			= 	np.array(f['Mass/M_Sub_Halo'], dtype = dt)
M_stars_bulge_2 		=	np.array(f['Mass/M_stars_bulge'], dtype = dt)					# Stellar mass Disk
M_stars_tot_2			=	np.array(f['Mass/M_stars_tot'], dtype = dt)					

V_bulge_2				=	np.array(f['Velocity_Calculated/V_bulge'], dtype = dt)			# Circular Velocity Bulge
V_disk_2				=	np.array(f['Velocity_Calculated/V_disk'], dtype = dt)			# Circular Velocity Disk
V_HI_2					=	np.array(f['Velocity_Calculated/V_HI'], dtype = dt)				# Circular Velocity Flat
V_halo_2				=	np.array(f['Velocity_Calculated/V_halo'], dtype = dt)			# Circular Velocity Halo
V_circ_2				=	np.array(f['Velocity_Calculated/V_max_circ'], dtype = dt)		# Circular Velocity total
W_20_2					=	np.array(f['Velocity_Calculated/W_20'], dtype = dt)				# W20
W_50_2					=	np.array(f['Velocity_Calculated/W_50'], dtype = dt)				# W50
W_peak_2				=	np.array(f['Velocity_Calculated/W_peak'], dtype = dt)			# Wpeak

theta_2					=	np.array(f['Theta'], dtype = dt)

V_halo_file_2			=	np.array(f['Velocity_File/Virial_velocity'], dtype = dt)		# Virial Velocityu


flux_central_2			=	np.array(f['flux/flux'], dtype = dt)							# Flux

M_atom_disk_2			=	np.array(f['Mass/M_atom_disk'], dtype = dt)						# Total cold gas in disk
M_atom_bulge_2			=	np.array(f['Mass/M_atom_bulge'], dtype = dt)						# Total cold gas in disk

s_peak_2 				= s_peak_2*flux_central_2


M_HI_2					=    M_atom_disk_2 + M_atom_bulge_2


B_T = M_stars_bulge_2/M_stars_tot_2

f.close()

#****************************************************************************************************************************************


f = h5.File("/media/garima/Seagate Backup Plus Drive/pleiades_icrar/Shark/medi_ALFALFA_SFR.h5", 'r')

#print(k)

M_stars_bulge_3         =   np.array(f['Mass/M_stars_bulge'], dtype = dt)                   # Stellar mass Disk
M_stars_tot_3           =   np.array(f['Mass/M_stars_tot'], dtype = dt)                 

W_20_3                  =   np.array(f['Velocity_Calculated/W_20'], dtype = dt)             # W20
W_50_3                  =   np.array(f['Velocity_Calculated/W_50'], dtype = dt)             # W50

flux_central_3          =   np.array(f['flux/flux'], dtype = dt)                            # Flux

M_atom_disk_3           =   np.array(f['Mass/M_atom_disk'], dtype = dt)                     # Total cold gas in disk
M_atom_bulge_3          =   np.array(f['Mass/M_atom_bulge'], dtype = dt)                        # Total cold gas in disk

sfr_disk_medi      =   np.array(f['All_else/sfr_disk'], dtype = dt) + np.array(f['All_else/sfr_burst'], dtype = dt)

M_HI_3                  =    M_atom_disk_3 + M_atom_bulge_3
z_cos_3             =   np.array(f['All_else/z_cos'], dtype = dt)                       # Cosmological redshift
V_halo_file_3         =   np.array(f['Velocity_File/Virial_velocity'], dtype = dt)
is_central_medi          =   np.array(f['is_central'], dtype = dt)
f.close()


#****************************************************************************************************************************************

f = h5.File("/media/garima/Seagate Backup Plus Drive/pleiades_icrar/Shark/micro_ALFALFA_SFR.h5", 'r')

#print(k)

M_stars_bulge_4         =   np.array(f['Mass/M_stars_bulge'], dtype = dt)                   # Stellar mass Disk
M_stars_tot_4           =   np.array(f['Mass/M_stars_tot'], dtype = dt)                 

W_20_4                  =   np.array(f['Velocity_Calculated/W_20'], dtype = dt)             # W20
W_50_4                  =   np.array(f['Velocity_Calculated/W_50'], dtype = dt)             # W50

flux_central_4          =   np.array(f['flux/flux'], dtype = dt)                            # Flux

M_atom_disk_4           =   np.array(f['Mass/M_atom_disk'], dtype = dt)                     # Total cold gas in disk
M_atom_bulge_4          =   np.array(f['Mass/M_atom_bulge'], dtype = dt)                        # Total cold gas in disk

sfr_disk_micro           =   np.array(f['All_else/sfr_disk'], dtype = dt) + np.array(f['All_else/sfr_burst'], dtype = dt)

M_HI_4                  =    M_atom_disk_4 + M_atom_bulge_4
z_cos_4                 =   np.array(f['All_else/z_cos'], dtype = dt)                       # Cosmological redshift
V_halo_file_4         =   np.array(f['Velocity_File/Virial_velocity'], dtype = dt)
is_central_micro          =   np.array(f['is_central'], dtype = dt)
f.close()


#****************************************************************************************************************************************

theta = theta[~np.isnan(W_50)]


flux_central 	= flux_central[~np.isnan(W_50)]#/theta_all


yo = 0.06*(W_50[~np.isnan(W_50)])**0.50 


is_central      = is_central[~np.isnan(W_50)]

M_cold_gas		= M_cold_gas[~np.isnan(V_circ)]/h
V_circ 			= V_circ[~np.isnan(V_circ)]
V_halo 			= V_halo[~np.isnan(V_halo)]
V_disk			= V_disk[~np.isnan(V_disk)]


M_HI  			= M_HI[~np.isnan(W_50)]
M_halo          = M_halo[~np.isnan(W_50)]/h
M_stars_tot		= M_stars_tot[~np.isnan(W_50)]/h
R_disk			= R_disk[~np.isnan(W_50)]
R_halo			= R_halo[~np.isnan(W_50)]

W_50  			= W_50[~np.isnan(W_50)]



#****************************************************************************************************************************************

theta_2 = theta_2[~np.isnan(W_50_2)]


flux_central_2 	= flux_central_2[~np.isnan(W_50_2)]#/theta_all

yo_2 = 0.06*(W_50_2[~np.isnan(W_50_2)])**0.50 

is_central_2      = is_central_2[~np.isnan(W_50_2)]

M_cold_gas_2		= M_cold_gas_2[~np.isnan(V_circ_2)]/h
V_circ_2 			= V_circ_2[~np.isnan(V_circ_2)]
V_halo_2 			= V_halo_2[~np.isnan(V_halo_2)]
V_disk_2			= V_disk_2[~np.isnan(V_disk_2)]

M_HI_2  			= M_HI_2[~np.isnan(W_50_2)]
M_halo_2            = M_halo_2[~np.isnan(M_halo_2)]/h
M_stars_tot_2		= M_stars_tot_2[~np.isnan(W_50_2)]/h
R_disk_2			= R_disk_2[~np.isnan(W_50_2)]
R_halo_2			= R_halo_2[~np.isnan(W_50_2)]


W_50_2  			= W_50_2[~np.isnan(W_50_2)]

#*************************************************************************************************************************************



flux_central_3  = flux_central_3[~np.isnan(W_50_3)]#/theta_all

yo_3 = 0.06*(W_50_3[~np.isnan(W_50_3)])**0.50 

is_central_medi      = is_central_medi[~np.isnan(W_50_3)]

M_HI_3              = M_HI_3[~np.isnan(W_50_3)]/h
z_cos_3             = z_cos_3[~np.isnan(W_50_3)]
M_stars_tot_3       = M_stars_tot_3[~np.isnan(W_50_3)]/h
V_halo_file_3       = V_halo_file_3[~np.isnan(W_50_3)]
sfr_disk_medi       = sfr_disk_medi[~np.isnan(W_50_3)]
W_50_3              = W_50_3[~np.isnan(W_50_3)]

#*************************************************************************************************************************************



flux_central_4  = flux_central_4[~np.isnan(W_50_4)]#/theta_all

yo_4 = 0.06*(W_50_4[~np.isnan(W_50_4)])**0.50 

is_central_micro      = is_central_micro[~np.isnan(W_50_4)]

M_HI_4              = M_HI_4[~np.isnan(W_50_4)]/h
z_cos_4             = z_cos_4[~np.isnan(W_50_4)]
M_stars_tot_4       = M_stars_tot_4[~np.isnan(W_50_4)]/h
V_halo_file_4       = V_halo_file_4[~np.isnan(W_50_4)]
sfr_disk_micro      = sfr_disk_micro[~np.isnan(W_50_4)]
W_50_4              = W_50_4[~np.isnan(W_50_4)]

#*************************************************************************************************************************************



######################################################################################################################################################
### OBSERVATION POINTS - ALFALFA 100 data
#############################################################################################################################################

data_file = "/media/garima/Seagate Backup Plus Drive/pleiades_icrar/Shark/Old_ALFALFA_Files/ALFALFA/a100_datafile.csv"

#data_100 = np.array(csv.reader(open(data_file, newline = ''), delimiter = ','))

data_100 = np.genfromtxt(data_file, delimiter=',')




W_50_100 = data_100[:,7]
W_20_100 = data_100[:,9]

M_HI_100 = data_100[:,16]



W_50_100  	= W_50_100[~np.isnan(W_50_100)]

W_20_100  	= W_20_100[~np.isnan(W_20_100)]

M_HI_100  	= M_HI_100[~np.isnan(M_HI_100)]



#############################################################################################################################################################
####### SELECTION OF ALFALFA
#************************************************************************************************************************************************************

W_50_sel_2 			= W_50_2[(flux_central_2 >yo_2) & (z_cos_2 < 0.06)]
W_20_sel_2 			= W_20_2[(flux_central_2 >yo_2) & (z_cos_2 < 0.06)]
W_peak_sel_2 		= W_peak_2[(flux_central_2 >yo_2) & (z_cos_2 < 0.06)]

M_HI_sel_2 			= M_HI_2[(flux_central_2 >yo_2) & (z_cos_2 < 0.06)]

V_circ_sel_2 		= V_circ_2[(flux_central_2 >yo_2) & (z_cos_2 < 0.06)]
V_disk_sel_2 		= V_disk_2[(flux_central_2 >yo_2) & (z_cos_2 < 0.06)]
V_halo_sel_2		= V_halo_2[(flux_central_2 >yo_2) & (z_cos_2 < 0.06)]
theta_sel_2       	= theta_2[(flux_central_2 >yo_2) & (z_cos_2 < 0.06)]

V_bulge_sel_2 		= V_bulge_2[(flux_central_2 >yo_2) & (z_cos_2 < 0.06)]
V_HI_sel_2 			= V_HI_2[(flux_central_2 >yo_2) & (z_cos_2 < 0.06)]



is_central_sel_2  	= is_central_2[(flux_central_2 >yo_2) & (z_cos_2 < 0.06)]

M_Sub_Halo_sel_2 	= M_Sub_Halo_2[(flux_central_2 >yo_2) & (z_cos_2 < 0.06)]
M_stars_tot_sel_2 	= M_stars_tot_2[(flux_central_2 >yo_2) & (z_cos_2 < 0.06)]
M_halo_sel_2      	= M_halo_2[(flux_central_2 >yo_2) & (z_cos_2 < 0.06)]

M_cold_gas_sel_2  	= M_cold_gas_2[(flux_central_2 >yo_2) & (z_cos_2 < 0.06)]

R_disk_star_sel_2 	= R_disk_star_2[(flux_central_2 >yo_2) & (z_cos_2 < 0.06)]*10**3
R_disk_sel_2		= R_disk_2[(flux_central_2 >yo_2) & (z_cos_2 < 0.06)]*10**3
R_halo_sel_2		= R_halo_2[(flux_central_2 >yo_2) & (z_cos_2 < 0.06)]*10**3

ra_sel_2 			= ra_2[(flux_central_2 >yo_2) & (z_cos_2 < 0.06)]
dec_sel_2 			= dec_2[(flux_central_2 >yo_2) & (z_cos_2 < 0.06)]
z_obs_sel_2			= z_obs_2[(flux_central_2 >yo_2) & (z_cos_2 < 0.06)]

z_cos_sel_2       	= z_cos_2[(flux_central_2 >yo_2) & (z_cos_2 < 0.06)]
B_T_ratio_2       	= M_stars_bulge_2/M_stars_tot_2

B_T_sel_2         	= B_T_ratio_2[(flux_central_2 >yo_2) & (z_cos_2 < 0.06)]

Distance_sel_2 		= Distance_2[(flux_central_2 >yo_2) & (z_cos_2 < 0.06)]

s_peak_sel_2      	= s_peak_2[flux_central_2 >= yo_2]

theta_sel_2 		= theta_2[(flux_central_2 >yo_2) & (z_cos_2 < 0.06)]
flux_sel_2 			= flux_central_2[(flux_central_2 >yo_2) & (z_cos_2 < 0.06)]



#############################################################################################################################################################
####### SELECTION OF ALFALFA
#************************************************************************************************************************************************************



W_50_sel 		= W_50[(flux_central >yo) & (z_cos < 0.06)]
W_20_sel 		= W_20[(flux_central >yo) & (z_cos < 0.06)]
W_peak_sel 		= W_peak[(flux_central >yo) & (z_cos < 0.06)]

M_HI_sel 		= M_HI[(flux_central >yo) & (z_cos < 0.06)]

V_circ_sel 		= V_circ[(flux_central >yo) & (z_cos < 0.06)]
V_disk_sel 		= V_disk[(flux_central >yo) & (z_cos < 0.06)]
V_halo_sel		= V_halo[(flux_central >yo) & (z_cos < 0.06)]
theta_sel       = theta[(flux_central >yo) & (z_cos < 0.06)]

V_bulge_sel 	= V_bulge[(flux_central >yo) & (z_cos < 0.06)]
V_HI_sel 		= V_HI[(flux_central >yo) & (z_cos < 0.06)]



is_central_sel  = is_central[(flux_central >yo) & (z_cos < 0.06)]

M_Sub_Halo_sel 	= M_Sub_Halo[(flux_central >yo) & (z_cos < 0.06)]
M_stars_tot_sel = M_stars_tot[(flux_central >yo) & (z_cos < 0.06)]
M_halo_sel      = M_halo[(flux_central >yo) & (z_cos < 0.06)]

M_cold_gas_sel  = M_cold_gas[(flux_central >yo) & (z_cos < 0.06)]

R_disk_star_sel = R_disk_star[(flux_central >yo) & (z_cos < 0.06)]*10**3
R_disk_sel		= R_disk[(flux_central >yo) & (z_cos < 0.06)]*10**3
R_halo_sel		= R_halo[(flux_central >yo) & (z_cos < 0.06)]*10**3

ra_sel 			= ra[(flux_central >yo) & (z_cos < 0.06)]
dec_sel 		= dec[(flux_central >yo) & (z_cos < 0.06)]
z_obs_sel		= z_obs[(flux_central >yo) & (z_cos < 0.06)]

z_cos_sel       = z_cos[(flux_central >yo) & (z_cos < 0.06)]
B_T_ratio       = M_stars_bulge/M_stars_tot

B_T_sel         = B_T_ratio[(flux_central >yo) & (z_cos < 0.06)]

Distance_sel 	= Distance[(flux_central >yo) & (z_cos < 0.06)]

s_peak_sel      = s_peak[flux_central >= yo]

theta_sel 			= theta[(flux_central >yo) & (z_cos < 0.06)]
flux_sel 		= flux_central[(flux_central >yo) & (z_cos < 0.06)]

#############################################################################################################################################################
####### SELECTION OF ALFALFA
#************************************************************************************************************************************************************

W_50_sel_3           = W_50_3[(flux_central_3 >yo_3) & (z_cos_3 < 0.06)]
#W_20_sel_3      = W_20_3[(flux_central_3 >yo_3) & (z_cos_3 < 0.06)]
 
M_HI_sel_3          = M_HI_3[(flux_central_3 >yo_3) & (z_cos_3 < 0.06)]

M_stars_tot_sel_3   = M_stars_tot_3[(flux_central_3 >yo_3) & (z_cos_3 < 0.06)]
#M_cold_gas_sel_3    = M_cold_gas_3[(flux_central_3 >yo_3) & (z_cos_3 < 0.06)]


sfr_disk_sel_medi        = sfr_disk_medi[(flux_central_3 > yo_3) & (z_cos_3 < 0.06)]
V_halo_sel_3      = V_halo_file_3[(flux_central_3 >yo_3) & (z_cos_3 < 0.06)]

is_central_sel_medi  = is_central_medi[(flux_central_3 >yo_3) & (z_cos_3 < 0.06)]


#************************************************************************************************************************************************************

W_50_sel_4           = W_50_4[(flux_central_4 >yo_4) & (z_cos_4 < 0.06)]
M_HI_sel_4          = M_HI_4[(flux_central_4 >yo_4) & (z_cos_4 < 0.06)]
M_stars_tot_sel_4   = M_stars_tot_4[(flux_central_4 >yo_4) & (z_cos_4 < 0.06)]
sfr_disk_sel_micro  = sfr_disk_micro[(flux_central_4 > yo_4) & (z_cos_4 < 0.06)]
V_halo_sel_4      = V_halo_file_4[(flux_central_4 >yo_4) & (z_cos_4 < 0.06)]
is_central_sel_micro  = is_central_micro[(flux_central_4 >yo_4) & (z_cos_4 < 0.06)]




#************************************************************************************************************************************************************
#************************************************************************************************************************************************************
#************************************************************************************************************************************************************

def mockCatalogueHdf5(file_name):
	
	
	vel_cal			=	file_name.create_group("Velocity_Calculated")
	vel_cal.create_dataset("V_halo", data = V_halo_sel)
	vel_cal.create_dataset("V_disk", data = V_disk_sel)
	vel_cal.create_dataset("V_bulge", data = V_bulge_sel)
	vel_cal.create_dataset("V_HI", data = V_HI_sel)
	vel_cal.create_dataset("V_max_circ", data = V_circ_sel)
	vel_cal.create_dataset("W_peak", data = W_peak_sel)
	vel_cal.create_dataset("W_20", data = W_20_sel)
	vel_cal.create_dataset("W_50",data = W_50_sel)

	file_name.create_dataset("Theta", data = theta_sel)

	flux_gal		= 	file_name.create_group("flux")
	flux_gal.create_dataset("flux", data = flux_sel)
	
	



	central = file_name.create_dataset('is_central', data = is_central_sel)

	G_Radius	=	file_name.create_group('Radius_file')

	G_Radius.create_dataset('R_vir', data = R_halo_sel)
	G_Radius.create_dataset('R_disk_star', data = R_disk_star_sel)
	# G_Radius.create_dataset('R_bulge_star', data = rstar_bulge)
	G_Radius.create_dataset('R_disk_gas', data = R_disk_sel)
	#G_Radius.create_dataset('R_bulge_gas', data = rgas_bulge)

	#*************************************************************************************************

	G_Mass		=	file_name.create_group('Mass')

	G_Mass.create_dataset('M_HI', data = M_HI_sel)
	G_Mass.create_dataset('M_cold_gas', data = M_cold_gas_sel)
	G_Mass.create_dataset('M_Host_Halo', data = M_halo_sel)
	G_Mass.create_dataset('M_Sub_Halo', data = M_Sub_Halo_sel)
	G_Mass.create_dataset('M_stars', data = M_stars_tot_sel)
	
	#*****************************************************************************************************

	G_cals		= file_name.create_group('All_else')

	G_cals.create_dataset('Distance', data = Distance_sel)
	G_cals.create_dataset('z_cos', data = z_cos_sel)
	G_cals.create_dataset('z_obs', data = z_obs_sel)
	G_cals.create_dataset('dec', data = dec_sel)
	G_cals.create_dataset('ra', data = ra_sel)

def mockCatalogueHdf5_2(file_name):
	
	
	vel_cal			=	file_name.create_group("Velocity_Calculated")
	vel_cal.create_dataset("V_halo", data = V_halo_sel_2)
	vel_cal.create_dataset("V_disk", data = V_disk_sel_2)
	vel_cal.create_dataset("V_bulge", data = V_bulge_sel_2)
	vel_cal.create_dataset("V_HI", data = V_HI_sel_2)
	vel_cal.create_dataset("V_max_circ", data = V_circ_sel_2)
	vel_cal.create_dataset("W_peak", data = W_peak_sel_2)
	vel_cal.create_dataset("W_20", data = W_20_sel_2)
	vel_cal.create_dataset("W_50",data = W_50_sel_2)

	file_name.create_dataset("Theta", data = theta_sel_2)

	flux_gal		= 	file_name.create_group("flux")
	flux_gal.create_dataset("flux", data = flux_sel_2)
	
	



	central = file_name.create_dataset('is_central', data = is_central_sel_2)

	G_Radius	=	file_name.create_group('Radius_file')

	G_Radius.create_dataset('R_vir', data = R_halo_sel_2)
	G_Radius.create_dataset('R_disk_star', data = R_disk_star_sel_2)
	# G_Radius.create_dataset('R_bulge_star', data = rstar_bulge)
	G_Radius.create_dataset('R_disk_gas', data = R_disk_sel_2)
	#G_Radius.create_dataset('R_bulge_gas', data = rgas_bulge)

	#*************************************************************************************************

	G_Mass		=	file_name.create_group('Mass')

	G_Mass.create_dataset('M_HI', data = M_HI_sel_2)
	G_Mass.create_dataset('M_cold_gas', data = M_cold_gas_sel_2)
	G_Mass.create_dataset('M_Host_Halo', data = M_halo_sel_2)
	G_Mass.create_dataset('M_Sub_Halo', data = M_Sub_Halo_sel_2)
	G_Mass.create_dataset('M_stars', data = M_stars_tot_sel_2)
	
	#*****************************************************************************************************

	G_cals		= file_name.create_group('All_else')

	G_cals.create_dataset('Distance', data = Distance_sel_2)
	G_cals.create_dataset('z_cos', data = z_cos_sel_2)
	G_cals.create_dataset('z_obs', data = z_obs_sel_2)
	G_cals.create_dataset('dec', data = dec_sel_2)
	G_cals.create_dataset('ra', data = ra_sel_2)




hf = h5.File('medi_alfalfa_catalogue.h5', 'w')

mockCatalogueHdf5_2(hf)
hf.close()


#############################################################################################################################################################################
### ENTIRE MEDIAN Properties
#****************************************************************************************************************************************************************************

 
# W_50_box_medi 		 					= {}
# W_20_box_medi							= {}
# M_HI_box_medi     	 					= {}
# V_circ_box_medi       					= {}
# M_stars_tot_box_medi  					= {}
# M_halo_box_medi       					= {}




# for k in range(64):

# 	#f = h5.File("/mnt/su3ctm/gchauhan/SHArk_Out/Plot_Dir/medi-SURFS-processed/medi_SURFS_subvolume_%s.h5" %k, 'r')
# 	f = h5.File("/media/garima/Seagate Backup Plus Drive/pleiades_icrar/Shark/Plot_Dir/medi-SURFS/199/%s/galaxies.hdf5" %k, 'r')
	
	
# 	M_halo_box_medi[k]				=	np.array(f['Mass/M_Host_Halo'], dtype = dt)/h							# Halo Mass
	
# 	M_stars_tot_box_medi[k]			=	np.array(f['Mass/M_stars_tot'], dtype = dt)/h					

# 	V_circ_box_medi[k]				=	np.array(f['Velocity_Calculated/V_max_circ'], dtype = dt)		# Circular Velocity total
# 	W_20_box_medi[k]					=	np.array(f['Velocity_Calculated/W_20'], dtype = dt)				# W20
# 	W_50_box_medi[k]					=	np.array(f['Velocity_Calculated/W_50'], dtype = dt)				# W50
	
# 	M_atom_disk				=	np.array(f['Mass/M_atom_disk'], dtype = dt)/h						# Total cold gas in disk
# 	M_atom_bulge			=	np.array(f['Mass/M_atom_bulge'], dtype = dt)/h					# Total cold gas in disk

# 	M_HI_box_medi[k]					=	M_atom_bulge + M_atom_disk

# 	f.close()


# #*************************************************************************************************************************************

# M_HI_medi 						= []
# M_halo_medi 					= []
# W_50_medi 						= []
# V_circ_medi						= []
# M_stars_tot_medi 				= []
# W_20_medi						= []


# #************************************************************************************************************************************


# for k in range(64): 
# 	#print(k)
# 	M_HI_medi 				= np.append(M_HI_medi, M_HI_box_medi[k])
# 	W_50_medi 				= np.append(W_50_medi, W_50_box_medi[k])
# 	W_20_medi 				= np.append(W_20_medi, W_20_box_medi[k])
# 	V_circ_medi 			= np.append(V_circ_medi, V_circ_box_medi[k])
# 	M_stars_tot_medi 		= np.append(M_stars_tot_medi, M_stars_tot_box_medi[k])
# 	M_halo_medi 			= np.append(M_halo_medi, M_halo_box_medi[k])
	


# #****************************************************************************************************************************************
 
# W_50_box_micro 		 					= {}
# W_20_box_micro							= {}
# M_HI_box_micro     	 					= {}
# V_circ_box_micro       					= {}
# M_stars_tot_box_micro  					= {}
# M_halo_box_micro       					= {}




# for k in range(64):

# 	#f = h5.File("/mnt/su3ctm/gchauhan/SHArk_Out/Plot_Dir/micro-SURFS-processed/micro_SURFS_subvolume_%s.h5" %k, 'r')
# 	f = h5.File("/media/garima/Seagate Backup Plus Drive/pleiades_icrar/Shark/Plot_Dir/micro-SURFS/199/%s/galaxies.hdf5" %k, 'r')
	
	
# 	M_halo_box_micro[k]				=	np.array(f['Mass/M_Host_Halo'], dtype = dt)/h							# Halo Mass
	
# 	M_stars_tot_box_micro[k]			=	np.array(f['Mass/M_stars_tot'], dtype = dt)/h					

# 	V_circ_box_micro[k]				=	np.array(f['Velocity_Calculated/V_max_circ'], dtype = dt)		# Circular Velocity total
# 	W_20_box_micro[k]					=	np.array(f['Velocity_Calculated/W_20'], dtype = dt)				# W20
# 	W_50_box_micro[k]					=	np.array(f['Velocity_Calculated/W_50'], dtype = dt)				# W50
	
# 	M_atom_disk				=	np.array(f['Mass/M_atom_disk'], dtype = dt)/h						# Total cold gas in disk
# 	M_atom_bulge			=	np.array(f['Mass/M_atom_bulge'], dtype = dt)/h					# Total cold gas in disk

# 	M_HI_box_micro[k]					=	M_atom_bulge + M_atom_disk

# 	f.close()


# #*************************************************************************************************************************************

# M_HI_micro 						= []
# M_halo_micro 					= []
# W_50_micro 						= []
# V_circ_micro					= []
# M_stars_tot_micro 				= []
# W_20_micro						= []


# #************************************************************************************************************************************


# for k in range(64): 
# 	#print(k)
# 	M_HI_micro 				= np.append(M_HI_micro, M_HI_box_micro[k])
# 	W_50_micro 				= np.append(W_50_micro, W_50_box_micro[k])
# 	W_20_micro 				= np.append(W_20_micro, W_20_box_micro[k])
# 	V_circ_micro 			= np.append(V_circ_micro, V_circ_box_micro[k])
# 	M_stars_tot_micro 		= np.append(M_stars_tot_micro, M_stars_tot_box_micro[k])
# 	M_halo_micro 			= np.append(M_halo_micro, M_halo_box_micro[k])


	


#****************************************************************************************************************************************

###------------------------------------------------------------------------------------------------------------------------------
# Angle of inclination generation
###------------------------------------------------------------------------------------------------------------------------------

# theta - sine of the angle of inclination for galaxies - 1-e4 is the lowest value I go for to avoid the division by zero problem, so when doing calculation, divide your thing with 2*theta
# if you want to add error 

# loop_length_1 = len(M_HI_medi)
# loop_length_2 = len(M_HI_micro)

# theta_1 = np.cos((np.array([random.uniform(0,360) for j in range(loop_length_1)]))*np.pi/180)
# theta_1 = np.arccos(theta_1)
# theta_1 = np.sin(theta_1)

# V_circ_medi = V_circ_medi*theta_1

# # for i in range(loop_length_1):
# #     theta_1[i] = max(0.0871, np.abs(theta[i]))

# theta_2 = np.cos((np.array([random.uniform(0,360) for j in range(loop_length_2)]))*np.pi/180)
# theta_2 = np.arccos(theta_2)
# theta_2 = np.sin(theta_2)

# V_circ_micro = V_circ_micro*theta_2


# V_circ_micro = V_circ_micro.tolist()

# W_50_micro 	=  V_circ_micro*6413
# W_50_micro_median = V_circ_micro*6413

# W_50_micro = np.array(W_50_micro)
# W_50_micro_median = np.array(W_50_micro_median)

# W_50_micro_median 	= V_circ_micro[M_HI_micro != 0] #np.append(W_50_4[W_50_4 <= 100],W_50[W_50 > 100])
# W_50_medi_median 	= V_circ_medi[M_HI_medi != 0]#np.append(W_50_3[W_50_3 <= 100],W_50_2[W_50_2 > 100])
# M_HI_micro_median 	= M_HI_micro[M_HI_micro != 0]#np.append(M_HI_4[W_50_4 <= 100],M_HI[W_50 > 100])
# M_HI_medi_median 	= M_HI_medi[M_HI_medi != 0]#np.append(M_HI_3[W_50_3 <= 100],M_HI_2[W_50_2 > 100])

# W_50_micro_median 	= np.append(V_circ_micro[V_circ_micro <= 100],W_50[W_50 > 100])
# W_50_medi_median 	= np.append(V_circ_medi[V_circ_medi <= 100],W_50_2[W_50_2 > 100])
# M_HI_micro_median 	= np.append(M_HI_micro[V_circ_micro <= 100],M_HI[W_50 > 100])
# W_50_medi_median 	= np.append(M_HI_medi[V_circ_medi <= 100],M_HI_2[W_50_2 > 100])



# W_50_micro 	= np.append(W_50_4,W_50)
# W_50_medi 	= np.append(W_50_3,W_50_2)
# M_HI_micro 	= np.append(M_HI_4,M_HI)
# M_HI_medi 	= np.append(M_HI_3,M_HI_2)

# W_50_micro_median 	= np.append(W_50_4[W_50_4 <= 130],W_50[W_50 > 130])
# W_50_medi_median 	= np.append(W_50_3[W_50_3 <= 250],W_50_2[W_50_2 > 250])
# M_HI_micro_median 	= np.append(M_HI_4[W_50_4 <= 130],M_HI[W_50 > 130])
# M_HI_medi_median 	= np.append(M_HI_3[W_50_3 <= 250],M_HI_2[W_50_2 > 250])




#############################################################################################################################################################################
### PLOTTING MASS FUNCTION
#****************************************************************************************************************************************************************************

# def plotting_MF(central,central_all, ALFALFA,area):
# 	central = central[~np.isnan(central)]
# 	central = central[central != 0]
	

# 	bins_host = np.arange(np.log10(min(central)),np.log10(max(central)),0.2)
# 	Z, bins = np.histogram(np.log10(central), bins = bins_host)


# 	central_all = central_all[~np.isnan(central_all)]
# 	central_all = central_all[central_all != 0]
	
# 	bins_host_all = np.arange(np.log10(min(central_all)),np.log10(max(central_all)),0.2)
# 	Z_all, bins = np.histogram(np.log10(central_all), bins = bins_host_all)


# 	afla_100 = ALFALFA[ALFALFA != 0]
	
# 	bins_100 = np.arange(min(ALFALFA),max(ALFALFA),0.2)

# 	Obs, bins = np.histogram(ALFALFA, bins = bins_100)

# 	Y = np.zeros(len(bins_host))
# 	Y_all = np.zeros(len(bins_host_all))

# 	err_medi_up = np.zeros(len(bins_host))
# 	err_medi_dn =np.zeros(len(bins_host))
# 	err_micro_up =np.zeros(len(bins_host_all))
# 	err_micro_dn =np.zeros(len(bins_host_all))

# 	Y_obs = np.zeros(len(bins_100))
# 	err_obs_up = np.zeros(len(bins_100))
# 	err_obs_dn = np.zeros(len(bins_100))

# 	for i in range(len(Z)):
# 		Y[i] = np.log10(Z[i]/area/0.2)
# 		err_medi_up[i] = np.log10((Z[i]+np.sqrt(Z[i]))/area/0.2) #- Y[i]
# 		err_medi_dn[i] = np.log10((Z[i]-np.sqrt(Z[i]))/area/0.2)

# 	for i in range(len(Z_all)):
# 		Y_all[i] = np.log10(Z_all[i]/area/0.2)
# 		err_micro_up[i] = np.log10((Z_all[i]+np.sqrt(Z_all[i]))/area/0.2)
# 		err_micro_dn[i] = np.log10((Z_all[i]-np.sqrt(Z_all[i]))/area/0.2)


# 	for i in range(len(Obs)):
# 		Y_obs[i] = np.log10(Obs[i]/7072.298/0.2)
# 		err_obs_up[i] = np.log10((Obs[i]+np.sqrt(Obs[i]))/7072.298/0.2) - Y_obs[i]
# 		err_obs_dn[i] = Y_obs[i] - np.log10((Obs[i]-np.sqrt(Obs[i]))/7072.298/0.2)

# 	X = bins_host
# 	X_obs = bins_100
# 	X_all = bins_host_all

# 	plt.figure(figsize = (12,8))
		
# 	M_HI_sel_handle = plt.plot(X[0:len(X) - 1],Y[0:len(X) - 1], linestyle='-', color='rebeccapurple', label = "$\\rm Medi-SURFS$")
# 	plt.fill_between(X[0:len(X) - 1], err_medi_up[0:len(X) - 1],err_medi_dn[0:len(X) - 1],hatch='\\',alpha=0.2,color='rebeccapurple')
# 	M_HI_all_handle = plt.plot(X_all[0:len(X_all) - 1],Y_all[0:len(X_all) - 1], linestyle='-', color='goldenrod', label = "$\\rm Micro-SURFS$")
# 	plt.fill_between(X_all[0:len(X_all) - 1], err_micro_up[0:len(X_all) - 1],err_micro_dn[0:len(X_all) - 1],hatch='|',alpha=0.2,color='goldenrod')

# 	#plt.scatter(X_obs[3:len(bins_100)-1], Y_obs[3:len(bins_100)-1], marker='*',s=15, color = 'black')
# 	ALFALFA = plt.errorbar(X_obs[3:len(bins_100)-1],Y_obs[3:len(bins_100)-1], yerr=[err_obs_dn[3:len(bins_100)-1], err_obs_up[3:len(bins_100)-1]],linestyle='--',linewidth=2,color='k', marker='*',ms=10,mfc='red',ecolor='r', capthick=2, label='$\\rm ALFALFA\ 100\ (Haynes+ 2018)$')
# 	plt.legend(frameon=False)
# 	plt.ylabel('$\\rm log_{10}(N$/$ area^{-1}dex^{-1})$')
# 	plt.xlabel('$\\rm log_{10}(M_{HI}$)[$M_{\odot}] $')
# 	#plt.title('HI Mass Distribution', size=20)
# 	plt.xlim(6,11)
# 	plt.ylim(-3.5,1)
# 	plt.xticks()
# 	plt.yticks()
# 	plt.show()


# #plotting_MF(M_HI_sel_2[W_50_sel_2 !=0]*h,M_HI_sel[W_50_sel != 0]*h, M_HI_100, 6300)



# #len(np.where(W_50_sel == 0 )[0])

# #############################################################################################################################################################################
# ### PLOTTING SKY MAP
# #****************************************************************************************************************************************************************************

# def skymap_new_test(ra,redshift, M_prop,lower_limit_theta,upper_limit_theta, max_z, rotate_a, rotate_b,ticks_number,title_name):
	
# 	tr_scale = Affine2D().scale(np.pi/180,1)
# 	tr_rotate = Affine2D().translate(rotate_a,rotate_b)
# 	theta = np.radians(ra[(M_prop != 0) & (M_prop > 10**5)])
# 	radius = redshift[(M_prop != 0) & (M_prop > 10**5)]

# 	#thetaLims = (np.radians(lower_limit_theta*15),np.radians(upper_limit_theta*15))
# 	#thetaLims = (lower_limit_theta,upper_limit_theta)
	
# 	lower_theta, upper_theta = lower_limit_theta*15, upper_limit_theta*15 

# 	zLims = (0.0,max_z)

# 	fig = plt.figure(figsize=(12,8))

# 	#grid_locator1 = angle_helper.LocatorDM(10)
	
# 	grid_locator1 = angle_helper.LocatorHMS(ticks_number)
# 	tick_formatter1 = angle_helper.FormatterHMS()

# 	grid_locator2 = MaxNLocator(6)

	

# 	tr = tr_rotate + tr_scale + PolarAxes.PolarTransform() 
	
# 	grid_helper = floating_axes.GridHelperCurveLinear(tr,
#                                         extremes=(lower_theta,upper_theta, *zLims),
#                                         grid_locator1=grid_locator1,
#                                         grid_locator2=grid_locator2,
#                                         tick_formatter1=tick_formatter1,
#                                         tick_formatter2=None
#                                         )

# 	ax = floating_axes.FloatingSubplot(fig, 111, grid_helper=grid_helper)
# 	fig.add_subplot(ax)

# 	# adjust axis
# 	ax.axis["left"].set_axis_direction("bottom")
# 	ax.axis["right"].set_axis_direction("top")

# 	ax.axis["bottom"].set_visible(False)
# 	ax.axis["top"].set_axis_direction("bottom")
# 	ax.axis["top"].toggle(ticklabels=True, label=True)
# 	ax.axis["top"].major_ticklabels.set_axis_direction("top")
# 	ax.axis["top"].major_ticklabels.set_fontsize(10)
# 	ax.axis["left"].major_ticklabels.set_fontsize(10)
# 	ax.axis["top"].label.set_axis_direction("top")

# 	ax.axis["left"].label.set_text(r"cz [kms$^{-1}$]")
# 	ax.axis["top"].label.set_text(r"Right Ascesion")
# 	ax.axis["left"].label.set_fontsize(10)
# 	ax.axis["top"].label.set_fontsize(10)

# 	# create a parasite axes whose transData in RA, cz

# 	aux_ax = ax.get_aux_axes(tr)
# 	aux_ax.patch = ax.patch  
# 	ax.patch.zorder = 0.9  

# 	scatter_yo = aux_ax.scatter(theta, radius, s= 0.05, c = np.log10(M_prop[(M_prop != 0) & (M_prop > 10**5)]), cmap='plasma_r')
# 	aux_ax.grid(True)
# 	aux_ax.set_axisbelow(True)
	
# 	cbar = plt.colorbar(scatter_yo, orientation='horizontal', shrink=0.5)
# 	cbar.ax.tick_params(labelsize=10)
# 	cbar.ax.set_title('$log_{10}[M_{HI}$ /$M_{\odot}$]',fontsize=12)
# 	plt.show()


# ra_plot = (np.append(ra_sel,ra_sel_2))*57.3
# z_cos_plot = (np.append(z_cos_sel, z_cos_sel_2))*299792
# M_HI_plot = np.append(M_HI_sel, M_HI_sel_2)

# skymap_new_test(ra_sel*57.3, z_cos_sel*299792, M_HI_sel, 7.5,16.5,18000,-90,0,10,'$\\rm Northern\ Hemisphere$')

# skymap_new_test(ra_sel*57.3, z_cos_sel*299792, M_HI_sel, 0,5,18000,45,180,5,'$\\rm Southern\ Hemisphere$')










#############################################################################################################################################################################
### PLOTTING VELOCITY FUNCTION
#****************************************************************************************************************************************************************************


# #def plotting_VF(width_1,velocity_1, velocity_2, velocity_3,ALFALFA, type_gal,area, name_width, name_velocity_1, name_velocity_2):
# def plotting_VF(width_1,velocity_1,ALFALFA, type_gal,area, name_width, name_velocity_1):

# 	width_1 = width_1[~np.isnan(width_1)]
# 	width_1 = width_1[width_1 != 0]
	
# 	bins_host = np.arange(np.log10(min(width_1)),np.log10(max(width_1)),0.1)
# 	Z, bins = np.histogram(np.log10(width_1), bins = bins_host)



# 	velocity_1 = velocity_1[~np.isnan(velocity_1)]
# 	velocity_1 = velocity_1[velocity_1 != 0]
# 	H, bins = np.histogram(np.log10(velocity_1), bins = bins_host)


# 	afla_100 = np.log10(ALFALFA[ALFALFA != 0])

# 	bins_100 = np.arange(min(np.log10(ALFALFA)),max(np.log10(ALFALFA)),0.1)
		
	
# 	Obs, bins = np.histogram(afla_100, bins = bins_100)
	
	
# 	Y = np.zeros(len(bins_host))
# 	Y_V_1 = np.zeros(len(bins_host))
	
# 	err_Y_up = np.zeros(len(bins_host))
# 	err_Y_dn =np.zeros(len(bins_host))
# 	err_Y_V_up =np.zeros(len(bins_host))
# 	err_Y_V_dn =np.zeros(len(bins_host))

# 	Y_obs = np.zeros(len(bins_100))
# 	err_obs_up = np.zeros(len(bins_100))
# 	err_obs_dn = np.zeros(len(bins_100))

	
# 	for i in range(len(Z)):
# 		Y[i] = np.log10(Z[i]/area/0.1)
# 		Y_V_1[i] = np.log10(H[i]/area/0.1)
# 		err_Y_up[i] = np.log10((Z[i]+np.sqrt(Z[i]))/area/0.1) #- Y[i]
# 		err_Y_dn[i] = np.log10((Z[i]-np.sqrt(Z[i]))/area/0.1)
# 		err_Y_V_up[i] = np.log10((H[i]+np.sqrt(H[i]))/area/0.1) #- Y[i]
# 		err_Y_V_dn[i] = np.log10((H[i]-np.sqrt(H[i]))/area/0.1)



# 	for i in range(len(Obs)):
	
# 		Y_obs[i] = np.log10(Obs[i]/6300/0.1)
# 		err_obs_up[i] = np.log10((Obs[i]+np.sqrt(Obs[i]))/6300/0.1) - Y_obs[i]
# 		err_obs_dn[i] = Y_obs[i] - np.log10((Obs[i]-np.sqrt(Obs[i]))/6300/0.1)


	
# 	X = bins_host
# 	X_obs = bins_100
	
# 	plt.figure(figsize = (12,8))
		
# 	width_handle = plt.plot(X[0:len(X) - 1],Y[0:len(X) - 1], linestyle='-', color='rebeccapurple' , label = "$\\rm Medi-SURFS$")
# 	plt.fill_between(X[0:len(X) - 1], err_Y_up[0:len(X) - 1],err_Y_dn[0:len(X) - 1],hatch='\\',alpha=0.2,color='rebeccapurple')

# 	velocity_handle_1 = plt.plot(X[0:len(X) - 1],Y_V_1[0:len(X) - 1], linestyle='-', color='goldenrod' , label = "$\\rm Micro-SURFS$")
# 	plt.fill_between(X[0:len(X) - 1], err_Y_V_up[0:len(X) - 1],err_Y_V_dn[0:len(X) - 1],hatch='|',alpha=0.1,color='goldenrod')


# 	plt.scatter(X_obs[3:len(bins_100)-1], Y_obs[3:len(bins_100)-1], marker='*',s=15, color = 'blue')
# 	ALFALFA = plt.errorbar(X_obs[3:len(bins_100)-1],Y_obs[3:len(bins_100)-1], yerr=[err_obs_dn[3:len(bins_100)-1], err_obs_up[3:len(bins_100)-1]], linestyle='--',linewidth=2,color='k',marker='*',ms=10,mfc='red', ecolor='r', capthick=2, label='$\\rm ALFALFA\ 100\  (Haynes+ 2018)$')

# 	plt.legend(frameon=False)

# 	plt.ylabel('$\\rm log_{10}(N$/$ area^{-1}dex^{-1})$')
# 	plt.xlabel('$\\rm log_{10}(W_{50}$)[$ km s^{-1}]$')
# 	#plt.title('Velocity Width Distribution', size=20)
# 	plt.xlim(1.2,2.8)
# 	plt.xticks()
# 	plt.yticks()
# 	plt.show()
	

# #plotting_VF(W_50_sel_2/np.sqrt(h), W_50_sel/np.sqrt(h), W_50_100, '', 6300, 'Medi-SURFS', 'Micro-SURFS' )


# #############################################################################################################################################################################
# ### PLOTTING VELOCITY FUNCTION
# #****************************************************************************************************************************************************************************


# def plot_all(velocity,width, name, name_width, title_name,color,log) :

# 	#path_save = path

# 	# Normal - W_50 for all

# 	plt.figure(figsize = (12,8))

# 	plt.scatter(velocity, width, s=2, alpha = 0.5, c = color, cmap='plasma_r')
# 	plt.plot(velocity, velocity, linestyle = ':', linewidth=0.2)
# 	cbar = plt.colorbar(pad = 0.1)
# 	cbar.ax.set_title('$(M_{HI}$  $M_{\odot})$', size=40)
# 	plt.xticks(size=40)
# 	plt.yticks(size=40)

	
	

# 	if log == 1:
# 		plt.xscale('log')
# 		plt.yscale('log')
# 		plt.xlabel('$V_{%s}$/$kms^{-1}$' %(name))
# 		plt.ylabel('%s/2sin(i) /$kms^{-1}$' %(name_width))
# 		#plt.title('$%s/2$ vs $Velocity_{%s}$' %(name_width,name),size=20)
# 		plt.title('%s'%title_name)
# 		plt.xlim(0.1*10**2,0.5*10**3)
# 		plt.ylim(0.1*10**2,0.5*10**3)
# 		#plt.savefig(path_save + name + name_width + 'log' + '.png')
# 		plt.show()
	
# 	else:		

# 		plt.xlabel('$Velocity_{%s}$ km/s' %(name))
# 		plt.ylabel('$%s/2sin(i)$/$kms^{-1}$' %(name_width))
# 		#plt.title('$%s/2$ vs $Velocity_{%s}$' %(name_width,name),size=20)
# 		plt.title('Micro',size=20)
# 		plt.xlim(0,200)
# 		plt.ylim(0,200)
# 		#plt.savefig(path_save + name + name_width + '.png')
# 		plt.show()




#plot_all(V_circ_sel_2, W_50_sel_2/2, 'max', '$W_{50}$', 'Medi-SURFS',np.log10(M_HI_sel_2), 1)







# def selection_plot():
# 	bins_host_medi = np.arange(np.log10(min(W_50_2[W_50_2 != 0])),np.log10(max(W_50_2[W_50_2 != 0])),0.1)
# 	Z_all_medi, bins = np.histogram(np.log10(W_50_2[W_50_2 != 0]), bins = bins_host_medi)
# 	Z_sel_medi, bins = np.histogram(np.log10(W_50_sel_2[W_50_sel_2 != 0]), bins = bins_host_medi)

# 	bins_host_medi = np.arange(np.log10(min(W_50[W_50 != 0])),np.log10(max(W_50[W_50 != 0])),0.1)
# 	Z_all_micro, bins = np.histogram(np.log10(W_50[W_50 != 0]), bins = bins_host_medi)
# 	Z_sel_micro, bins = np.histogram(np.log10(W_50_sel[W_50_sel != 0]), bins = bins_host_medi)

# 	fraction_micro = Z_sel_micro/Z_all_micro
# 	fraction_medi  = Z_sel_medi/Z_all_medi

# 	#plt.plot(bins_host_medi[0:15],fraction_micro[0:15],label='Micro')
# 	#plt.plot(bins_host_medi[0:15],fraction_medi[0:15],label='Medi')

# 	plt.hist(np.log10(W_50_2[W_50_2 != 0]),bins=bins_host_medi, label='All galaxies',alpha=0.8,color='darkseagreen')
# 	plt.hist(np.log10(W_50_sel_2[W_50_sel_2 != 0]),bins=bins_host_medi, label='Mock-survey galaxies',alpha=0.8,color='palevioletred')
# 	plt.xlabel(' $\\rm log_{10}(W_{50})$ km/s')
# 	plt.ylabel('Number of galaxies')
# 	plt.title('Medi-SURFS')
# 	plt.legend()
# 	plt.show()

# 	plt.hist(np.log10(W_50[W_50 != 0]),bins=bins_host_medi, label='All galaxies',alpha=0.8,color='palegoldenrod')
# 	plt.hist(np.log10(W_50_sel[W_50_sel != 0]),bins=bins_host_medi, label='Mock-survey galaxies',alpha=0.8,color='indianred')
# 	plt.xlabel('$\\rm log_{10}(W_{50})$ km/s')
# 	plt.ylabel('Number of galaxies')
# 	plt.title('Micro-SURFS')
# 	plt.legend()
# 	plt.show()



# #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ########### SUBPLOT
# #********************************************************************************************************************************************************************

# median_velocity,bin_edges, some = scipy.binned_statistic(np.log10(V_circ[W_50 != 0]),np.log10(W_50[W_50 != 0]/2),statistic='median',bins=50)
# median_velocity_2 = scipy.binned_statistic(np.log10(V_circ_2[W_50_2 !=0]/4.5),np.log10(W_50_2[W_50_2 !=0]/4.5), statistic='mean', bins=50)[0]

# median_velocity_20,bin_edges_20, some = scipy.binned_statistic(np.log10(V_circ[W_20 != 0]),np.log10(W_20[W_20 != 0]/2),statistic='median',bins=50)
# median_velocity_20_2 = scipy.binned_statistic(np.log10(V_circ_2[W_20_2 !=0]/4.5),np.log10(W_20_2[W_20_2 !=0]/4.5), statistic='mean', bins=50)[0]

# f,axarr = plt.subplots(2,2,sharex='all',sharey='all')
# axarr[0,0].set_ylim(1,3)
# axarr[0,0].set_xlim(1,3)

# axarr[0,0].set_ylabel("$\\rm log_{10}(W_{50}^{edge})\ [km/s]$")
# axarr[1,0].set_ylabel("$\\rm log_{10}(W_{20}^{edge})\ [km/s]$")

# #axarr[0,0].set_xlabel("$\\rm V_{max}$")
# axarr[1,0].set_xlabel("$\\rm log_{10}(V_{max})\ [km/s]$")
# axarr[1,1].set_xlabel("$\\rm log_{10}(V_{max})\ [km/s]$")


# a=axarr[0,0].scatter(np.log10(V_circ[W_50 != 0]),np.log10(W_50[W_50 != 0]/2),s=0.5,c=np.log10(M_HI[W_50 != 0]),cmap='plasma_r')
# axarr[0,0].plot([1,3],[1,3],'--k',linewidth=1)
# axarr[0,0].plot(bin_edges[0:50],median_velocity,'-k',linewidth=1)
# axarr[0,0].text(1.1,2.8,'$\\rm Micro-SURFS$', fontsize=12)
# axarr[1,0].scatter(np.log10(V_circ[W_20 != 0]),np.log10(W_20[W_20 != 0]/2),s=0.5,c=np.log10(M_HI[W_20 != 0]),cmap='plasma_r')
# axarr[1,0].plot([1,3],[1,3],'--k',linewidth=1)
# axarr[1,0].plot(bin_edges_20[0:50],median_velocity_20,'-k',linewidth=1)
# axarr[1,0].text(1.1,2.8,'$\\rm Micro-SURFS$', fontsize=12)
# axarr[0,1].scatter(np.log10(V_circ_2[W_50_2 != 0]),np.log10(W_50_2[W_50_2 != 0]/2),s=0.5,c=np.log10(M_HI_2[W_50_2 != 0]),cmap='plasma_r')
# axarr[0,1].plot([1,3],[1,3],'--k',linewidth=1)
# axarr[0,1].plot(bin_edges[0:50],median_velocity_2,'-k',linewidth=1)
# axarr[0,1].text(1.1,2.8,'$\\rm Medi-SURFS$', fontsize=12)
# axarr[1,1].scatter(np.log10(V_circ_2[W_20_2 != 0]),np.log10(W_20_2[W_20_2 != 0]/2),s=0.5,c=np.log10(M_HI_2[W_20_2 != 0]),cmap='plasma_r')
# axarr[1,1].plot([1,3],[1,3],'--k',linewidth=1)
# axarr[1,1].plot(bin_edges_20[0:50],median_velocity_20_2,'-k',linewidth=1)
# axarr[1,1].text(1.1,2.8,'$\\rm Medi-SURFS$', fontsize=12)

# f.subplots_adjust(right=0.8)
# cbar_ax = f.add_axes([0.85,0.13,0.02,0.8])
# f.colorbar(a,cax=cbar_ax)
# cbar_ax.set_ylabel('$\\rm log_{10}(M_{HI})\ [M_{\odot}]$')
# plt.show()




#############################################################################################################################################################################
### PLOTTING COMPARISON FUNCTIONS
#****************************************************************************************************************************************************************************


# def plot_2d_all():

# 	from matplotlib import colors
# 	#import matplotlib as mpl
# 	f,axarr = plt.subplots(3,1,sharey='all',sharex='all')

# 	bins_x_100 = np.arange(np.log10(min(W_50_100)) - 1, np.log10(max(W_50_100)) + 1, 0.1)
# 	bins_y_100 = np.arange(min(M_HI_100) - 1, max(M_HI_100) + 1, 0.2)

# 	median_w50_100 = scipy.binned_statistic(np.log10(W_50_100),M_HI_100,statistic='median',bins=bins_x_100)[0]

	
# 	bins_x = np.arange(np.log10(min(W_50_sel[W_50_sel != 0])) - 1, np.log10(max(W_50_sel[W_50_sel != 0])) + 1, 0.1)
# 	bins_y = np.arange(np.log10(min(M_HI_sel[W_50_sel != 0])) - 1, np.log10(max(M_HI_sel[W_50_sel != 0])) + 1, 0.2)

# 	#H, xbins, ybins = np.histogram2d(np.log10(W_50_sel[W_50_sel != 0]), np.log10(M_HI_sel[W_50_sel != 0]), bins = (bins_x, bins_y), normed = True)
# 	median_w50 = scipy.binned_statistic(np.log10(W_50_micro_median[W_50_micro_median != 0]),np.log10(M_HI_micro_median[W_50_micro_median != 0]),statistic='median',bins=bins_x)[0]

# 	a =axarr[1].hist2d(np.log10(W_50_micro[W_50_micro != 0]), np.log10(M_HI_micro[W_50_micro != 0]), bins = (bins_x, bins_y), norm=colors.LogNorm(vmin=1,vmax=10000), cmap = plt.cm.jet)
# 	#axarr[1].hist2d(np.log10(W_50_4[W_50_4 != 0]), np.log10(M_HI_4[W_50_4 != 0]), bins = (bins_x, bins_y), norm=colors.LogNorm(vmin=1,vmax=10000), cmap = plt.cm.jet)
# 	axarr[1].plot(bins_x[0:len(bins_x)-4],median_w50[0:len(bins_x)-4],'-k')
# 	axarr[1].plot(bins_x_100[0:len(bins_x_100)-1],median_w50_100[0:len(bins_x_100)-1],'--k')
# 	axarr[1].set_ylabel('$\\rm log_{10}(M_{HI})\ [M_{\odot}]$')
# 	#axarr[0].set_xlabel('$\\rm log_{10}(W_{50})\ [km/s]$')
# 	axarr[1].set_title('Micro-SURFS (Lightcone)')
# 	# cbar = plt.colorbar(a[3],orientation='horizontal',pad=0.5)
# 	# cbar.ax.set_title('Number of galaxies')
# 	axarr[1].set_xlim(1.5,3)
# 	axarr[1].set_ylim(6.5,11)
	
# 	bins_x = np.arange(np.log10(min(W_50_sel[W_50_sel != 0])) - 1, np.log10(max(W_50_sel[W_50_sel != 0])) + 1, 0.1)
# 	bins_y = np.arange(np.log10(min(M_HI_sel[W_50_sel != 0])) - 1, np.log10(max(M_HI_sel[W_50_sel != 0])) + 1, 0.2)

# 	median_w50 = scipy.binned_statistic(np.log10(W_50_medi_median[W_50_medi_median != 0]),np.log10(M_HI_medi_median[W_50_medi_median != 0]),statistic='median',bins=bins_x)[0]

# 	a =axarr[2].hist2d(np.log10(W_50_medi[W_50_medi != 0]), np.log10(M_HI_medi[W_50_medi != 0]), bins = (bins_x, bins_y), norm=colors.LogNorm(vmin=1,vmax=10000), cmap = plt.cm.jet)
# 	#axarr[2].hist2d(np.log10(W_50_2[W_50_2 != 0]), np.log10(M_HI_2[W_50_2 != 0]), bins = (bins_x, bins_y), normed=True, cmap = plt.cm.jet,cmin=0.01)
# 	axarr[2].plot(bins_x[0:len(bins_x)-4],median_w50[0:len(bins_x)-4],'-k')
# 	axarr[2].plot(bins_x_100[0:len(bins_x_100)-1],median_w50_100[0:len(bins_x_100)-1],'--k')
# 	axarr[2].set_ylabel('$\\rm log_{10}(M_{HI})\ [M_{\odot}]$')
# 	axarr[2].set_xlabel('$\\rm log_{10}(W_{50})\ [km/s]$')
# 	axarr[2].set_title('Medi-SURFS (Lightcone)')
# 	#cbar = plt.colorbar(a[3],orientation='horizontal')
# 	# cbar.ax.set_title('Number of galaxies')
# 	axarr[2].set_xlim(1.5,3)
# 	axarr[2].set_ylim(6.5,11)
	


# 	plt.show()
# 	#plt.savefig('Lightcone_2d_hist.eps')

	
# 	f,axarr = plt.subplots(3,1,sharex='all',sharey='all')

# 	bins_x = np.arange(np.log10(min(W_50_2[W_50_2 != 0])) - 1, np.log10(max(W_50_2[W_50_2 != 0])) + 1, 0.1)
# 	bins_y = np.arange(np.log10(min(M_HI_2[W_50_2 != 0])) - 1, np.log10(max(M_HI_2[W_50_2 != 0])) + 1, 0.2)

# 	median_w50 = scipy.binned_statistic(np.log10(W_50_sel[W_50_sel != 0]),np.log10(M_HI_sel[W_50_sel != 0]),statistic='median',bins=bins_x)[0]

# 	a=axarr[0].hist2d(np.log10(W_50_sel[W_50_sel != 0]), np.log10(M_HI_sel[W_50_sel != 0]*h), bins = (bins_x, bins_y), norm=colors.LogNorm(vmin=1,vmax=10000), cmap = plt.cm.jet)
# 	#axarr[0].hist2d(np.log10(W_50_sel[W_50_sel != 0]), np.log10(M_HI_sel[W_50_sel != 0]), bins = (bins_x, bins_y), normed=True, cmap = plt.cm.jet,cmin=0.01)
# 	axarr[0].plot(bins_x[0:len(bins_x)-4],median_w50[0:len(bins_x)-4],'-k')
# 	axarr[0].plot(bins_x_100[0:len(bins_x_100)-1],median_w50_100[0:len(bins_x_100)-1],'--k')
# 	axarr[0].set_ylabel('$\\rm log_{10}(M_{HI})\ [M_{\odot}]$')
# 	#axarr[0].set_xlabel('$\\rm log_{10}(W_{50})\ [km/s]$')
# 	axarr[0].set_title('Micro-SURFS (ALFALFA Selection)')
# 	# cbar = plt.colorbar(a[3],orientation='horizontal')
# 	# cbar.ax.set_title('Number of galaxies')
# 	axarr[0].set_xlim(1.5,3)
# 	axarr[0].set_ylim(6.5,11)
	

# 	bins_x = np.arange(np.log10(min(W_50_sel_2[W_50_sel_2 != 0])) - 1, np.log10(max(W_50_sel_2[W_50_sel_2 != 0])) + 1, 0.1)
# 	bins_y = np.arange(np.log10(min(M_HI_sel_2[W_50_sel_2 != 0])) - 1, np.log10(max(M_HI_sel_2[W_50_sel_2 != 0])) + 1, 0.2)

# 	median_w50 = scipy.binned_statistic(np.log10(W_50_sel_2[W_50_sel_2 != 0]),np.log10(M_HI_sel_2[W_50_sel_2 != 0]),statistic='median',bins=bins_x)[0]

# 	a = axarr[1].hist2d(np.log10(W_50_sel_2[W_50_sel_2 != 0]), np.log10(M_HI_sel_2[W_50_sel_2 != 0]*h), bins = (bins_x, bins_y), norm=colors.LogNorm(vmin=1,vmax=10000), cmap = plt.cm.jet)
# 	axarr[1].plot(bins_x_100[0:len(bins_x_100)-1],median_w50_100[0:len(bins_x_100)-1],'--k')
# 	#axarr[1].hist2d(np.log10(W_50_sel_2[W_50_sel_2 != 0]), np.log10(M_HI_sel_2[W_50_sel_2 != 0]), bins = (bins_x, bins_y), normed=True, cmap = plt.cm.jet,cmin=0.01)
# 	axarr[1].plot(bins_x[0:len(bins_x)-4],median_w50[0:len(bins_x)-4],'-k')
# 	axarr[1].set_ylabel('$\\rm log_{10}(M_{HI})\ [M_{\odot}]$')
# 	#axarr[1].set_xlabel('$\\rm log_{10}(W_{50})\ [km/s]$')
# 	axarr[1].set_title('Medi-SURFS (ALFALFA selection)')
# 	axarr[1].set_xlim(1.5,3)
# 	axarr[1].set_ylim(6.5,11)
	


	
# 	axarr[2].hist2d(np.log10(W_50_100),M_HI_100, bins = (bins_x, bins_y), norm=colors.LogNorm(vmin=1,vmax=10000), cmap = plt.cm.jet)
# 	#axarr[2].hist2d(np.log10(W_50_100),M_HI_100, bins = (bins_x, bins_y), normed=True, cmap = plt.cm.jet,cmin=0.01)
# 	axarr[2].plot(bins_x_100[0:len(bins_x_100)-1],median_w50_100[0:len(bins_x_100)-1],'--k')
# 	axarr[2].set_ylabel('$\\rm log_{10}(M_{HI})\ [M_{\odot}]$')
# 	axarr[2].set_xlabel('$\\rm log_{10}(W_{50})\ [km/s]$')
# 	axarr[2].set_title('ALFALFA Observations')
# 	axarr[2].set_xlim(1.5,3)
# 	axarr[2].set_ylim(6.5,11)

# 	plt.show()
# 	#plt.savefig('Selection_2d_hist.eps')


# plt.scatter(np.log10(V_circ[W_50 != 0]),np.log10(W_50[W_50 != 0]/2),s=0.5,c=np.log10(M_HI[W_50 != 0]),cmap='plasma_r')
# plt.plot([1,3],[1,3],'--k',linewidth=1)
# plt.xlabel("$\\rm V_{max}\ [km/s]$")
# plt.ylabel("$\\rm W_{50}^{edge}\ [km/s]$")
# cbar = plt.colorbar(pad = 0.1)
# cbar.ax.set_title('$(M_{HI}$  $M_{\odot})$')
# plt.xticks()
# plt.yticks()
# plt.show()


#exec(open("SHARK_properties.py").read())

#exec(open("Plots_Shark_reading.py").read())



#*****************************************************************************************************************************************************************************

#exec(open("Plots_Shark_reading.py").read())



##############################################################################################################################################################################
###############################################################################################################################################################################
#### COMMENTED LINES
###############################################################################################################################################################################
###############################################################################################################################################################################

#*****************************************************************************************************************************************************************************



# volume_lightcone = 45787520.0

# bins_x_2 = np.arange(np.log10(min(V_circ[V_circ != 0])), np.log10(max(V_circ[V_circ != 0])), 0.1)

# H_all_2, bin_edges_all	= np.histogram(np.log10(V_circ[V_circ != 0]), bins = bins_x_2)
# H_sel_2, bin_edges_sel	= np.histogram(np.log10(V_circ_sel[V_circ_sel != 0]), bins =bins_x_2)

# H_final_2 = H_sel_2/H_all_2

# H_distribution_2 = np.zeros(len(bins_x_2))

# for i in range(len(H_final_2)):
# 	H_distribution_2[i] = H_final_2[i]/volume_lightcone/0.1


# plt.plot(bins_x_2,H_distribution_2)
# plt.show()


# volume_shark = 64000

# H_all, bin_edges_all	= np.histogram(np.log10(vmax_subhalo_all_2[vmax_subhalo_all_2 != 0]), bins = bins_x_2)
# H_sel, bin_edges_sel	= np.histogram(np.log10(V_circ_sel[V_circ_sel != 0]), bins =bins_x_2)

# H_final = H_sel/volume_lightcone/H_all/volume_shark


# H_distribution = np.zeros(len(bins_x_2))

# for i in range(len(H_final)):
# 	H_distribution[i] = H_final[i]/0.1/(volume_shark/volume_lightcone)

# plt.plot(bins_x_2,H_distribution)
# plt.show()


# H_trial = H_distribution*H_distribution_2

# plt.plot(bins_x_2,H_trial)
# plt.show()


# bins_y_2 = np.arange(np.log10(min(M_HI_sel_2[W_50_sel_2 != 0])), np.log10(max(M_HI_sel_2[W_50_sel_2 != 0])), 0.1)

# H_all_2, bin_edges_all, binnumber_all = np.histogram2d(np.log10(W_50_2), np.log10(M_HI_2), bins = (bins_x_2, bins_y_2))
# H_sel_2, bin_edges_sel, binnumber_sel = np.histogram2d(np.log10(W_50_sel_2), np.log10(M_HI_sel_2), bins = (bins_x_2, bins_y_2))

# for i in range(len(bin_edges_all)-1):
#   for j in range(len(binnumber_all)-1):
#       if H_all_2[i,j] <= 5:
#           H_all_2[i,j] = 0


# H_final_2 = H_sel_2/H_all_2 

# plt.figure(figsize=(12,12))
# plt.imshow(H_final_2, extent=(bins_y_2[0],bins_y_2[-1],bins_x_2[0],bins_x_2[-1]), aspect = 'auto', cmap = 'gist_ncar')
# plt.ylabel('$log_{10}\ (W_{50}\ /kms^{-1})$', size=30)
# plt.xlabel('$log_{10}(M_{HI}/ M_{\odot})$', size=30)
# plt.title('Medi SURFS', size = 30)
# cbar = plt.colorbar(orientation='horizontal', shrink=0.7, pad = 0.2)
# cbar.ax.set_title('$(Galaxies^{ALFALFA}$ / $Galaxies^{Lightcone})$', size=20)
# plt.clim(0,1)
# plt.xlim(6.5,11)
# plt.savefig('Plots_Yearly/Final Plots/comparison_medi.png')
# plt.show()


# # In[ ]:


# xbins = np.arange(np.log10(min(W_50_sel[W_50_sel != 0])), np.log10(max(W_50_sel[W_50_sel != 0])), 0.1)
# ybins = np.arange(np.log10(min(M_HI_sel[W_50_sel != 0])), np.log10(max(M_HI_sel[W_50_sel != 0])), 0.1)

# H_all, bin_edges_all, binnumber_all = np.histogram2d(np.log10(W_50), np.log10(M_HI), bins = (xbins, ybins))
# H_sel, bin_edges_sel, binnumber_sel = np.histogram2d(np.log10(W_50_sel), np.log10(M_HI_sel), bins = (xbins, ybins))

# for i in range(len(bin_edges_all)-1):
#   for j in range(len(binnumber_all)-1):
#       if H_all[i,j] <= 5:
#           H_all[i,j] = 0


# H_final = H_sel/H_all 

# plt.figure(figsize=(12,12))
# plt.imshow(H_final, extent=(ybins[0],ybins[-1],xbins[0],xbins[-1]), aspect = 'auto',cmap='gist_ncar',        )
# plt.ylabel('$log_{10}\ (W_{50}\ /kms^{-1})$', size=30)
# plt.xlabel('$log_{10}(M_{HI}/ M_{\odot})$', size=30)
# cbar = plt.colorbar(orientation='horizontal', shrink=0.7, pad = 0.2)
# cbar.ax.set_title('$(Galaxies^{ALFALFA}$ / $Galaxies^{Lightcone})$', size=20)
# #plt.clim(0,1)
# plt.xlim(6.5,11)
# plt.ylim(1.4,2.8)
# plt.title('Micro-SURFS', size = 30)
# plt.savefig('Plots_Yearly/Final Plots/comparison_micro.png')
# plt.show()


# # In[ ]:

# ##################################################################################################################################################
# ### Adding Beam Confusion
# ###################################################################################################################################################


# maxR = 0.063
# #maxR = 0.00110538

# #maxR = 1.0110538

# corr_W50 = W_50_sel
# corr_W20 = W_20_sel
# corr_speak = s_peak_sel
# corr_zobs  = z_obs_sel
# new_MHI    = M_HI_sel
# new_zobs   = z_obs_sel


# corr_flux = flux_central[flux_central >= yo]

# indices_removing = []

# #############################################################################################################
# indicesSorted = np.argsort(ra_sel) # Sort the RA array and return the indices of the sorted array (not the values)
# sep = maxR # The separation

# for n, ii in enumerate(indicesSorted):
#     if n%1000==0:
#         print(n)
#     for jj in indicesSorted[n:]:
#         if (ii != jj):
							   
#             if ((ra_sel[jj])-ra_sel[ii]>sep): # Check if the separation in RA is beyond the maximum separation (i.e. no point continuing) 
				
#                 continue # move on to the next item
			
#             if (np.sqrt((ra_sel[jj]-ra_sel[ii])**2+(np.abs(dec_sel[jj])-np.abs(dec_sel[ii]))**2) <= sep): # Check whether the next galaxy falls within the circle

#                 del_V = np.abs(W_50_sel[ii] + W_50_sel[jj])/2
#                 del_V_2 = np.abs(W_20_sel[ii] + W_20_sel[jj])/2
				
#                 del_z = np.abs(z_obs_sel[ii]*c - z_obs_sel[jj]*c)

#                 if del_V < del_z:

#                     corr_W50[ii] = (corr_W50[ii] + corr_W50[jj]) + del_V
#                     corr_W20[ii] = (corr_W20[ii] + corr_W20[jj]) + del_V_2
#                     corr_speak[ii] = max(s_peak_sel[ii], s_peak_sel[jj])
#                     corr_zobs[ii] = (flux_central[ii]*z_obs_sel[ii] +                                  flux_central[jj]*z_obs_sel[jj])/(2*(flux_central[ii] + flux_central[jj]))
#                     corr_flux[ii] = corr_flux[ii] + corr_flux[jj]
#                     new_MHI[ii] = M_HI_sel[ii] + M_HI_sel[jj]
#                     new_zobs[ii] = z_obs_sel[ii]*c - W_50_sel[ii]/2 + corr_W50[ii]/2


#                     indices_removing = np.append(indices_removing,jj)
					
					


# print(len(indices_removing))
# print(len(corr_W50))


# print(len(corr_W50))
# print(indices_removing)

# indices_final = list(map(int, indices_removing))

# indices_final = np.argsort(indices_final)

# new_W50 = corr_W50.tolist()
# new_W20 = corr_W20.tolist()
# new_MHI = new_MHI.tolist()


# for i in sorted(indices_final):
#     del new_W50[i]
#     del new_W20[i]
#     del new_MHI[i]

# len(new_W50)


# ##################################################################################################################################################
# ### Adding Beam Confusion
# ###################################################################################################################################################


# maxR = 0.063
# #maxR = 0.00110538

# #maxR = 1.0110538

# corr_W50_2 = W_50_sel_2
# corr_W20_2 = W_20_sel_2
# #corr_speak_2 = s_peak_sel_2
# corr_zobs_2  = z_obs_sel_2
# new_MHI_2    = M_HI_sel_2
# new_zobs_2   = z_obs_sel_2


# corr_flux_2 = flux_central_2[flux_central_2 >= yo_2]

# indices_removing = []

# #############################################################################################################
# indicesSorted = np.argsort(ra_sel_2) # Sort the RA array and return the indices of the sorted array (not the values)
# sep = maxR # The separation

# for n, ii in enumerate(indicesSorted):
#     if n%1000==0:
#         print(n)
#     for jj in indicesSorted[n:]:
#         if (ii != jj):
							   
#             if ((ra_sel_2[jj])-ra_sel_2[ii]>sep): # Check if the separation in RA is beyond the maximum separation (i.e. no point continuing) 
				
#                 continue # move on to the next item
			
#             if (np.sqrt((ra_sel_2[jj]-ra_sel_2[ii])**2+(np.abs(dec_sel_2[jj])-np.abs(dec_sel_2[ii]))**2) <= sep): # Check whether the next galaxy falls within the circle

#                 del_V = np.abs(W_50_sel_2[ii] + W_50_sel_2[jj])/2
#                 del_V_2 = np.abs(W_20_sel_2[ii] + W_20_sel_2[jj])/2
				
#                 del_z = np.abs(z_obs_sel_2[ii]*c - z_obs_sel_2[jj]*c)

#                 if del_V < del_z:

#                     corr_W50_2[ii] = (corr_W50_2[ii] + corr_W50_2[jj]) + del_V
#                     corr_W20_2[ii] = (corr_W20_2[ii] + corr_W20_2[jj]) + del_V_2
#                     #corr_speak_2[ii] = max(s_peak_sel_2[ii], s_peak_sel_2[jj])
#                     corr_zobs_2[ii] = (flux_central_2[ii]*z_obs_sel_2[ii] +                                    flux_central_2[jj]*z_obs_sel_2[jj])/(2*(flux_central_2[ii] + flux_central_2[jj]))
#                     corr_flux_2[ii] = corr_flux_2[ii] + corr_flux_2[jj]
#                     new_MHI_2[ii] = M_HI_sel_2[ii] + M_HI_sel_2[jj]
#                     new_zobs_2[ii] = z_obs_sel_2[ii]*c - W_50_sel_2[ii]/2 + corr_W50_2[ii]/2


#                     indices_removing = np.append(indices_removing,jj)
					

# print(len(corr_W50_2))
# print(indices_removing)

# indices_final = list(map(int, indices_removing))

# indices_final = np.argsort(indices_final)

# new_W50_2 = corr_W50_2.tolist()
# new_W20_2 = corr_W20_2.tolist()
# new_MHI_2 = new_MHI_2.tolist()

# for i in sorted(indices_final):
#     del new_W50_2[i]
#     del new_W20_2[i]
#     del new_MHI_2[i]

# len(new_W50_2)


# new_MHI_2 = np.array(new_MHI_2)

