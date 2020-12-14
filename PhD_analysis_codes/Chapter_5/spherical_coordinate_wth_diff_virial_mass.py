import h5py as h5py
import numpy as np
import os
import scipy.stats as scipy
import re
from scipy import stats
from astropy.stats import bootstrap
from astropy.utils import NumpyRNGContext
#import seaborn as sns
import time 
import collections
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.colors as colors
import splotch as spl
import statistics as statistics
import csv as csv
from Common_module import SharkDataReading
import Common_module

import pandas as pd
import splotch as spl

############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
### Defining data_type and constants
######--------------------------------------------------------------------------------------------------------------------------------------
####****************************************************************************************************************************************

df  = np.dtype(float)
G   = 4.301e-9 #Mpc
h   = 0.6751
M_solar_2_g   = 1.99e33
dt = int
df = float

############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
###                                                             Function 
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************

def Mvir2Rvir(mass, crit_fac=200., z=0, H_0=67.51, Omega_R=0, Omega_M=0.3121, Omega_L=0.6879):

	def critdens(z,H_0=67.51,Omega_R=0,Omega_M=0.3121,Omega_Lambda=0.6879):
		# Find critical density at redshift z.  Will probably need alteration as well
		G = 6.67384e-11 # Gravitational constant
		Mpc_km = 3.08567758e19 # Number of km in 1 Mpc
		pc = 3.08567758e16 # Number of m in 1 pc
		M_sun = 1.9891e30 # Number of kg in 1 solar mass
		
		Hsqr = (H_0**2)*(Omega_R*(1+z)**4 + Omega_M*(1+z)**3 + Omega_Lambda) # Square of Hubble parameter at redshift z
		#print 'H(z)', np.sqrt(Hsqr)
		
		Hsqr = Hsqr/(Mpc_km**2) # Convert Hubble parameter to seconds
		rho_crit = 3*Hsqr/(8*np.pi*G) # Critical density in kg/m^3
		return (rho_crit*(pc)**3)/M_sun # Critical density in M_sun/pc^3


	# convert virial mass [Msun] to virial radius [Mpc]
	vol = mass / critdens(z,H_0,Omega_R,Omega_M,Omega_L) / crit_fac
	return (3.*vol/4./np.pi)**(1./3.) * 1e-6




############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
###                                                             Observational Data
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************

Guo_halo_isolated 		= np.array([11.118972,11.383127,11.616222,11.888279,12.125619,12.384735,12.630501,12.885231,13.258829,13.750053]) - np.log10(h)
Guo_HI_isolated 		= np.array([8.896476,9.197137,9.401432,9.436123,9.586453,9.678965,9.732929,9.856277,10.068282,10.09141]) - np.log10(h)

Guo_halo_group 			= np.array([11.375638,11.635152,11.894217,12.144291,12.372295,12.635426,12.872547,13.258932,13.754722]) - np.log10(h)
Guo_HI_group 			= np.array([9.601872,9.806168,9.902533,9.925661,9.906387,9.906387,9.991189,10.099119,10.153084]) - np.log10(h)

Guo_HI_isolated_upper 	= np.array([8.925696,9.208094,9.41009,9.435954,9.595778,9.694416,9.743239,9.883966,10.062517,10.1294775]) - np.log10(h) - Guo_HI_isolated
Guo_HI_isolated_lower 	= Guo_HI_isolated - np.array([8.887403,9.188963,9.390961,9.432108,9.572819,9.663798,9.708758,9.834168,10.031865,10.022272]) + np.log10(h)

Guo_HI_group_upper 		= np.array([9.705927,9.877222,9.9605255,9.990219,9.924178,9.938503,10.010336,10.116128,10.2214155]) - np.log10(h) - Guo_HI_group
Guo_HI_group_lower 		= Guo_HI_group - np.array([9.491466,9.739415,9.853371,9.9059725,9.85147,9.869625,9.945203,10.070175,10.114194]) + np.log10(h)

Guo_halo_central 		= np.array([11.128378,11.385135,11.621622,11.864865,12.121622,12.364865,12.594595,12.8918915,13.25,13.756757]) - np.log10(h)
Guo_HI_central 			= np.array([8.887856,9.171226,9.37035,9.420132,9.531181,9.580963,9.623085,9.695843,9.695843,9.607768]) - np.log10(h)

Guo_halo_satellite 		= np.array([12.140271,12.377828,12.608598,12.880091,13.239819,13.755656]) - np.log10(h)
Guo_HI_satellite 		= np.array([8.677083,8.964912,9.076206,9.337171,9.805373,9.8936405]) - np.log10(h)

Guo_HI_ng_2 			= np.array([9.167,9.445,9.660,9.733,9.798,9.805,9.858,9.941,10.104,10.165]) - np.log10(h)




############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
###                                                             Reading Data
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************

# path_GAMA 	= '/mnt/su3ctm/gchauhan/HI_stacking_paper/GAMA_files_Matias/'

path_GAMA 	= '/mnt/su3ctm/gchauhan/HI_stacking_paper/Matias_GAMA_new/'

# path_GAMA 	= '/mnt/su3ctm/gchauhan/HI_stacking_paper/Old_GAMA_files/GAMA_files_Matias/'

# path_GAMA 	= '/mnt/sshfs/pleiades_gchauhan/HI_stacking_paper/Matias_GAMA_new/'

path 		= '/mnt/su3ctm/clagos/SHARK_Out/'

# path 		= '/mnt/sshfs/pleiades_gchauhan/SHArk_Out/HI_haloes/'

path_plot   = '/mnt/su3ctm/gchauhan/HI_stacking_paper/Plots/Paper_plots/Distribution_plots/'

shark_runs 		= ['Shark-TreeFixed-ReincPSO-kappa0p002','Shark-Lagos18-default-br06','Shark-Lagos18-Kappa-10','Shark-Lagos18-Kappa-0','Shark-Lagos18-Kappa-1','Shark-Lagos18-default-br06-stripping-off']

shark_labels	= ['SHARK-ref','Kappa = 0.02','Kappa = 0','Kappa = 1','Lagos18 (stripping off)']

with open("/home/ghauhan/Parameter_files/redshift_list_medi.txt", 'r') as csv_file:  
# with open("/home/garima/Desktop/redshift_list_medi.txt", 'r') as csv_file:  
	trial = list(csv.reader(csv_file, delimiter=',')) 
	trial = np.array(trial[1:], dtype = np.float) 



simulation 		= ['medi-SURFS', 'micro-SURFS']
snapshot_avail	= [x for x in range(100,200,1)]
z_values 		= ["%0.2f" %x for x in trial[:,1]]

subvolumes 		= 64

############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
###                                                             Reading SHARK-ref File
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************

medi_Kappa_original 	= {}

for snapshot in snapshot_avail:

	medi_Kappa_original[snapshot]	 	= SharkDataReading(path,simulation[0],shark_runs[0],snapshot,subvolumes)


medi_HI, medi_HI_central, medi_HI_satellite,medi_HI_orphan,medi_vir,a = medi_Kappa_original[199].mergeValues_satellite('id_halo','matom_bulge','matom_disk')

medi_stellar, medi_stellar_central, medi_stellar_satellite,medi_stellar_orphan,a,b = medi_Kappa_original[199].mergeValues_satellite('id_halo','mstars_bulge','mstars_disk')


############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
###                                                             Reading GAMA Files
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************

GAMA_gals_file 		= pd.read_csv(path_GAMA + "Garima_T19-RR14_gals.csv")

GAMA_group_file 	= pd.read_csv(path_GAMA + "Garima_T19-RR14_group.csv")

GAMA_censat_file 	= pd.read_csv(path_GAMA +"Garima_T19-RR14_abmatch_magerr_censat85.csv" )


# GAMA_gals_file 		= pd.read_csv(path_GAMA + "GAMA_T19-RR14_gals.csv")

# GAMA_group_file 	= pd.read_csv(path_GAMA + "GAMA_T19-RR14_group.csv")

# GAMA_censat_file 	= pd.read_csv(path_GAMA +"GAMA_T19-RR14_abmatch_magerr_censat85.csv" )



############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
###                                                             Merging GAMA Files
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************

grand_gals		= GAMA_gals_file.merge(GAMA_censat_file,left_on='CATAID',right_on='id_galaxy_sky',how='inner')  # Merges Galaxy file with censat (merging CATAID and id_galaxy_sky)

grand_isolated 	= grand_gals.loc[(grand_gals['RankIterCen']==-999) & (grand_gals['Z']<=0.06)].copy() # only isolated centrals

grand_group 	= grand_gals.loc[(grand_gals['RankIterCen']==1) & (grand_gals['Z']<=0.06)].copy()   # separates the central of groups

grand_group 	= grand_group.merge(GAMA_group_file,on='GroupID',how='inner')  # merges group file with galaxy file


############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
###                                                   Reading Simulation Box and Matching up
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************

def prepare_data(path,simulation,run,snapshot,subvolumes):

	fields_read 	= {'galaxies':('id_halo', 'id_galaxy', 'type', 'mvir_hosthalo','mvir_subhalo', 'matom_bulge', 'matom_disk', 'position_x', 'position_y', 'position_z')}

	data_reading = SharkDataReading(path,simulation,run,snapshot,subvolumes)

	data_plotting = data_reading.readIndividualFiles_GAMA(fields=fields_read, snapshot_group=snapshot, subvol_group=subvolumes)

	return data_plotting





############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
###                                                   HI Matching  --- Group - ABMATCH
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************


matched_snapshot 		= np.array(grand_group['snapshot']) 
matched_subvolume 		= np.array(grand_group['subvolume']) 
matched_id_galaxy_sam	= np.array(grand_group['id_galaxy_sam']) 
matched_group_id 		= np.array(grand_group['GroupID']) 

M_HI_sam_mass_rvir_2_ab 		= np.zeros(len(matched_id_galaxy_sam))
M_HI_sam_mass_rvir_5_ab 		= np.zeros(len(matched_id_galaxy_sam))
M_HI_sam_mass_rvir_10_ab 		= np.zeros(len(matched_id_galaxy_sam))

M_HI_sam_central_group 		= np.zeros(len(matched_id_galaxy_sam))
central_group 		= np.zeros(len(matched_id_galaxy_sam))

Mvir_abundance_match 	= []


for snapshot_run, subvolume_run, i, group_id in zip(matched_snapshot, matched_subvolume, range(len(matched_id_galaxy_sam)), matched_group_id):

	(h0,_, id_halo_all_sam, id_galaxy_all_sam, is_central_all_sam, mvir_hosthalo,mvir_subhalo, matom_bulge, matom_disk, position_x_sam, position_y_sam, position_z_sam) = prepare_data(path,simulation[0],shark_runs[0],int(snapshot_run),int(subvolume_run))
	

	trial 			= np.where(id_galaxy_all_sam == int(matched_id_galaxy_sam[i]))[0]
	


	if len(trial) == 0 :

		HI_unique_sam = 0
		Mvir_add 	= 0
		M_HI_sam_mass[i] 		= HI_unique_sam
		Mvir_abundance_match	= np.append(Mvir_abundance_match, Mvir_add)


	else:

		print(i)

		HI_unique_sam 		= (matom_bulge + matom_disk)/h/1.35
		is_central_sam  	= is_central_all_sam[trial]
		central_group[i]	= is_central_sam

		position_x_cen 		= np.ones((len(position_x_sam)))*position_x_sam[trial]
		position_y_cen 		= np.ones((len(position_x_sam)))*position_y_sam[trial]
		position_z_cen 		= np.ones((len(position_x_sam)))*position_z_sam[trial]

		distance 			= np.sqrt((position_x_sam - position_x_cen)**2 + (position_y_sam - position_y_cen)**2 + (position_z_sam - position_z_cen)**2)					


		######------------------------------------------
		###   Mvir_mass Abundance Matching
		######------------------------------------------

		Mvir_abundance_match 		= np.append(Mvir_abundance_match, 10**(Common_module.abundanceMatchingMvir_luminosity(np.array(grand_group[grand_group['GroupID'] == group_id]['r_ab'])))/h)
		Mvir_here					= 10**(Common_module.abundanceMatchingMvir_luminosity(np.array(grand_group[grand_group['GroupID'] == group_id]['r_ab'])))/h
	
		######------------------------------------------
		###   R_vir = 2
		######------------------------------------------

		# merge_array 		= np.where(distance <  Mvir2Rvir(mass=grand_group['MassAfunc'][i], z=grand_group['Zfof'][i]))[0]
		merge_array 		= np.where(distance <  Mvir2Rvir(mass=Mvir_here, z=np.array(grand_group[grand_group['GroupID'] == group_id]['Zfof'])))[0]

		a = [int(k) for k in merge_array]

		add_HI 						= np.sum(HI_unique_sam[a]) #+ HI_unique_sam[trial]
		M_HI_sam_mass_rvir_2_ab[i] 	= add_HI

		
		######------------------------------------------
		###   R_vir = 1.5
		######------------------------------------------

		merge_array 		= np.where(distance <  0.75*Mvir2Rvir(mass=Mvir_here, z=np.array(grand_group[grand_group['GroupID'] == group_id]['Zfof'])))[0]

		a = [int(k) for k in merge_array]

		add_HI 						= np.sum(HI_unique_sam[a]) #+ HI_unique_sam[trial]
		M_HI_sam_mass_rvir_5_ab[i] 	= add_HI

		######------------------------------------------
		###   R_vir = 1
		######------------------------------------------

		merge_array 		= np.where(distance <  0.5*Mvir2Rvir(mass=Mvir_here, z=np.array(grand_group[grand_group['GroupID'] == group_id]['Zfof'])))[0]

		a = [int(k) for k in merge_array]

		add_HI						= np.sum(HI_unique_sam[a]) #+ HI_unique_sam[trial]
		M_HI_sam_mass_rvir_10_ab[i] 	= add_HI

		M_HI_sam_central_group[i] 	= HI_unique_sam[trial]

		

############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
###                                                   HI Matching  --- Group - Dynamical Mass
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************


matched_snapshot 		= np.array(grand_group['snapshot']) 
matched_subvolume 		= np.array(grand_group['subvolume']) 
matched_id_galaxy_sam	= np.array(grand_group['id_galaxy_sam']) 
matched_group_id 		= np.array(grand_group['GroupID']) 

M_HI_sam_mass_rvir_2_dyn 		= np.zeros(len(matched_id_galaxy_sam))
M_HI_sam_mass_rvir_5_dyn 		= np.zeros(len(matched_id_galaxy_sam))
M_HI_sam_mass_rvir_10_dyn 		= np.zeros(len(matched_id_galaxy_sam))

M_HI_sam_central_group 		= np.zeros(len(matched_id_galaxy_sam))
central_group 		= np.zeros(len(matched_id_galaxy_sam))

Mvir_dynamical_mass 	= []


for snapshot_run, subvolume_run, i, group_id in zip(matched_snapshot, matched_subvolume, range(len(matched_id_galaxy_sam)), matched_group_id):

	(h0,_, id_halo_all_sam, id_galaxy_all_sam, is_central_all_sam, mvir_hosthalo,mvir_subhalo, matom_bulge, matom_disk, position_x_sam, position_y_sam, position_z_sam) = prepare_data(path,simulation[0],shark_runs[0],int(snapshot_run),int(subvolume_run))
	

	trial 			= np.where(id_galaxy_all_sam == int(matched_id_galaxy_sam[i]))[0]
	


	if len(trial) == 0 :

		HI_unique_sam = 0
		Mvir_add 	= 0
		M_HI_sam_mass[i] 		= HI_unique_sam
		Mvir_abundance_match	= np.append(Mvir_dynamical_mass, Mvir_add)


	else:

		print(i)

		HI_unique_sam 		= (matom_bulge + matom_disk)/h/1.35
		is_central_sam  	= is_central_all_sam[trial]
		central_group[i]	= is_central_sam

		position_x_cen 		= np.ones((len(position_x_sam)))*position_x_sam[trial]
		position_y_cen 		= np.ones((len(position_x_sam)))*position_y_sam[trial]
		position_z_cen 		= np.ones((len(position_x_sam)))*position_z_sam[trial]

		distance 			= np.sqrt((position_x_sam - position_x_cen)**2 + (position_y_sam - position_y_cen)**2 + (position_z_sam - position_z_cen)**2)					


		######------------------------------------------
		###   Mvir_mass Abundance Matching
		######------------------------------------------

		Mvir_dynamical_mass 		= np.append(Mvir_dynamical_mass, np.array(grand_group[grand_group['GroupID'] == group_id]['MassAfunc'])/h)
		Mvir_here					= np.array(grand_group[grand_group['GroupID'] == group_id]['MassAfunc'])/h
	
		######------------------------------------------
		###   R_vir = 2
		######------------------------------------------

		# merge_array 		= np.where(distance <  Mvir2Rvir(mass=grand_group['MassAfunc'][i], z=grand_group['Zfof'][i]))[0]
		merge_array 		= np.where(distance <  Mvir2Rvir(mass=Mvir_here, z=np.array(grand_group[grand_group['GroupID'] == group_id]['Zfof'])))[0]

		a = [int(k) for k in merge_array]

		add_HI 							= np.sum(HI_unique_sam[a]) #+ HI_unique_sam[trial]
		M_HI_sam_mass_rvir_2_dyn[i] 	= add_HI

		
		######------------------------------------------
		###   R_vir = 1.5
		######------------------------------------------

		merge_array 					= np.where(distance <  0.75*Mvir2Rvir(mass=Mvir_here, z=np.array(grand_group[grand_group['GroupID'] == group_id]['Zfof'])))[0]

		a = [int(k) for k in merge_array]

		add_HI 							= np.sum(HI_unique_sam[a]) #+ HI_unique_sam[trial]
		M_HI_sam_mass_rvir_5_dyn[i] 	= add_HI

		######------------------------------------------
		###   R_vir = 1
		######------------------------------------------

		merge_array 		= np.where(distance <  0.5*Mvir2Rvir(mass=Mvir_here, z=np.array(grand_group[grand_group['GroupID'] == group_id]['Zfof'])))[0]

		a = [int(k) for k in merge_array]

		add_HI						= np.sum(HI_unique_sam[a]) #+ HI_unique_sam[trial]
		M_HI_sam_mass_rvir_10_dyn[i] 	= add_HI

		M_HI_sam_central_group[i] 	= HI_unique_sam[trial]

				
############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
###                                                   Plotting : HI-Halo Scaling Relation [N_g >= 2]
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************



plt = Common_module.load_matplotlib(12,8)
colour_plot = Common_module.colour_scheme(colourmap = plt.cm.Set2)

legendHandles = list()

####*********************************************************

Halo_mass_plotting, HI_mass_plotting, HI_mass_low, HI_mass_high = Common_module.halo_value_list(Mvir_abundance_match, M_HI_sam_mass_rvir_2_ab, mean=True)
Error_SAM_2 		= plt.errorbar(Halo_mass_plotting[0:len(HI_mass_plotting)-1], HI_mass_plotting[0:len(HI_mass_plotting)-1], yerr=None,label='Aperture: $2 \\times R_{vir}$', marker = "o", mfc = colour_plot[1], mec = colour_plot[1], c = colour_plot[1], elinewidth=2,ls = ':', markersize=10) # Plotting Mean HI-Halo values for plotting (isolated + group)

####*********************************************************

Halo_mass_plotting, HI_mass_plotting, HI_mass_low, HI_mass_high = Common_module.halo_value_list(Mvir_abundance_match, M_HI_sam_mass_rvir_5_ab, mean=True)
Error_SAM_5 		= plt.errorbar(Halo_mass_plotting[0:len(HI_mass_plotting)-1], HI_mass_plotting[0:len(HI_mass_plotting)-1], yerr=None,label='Aperture: $1.5 \\times R_{vir}$', marker = "o", mfc = colour_plot[2], mec = colour_plot[2], c = colour_plot[2], elinewidth=2,ls = ':', markersize=10) # Plotting Mean HI-Halo values for plotting (isolated + group)


####*********************************************************

Halo_mass_plotting, HI_mass_plotting, HI_mass_low, HI_mass_high = Common_module.halo_value_list(Mvir_abundance_match, M_HI_sam_mass_rvir_10_ab, mean=True)
Error_SAM_10 		= plt.errorbar(Halo_mass_plotting[0:len(HI_mass_plotting)-1], HI_mass_plotting[0:len(HI_mass_plotting)-1], yerr=None,label='Aperture: $1 \\times R_{vir}$', marker = "o", mfc = colour_plot[3], mec = colour_plot[3], c = colour_plot[3], elinewidth=2,ls = ':', markersize=10) # Plotting Mean HI-Halo values for plotting (isolated + group)



######--------------------------------------------------------------------------------------------------------------------------------------
###                                                   Out-of-box
######--------------------------------------------------------------------------------------------------------------------------------------


HI_all 			= medi_HI#[medi_vir >= 10**11.15]
Stellar_all 	= medi_vir#[medi_vir >= 10**11.15]

property_plot 	= HI_all#/medi_vir[medi_vir >= 10**11.15]

Common_module.plotting_properties_halo(Stellar_all,property_plot,mean=True,legend_handles=legendHandles,colour_line='maroon',alpha=0.5,fill_between=False,xlim_lower=11,xlim_upper=14,ylim_lower=-5,ylim_upper=-0.5,legend_name=shark_labels[0], property_name_x='log_{10}(M_{vir}[M_{\odot}])', property_name_y='log_{10}(M_{HI} [M_{\odot}])', resolution = False, first_legend=False)




######--------------------------------------------------------------------------------------------------------------------------------------
###                                                   Observations
######--------------------------------------------------------------------------------------------------------------------------------------


extra = mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="Abundance Matching ($N_{g} \\geq 2$)")

legendHandles.append(Error_SAM_2)
legendHandles.append(Error_SAM_5)
legendHandles.append(Error_SAM_10)
legendHandles.append(extra)
legendHandles.append(extra)


plt.xlabel('$log_{10}(M_{vir}[M_{\odot}])$')
plt.ylabel('$log_{10}(M_{HI}[M_{\odot}])$')
plt.ylim(8.01,11)

leg = plt.legend(handles=legendHandles[0:4],loc='upper left', frameon=False)
plt.gca().add_artist(leg)
leg = plt.legend(handles=legendHandles[4:5],loc='lower right', frameon=False)
plt.gca().add_artist(leg)

# plt.legend(handles=legendHandles)
plt.savefig("Plots/Spherical_coordinates/abmatch_spherical_coordinates.png")
# plt.show()
plt.close()



############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
###                                                   Plotting : HI-Halo Scaling Relation [N_g >= 2]
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************



plt = Common_module.load_matplotlib(12,8)
colour_plot = Common_module.colour_scheme(colourmap = plt.cm.Set2)

legendHandles = list()

####*********************************************************

Halo_mass_plotting, HI_mass_plotting, HI_mass_low, HI_mass_high = Common_module.halo_value_list(Mvir_dynamical_mass, M_HI_sam_mass_rvir_2_dyn, mean=True)
Error_SAM_2 		= plt.errorbar(Halo_mass_plotting[0:len(HI_mass_plotting)-1], HI_mass_plotting[0:len(HI_mass_plotting)-1], yerr=None,label='Aperture: $2 \\times R_{vir}$', marker = "o", mfc = colour_plot[1], mec = colour_plot[1], c = colour_plot[1], elinewidth=2,ls = ':', markersize=10) # Plotting Mean HI-Halo values for plotting (isolated + group)

####*********************************************************

Halo_mass_plotting, HI_mass_plotting, HI_mass_low, HI_mass_high = Common_module.halo_value_list(Mvir_dynamical_mass, M_HI_sam_mass_rvir_5_dyn, mean=True)
Error_SAM_5 		= plt.errorbar(Halo_mass_plotting[0:len(HI_mass_plotting)-1], HI_mass_plotting[0:len(HI_mass_plotting)-1], yerr=None,label='Aperture: $1.5 \\times R_{vir}$', marker = "o", mfc = colour_plot[2], mec = colour_plot[2], c = colour_plot[2], elinewidth=2,ls = ':', markersize=10) # Plotting Mean HI-Halo values for plotting (isolated + group)


####*********************************************************

Halo_mass_plotting, HI_mass_plotting, HI_mass_low, HI_mass_high = Common_module.halo_value_list(Mvir_dynamical_mass, M_HI_sam_mass_rvir_10_dyn, mean=True)
Error_SAM_10 		= plt.errorbar(Halo_mass_plotting[0:len(HI_mass_plotting)-1], HI_mass_plotting[0:len(HI_mass_plotting)-1], yerr=None,label='Aperture: $1 \\times R_{vir}$', marker = "o", mfc = colour_plot[3], mec = colour_plot[3], c = colour_plot[3], elinewidth=2,ls = ':', markersize=10) # Plotting Mean HI-Halo values for plotting (isolated + group)



######--------------------------------------------------------------------------------------------------------------------------------------
###                                                   Out-of-box
######--------------------------------------------------------------------------------------------------------------------------------------


HI_all 			= medi_HI#[medi_vir >= 10**11.15]
Stellar_all 	= medi_vir#[medi_vir >= 10**11.15]

property_plot 	= HI_all#/medi_vir[medi_vir >= 10**11.15]

Common_module.plotting_properties_halo(Stellar_all,property_plot,mean=True,legend_handles=legendHandles,colour_line='maroon',alpha=0.5,fill_between=False,xlim_lower=11,xlim_upper=14,ylim_lower=-5,ylim_upper=-0.5,legend_name=shark_labels[0], property_name_x='log_{10}(M_{vir}[M_{\odot}])', property_name_y='log_{10}(M_{HI} [M_{\odot}])', resolution = False, first_legend=False)




######--------------------------------------------------------------------------------------------------------------------------------------
###                                                   Observations
######--------------------------------------------------------------------------------------------------------------------------------------


extra = mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="Dynamical Mass ($N_{g} \\geq 2$)")

legendHandles.append(Error_SAM_2)
legendHandles.append(Error_SAM_5)
legendHandles.append(Error_SAM_10)
legendHandles.append(extra)
legendHandles.append(extra)


plt.xlabel('$log_{10}(M_{vir}[M_{\odot}])$')
plt.ylabel('$log_{10}(M_{HI}[M_{\odot}])$')
plt.ylim(8.01,11)

leg = plt.legend(handles=legendHandles[0:4],loc='upper left', frameon=False)
plt.gca().add_artist(leg)
leg = plt.legend(handles=legendHandles[4:5],loc='lower right', frameon=False)
plt.gca().add_artist(leg)

# plt.legend(handles=legendHandles)
plt.savefig("Plots/Spherical_coordinates/dynamical_spherical_coordinates.png")
# plt.show()
plt.close()
