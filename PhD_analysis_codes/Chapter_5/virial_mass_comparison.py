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
from Common_module import LightconeReading
import matplotlib as mpl
import pandas as pd
import splotch as spl

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck15

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


def nanpercentile_lower(arr,q=15.87): 
	arr = arr[~np.isnan(arr)]
	
	if (len(arr) != 0): return (np.nanpercentile(a=arr,q=q))
	else: return None
	

def nanpercentile_upper(arr,q=84.13): 
	arr = arr[~np.isnan(arr)]
	
	if (len(arr) != 0): return (np.nanpercentile(a=arr,q=q))
	else: return None




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

medi_stellar, medi_stellar_central, medi_stellar_satellite,medi_stellar_orphan,a,a = medi_Kappa_original[199].mergeValues_satellite('id_halo','mstars_bulge','mstars_disk')


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
###                                                   Reading Lightcone 
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************

def prepare_data(path,subvolumes,filename):

	fields_read 	= {'galaxies':('id_halo_sam', 'id_galaxy_sam','id_galaxy_sky', 'type', 'mvir_hosthalo','mvir_subhalo', 'matom_bulge', 'matom_disk', 'ra', 'dec', 'vpec_r','vpec_x','vpec_y','vpec_z', 'zcos', 'zobs', 'mstars_bulge','mstars_disk')} #'vvir_hosthalo', 'vvir_subhalo',

	data_reading = LightconeReading(path,subvolumes,filename)

	data_plotting = data_reading.readIndividualFiles_test(fields=fields_read)

	return data_plotting


id_halo_sam_all  	= []
id_galaxy_sam_all	= []
id_galaxy_sky_all	= []
type_g_all			= []
mvir_hosthalo_all	= []
mvir_subhalo_all	= []
matom_sam_all		= []
mstars_sam_all		= []
ra_all				= []
dec_all				= []
vpec_r_all			= []
vpec_x_all			= []
vpec_y_all			= []
vpec_z_all			= []
# vvir_hosthalo_all	= []
# vvir_subhalo_all	= []
zcos_all			= []
zobs_all			= []


for i in range(subvolumes):
	print(i)

	(id_halo_sam, id_galaxy_sam, id_galaxy_sky, type_g, mvir_hosthalo, mvir_subhalo, matom_bulge, matom_disk, ra, dec, vpec_r,vpec_x,vpec_y,vpec_z, zcos, zobs,mstars_bulge,mstars_disk) = prepare_data(path_GAMA, i, 'mock_medi_alfalfa') #'mock_medi_alfalfa') #'mocksky')# #, vvir_hosthalo, vvir_subhalo

	id_halo_sam_all		= np.append(id_halo_sam_all,id_halo_sam[zobs <= 0.06])
	id_galaxy_sam_all	= np.append(id_galaxy_sam_all,id_galaxy_sam[zobs <= 0.06])
	id_galaxy_sky_all	= np.append(id_galaxy_sky_all,id_galaxy_sky[zobs <= 0.06])
	type_g_all			= np.append(type_g_all,type_g[zobs <= 0.06])
	mvir_hosthalo_all	= np.append(mvir_hosthalo_all,mvir_hosthalo[zobs <= 0.06])
	mvir_subhalo_all	= np.append(mvir_subhalo_all,mvir_subhalo[zobs <= 0.06])
	matom_sam_all		= np.append(matom_sam_all,matom_bulge[zobs <= 0.06] + matom_disk[zobs <= 0.06])
	mstars_sam_all		= np.append(mstars_sam_all,mstars_bulge[zobs <= 0.06]+mstars_disk[zobs <= 0.06])
	ra_all				= np.append(ra_all,ra[zobs <= 0.06])
	dec_all				= np.append(dec_all,dec[zobs <= 0.06])
	vpec_r_all			= np.append(vpec_r_all,vpec_r[zobs <= 0.06])
	vpec_x_all			= np.append(vpec_x_all,vpec_x[zobs <= 0.06])
	vpec_y_all			= np.append(vpec_y_all,vpec_y[zobs <= 0.06])
	vpec_z_all			= np.append(vpec_z_all,vpec_z[zobs <= 0.06])
	# vvir_hosthalo_all	= np.append(vvir_hosthalo_all,vvir_hosthalo[zobs <= 0.06])
	# vvir_subhalo_all	= np.append(vvir_subhalo_all,vvir_subhalo[zobs <= 0.06])
	zcos_all			= np.append(zcos_all,zcos[zobs <= 0.06])
	zobs_all			= np.append(zobs_all,zobs[zobs <= 0.06])



############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
###                                                   Making dataframe
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************

id_halo_sam_all 	= [int(k) for k in id_halo_sam_all]
id_galaxy_sam_all	= [int(k) for k in id_galaxy_sam_all]
id_galaxy_sky_all 	= [int(k) for k in id_galaxy_sky_all]
type_g_all			= [int(k) for k in type_g_all]
mvir_hosthalo_all	= [int(k) for k in mvir_hosthalo_all]
mvir_subhalo_all	= [int(k) for k in mvir_subhalo_all] 
matom_sam_all 		= [int(k) for k in matom_sam_all]
mstars_sam_all 		= [int(k) for k in mstars_sam_all]




dictionary_df = {'id_halo_sam':id_halo_sam_all, 'id_galaxy_sam':id_galaxy_sam_all, 'id_galaxy_sky':id_galaxy_sky_all, 'type':type_g_all, 'mvir_hosthalo':mvir_hosthalo_all, 'mvir_subhalo':mvir_subhalo_all, 'matom_all':matom_sam_all, 'ra':ra_all, 'dec':dec_all, 'vpec_r':vpec_r_all,'vpec_x':vpec_x_all,'vpec_y':vpec_y_all,'vpec_z':vpec_z_all, 'zcos':zcos_all, 'zobs':zcos_all,'mstars_all':mstars_sam_all} #, 'vvir_hosthalo':vvir_hosthalo_all, 'vvir_subhalo':vvir_subhalo_all


lightcone_alfalfa = pd.DataFrame.from_dict(dictionary_df)


############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
###                                                   Merging GAMA files
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************

grand_gals					= GAMA_gals_file.merge(GAMA_censat_file,left_on='CATAID', right_on='id_galaxy_sky', how='inner')  # Merges Galaxy file with censat (merging CATAID and id_galaxy_sky)

grand_isolated 				= grand_gals.loc[(grand_gals['RankIterCen']==-999) & (grand_gals['Z']<=0.06)].copy() 			  # only isolated centrals	
grand_group_all				= grand_gals.loc[(grand_gals['GroupID'] != 0) & (grand_gals['Z']<=0.06)].copy()  				  # separates the central of groups	

grand_group 				= grand_gals.loc[(grand_gals['RankIterCen'] == 1) & (grand_gals['Z']<=0.06)].copy()  				  # separates the central of groups	
grand_group 				= grand_group.merge(GAMA_group_file,on='GroupID',how='inner')
grand_group_nfof			= grand_group.loc[(grand_group['Nfof'] > 5)].copy()


grand_isolated_lightcone	= grand_isolated.merge(lightcone_alfalfa, on='id_galaxy_sky', how='inner') 						# matching isolated centrals to lightcone

grand_group_lightcone 		= grand_group.merge(lightcone_alfalfa, on='id_galaxy_sky', how='inner') 						# matching group galaxies to lightcone
grand_group_lightcone_all	= grand_group_all.merge(lightcone_alfalfa, on='id_galaxy_sky', how='inner') 						# matching group galaxies to lightcone

grand_gals_lightcone 		= grand_gals.merge(lightcone_alfalfa, on='id_galaxy_sky', how='inner')


############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
###                                                   Distribution values - GAMA
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************


Mstar_GAMA_nfof 		= []
Mvir_GAMA_nfof 		= []
MHI_GAMA_nfof 		= []


# matching_ID 	 = np.unique(np.array(grand_group_lightcone['GroupID'])) 

matching_ID 	 = np.unique(np.array(grand_group_nfof['GroupID'])) 

abs_rband_mag	 = np.array(grand_gals_lightcone['r_ab'])


print('Started matching GAMA')

for i in matching_ID:

	# print(i)

	trial 				= np.where(grand_gals_lightcone['GroupID'] == i)[0]

	if len(grand_gals_lightcone[(grand_gals_lightcone['GroupID'] == i) & (grand_gals_lightcone['RankIterCen'] == 1)]) != 0:

		Mstar_add = int(np.sum(grand_gals_lightcone[(grand_gals_lightcone['GroupID'] == i)]['mstar']))
		Mvir_GAMA_nfof = np.append(Mvir_GAMA_nfof, int(grand_group_lightcone[(grand_group_lightcone['GroupID'] == i) & (grand_group_lightcone['RankIterCen'] == 1)]['MassAfunc']))
		MHI_add   = int(np.sum(grand_gals_lightcone[(grand_gals_lightcone['GroupID'] == i)]['matom_all']))/1.35/h


		Mstar_GAMA_nfof = np.append(Mstar_GAMA_nfof, Mstar_add)
		MHI_GAMA_nfof   = np.append(MHI_GAMA_nfof, MHI_add)

	else:

		continue


print('Finished matching GAMA')



Mstar_GAMA 		= []
Mvir_GAMA 		= []
MHI_GAMA 		= []


# matching_ID 	 = np.unique(np.array(grand_group_lightcone['GroupID'])) 

matching_ID 	 = np.unique(np.array(grand_group['GroupID'])) 

abs_rband_mag	 = np.array(grand_gals_lightcone['r_ab'])


print('Started matching GAMA')

for i in matching_ID:

	# print(i)

	trial 				= np.where(grand_gals_lightcone['GroupID'] == i)[0]

	if len(grand_gals_lightcone[(grand_gals_lightcone['GroupID'] == i) & (grand_gals_lightcone['RankIterCen'] == 1)]) != 0:

		Mstar_add = int(np.sum(grand_gals_lightcone[(grand_gals_lightcone['GroupID'] == i)]['mstar']))
		Mvir_GAMA = np.append(Mvir_GAMA, int(grand_group_lightcone[(grand_group_lightcone['GroupID'] == i) & (grand_group_lightcone['RankIterCen'] == 1)]['MassAfunc']))
		MHI_add   = int(np.sum(grand_gals_lightcone[(grand_gals_lightcone['GroupID'] == i)]['matom_all']))/1.35/h


		Mstar_GAMA = np.append(Mstar_GAMA, Mstar_add)
		MHI_GAMA   = np.append(MHI_GAMA, MHI_add)

	else:

		continue


print('Finished matching GAMA')




############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
###                                                   Merging SDSS files
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************

grand_gals_SDSS 				= grand_gals.loc[(grand_gals['Z']<=0.06) & (grand_gals['mag'] <= 17.77)].copy()

grand_isolated_SDSS 			= grand_gals_SDSS.loc[(grand_gals_SDSS['RankIterCen']==-999)].copy() 	
grand_group_SDSS_all 			= grand_gals_SDSS.loc[(grand_gals_SDSS['GroupID'] != 0)].copy()  

grand_group_SDSS 				= grand_gals_SDSS.loc[(grand_gals_SDSS['RankIterCen'] == 1) & (grand_gals_SDSS['Z']<=0.06) & (grand_gals_SDSS['mag'] <= 17.77)].copy()  
grand_group_SDSS 				= grand_group_SDSS.merge(GAMA_group_file,on='GroupID',how='inner')
grand_group_SDSS_nfof			= grand_group_SDSS.loc[(grand_group_SDSS['Nfof'] > 5)].copy()


grand_isolated_lightcone_SDSS	= grand_isolated_SDSS.merge(lightcone_alfalfa, on='id_galaxy_sky', how='inner') 		# matching isolated centrals to lightcone

grand_group_lightcone_SDSS 		= grand_group_SDSS.merge(lightcone_alfalfa, on='id_galaxy_sky', how='inner') 						# matching group galaxies to lightcone
grand_group_lightcone_SDSS_all 	= grand_group_SDSS_all.merge(lightcone_alfalfa, on='id_galaxy_sky', how='inner') 			# matching group galaxies to lightcone

grand_gals_lightcone_SDSS 		= grand_gals_SDSS.merge(lightcone_alfalfa, on='id_galaxy_sky', how='inner')


grand_group_lightcone_SDSS_nfof 		= grand_group_SDSS_nfof.merge(lightcone_alfalfa, on='id_galaxy_sky', how='inner')

############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
###                                                   Distribution values - SDSS
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************


Mstar_SDSS_nfof 		= []
Mvir_SDSS_nfof 		= []
MHI_SDSS_nfof 		= []

matching_ID 	 = np.unique(np.array(grand_group_SDSS_nfof['GroupID']))
# matching_ID 	 = np.unique(np.array(grand_group_lightcone_SDSS['GroupID'])) 

abs_rband_mag	 = np.array(grand_gals_lightcone_SDSS['r_ab'])

print('Started matching SDSS')

for i in matching_ID:

	# print(i)

	trial 				= np.where(grand_gals_lightcone_SDSS['GroupID'] == i)[0]

	#if len(grand_gals_lightcone_SDSS[(grand_gals_lightcone_SDSS['GroupID'] == i) & (grand_gals_lightcone_SDSS['RankIterCen'] == 1)]) != 0:
	if len(grand_gals_lightcone_SDSS[(grand_gals_lightcone_SDSS['GroupID'] == i) & (grand_gals_lightcone_SDSS['RankIterCen'] == 1)]) != 0:

		Mstar_add = int(np.sum(grand_gals_lightcone_SDSS[(grand_gals_lightcone_SDSS['GroupID'] == i)]['mstar']))
		# abs_high = np.array(grand_group_lightcone_SDSS[(grand_group_lightcone_SDSS['GroupID'] == i) & (grand_group_lightcone_SDSS['RankIterCen'] == 1)]['r_ab'])
		abs_high = np.array(grand_group_lightcone_SDSS[(grand_group_lightcone_SDSS['GroupID'] == i) & (grand_group_lightcone_SDSS['RankIterCen'] == 1)]['r_ab'])

		abs_low = np.array(grand_gals_lightcone_SDSS[(grand_gals_lightcone_SDSS['GroupID'] == i) & (grand_gals_lightcone_SDSS['RankIterCen'] == max(grand_gals_lightcone_SDSS[(grand_gals_lightcone_SDSS['GroupID'] == i)]['RankIterCen']))]['r_ab'])

		Mvir_add  = 10**(Common_module.abundanceMatchingMvir_luminosity(abs_high, abs_low))

		MHI_add   = int(np.sum(grand_gals_lightcone_SDSS[(grand_gals_lightcone_SDSS['GroupID'] == i)]['matom_all']))/1.35/h
		Mvir_SDSS_nfof = np.append(Mvir_SDSS_nfof, Mvir_add) 
		Mstar_SDSS_nfof = np.append(Mstar_SDSS_nfof, Mstar_add)
		MHI_SDSS_nfof   = np.append(MHI_SDSS_nfof, MHI_add)

	else:

		continue

print('Finished matching SDSS')




Mstar_SDSS 		= []
Mvir_SDSS 		= []
MHI_SDSS 		= []

matching_ID 	 = np.unique(np.array(grand_group_SDSS['GroupID']))
# matching_ID 	 = np.unique(np.array(grand_group_lightcone_SDSS['GroupID'])) 

abs_rband_mag	 = np.array(grand_gals_lightcone_SDSS['r_ab'])

print('Started matching SDSS')

for i in matching_ID:

	# print(i)

	trial 				= np.where(grand_gals_lightcone_SDSS['GroupID'] == i)[0]

	if len(grand_gals_lightcone_SDSS[(grand_gals_lightcone_SDSS['GroupID'] == i) & (grand_gals_lightcone_SDSS['RankIterCen'] == 1)]) != 0:

		Mstar_add = int(np.sum(grand_gals_lightcone_SDSS[(grand_gals_lightcone_SDSS['GroupID'] == i)]['mstar']))
		# abs_high = np.array(grand_group_lightcone_SDSS[(grand_group_lightcone_SDSS['GroupID'] == i) & (grand_group_lightcone_SDSS['RankIterCen'] == 1)]['r_ab'])
		abs_high = np.array(grand_group_lightcone_SDSS[(grand_group_lightcone_SDSS['GroupID'] == i) & (grand_group_lightcone_SDSS['RankIterCen'] == 1)]['r_ab'])

		abs_low = np.array(grand_gals_lightcone_SDSS[(grand_gals_lightcone_SDSS['GroupID'] == i) & (grand_gals_lightcone_SDSS['RankIterCen'] == max(grand_gals_lightcone_SDSS[(grand_gals_lightcone_SDSS['GroupID'] == i)]['RankIterCen']))]['r_ab'])

		Mvir_add  = 10**(Common_module.abundanceMatchingMvir_luminosity(abs_high, abs_low))

		MHI_add   = int(np.sum(grand_gals_lightcone_SDSS[(grand_gals_lightcone_SDSS['GroupID'] == i)]['matom_all']))/1.35/h
		Mvir_SDSS = np.append(Mvir_SDSS, Mvir_add) 
		Mstar_SDSS = np.append(Mstar_SDSS, Mstar_add)
		MHI_SDSS   = np.append(MHI_SDSS, MHI_add)

	else:

		continue

print('Finished matching SDSS')



############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
###                                                   Distribution values - MOCk
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************


Mstar_Mock_GAMA 		= []
Mvir_Mock_GAMA 		= []
MHI_Mock_GAMA 		= []
type_gal_GAMA 		= []

# matching_ID 	 = np.unique(np.array(grand_group_lightcone['GroupID'])) 

matching_ID 	 = np.unique(np.array(grand_group['GroupID']))

abs_rband_mag	 = np.array(grand_gals_lightcone_SDSS['r_ab'])


print('Started matching mocks')

for i in matching_ID:

	trial 				= np.where(grand_gals_lightcone['GroupID'] == i)[0]

	if len(grand_gals_lightcone[(grand_gals_lightcone['GroupID'] == i)]) > 0:

		Mstar_add = int(np.sum(grand_gals_lightcone[(grand_gals_lightcone['GroupID'] == i)]['mstars_all']))/h
		# Mvir_add  = int(np.array(grand_gals_lightcone[(grand_gals_lightcone['GroupID'] == i) ]['mvir_hosthalo'])[0])/h
		Mvir_add  = max(np.array(grand_gals_lightcone[(grand_gals_lightcone['GroupID'] == i) ]['mvir_hosthalo']))/h

		MHI_add   = int(np.sum(grand_gals_lightcone[(grand_gals_lightcone['GroupID'] == i)]['matom_all']))/1.35/h
		type_add  = int(np.array(grand_gals_lightcone[(grand_gals_lightcone['GroupID'] == i) ]['type_x'])[0])

		Mvir_Mock_GAMA = np.append(Mvir_Mock_GAMA, Mvir_add) 
		Mstar_Mock_GAMA = np.append(Mstar_Mock_GAMA, Mstar_add)
		MHI_Mock_GAMA   = np.append(MHI_Mock_GAMA, MHI_add)
		type_gal_GAMA  	= np.append(type_gal_GAMA, type_add)
	else:

		continue


print('Finished matching mocks')



Mstar_Mock_GAMA_nfof 		= []
Mvir_Mock_GAMA_nfof 		= []
MHI_Mock_GAMA_nfof 		= []
type_gal_GAMA_nfof 		= []

# matching_ID 	 = np.unique(np.array(grand_group_lightcone['GroupID'])) 

matching_ID 	 = np.unique(np.array(grand_group_nfof['GroupID']))

abs_rband_mag	 = np.array(grand_gals_lightcone_SDSS['r_ab'])


print('Started matching mocks')

for i in matching_ID:

	trial 				= np.where(grand_gals_lightcone['GroupID'] == i)[0]

	if len(grand_gals_lightcone[(grand_gals_lightcone['GroupID'] == i)]) > 0:

		Mstar_add = int(np.sum(grand_gals_lightcone[(grand_gals_lightcone['GroupID'] == i)]['mstars_all']))/h
		# Mvir_add  = int(np.array(grand_gals_lightcone[(grand_gals_lightcone['GroupID'] == i) ]['mvir_hosthalo'])[0])/h
		Mvir_add  = max(np.array(grand_gals_lightcone[(grand_gals_lightcone['GroupID'] == i) ]['mvir_hosthalo']))/h

		MHI_add   = int(np.sum(grand_gals_lightcone[(grand_gals_lightcone['GroupID'] == i)]['matom_all']))/1.35/h
		type_add  = int(np.array(grand_gals_lightcone[(grand_gals_lightcone['GroupID'] == i) ]['type_x'])[0])

		Mvir_Mock_GAMA_nfof = np.append(Mvir_Mock_GAMA_nfof, Mvir_add) 
		Mstar_Mock_GAMA_nfof = np.append(Mstar_Mock_GAMA_nfof, Mstar_add)
		MHI_Mock_GAMA_nfof   = np.append(MHI_Mock_GAMA_nfof, MHI_add)
		type_gal_GAMA_nfof  	= np.append(type_gal_GAMA_nfof, type_add)
	else:

		continue


print('Finished matching mocks')





Mstar_Mock_SDSS 		= []
Mvir_Mock_SDSS 		= []
MHI_Mock_SDSS 		= []
type_gal 			= []

# matching_ID 	 = np.unique(np.array(grand_group_SDSS_nfof['GroupID'])) 

matching_ID 	 = np.unique(np.array(grand_group_SDSS['GroupID'])) 

abs_rband_mag	 = np.array(grand_gals_lightcone_SDSS['r_ab'])


print('Started matching mocks')

for i in matching_ID:

	trial 				= np.where(grand_gals_lightcone_SDSS['GroupID'] == i)[0]

	# if len(grand_gals_lightcone_SDSS[(grand_gals_lightcone_SDSS['GroupID'] == i) & (grand_gals_lightcone_SDSS['RankIterCen'] == 1)]) > 0:
	if len(grand_gals_lightcone_SDSS[(grand_gals_lightcone_SDSS['GroupID'] == i) & (grand_gals_lightcone_SDSS['RankIterCen'] == 1)]) != 0:

		Mstar_add = int(np.sum(grand_gals_lightcone_SDSS[(grand_gals_lightcone_SDSS['GroupID'] == i)]['mstars_all']))/h
		# Mvir_add  = int(np.array(grand_gals_lightcone_SDSS[(grand_gals_lightcone_SDSS['GroupID'] == i) ]['mvir_hosthalo'])[0])/h
		# Mvir_add  = int(np.array(grand_gals_lightcone_SDSS[(grand_gals_lightcone_SDSS['GroupID'] == i) ]['mvir_hosthalo']))/h
		Mvir_add  = max(np.array(grand_gals_lightcone_SDSS[(grand_gals_lightcone_SDSS['GroupID'] == i) ]['mvir_hosthalo']))/h/h
		type_add  = int(np.array(grand_gals_lightcone_SDSS[(grand_gals_lightcone_SDSS['GroupID'] == i)& (grand_gals_lightcone_SDSS['RankIterCen'] == 1) ]['type_x'])[0])
		MHI_add   = int(np.sum(grand_gals_lightcone_SDSS[(grand_gals_lightcone_SDSS['GroupID'] == i)]['matom_all']))/1.35/h

		Mvir_Mock_SDSS = np.append(Mvir_Mock_SDSS, Mvir_add) 
		Mstar_Mock_SDSS = np.append(Mstar_Mock_SDSS, Mstar_add)
		MHI_Mock_SDSS   = np.append(MHI_Mock_SDSS, MHI_add)
		type_gal = np.append(type_gal, type_add)
	else:

		continue


print('Finished matching mocks')






Mstar_Mock_SDSS_nfof 		= []
Mvir_Mock_SDSS_nfof 		= []
MHI_Mock_SDSS_nfof 		= []
type_gal_nfof 			= []

# matching_ID 	 = np.unique(np.array(grand_group_SDSS_nfof['GroupID'])) 

matching_ID 	 = np.unique(np.array(grand_group_SDSS_nfof['GroupID'])) 

abs_rband_mag	 = np.array(grand_gals_lightcone_SDSS['r_ab'])


print('Started matching mocks')

for i in matching_ID:

	trial 				= np.where(grand_gals_lightcone_SDSS['GroupID'] == i)[0]

	# if len(grand_gals_lightcone_SDSS[(grand_gals_lightcone_SDSS['GroupID'] == i) & (grand_gals_lightcone_SDSS['RankIterCen'] == 1)]) > 0:
	if len(grand_gals_lightcone_SDSS[(grand_gals_lightcone_SDSS['GroupID'] == i) & (grand_gals_lightcone_SDSS['RankIterCen'] == 1)]) != 0:

		Mstar_add = int(np.sum(grand_gals_lightcone_SDSS[(grand_gals_lightcone_SDSS['GroupID'] == i)]['mstars_all']))/h
		# Mvir_add  = int(np.array(grand_gals_lightcone_SDSS[(grand_gals_lightcone_SDSS['GroupID'] == i) ]['mvir_hosthalo'])[0])/h
		# Mvir_add  = int(np.array(grand_gals_lightcone_SDSS[(grand_gals_lightcone_SDSS['GroupID'] == i) ]['mvir_hosthalo']))/h
		Mvir_add  = max(np.array(grand_gals_lightcone_SDSS[(grand_gals_lightcone_SDSS['GroupID'] == i) ]['mvir_hosthalo']))/h/h
		type_add  = int(np.array(grand_gals_lightcone_SDSS[(grand_gals_lightcone_SDSS['GroupID'] == i)& (grand_gals_lightcone_SDSS['RankIterCen'] == 1) ]['type_x'])[0])
		MHI_add   = int(np.sum(grand_gals_lightcone_SDSS[(grand_gals_lightcone_SDSS['GroupID'] == i)]['matom_all']))/1.35/h

		Mvir_Mock_SDSS_nfof = np.append(Mvir_Mock_SDSS_nfof, Mvir_add) 
		Mstar_Mock_SDSS_nfof = np.append(Mstar_Mock_SDSS_nfof, Mstar_add)
		MHI_Mock_SDSS_nfof   = np.append(MHI_Mock_SDSS_nfof, MHI_add)
		type_gal_nfof = np.append(type_gal_nfof, type_add)
	else:

		continue


print('Finished matching mocks')


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

plt = Common_module.load_matplotlib(12,8)
colour_plot = Common_module.colour_scheme(colourmap = plt.cm.Set2)
legendHandles = []

fig, ax = plt.subplots(figsize=(12,8))
colour = np.array(grand_group['Nfof'])
colour = colour[Mvir_GAMA != 0]
Mvir_GAMA_plot = np.log10(Mvir_GAMA[Mvir_GAMA != 0]/h)
Mvir_Mock_plot = np.log10(Mvir_Mock_GAMA[Mvir_GAMA !=0])
plt.scatter(Mvir_Mock_plot[colour < 5],Mvir_GAMA_plot[colour < 5],  s=5, c='lightgrey', alpha=0.2)

colour = np.log10(np.array(grand_group_nfof['Nfof']))
Mvir_GAMA_plot = np.log10(Mvir_GAMA_nfof[Mvir_GAMA_nfof != 0]/h)
Mvir_Mock_plot = np.log10(Mvir_Mock_GAMA_nfof[Mvir_GAMA_nfof !=0])
plt.scatter(Mvir_Mock_plot,Mvir_GAMA_plot,  s=15*(colour), c=colour, cmap='Spectral_r', label='Trial')#, cmap='RdPu')

cbar=plt.colorbar()
cbar.set_label('$log_{10}$($N_{g}$)', rotation=270, labelpad = 20)
plt.plot([8,15],[8,15], '--k', linewidth=1)
plt.xlabel('$log_{10}(M^{SHARK}_{vir} [M_{\odot}])$')#' (SHARK-ref)')
plt.ylabel('$log_{10}(M^{Dyn}_{vir} [M_{\odot}])$')#' (Dynamical Mass) ')


l1, = ax.plot([0,0],[0,0], marker = 'o', linestyle=' ', color ='lightgrey', label = '$N_{g}$ $<$ 5', markersize=10)
l2, = ax.plot([0,0],[0,0], marker = 'o', linestyle=' ', color ='darkslateblue', label = '$N_{g}$ $\\geq$ 5', markersize=10)
l3, = ax.plot([0,0],[0,0], marker = 'o', linestyle=' ', color ='maroon')

extra = mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="Dynamical Mass")


handles = [(l1), (l2,l3),(extra), (extra)]
_, labels = ax.get_legend_handles_labels()

# plt.legend(handles = handles, labels = labels, loc='upper left', handler_map={tuple:mpl.legend_handler.HandlerTuple(None)})

leg = plt.legend(handles = handles[0:2], labels = labels, loc='upper left', handler_map={tuple:mpl.legend_handler.HandlerTuple(None)})
plt.gca().add_artist(leg)
leg = plt.legend(handles=handles[2:3],loc='lower right', frameon=False, handler_map={tuple:mpl.legend_handler.HandlerTuple(None)})
plt.gca().add_artist(leg)

plt.xlim(11,15)
plt.ylim(11.01,15)
plt.savefig(path_plot+'new_GAMA-match-Mvir-NFOF.png')
plt.close()

# extra = mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="Individual Stacking")

# legendHandles.append(extra)
# legendHandles.append(extra)

# leg = plt.legend(handles=legendHandles[0:3],loc='upper left', frameon=False)
# plt.gca().add_artist(leg)
# leg = plt.legend(handles=legendHandles[3:4],loc='lower right', frameon=False)
# plt.gca().add_artist(leg)


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------



#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------


plt = Common_module.load_matplotlib(12,8)
colour_plot = Common_module.colour_scheme(colourmap = plt.cm.Set1)
plt.figure(figsize=(12,8))
bins_plot = np.arange(10,15,0.2)
colour = np.array(grand_group['Nfof'])
colour = colour[Mvir_GAMA != 0]

plt.hist(np.log10(Mvir_Mock_GAMA[Mvir_Mock_GAMA != 0]),bins=bins_plot,log=True, label='Intrinsic', color='lightgrey', alpha=1,fill=False, histtype='step', linewidth=3,hatch='/')#, density=True)
plt.hist(np.log10(Mvir_GAMA[Mvir_GAMA != 0]/h),bins=bins_plot,log=True, label= 'Dynamical Mass' , color='maroon', alpha=1,fill=False, histtype='step', linewidth=3)#, density=True)

plt.xlabel('$M_{vir}[M_{\odot}]$ ($N_{g}$ $\\geq$ 2)')
plt.ylabel('$N_{Groups}$')
plt.legend()
plt.xlim(10,15)
plt.ylim(2,7000)
plt.savefig(path_plot+"GAMA_Histogram_HALO.png")
plt.close()


plt.figure(figsize=(12,8))
bin_to_plot = np.arange(10,15,0.2)


plt.hist(np.log10(Mvir_Mock_GAMA[Mvir_Mock_GAMA != 0]),bins=bin_to_plot, label='SHARK', color=colour_plot[3], alpha=1, log=True, fill=False, histtype='step', linewidth=3)#, density=True) 
plt.hist(np.log10(Mvir_GAMA[Mvir_GAMA != 0]/h),bins=bin_to_plot, label='Dynamical Mass', color=colour_plot[1], alpha=1, log=True, fill=False, histtype='step', linewidth=3,hatch='/')#, density=True)
plt.hist(np.log10(Mvir_SDSS[Mvir_SDSS != 0]/h),bins=bin_to_plot, label='Abundance Matching', color=colour_plot[2], alpha=1, log=True, fill=False, histtype='step', linewidth=3, hatch='o')#, density=True)

plt.xlabel('$M_{vir}[M_{\odot}]$ ($N_{g}$ $\\geq$ 2)')
plt.ylabel('$N_{Groups}$')
plt.legend()
plt.xlim(10,15)
plt.ylim(2,7000)
plt.savefig(path_plot+"Halo_Histogram_ng2.png")
plt.close()





plt.figure(figsize=(12,8))
bin_to_plot = np.arange(10,15,0.2)


plt.hist(np.log10(Mvir_Mock_GAMA_nfof[Mvir_Mock_GAMA_nfof != 0]),bins=bin_to_plot, label='SHARK', color=colour_plot[3], alpha=1, log=True, fill=False, histtype='step', linewidth=3)#, density=True) 
plt.hist(np.log10(Mvir_GAMA_nfof[Mvir_GAMA_nfof != 0]/h),bins=bin_to_plot, label='Dynamical Mass', color=colour_plot[1], alpha=1, log=True, fill=False, histtype='step', linewidth=3,hatch='/')#, density=True)
plt.hist(np.log10(Mvir_SDSS_nfof[Mvir_SDSS_nfof != 0]/h),bins=bin_to_plot, label='Abundance Matching', color=colour_plot[2], alpha=1, log=True, fill=False, histtype='step', linewidth=3, hatch='o')#, density=True)

plt.xlabel('$M_{vir}[M_{\odot}]$ ($N_{g}$ $\\geq$ 5)')
plt.ylabel('$N_{Groups}$')
plt.legend()
plt.xlim(10,15)
plt.ylim(2,7000)
plt.savefig(path_plot+"Halo_Histogram_ng5.png")
plt.close()


plt = Common_module.load_matplotlib(12,8)
colour_plot = Common_module.colour_scheme(colourmap = plt.cm.Set1)
plt.figure(figsize=(12,8))
bins_plot = np.arange(10,15,0.2)
colour = np.array(grand_group['Nfof'])
colour = colour[Mvir_GAMA != 0]

plt.hist(np.log10(Mvir_Mock_GAMA_nfof[Mvir_Mock_GAMA_nfof != 0]),bins=bins_plot,log=True, label='Intrinsic', color='lightgrey', alpha=1,fill=False, histtype='step', linewidth=3,hatch='/')#, density=True)
plt.hist(np.log10(Mvir_GAMA_nfof[Mvir_GAMA_nfof != 0]/h),bins=bins_plot,log=True, label='Dynamical Mass', color='maroon', alpha=1,fill=False, histtype='step', linewidth=3)#, density=True)

plt.xlabel('$M_{vir}[M_{\odot}]$ ($N_{g}$ $\\geq$ 5)')
plt.ylabel('$N_{Groups}$')
plt.legend(loc='upper left')
plt.xlim(10,15)
plt.ylim(2,500)
plt.savefig(path_plot+"Mock-GAMA_Histogram_HALO.png")
plt.close()



############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------

# plt = Common_module.load_matplotlib(12,8)
# colour_plot = Common_module.colour_scheme(colourmap = plt.cm.Set2)

# colour = np.array(grand_group_nfof['Nfof']) 
# # colour = colour[Mvir_GAMA != 0]

# Mvir_GAMA_plot = np.log10(Mvir_GAMA[Mvir_GAMA != 0])
# Mvir_Mock_plot = np.log10(Mvir_Mock_GAMA[Mvir_GAMA !=0])

# plt.scatter(Mvir_Mock_plot[colour <=15],Mvir_GAMA_plot[colour <=15],  s=10, c=colour[colour <=15], cmap='Purples')
# cbar=plt.colorbar()
# cbar.set_label('(Nfof)', rotation=270, labelpad = 20)
# plt.scatter(Mvir_GAMA_plot[colour > 15], Mvir_Mock_plot[colour > 15], s=10, c='rebeccapurple')
# plt.plot([10,15],[10,15], '--k', linewidth=1)
# plt.xlabel('$log_{10}(M_{vir} [M_{\odot}])$ (SHARK-ref)')
# plt.ylabel('$log_{10}(M_{vir} [M_{\odot}])$ (Dynamical Mass) ')
# plt.xlim(11,15)
# plt.ylim(11,15)
# plt.savefig(path_plot+'GAMA-match-Mvir-NFOF-pretty.png')
# plt.close()


############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------

plt = Common_module.load_matplotlib(12,8)
colour_plot = Common_module.colour_scheme(colourmap = plt.cm.Set2)

fig, ax = plt.subplots(figsize=(12,8))
Mvir_GAMA_plot = np.log10(Mvir_GAMA[Mvir_GAMA != 0]/h)
Mvir_Mock_plot = np.log10(Mvir_Mock_GAMA[Mvir_GAMA !=0])
colour = np.array(grand_group['Nfof'])
colour = colour[Mvir_GAMA != 0]
plt.scatter(Mvir_Mock_plot[colour < 5],Mvir_GAMA_plot[colour < 5], s=5, c='silver')


colour = type_gal
norm = colors.Normalize(vmin=0,vmax=2)
import matplotlib.cm as cm

colour_plot = Common_module.colour_scheme(colourmap = plt.cm.RdPu) # Accent Set1 RdPu Pastel1
# cmap, norm = mpl.colors.from_levels_and_colors([0, 1, 2,3], [colour_plot[0], colour_plot[1], colour_plot[2]])

cmap, norm = mpl.colors.from_levels_and_colors([0, 1, 2,3], [colour_plot[2], colour_plot[5], colour_plot[8]])
# cmap, norm = mpl.colors.from_levels_and_colors([0, 1, 2,3], ['gold', 'green', 'darkred'])

Mvir_GAMA_plot = np.log10(Mvir_GAMA_nfof[Mvir_GAMA_nfof != 0]/h)
Mvir_Mock_plot = np.log10(Mvir_Mock_GAMA_nfof[Mvir_GAMA_nfof !=0])

plt.scatter(Mvir_Mock_plot,Mvir_GAMA_plot, s=15, c=type_gal_GAMA_nfof, cmap=cmap, norm=norm)

cbar = plt.colorbar(ticks=[0.5,1.5,2.5], format='%1i')
cbar.set_label('Galaxy Type in SHARK', rotation=270, labelpad = 20)
plt.plot([10,15],[10,15], '--k', linewidth=1)
plt.xlabel('$log_{10}(M^{SHARK}_{vir} [M_{\odot}])$')#' (SHARK-ref)')
plt.ylabel('$log_{10}(M^{Dyn}_{vir} [M_{\odot}])$')#' (Dynamical Mass) ')


l1, = ax.plot([0,0],[0,0], marker = 'o', linestyle=' ', color ='lightgrey', label = '$N_{g}$ $<$ 5', markersize=10)
l2, = ax.plot([0,0],[0,0], marker = 'o', linestyle=' ', color =colour_plot[2], label = '$N_{g}$ $\\geq$ 5', markersize=10)
l3, = ax.plot([0,0],[0,0], marker = 'o', linestyle=' ', color =colour_plot[5], markersize=10)
l4, = ax.plot([0,0],[0,0], marker = 'o', linestyle=' ', color =colour_plot[8], markersize=10)

# l2, = ax.plot([0,0],[0,0], marker = 'o', linestyle=' ', color ='gold', label = '$N_{g}$ $\\geq$ 5', markersize=10)
# l3, = ax.plot([0,0],[0,0], marker = 'o', linestyle=' ', color ='green', markersize=10)
# l4, = ax.plot([0,0],[0,0], marker = 'o', linestyle=' ', color ='darkred', markersize=10)

extra = mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="Dynamical Mass")


handles = [(l1), (l2,l3, l4),(extra), (extra)]
_, labels = ax.get_legend_handles_labels()

# plt.legend(handles = handles, labels = labels, loc='upper left', handler_map={tuple:mpl.legend_handler.HandlerTuple(None)})

leg = plt.legend(handles = handles[0:2], labels = labels, loc='upper left', handler_map={tuple:mpl.legend_handler.HandlerTuple(None)})
plt.gca().add_artist(leg)
leg = plt.legend(handles=handles[2:3],loc='lower right', frameon=False, handler_map={tuple:mpl.legend_handler.HandlerTuple(None)})
plt.gca().add_artist(leg)


# handles = [(l1), (l2,l3,l4)]
# _, labels = ax.get_legend_handles_labels()

# plt.legend(handles = handles, labels = labels, loc='upper left', handler_map={tuple:mpl.legend_handler.HandlerTuple(None)})

plt.xlim(11,15)
plt.ylim(11.01,15)
plt.savefig(path_plot+'new_GAMA-match-Mvir-satvscen.png')
plt.close()



plt.figure(figsize=(8,8))
labels = 'Type = 0', 'Type = 1', 'Type = 2'
sizes = [77.25, 19.47, 3.27 ]   ## 80.186, 19.067, 0.745    ## SDSS-all [78.045, 19.606, 2.347 ]  SDSS_nfof = [74.485, 23.314, 2.2003 ]
explode = (0.0,0.0,0.1)
colour_pie = [colour_plot[2], colour_plot[5], colour_plot[8]]
fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, colors=colour_pie,autopct='%1.1f%%', shadow=True, startangle=50, wedgeprops={'linewidth' :1}, textprops={'size':15})
ax1.axis('equal')

extra = mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="Dynamical Mass")

legendHandles = []

legendHandles.append(extra)
legendHandles.append(extra)

leg = plt.legend(handles=handles[0:1],loc='lower right', frameon=False, handler_map={tuple:mpl.legend_handler.HandlerTuple(None)})
plt.gca().add_artist(leg)

plt.title('$N_{g} \\geq 2$')
plt.savefig(path_plot+'GAMA_Type_pieChart_all.png')
plt.close()


plt.figure(figsize=(8,8))
labels = 'Type = 0', 'Type = 1', 'Type = 2'
sizes = [80.186, 19.067, 0.745]   ## 80.186, 19.067, 0.745    ## SDSS-all [78.045, 19.606, 2.347 ]  SDSS_nfof = [74.485, 23.314, 2.2003 ]
explode = (0.0,0.0,0.1)
colour_pie = [colour_plot[2], colour_plot[5], colour_plot[8]]
fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, colors=colour_pie,autopct='%1.1f%%', shadow=True, startangle=50, wedgeprops={'linewidth' :1}, textprops={'size':15})
ax1.axis('equal')

extra = mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="Dynamical Mass")

legendHandles = []

legendHandles.append(extra)
legendHandles.append(extra)

leg = plt.legend(handles=handles[0:1],loc='lower right', frameon=False, handler_map={tuple:mpl.legend_handler.HandlerTuple(None)})
plt.gca().add_artist(leg)


plt.title('$N_{g} \\geq 5$')
plt.savefig(path_plot+'GAMA_Type_pieChart_nfof.png')
plt.close()




plt.figure(figsize=(8,8))
labels = 'Type = 0', 'Type = 1', 'Type = 2'
sizes = [78.045, 19.606, 2.347]   ## 80.186, 19.067, 0.745    ## SDSS-all [78.045, 19.606, 2.347 ]  SDSS_nfof = [74.485, 23.314, 2.2003 ]
explode = (0.0,0.0,0.1)
colour_pie = [colour_plot[2], colour_plot[5], colour_plot[8]]
fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, colors=colour_pie,autopct='%1.1f%%', shadow=True, startangle=50, wedgeprops={'linewidth' :1}, textprops={'size':15})
ax1.axis('equal')

extra = mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="Abundance Matching")

legendHandles = []

legendHandles.append(extra)
legendHandles.append(extra)

leg = plt.legend(handles=handles[0:1],loc='lower right', frameon=False, handler_map={tuple:mpl.legend_handler.HandlerTuple(None)})
plt.gca().add_artist(leg)
plt.title('$N_{g} \\geq 2$')
plt.savefig(path_plot+'SDSS_Type_pieChart_all.png')
plt.close()


plt.figure(figsize=(8,8))
labels = 'Type = 0', 'Type = 1', 'Type = 2'
sizes = [74.485, 23.314, 2.2003]   ## 80.186, 19.067, 0.745    ## SDSS-all [78.045, 19.606, 2.347 ]  SDSS_nfof = [74.485, 23.314, 2.2003 ]
explode = (0.0,0.0,0.1)
colour_pie = [colour_plot[2], colour_plot[5], colour_plot[8]]
fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, colors=colour_pie,autopct='%1.1f%%', shadow=True, startangle=50, wedgeprops={'linewidth' :1}, textprops={'size':15})
ax1.axis('equal')

extra = mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="Abundance Matching")
legendHandles = []
legendHandles.append(extra)
legendHandles.append(extra)

leg = plt.legend(handles=handles[0:1],loc='lower right', frameon=False, handler_map={tuple:mpl.legend_handler.HandlerTuple(None)})
plt.gca().add_artist(leg)

plt.title('$N_{g} \\geq 5$')
plt.savefig(path_plot+'SDSS_Type_pieChart_nfof.png')
plt.close()





plt = Common_module.load_matplotlib(12,8)
colour_plot = Common_module.colour_scheme(colourmap = plt.cm.Set1)
plt.figure(figsize=(12,8))
bins_plot = np.arange(10,15,0.2)

Mvir_GAMA_plot 	= np.log10(Mvir_GAMA[Mvir_GAMA != 0]/h)
Mvir_Mock_plot 	= np.log10(Mvir_Mock_GAMA[Mvir_GAMA !=0])
colour 			= type_gal_GAMA[Mvir_GAMA != 0]
colour_plot = Common_module.colour_scheme(colourmap = plt.cm.RdPu) # Accent Set1 RdPu Pastel1

plt.hist(Mvir_GAMA_plot[colour == 0],bins=bins_plot,log=True, label='Type = 0', color=colour_plot[2], alpha=0.5)#, density=True)
plt.hist(Mvir_GAMA_plot[colour == 1],bins=bins_plot,log=True, label='Type = 1', color=colour_plot[5], alpha=0.5)#, density=True)
plt.hist(Mvir_GAMA_plot[colour == 2],bins=bins_plot,log=True, label='Type = 2', color=colour_plot[8], alpha=0.5)#, density=True)

plt.xlabel('$M_{vir}[M_{\odot}]$ (Dynamical Mass)')
plt.ylabel('$N_{Groups}$')
plt.legend()
plt.xlim(11,15)

plt.savefig(path_plot+"GAMA_Histogram_HALO_cenvssat_all.png")
plt.close()



plt = Common_module.load_matplotlib(12,8)
colour_plot = Common_module.colour_scheme(colourmap = plt.cm.Set1)
plt.figure(figsize=(12,8))
bins_plot = np.arange(10,15,0.2)

Mvir_GAMA_plot 	= np.log10(Mvir_GAMA_nfof[Mvir_GAMA_nfof != 0]/h)
Mvir_Mock_plot 	= np.log10(Mvir_Mock_GAMA_nfof[Mvir_GAMA_nfof !=0])
colour 			= type_gal_GAMA_nfof[Mvir_GAMA_nfof != 0]
colour_plot = Common_module.colour_scheme(colourmap = plt.cm.RdPu) # Accent Set1 RdPu Pastel1
#
plt.hist(Mvir_GAMA_plot[colour == 0],bins=bins_plot,log=True, label='Type = 0', color=colour_plot[2], alpha=0.5)#, density=True)
plt.hist(Mvir_GAMA_plot[colour == 1],bins=bins_plot,log=True, label='Type = 1', color=colour_plot[5], alpha=0.5)#, density=True)
plt.hist(Mvir_GAMA_plot[colour == 2],bins=bins_plot,log=True, label='Type = 2', color=colour_plot[8], alpha=0.5)#, density=True)

plt.xlabel('$M_{vir}[M_{\odot}]$ (Dynamical Mass)')
plt.ylabel('$N_{Groups}$')
plt.legend()
plt.xlim(11,15)

plt.savefig(path_plot+"GAMA_Histogram_HALO_cenvssat_nfof.png")
plt.close()



############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
######--------------------------------------------------------------------------------------------------------------------------------------
############################################################################################################################################

plt = Common_module.load_matplotlib(12,8)
colour_plot = Common_module.colour_scheme(colourmap = plt.cm.Set2)
legendHandles = []

fig, ax = plt.subplots(figsize=(12,8))
colour = np.array(grand_group_SDSS['Nfof'])
colour = colour[Mvir_SDSS != 0]
Mvir_GAMA_plot = np.log10(Mvir_SDSS[Mvir_SDSS != 0]/h)
Mvir_Mock_plot = np.log10(Mvir_Mock_SDSS[Mvir_SDSS !=0]*h)
plt.scatter(Mvir_Mock_plot[colour < 5],Mvir_GAMA_plot[colour < 5],  s=5, c='lightgrey', alpha=0.2)

colour = np.log10(np.array(grand_group_SDSS_nfof['Nfof']))
Mvir_GAMA_plot = np.log10(Mvir_SDSS_nfof[Mvir_SDSS_nfof != 0]/h)
Mvir_Mock_plot = np.log10(Mvir_Mock_SDSS_nfof[Mvir_SDSS_nfof !=0]*h)
plt.scatter(Mvir_Mock_plot,Mvir_GAMA_plot,  s=15*(colour), c=colour, cmap='Spectral_r', label='Trial')#, cmap='RdPu')

cbar=plt.colorbar()
cbar.set_label('$log_{10}$($N_{g}$)', rotation=270, labelpad = 20)
plt.plot([8,15],[8,15], '--k', linewidth=1)
plt.xlabel('$log_{10}(M^{SHARK}_{vir} [M_{\odot}])$')#' (SHARK-ref)')
plt.ylabel('$log_{10}(M^{Ab-Mat}_{vir} [M_{\odot}])$')#' (Dynamical Mass) ')

l1, = ax.plot([0,0],[0,0], marker = 'o', linestyle=' ', color ='lightgrey', label = '$N_{g}$ $<$ 5', markersize=10)
l2, = ax.plot([0,0],[0,0], marker = 'o', linestyle=' ', color ='darkslateblue', label = '$N_{g}$ $\\geq$ 5', markersize=10)
l3, = ax.plot([0,0],[0,0], marker = 'o', linestyle=' ', color ='maroon')

# handles = [(l1), (l2,l3)]
# _, labels = ax.get_legend_handles_labels()

# plt.legend(handles = handles, labels = labels, loc='upper left', handler_map={tuple:mpl.legend_handler.HandlerTuple(None)})

extra = mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="Abundance Matching")


handles = [(l1), (l2,l3),(extra), (extra)]
_, labels = ax.get_legend_handles_labels()

# plt.legend(handles = handles, labels = labels, loc='upper left', handler_map={tuple:mpl.legend_handler.HandlerTuple(None)})

leg = plt.legend(handles = handles[0:2], labels = labels, loc='upper left', handler_map={tuple:mpl.legend_handler.HandlerTuple(None)})
plt.gca().add_artist(leg)
leg = plt.legend(handles=handles[2:3],loc='lower right', frameon=False, handler_map={tuple:mpl.legend_handler.HandlerTuple(None)})
plt.gca().add_artist(leg)




plt.xlim(11,15)
plt.ylim(11.01,15)
plt.savefig(path_plot+'new_SDSS-match-Mvir-NFOF.png')
plt.close()



plt = Common_module.load_matplotlib(12,8)
colour_plot = Common_module.colour_scheme(colourmap = plt.cm.Set1)
plt.figure(figsize=(12,8))
bins_plot = np.arange(10,15,0.2)
colour = np.array(grand_group['Nfof'])
colour = colour[Mvir_GAMA != 0]

plt.hist(np.log10(Mvir_Mock_SDSS[Mvir_Mock_SDSS != 0]*h),bins=bins_plot,log=True, label='Intrinsic', color='lightgrey',fill=False, histtype='step', linewidth=3,hatch='/')#, density=True)
plt.hist(np.log10(Mvir_SDSS[Mvir_SDSS != 0]/h),bins=bins_plot,log=True, label='Abundance Matching', color='maroon', fill=False, histtype='step', linewidth=3)#, density=True)

plt.xlabel('$M_{vir}[M_{\odot}]$ ($N_{g}$ $\\geq$ 2)')
plt.ylabel('$N_{Groups}$')
plt.legend()
plt.xlim(10,15)
plt.ylim(2,7000)
plt.savefig(path_plot+"SDSS_Histogram_HALO.png")
plt.close()



plt = Common_module.load_matplotlib(12,8)
colour_plot = Common_module.colour_scheme(colourmap = plt.cm.Set1)
plt.figure(figsize=(12,8))
bins_plot = np.arange(10,15,0.2)
colour = np.array(grand_group['Nfof'])
colour = colour[Mvir_GAMA != 0]

plt.hist(np.log10(Mvir_Mock_SDSS_nfof[Mvir_Mock_SDSS_nfof != 0]*h),bins=bins_plot,log=True, label='Intrinsic', color='lightgrey',fill=False, histtype='step', linewidth=3,hatch='/')#, density=True)
plt.hist(np.log10(Mvir_SDSS_nfof[Mvir_SDSS_nfof != 0]/h),bins=bins_plot,log=True, label='Abundance Matching', color='maroon',fill=False, histtype='step', linewidth=3)#, density=True)

plt.xlabel('$M_{vir}[M_{\odot}]$ ($N_{g}$ $\\geq$ 5)')
plt.ylabel('$N_{Groups}$')
plt.legend(loc='upper left')
plt.xlim(10,15)
plt.ylim(2,500)
plt.savefig(path_plot+"Mock-SDSS_Histogram_HALO.png")
plt.close()



#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------



plt = Common_module.load_matplotlib(12,8)
colour_plot = Common_module.colour_scheme(colourmap = plt.cm.Set2)

plt.figure(figsize=(12,8))
Mvir_GAMA_plot = np.log10(Mvir_SDSS[Mvir_SDSS != 0]/h)
Mvir_Mock_plot = np.log10(Mvir_Mock_SDSS[Mvir_SDSS !=0]*h)
colour = np.array(grand_group_SDSS['Nfof'])
colour = colour[Mvir_SDSS != 0]
plt.scatter(Mvir_Mock_plot[colour < 5],Mvir_GAMA_plot[colour < 5], s=5, c='silver')


colour = type_gal
norm = colors.Normalize(vmin=0,vmax=2)
import matplotlib.cm as cm
# # cmap, norm = mpl.colors.from_levels_and_colors([0, 1, 2,3], [colour_plot[1], colour_plot[2], colour_plot[3]])
# cmap, norm = mpl.colors.from_levels_and_colors([0, 1, 2,3], ['gold', 'green', 'darkred'])

colour_plot = Common_module.colour_scheme(colourmap = plt.cm.RdPu) # Accent Set1 RdPu Pastel1
cmap, norm = mpl.colors.from_levels_and_colors([0, 1, 2,3], [colour_plot[2], colour_plot[5], colour_plot[8]])

Mvir_GAMA_plot = np.log10(Mvir_SDSS_nfof[Mvir_SDSS_nfof != 0]/h)
Mvir_Mock_plot = np.log10(Mvir_Mock_SDSS_nfof[Mvir_SDSS_nfof !=0]*h)

plt.scatter(Mvir_Mock_plot,Mvir_GAMA_plot, s=15, c=type_gal_nfof, cmap=cmap, norm=norm)

cbar = plt.colorbar(ticks=[0.5,1.5,2.5], format='%1i')
cbar.set_label('Galaxy Type in SHARK', rotation=270, labelpad = 20)
plt.plot([10,15],[10,15], '--k', linewidth=1)
plt.xlabel('$log_{10}(M^{SHARK}_{vir} [M_{\odot}])$')#' (SHARK-ref)')
plt.ylabel('$log_{10}(M^{Ab-Mat}_{vir} [M_{\odot}])$')#' (Dynamical Mass) ')


l1, = ax.plot([0,0],[0,0], marker = 'o', linestyle=' ', color ='lightgrey', label = '$N_{g}$ $<$ 5', markersize=10)
l2, = ax.plot([0,0],[0,0], marker = 'o', linestyle=' ', color =colour_plot[2], label = '$N_{g}$ $\\geq$ 5', markersize=10)
l3, = ax.plot([0,0],[0,0], marker = 'o', linestyle=' ', color =colour_plot[5], markersize=10)
l4, = ax.plot([0,0],[0,0], marker = 'o', linestyle=' ', color =colour_plot[8], markersize=10)

# l2, = ax.plot([0,0],[0,0], marker = 'o', linestyle=' ', color ='gold', label = '$N_{g}$ $\\geq$ 5', markersize=10)
# l3, = ax.plot([0,0],[0,0], marker = 'o', linestyle=' ', color ='green', markersize=10)
# l4, = ax.plot([0,0],[0,0], marker = 'o', linestyle=' ', color ='darkred', markersize=10)
extra = mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="Abundance Matching")


handles = [(l1), (l2,l3,l4),(extra), (extra)]
_, labels = ax.get_legend_handles_labels()

# plt.legend(handles = handles, labels = labels, loc='upper left', handler_map={tuple:mpl.legend_handler.HandlerTuple(None)})

leg = plt.legend(handles = handles[0:2], labels = labels, loc='upper left', handler_map={tuple:mpl.legend_handler.HandlerTuple(None)})
plt.gca().add_artist(leg)
leg = plt.legend(handles=handles[2:3],loc='lower right', frameon=False, handler_map={tuple:mpl.legend_handler.HandlerTuple(None)})
plt.gca().add_artist(leg)



plt.xlim(11,15)
plt.ylim(11.01,15)
plt.savefig(path_plot+'new_SDSS-match-Mvir-satvscen.png')
plt.close()



#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

plt = Common_module.load_matplotlib(12,8)
colour_plot = Common_module.colour_scheme(colourmap = plt.cm.RdPu) # Accent Set1 RdPu Pastel1

plt.figure(figsize=(12,8))
bins_plot = np.arange(10,15,0.2)

Mvir_GAMA_plot 	= np.log10(Mvir_SDSS[Mvir_SDSS != 0]/h)
Mvir_Mock_plot 	= np.log10(Mvir_Mock_SDSS[Mvir_SDSS !=0]*h)
colour 			= type_gal[Mvir_SDSS != 0]

plt.hist(Mvir_GAMA_plot[colour == 0],bins=bins_plot,log=True, label='Type = 0', color=colour_plot[2], alpha=0.5)#, density=True)
plt.hist(Mvir_GAMA_plot[colour == 1],bins=bins_plot,log=True, label='Type = 1', color=colour_plot[5], alpha=0.5)#, density=True)
plt.hist(Mvir_GAMA_plot[colour == 2],bins=bins_plot,log=True, label='Type = 2', color=colour_plot[8], alpha=0.5)#, density=True)

plt.xlabel('$M_{vir}[M_{\odot}]$ ($N_{g}$ $\\geq$ 2)')
plt.ylabel('Density')
plt.legend()
plt.xlim(11,15)

plt.savefig(path_plot+"SDSS_Histogram_HALO_cenvssat_all.png")
plt.close()



plt = Common_module.load_matplotlib(12,8)
colour_plot = Common_module.colour_scheme(colourmap = plt.cm.RdPu) # Accent Set1 RdPu Pastel1

plt.figure(figsize=(12,8))
bins_plot = np.arange(10,15,0.2)

Mvir_GAMA_plot 	= np.log10(Mvir_SDSS_nfof[Mvir_SDSS_nfof != 0]/h)
Mvir_Mock_plot 	= np.log10(Mvir_Mock_SDSS_nfof[Mvir_SDSS_nfof !=0]*h)
colour 			= type_gal_nfof[Mvir_SDSS_nfof != 0]

plt.hist(Mvir_GAMA_plot[colour == 0],bins=bins_plot,log=True, label='Type = 0', color=colour_plot[2], alpha=0.5)#, density=True)
plt.hist(Mvir_GAMA_plot[colour == 1],bins=bins_plot,log=True, label='Type = 1', color=colour_plot[5], alpha=0.5)#, density=True)
plt.hist(Mvir_GAMA_plot[colour == 2],bins=bins_plot,log=True, label='Type = 2', color=colour_plot[8], alpha=0.5)#, density=True)

plt.xlabel('$M_{vir}[M_{\odot}]$ ($N_{g}$ $\\geq$ 5)')
plt.ylabel('Density')
plt.legend()
plt.xlim(11,15)

plt.savefig(path_plot+"SDSS_Histogram_HALO_cenvssat_nfof.png")
plt.close()







#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def halo_value_list(virial_mass,property_plot,mean):
	
	bin_for_disk = np.arange(11,16,0.4)

	halo_mass 		= np.zeros(len(bin_for_disk))
	prop_mass 		= np.zeros(len(bin_for_disk))

	prop_mass_low 		= np.zeros(len(bin_for_disk))
	prop_mass_high		= np.zeros(len(bin_for_disk))
	
	bin_members 		= np.zeros(len(bin_for_disk))

	if mean == True:
		for i in range(1,len(bin_for_disk)):
			halo_mass[i-1] 		= np.log10(np.mean(virial_mass[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])]))
			prop_mass[i-1]			= np.log10(np.mean(property_plot[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])]))

			bootarr 			= np.log10(property_plot[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])])
			bootarr = bootarr[bootarr != float('-inf')]
			
			print(len(virial_mass[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])]))

			bin_members[i-1] = len(virial_mass[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])])


			if len(bootarr) > 10:
				if bootarr != []:
					with NumpyRNGContext(1):
						bootresult 			= bootstrap(bootarr,10,bootfunc=np.mean)
						bootresult_error	= bootstrap(bootarr,10,bootfunc=stats.tstd)/2

					prop_mass_low[i-1]	= prop_mass[i-1] - np.average(bootresult_error)
					prop_mass_high[i-1]	= np.average(bootresult_error) + prop_mass[i-1]
					
			
	
	else:
		for i in range(1,len(bin_for_disk)):
			halo_mass[i-1] 		= np.log10(np.median(virial_mass[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])]))
			prop_mass[i-1]			= np.log10(np.median(property_plot[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])]))
			
			bootarr 			= np.log10(property_plot[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])])
			bootarr = bootarr[bootarr != float('-inf')]

			print(len(virial_mass[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])]))
			
			bin_members[i-1] = len(virial_mass[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])])

			if len(bootarr) > 10 :
				if bootarr != []:
					with NumpyRNGContext(1):
						bootresult 			= bootstrap(bootarr,10,bootfunc=np.median)
						bootresult_lower	= bootstrap(bootarr,10,bootfunc=nanpercentile_lower)
						bootresult_upper	= bootstrap(bootarr,10,bootfunc=nanpercentile_upper)

					prop_mass_low[i-1]	= np.mean(bootresult_lower)

					prop_mass_high[i-1]	= np.mean(bootresult_upper)


	return halo_mass, prop_mass, prop_mass_low, prop_mass_high, bin_members





def plotting_properties_halo_fill_between(virial_mass,property_plot,mean,legend_handles,property_name_x='X-Label-math-mode',legend_name='name-of-run', property_name_y='Y-Label-math-mode', xlim_lower=6,xlim_upper=15,ylim_lower=-4,ylim_upper=11,colour_line='r',fill_between=False, first_legend = False, resolution=False, halo_bins=0):
	

	if halo_bins == 0:
		bin_for_disk = np.arange(7,15.2,0.4)

	elif halo_bins == 1:
		bin_for_disk = [10,11.1,11.2,11.3,11.5,11.7,13.1,15] - np.log10(h)

	else:
		bin_for_disk = [9,11.7,12.4,13.0,14.6] - np.log10(h)


	halo_mass 		= np.zeros(len(bin_for_disk))
	prop_mass 		= np.zeros(len(bin_for_disk))

	prop_mass_low 		= np.zeros(len(bin_for_disk))
	prop_mass_high		= np.zeros(len(bin_for_disk))
	
	halo_mass_plot 		= []
	prop_mass_plot 		= []
	if mean == True:
		for i in range(1,len(bin_for_disk)):
			halo_mass[i-1] 		= np.log10(np.mean(virial_mass[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])]))
			prop_mass[i-1]			= np.log10(np.mean(property_plot[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])]))

			bootarr 			= np.log10(property_plot[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])])
			bootarr = bootarr[bootarr != float('-inf')]
			

			if len(bootarr) > 10:
				if bootarr != []:
					with NumpyRNGContext(1):
						bootresult 			= bootstrap(bootarr,100,bootfunc=np.mean)
						bootresult_error	= bootstrap(bootarr,100,bootfunc=stats.tstd)#/2
						# bootresult_error	= bootstrap(bootarr,10,bootfunc=stats.sem)
						# bootresult_error	= bootstrap(bootarr,10,bootfunc=statistics.median_grouped)
						# bootresult_error_low	= bootstrap(bootarr,10,bootfunc=statistics.median_low)
						# bootresult_error_high	= bootstrap(bootarr,10,bootfunc=statistics.median_high)
				
					prop_mass_low[i-1]	= prop_mass[i-1] - np.average(bootresult_error)
					prop_mass_high[i-1]	= np.average(bootresult_error) + prop_mass[i-1]
					
					# prop_mass_low[i-1]	= np.average(bootresult_error_low)
					# prop_mass_high[i-1]	= np.average(bootresult_error_high)
			
	
	else:
		for i in range(1,len(bin_for_disk)):
			halo_mass[i-1] 		= np.log10(np.median(virial_mass[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])]))
			prop_mass[i-1]			= np.log10(np.median(property_plot[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])]))
			
			bootarr 			= np.log10(property_plot[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])])
			bootarr = bootarr[bootarr != float('-inf')]


			if len(bootarr) > 10 :
				if bootarr != []:
					with NumpyRNGContext(1):
						bootresult 			= bootstrap(bootarr,100,bootfunc=np.median)
						bootresult_lower	= bootstrap(bootarr,100,bootfunc=nanpercentile_lower)
						bootresult_upper	= bootstrap(bootarr,100,bootfunc=nanpercentile_upper)

					prop_mass_low[i-1]	= np.mean(bootresult_lower)
					prop_mass_high[i-1]	= np.mean(bootresult_upper)
						
					

	
	if fill_between == True:
		plt.fill_between(halo_mass[(prop_mass_high != 0)],prop_mass_low[ (prop_mass_high != 0)],prop_mass_high[(prop_mass_high != 0)],color=colour_line,alpha=0.1, label='$\\rm %s$' %legend_name)
	
	plt.xlabel('$\\rm %s$ '%property_name_x)
	plt.ylabel('$\\rm %s$ '%property_name_y)

	print(halo_mass[prop_mass_high != 0])
	print(prop_mass_low[prop_mass_high != 0])
	print(prop_mass_high[prop_mass_high != 0])
	
	plt.xlim(xlim_lower,xlim_upper)
	plt.ylim(ylim_lower,ylim_upper)

	#plt.legend(frameon=False)
	# plt.savefig('%s.png'%figure_name)
	# plt.show()





GAMA_mock_all, GAMA_all, GAMA_all_error_low, GAMA_all_error_high,yoyo = halo_value_list(Mvir_Mock_GAMA, Mvir_GAMA/h, mean=False)

GAMA_mock_nfof, GAMA_nfof, GAMA_nfof_error_low, GAMA_nfof_error_high,yoyo = halo_value_list(Mvir_Mock_GAMA_nfof, Mvir_GAMA_nfof/h, mean=False)

SDSS_mock_all, SDSS_all, SDSS_all_error_low, SDSS_all_error_high,yoyo = halo_value_list(Mvir_Mock_SDSS*h, Mvir_SDSS/h, mean=False)

SDSS_mock_nfof, SDSS_nfof, SDSS_nfof_error_low, SDSS_nfof_error_high,yoyo = halo_value_list(Mvir_Mock_SDSS_nfof*h, Mvir_SDSS_nfof/h, mean=False)


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# def plotting_properties_halo_fill_between(virial_mass,property_plot,mean,legend_handles,property_name_x='X-Label-math-mode',legend_name='name-of-run', property_name_y='Y-Label-math-mode', xlim_lower=6,xlim_upper=15,ylim_lower=-4,ylim_upper=11,colour_line='r',fill_between=False, first_legend = False, resolution=False, halo_bins=0):



plt = Common_module.load_matplotlib(12,8)
colour_plot = Common_module.colour_scheme(colourmap = plt.cm.Set1)

plt.figure(figsize=(12,8))

legendHandles = []

Halo_GAMA_mean, HI_GAMA_mean, a,b, number = halo_value_list(Mvir_Mock_GAMA, Mvir_GAMA/h, mean=False)

Error_SAM_2         = plt.errorbar(Halo_GAMA_mean[0:len(Halo_GAMA_mean)-1], (HI_GAMA_mean[0:len(Halo_GAMA_mean)-1]), yerr=None,marker = "d", mfc = 'rebeccapurple', mec = 'k', c = 'k', capsize=2,ls = '--', markersize=15, linewidth=1, barsabove=True, capthick=2, label='$N_{g} \\geq 2$ (Median)')

plotting_properties_halo_fill_between(Mvir_Mock_GAMA, Mvir_GAMA/h, mean=False, legend_handles=legendHandles, fill_between=True, colour_line='rebeccapurple')

Halo_GAMA_mean, HI_GAMA_mean, a,b, number = halo_value_list(Mvir_Mock_GAMA_nfof, Mvir_GAMA_nfof/h, mean=False)

Error_SAM_3         = plt.errorbar(Halo_GAMA_mean[2:len(Halo_GAMA_mean)-1], (HI_GAMA_mean[2:len(Halo_GAMA_mean)-1]), yerr=None,marker = "s", mfc = 'goldenrod', mec = 'k', c = 'k', capsize=2,ls = '--', markersize=15, linewidth=1, barsabove=True, capthick=2, label='$N_{g} \\geq 5$ (Median)')

plotting_properties_halo_fill_between(Mvir_Mock_GAMA_nfof, Mvir_GAMA_nfof/h, mean=False, legend_handles=legendHandles, fill_between=True, colour_line='goldenrod')



plt.plot([10,15],[10,15], ':k', linewidth=1)

plt.xlabel('$log_{10}(M^{SHARK}_{vir} [M_{\odot}])$')#' (SHARK-ref)')
plt.ylabel('$log_{10}(M^{Dyn}_{vir} [M_{\odot}])$')#' (Dynamical Mass) ')

legendHandles.append(Error_SAM_2)
legendHandles.append(Error_SAM_3)

extra = mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="Dynamical Mass")

legendHandles.append(extra)
legendHandles.append(extra)

leg = plt.legend(handles=legendHandles[0:2],loc='upper left', frameon=False)
plt.gca().add_artist(leg)
leg = plt.legend(handles=legendHandles[2:3],loc='lower right', frameon=False)
plt.gca().add_artist(leg)


# plt.legend(handles=legendHandles)
plt.xlim(11,15)
plt.ylim(11.01,15)
plt.savefig(path_plot+"GAMA_median_values.png")
plt.close()




plt = Common_module.load_matplotlib(12,8)
colour_plot = Common_module.colour_scheme(colourmap = plt.cm.Set1)
plt.figure(figsize=(12,8))
legendHandles = []

Halo_GAMA_mean, HI_GAMA_mean, a,b ,c = halo_value_list(Mvir_Mock_SDSS*h, Mvir_SDSS/h, mean=False)

Error_SAM_2         = plt.errorbar(Halo_GAMA_mean[0:len(Halo_GAMA_mean)-1], (HI_GAMA_mean[0:len(Halo_GAMA_mean)-1]), yerr=None,marker = "d", mfc = 'rebeccapurple', mec = 'k', c = 'k', capsize=2,ls = '--', markersize=15, linewidth=1, barsabove=True, capthick=2, label='$N_{g} \\geq 2$ (Median)')


plotting_properties_halo_fill_between(Mvir_Mock_SDSS*h, Mvir_SDSS/h, mean=False, legend_handles=legendHandles, fill_between=True, colour_line='rebeccapurple')


Halo_GAMA_mean, HI_GAMA_mean, a,b,c = halo_value_list(Mvir_Mock_SDSS_nfof*h, Mvir_SDSS_nfof/h, mean=False)

Error_SAM_3         = plt.errorbar(Halo_GAMA_mean[2:len(Halo_GAMA_mean)-1], (HI_GAMA_mean[2:len(Halo_GAMA_mean)-1]), yerr=None,marker = "s", mfc = 'goldenrod', mec = 'k', c = 'k', capsize=2,ls = '--', markersize=15, linewidth=1, barsabove=True, capthick=2, label='$N_{g} \\geq 5$ (Median)')


plotting_properties_halo_fill_between(Mvir_Mock_SDSS_nfof*h, Mvir_SDSS_nfof/h, mean=False, legend_handles=legendHandles, fill_between=True, colour_line='goldenrod')


plt.plot([10,15],[10,15], ':k', linewidth=1)

plt.xlabel('$log_{10}(M^{SHARK}_{vir} [M_{\odot}])$')#' (SHARK-ref)')
plt.ylabel('$log_{10}(M^{Ab-Mat}_{vir} [M_{\odot}])$')#' (Dynamical Mass) ')


legendHandles.append(Error_SAM_2)
legendHandles.append(Error_SAM_3)

extra = mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="Abundance Matching")

legendHandles.append(extra)
legendHandles.append(extra)

leg = plt.legend(handles=legendHandles[0:2],loc='upper left', frameon=False)
plt.gca().add_artist(leg)
leg = plt.legend(handles=legendHandles[2:3],loc='lower right', frameon=False)
plt.gca().add_artist(leg)

# plt.legend(handles=legendHandles)
plt.xlim(11,15)
plt.ylim(11.01,15)
plt.savefig(path_plot+"SDSS_median_values.png")
plt.close()
















# plt = Common_module.load_matplotlib(12,8)
# colour_plot = Common_module.colour_scheme(colourmap = plt.cm.Set1)

# bins_plot = np.arange(10,16,0.2)

# plt.hist(np.log10(Mvir_GAMA[Mvir_GAMA != 0]*h),bins=bins_plot,log=False, label='GAMA', color=colour_plot[1], alpha=0.5, density=True)
# plt.hist(np.log10(Mvir_SDSS[Mvir_SDSS != 0]),bins=bins_plot,log=False, label='SDSS', color=colour_plot[2], alpha=0.5, density=True)
# plt.hist(np.log10(Mvir_Mock_SDSS[Mvir_Mock_SDSS != 0]*h*h),bins=bins_plot,log=False, label='Mock', color=colour_plot[3], alpha=0.5, density=True)


# plt.xlabel('$M_{vir}[h^{-1} M_{\odot}]$')
# plt.ylabel('Density')
# plt.legend()
# plt.xlim(11,15)

# plt.savefig(path_plot+"GRP_HaloMass_Histogram.png")
# plt.close()


# bins_plot = np.arange(7,12,0.2)

# plt = Common_module.load_matplotlib(12,8)
# colour_plot = Common_module.colour_scheme(colourmap = plt.cm.Set1)

# plt.hist(np.log10(Mstar_GAMA[Mstar_GAMA != 0]),bins=bins_plot,log=False, label='GAMA', color=colour_plot[1], alpha=0.5, density=True)
# plt.hist(np.log10(Mstar_SDSS[Mstar_SDSS != 0]),bins=bins_plot,log=False, label='SDSS', color=colour_plot[2], alpha=0.5, density=True)
# plt.hist(np.log10(Mstar_Mock[Mstar_Mock != 0]),bins=bins_plot,log=False, label='Mock', color=colour_plot[3], alpha=0.5, density=True)


# plt.xlabel('$M_{star}[M_{\odot}]$')
# plt.ylabel('Number Density')
# plt.legend()
# # plt.xlim(10,15)

# plt.savefig("GRP_StellarMass_Histogram.png")
# plt.close()


# bins_plot = np.arange(6,12,0.2)

# plt = Common_module.load_matplotlib(12,8)
# colour_plot = Common_module.colour_scheme(colourmap = plt.cm.Set1)

# plt.hist(np.log10(MHI_GAMA[MHI_GAMA != 0]),bins=bins_plot,log=False, label='GAMA', color=colour_plot[1], alpha=0.5, density=True)
# plt.hist(np.log10(MHI_SDSS[MHI_SDSS != 0]),bins=bins_plot,log=False, label='SDSS', color=colour_plot[2], alpha=0.5, density=True)
# plt.hist(np.log10(MHI_Mock[MHI_Mock != 0]),bins=bins_plot,log=False, label='Mock', color=colour_plot[3], alpha=0.5, density=True)


# plt.xlabel('$M_{HI}[M_{\odot}]$')
# plt.ylabel('Number Density')
# plt.legend()
# # plt.xlim(10,15)

# plt.savefig("GRP_GasMass_Histogram.png")
# plt.close()


