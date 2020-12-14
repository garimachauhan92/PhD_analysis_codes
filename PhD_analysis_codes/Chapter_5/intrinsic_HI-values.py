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
from Common_module import LightconeReading
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
###                                                   Reading Lightcone 
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************

def prepare_data(path,subvolumes,filename):

    fields_read     = {'galaxies':('id_halo_sam', 'id_galaxy_sam','id_galaxy_sky', 'type', 'mvir_hosthalo','mvir_subhalo', 'matom_bulge', 'matom_disk', 'ra', 'dec', 'vpec_r','vpec_x','vpec_y','vpec_z', 'zcos', 'zobs', 'mstars_bulge','mstars_disk', 'id_group_sky', 'mag', 'snapshot', 'subvolume')} #'vvir_hosthalo', 'vvir_subhalo',

    data_reading = LightconeReading(path,subvolumes,filename)

    data_plotting = data_reading.readIndividualFiles_test(fields=fields_read)

    return data_plotting


id_halo_sam_all     = []
id_galaxy_sam_all   = []
id_galaxy_sky_all   = []
type_g_all          = []
mvir_hosthalo_all   = []
mvir_subhalo_all    = []
matom_sam_all       = []
mstars_sam_all      = []
ra_all              = []
dec_all             = []
vpec_r_all          = []
vpec_x_all          = []
vpec_y_all          = []
vpec_z_all          = []
# vvir_hosthalo_all = []
# vvir_subhalo_all  = []
zcos_all            = []
zobs_all            = []
id_group_sky_all     = []
mag_all             = []
snapshot_all        = []
subvolume_all       = []


for i in range(subvolumes):
    print(i)

    (id_halo_sam, id_galaxy_sam, id_galaxy_sky, type_g, mvir_hosthalo, mvir_subhalo, matom_bulge, matom_disk, ra, dec, vpec_r,vpec_x,vpec_y,vpec_z, zcos, zobs,mstars_bulge,mstars_disk, id_group_sky, mag, snapshot, subvolume) = prepare_data(path_GAMA, i, 'mock_medi_alfalfa') #'mock_medi_alfalfa') #'mocksky')# #, vvir_hosthalo, vvir_subhalo

    mag_yo              = mag

    id_halo_sam_all     = np.append(id_halo_sam_all,id_halo_sam[(zobs <= 0.06) & (mag_yo <= 20)])
    id_galaxy_sam_all   = np.append(id_galaxy_sam_all,id_galaxy_sam[(zobs <= 0.06) & (mag_yo <= 20)])
    id_galaxy_sky_all   = np.append(id_galaxy_sky_all,id_galaxy_sky[(zobs <= 0.06) & (mag_yo <= 20)])
    type_g_all          = np.append(type_g_all,type_g[(zobs <= 0.06) & (mag_yo <= 20)])
    mvir_hosthalo_all   = np.append(mvir_hosthalo_all,mvir_hosthalo[(zobs <= 0.06) & (mag_yo <= 20)])
    mvir_subhalo_all    = np.append(mvir_subhalo_all,mvir_subhalo[(zobs <= 0.06) & (mag_yo <= 20)])
    matom_sam_all       = np.append(matom_sam_all,matom_bulge[(zobs <= 0.06) & (mag_yo <= 20)] + matom_disk[(zobs <= 0.06) & (mag_yo <= 20)])
    mstars_sam_all      = np.append(mstars_sam_all,mstars_bulge[(zobs <= 0.06) & (mag_yo <= 20)]+mstars_disk[(zobs <= 0.06) & (mag_yo <= 20)])
    ra_all              = np.append(ra_all,ra[(zobs <= 0.06) & (mag_yo <= 20)])
    dec_all             = np.append(dec_all,dec[(zobs <= 0.06) & (mag_yo <= 20)])
    vpec_r_all          = np.append(vpec_r_all,vpec_r[(zobs <= 0.06) & (mag_yo <= 20)])
    vpec_x_all          = np.append(vpec_x_all,vpec_x[(zobs <= 0.06) & (mag_yo <= 20)])
    vpec_y_all          = np.append(vpec_y_all,vpec_y[(zobs <= 0.06) & (mag_yo <= 20)])
    vpec_z_all          = np.append(vpec_z_all,vpec_z[(zobs <= 0.06) & (mag_yo <= 20)])
    # vvir_hosthalo_all = np.append(vvir_hosthalo_all,vvir_hosthalo[zobs <= 0.06])
    # vvir_subhalo_all  = np.append(vvir_subhalo_all,vvir_subhalo[zobs <= 0.06])
    id_group_sky_all    = np.append(id_group_sky_all, id_group_sky[(zobs <= 0.06) & (mag_yo <= 20)])
    mag_all             = np.append(mag_all, mag[(zobs <= 0.06) & (mag_yo <= 20)])
    snapshot_all        = np.append(snapshot_all, snapshot[(zobs <= 0.06) & (mag_yo <= 20)])
    subvolume_all        = np.append(subvolume_all, subvolume[(zobs <= 0.06) & (mag_yo <= 20)])

    zcos_all            = np.append(zcos_all,zcos[(zobs <= 0.06) & (mag_yo <= 20)])
    zobs_all            = np.append(zobs_all,zobs[(zobs <= 0.06) & (mag_yo <= 20)])



############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
###                                                   Making dataframe
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************

id_halo_sam_all     = [int(k) for k in id_halo_sam_all]
id_galaxy_sam_all   = [int(k) for k in id_galaxy_sam_all]
id_galaxy_sky_all   = [int(k) for k in id_galaxy_sky_all]
type_g_all          = [int(k) for k in type_g_all]
mvir_hosthalo_all   = [int(k) for k in mvir_hosthalo_all]
mvir_subhalo_all    = [int(k) for k in mvir_subhalo_all] 
matom_sam_all       = [int(k) for k in matom_sam_all]
mstars_sam_all      = [int(k) for k in mstars_sam_all]
new_group_id        = np.zeros(len(snapshot_all))

for i in range(len(snapshot_all)):
    new_group_id[i]        = snapshot_all[i]*10**10 + subvolume_all[i]*10**8 + id_halo_sam_all[i] 

dictionary_df = {'id_halo_sam':id_halo_sam_all, 'id_galaxy_sam':id_galaxy_sam_all, 'id_galaxy_sky':id_galaxy_sky_all, 'type':type_g_all, 'mvir_hosthalo':mvir_hosthalo_all, 'mvir_subhalo':mvir_subhalo_all, 'matom_all':matom_sam_all, 'ra':ra_all, 'dec':dec_all, 'vpec_r':vpec_r_all,'vpec_x':vpec_x_all,'vpec_y':vpec_y_all,'vpec_z':vpec_z_all, 'zcos':zcos_all, 'zobs':zcos_all,'mstars_all':mstars_sam_all, 'id_group_sky':new_group_id} #, 'vvir_hosthalo':vvir_hosthalo_all, 'vvir_subhalo':vvir_subhalo_all


lightcone_alfalfa = pd.DataFrame.from_dict(dictionary_df)


############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
###                                                   Reading Lightcone 
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************

def prepare_data(path,subvolumes,filename):

    fields_read     = {'galaxies':('id_halo_sam', 'id_galaxy_sam','id_galaxy_sky', 'type', 'mvir_hosthalo','mvir_subhalo', 'matom_bulge', 'matom_disk', 'ra', 'dec', 'vpec_r','vpec_x','vpec_y','vpec_z', 'zcos', 'zobs', 'mstars_bulge','mstars_disk', 'id_group_sky', 'mag', 'snapshot', 'subvolume')} #'vvir_hosthalo', 'vvir_subhalo',

    data_reading = LightconeReading(path,subvolumes,filename)

    data_plotting = data_reading.readIndividualFiles_test(fields=fields_read)

    return data_plotting

id_halo_sam_all     = []
id_galaxy_sam_all   = []
id_galaxy_sky_all   = []
type_g_all          = []
mvir_hosthalo_all   = []
mvir_subhalo_all    = []
matom_sam_all       = []
mstars_sam_all      = []
ra_all              = []
dec_all             = []
vpec_r_all          = []
vpec_x_all          = []
vpec_y_all          = []
vpec_z_all          = []
# vvir_hosthalo_all = []
# vvir_subhalo_all  = []
zcos_all            = []
zobs_all            = []
id_group_sky_all     = []
mag_all             = []
snapshot_all        = []
subvolume_all       = []


for i in range(subvolumes):
    print(i)

    (id_halo_sam, id_galaxy_sam, id_galaxy_sky, type_g, mvir_hosthalo, mvir_subhalo, matom_bulge, matom_disk, ra, dec, vpec_r,vpec_x,vpec_y,vpec_z, zcos, zobs,mstars_bulge,mstars_disk, id_group_sky, mag, snapshot, subvolume) = prepare_data(path_GAMA, i, 'mock_medi_alfalfa') #'mock_medi_alfalfa') #'mocksky')# #, vvir_hosthalo, vvir_subhalo

    id_halo_sam_all     = np.append(id_halo_sam_all,id_halo_sam[zobs <= 0.06])
    id_galaxy_sam_all   = np.append(id_galaxy_sam_all,id_galaxy_sam[zobs <= 0.06])
    id_galaxy_sky_all   = np.append(id_galaxy_sky_all,id_galaxy_sky[zobs <= 0.06])
    type_g_all          = np.append(type_g_all,type_g[zobs <= 0.06])
    mvir_hosthalo_all   = np.append(mvir_hosthalo_all,mvir_hosthalo[zobs <= 0.06])
    mvir_subhalo_all    = np.append(mvir_subhalo_all,mvir_subhalo[zobs <= 0.06])
    matom_sam_all       = np.append(matom_sam_all,matom_bulge[zobs <= 0.06] + matom_disk[zobs <= 0.06])
    mstars_sam_all      = np.append(mstars_sam_all,mstars_bulge[zobs <= 0.06]+mstars_disk[zobs <= 0.06])
    ra_all              = np.append(ra_all,ra[zobs <= 0.06])
    dec_all             = np.append(dec_all,dec[zobs <= 0.06])
    vpec_r_all          = np.append(vpec_r_all,vpec_r[zobs <= 0.06])
    vpec_x_all          = np.append(vpec_x_all,vpec_x[zobs <= 0.06])
    vpec_y_all          = np.append(vpec_y_all,vpec_y[zobs <= 0.06])
    vpec_z_all          = np.append(vpec_z_all,vpec_z[zobs <= 0.06])
    # vvir_hosthalo_all = np.append(vvir_hosthalo_all,vvir_hosthalo[zobs <= 0.06])
    # vvir_subhalo_all  = np.append(vvir_subhalo_all,vvir_subhalo[zobs <= 0.06])

    id_group_sky_all    = np.append(id_group_sky_all, id_group_sky[(zobs <= 0.06)])
    mag_all             = np.append(mag_all, mag[(zobs <= 0.06)])
    snapshot_all        = np.append(snapshot_all, snapshot[(zobs <= 0.06)])
    subvolume_all        = np.append(subvolume_all, subvolume[(zobs <= 0.06)])
    zcos_all            = np.append(zcos_all,zcos[zobs <= 0.06])
    zobs_all            = np.append(zobs_all,zobs[zobs <= 0.06])





############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
###                                                   Making dataframe
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************

id_halo_sam_all     = [int(k) for k in id_halo_sam_all]
id_galaxy_sam_all   = [int(k) for k in id_galaxy_sam_all]
id_galaxy_sky_all   = [int(k) for k in id_galaxy_sky_all]
type_g_all          = [int(k) for k in type_g_all]
mvir_hosthalo_all   = [int(k) for k in mvir_hosthalo_all]
mvir_subhalo_all    = [int(k) for k in mvir_subhalo_all] 
matom_sam_all       = [int(k) for k in matom_sam_all]
mstars_sam_all      = [int(k) for k in mstars_sam_all]
new_group_id        = np.zeros(len(snapshot_all))

for i in range(len(snapshot_all)):
    new_group_id[i]        = snapshot_all[i]*10**10 + subvolume_all[i]*10**8 + id_halo_sam_all[i] 

dictionary_df = {'id_halo_sam':id_halo_sam_all, 'id_galaxy_sam':id_galaxy_sam_all, 'id_galaxy_sky':id_galaxy_sky_all, 'type':type_g_all, 'mvir_hosthalo':mvir_hosthalo_all, 'mvir_subhalo':mvir_subhalo_all, 'matom_all':matom_sam_all, 'ra':ra_all, 'dec':dec_all, 'vpec_r':vpec_r_all,'vpec_x':vpec_x_all,'vpec_y':vpec_y_all,'vpec_z':vpec_z_all, 'zcos':zcos_all, 'zobs':zcos_all,'mstars_all':mstars_sam_all, 'id_group_sky':new_group_id} #, 'vvir_hosthalo':vvir_hosthalo_all, 'vvir_subhalo':vvir_subhalo_all



lightcone_alfalfa_all = pd.DataFrame.from_dict(dictionary_df)



############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
###                                                   Merging GAMA files
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************

grand_gals                  = GAMA_gals_file.merge(GAMA_censat_file,left_on='CATAID', right_on='id_galaxy_sky', how='inner')  # Merges Galaxy file with censat (merging CATAID and id_galaxy_sky)

grand_isolated              = grand_gals.loc[(grand_gals['RankIterCen']==-999) & (grand_gals['Z']<=0.06)].copy()              # only isolated centrals  
grand_group_all             = grand_gals.loc[(grand_gals['GroupID'] != 0) & (grand_gals['Z']<=0.06)].copy()                   # separates the central of groups 

grand_group                 = grand_gals.loc[(grand_gals['RankIterCen'] == 1) & (grand_gals['Z']<=0.06)].copy()                   # separates the central of groups 
grand_group                 = grand_group.merge(GAMA_group_file,on='GroupID',how='inner')
grand_group_nfof            = grand_group.loc[(grand_group['Nfof'] > 5)].copy()


grand_isolated_lightcone    = grand_isolated.merge(lightcone_alfalfa_all, on='id_galaxy_sky', how='inner')                      # matching isolated centrals to lightcone

grand_group_lightcone       = grand_group.merge(lightcone_alfalfa_all, on='id_galaxy_sky', how='inner')                         # matching group galaxies to lightcone
grand_group_lightcone_all   = grand_group_all.merge(lightcone_alfalfa_all, on='id_galaxy_sky', how='inner')                         # matching group galaxies to lightcone

grand_gals_lightcone        = grand_gals.merge(lightcone_alfalfa_all, on='id_galaxy_sky', how='inner')

grand_group_lightcone_nfof  = grand_group_nfof.merge(lightcone_alfalfa_all, on='id_galaxy_sky', how='inner')


############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
###                                                             Merging GAMA Files
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************

grand_gals		= GAMA_gals_file.merge(GAMA_censat_file,left_on='CATAID',right_on='id_galaxy_sky',how='inner')  # Merges Galaxy file with censat (merging CATAID and id_galaxy_sky)
grand_isolated 	= grand_gals.loc[(grand_gals['RankIterCen']==-999) & (grand_gals['Z']<=0.06)].copy() # only isolated centrals
grand_group 	= grand_gals.loc[(grand_gals['RankIterCen']==1) & (grand_gals['Z']<=0.06)].copy()   # separates the central of groups
grand_group 	= grand_group.merge(GAMA_group_file,on='GroupID',how='inner')  # merges group file with galaxy file


grand_gals_SDSS 				= grand_gals.loc[(grand_gals['Z']<=0.06) & (grand_gals['mag'] <= 17.77)].copy()

grand_isolated_SDSS 			= grand_gals_SDSS.loc[(grand_gals_SDSS['RankIterCen']==-999)].copy() 	
grand_group_SDSS_all 			= grand_gals_SDSS.loc[(grand_gals_SDSS['GroupID'] != 0)].copy()  

grand_group_SDSS 				= grand_gals_SDSS.loc[(grand_gals_SDSS['RankIterCen'] == 1) & (grand_gals_SDSS['Z']<=0.06) & (grand_gals_SDSS['mag'] <= 17.77)].copy()  
grand_group_SDSS 				= grand_group_SDSS.merge(GAMA_group_file,on='GroupID',how='inner')
grand_group_SDSS_nfof			= grand_group_SDSS.loc[(grand_group_SDSS['Nfof'] > 5)].copy()

grand_isolated_lightcone	= grand_isolated.merge(lightcone_alfalfa, on='id_galaxy_sky', how='inner') 						# matching isolated centrals to lightcone



############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
###                                                   HI Matching  --- NFOF >= 1
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************

############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
###                                                   Reading Simulation Box and Matching up
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************

matched_id_group        = np.unique(np.array(lightcone_alfalfa['id_group_sky'])) 
matched_id_group        = matched_id_group[matched_id_group > 0]


Mvir_spherical_group_5 = []
MHI_spherical_group_5  = []

Mvir_spherical_group_2 = []
MHI_spherical_group_2  = []


Mvir_spherical_group_1 = []
MHI_spherical_group_1  = []


for group_sky, i in zip(matched_id_group, range(len(matched_id_group))):

    trial           = np.where(lightcone_alfalfa['id_group_sky'] == int(group_sky))[0]
    
    if len(trial) >= 5 :
        print(i)

        MHI_spherical_group_5 = np.append(MHI_spherical_group_5, np.sum(np.array(lightcone_alfalfa_all[(lightcone_alfalfa_all['id_group_sky'] == group_sky)]['matom_all']))/1.35/h)
        
        Mvir_spherical_group_5 = np.append(Mvir_spherical_group_5, max(np.array(lightcone_alfalfa_all[(lightcone_alfalfa_all['id_group_sky'] == group_sky)]['mvir_hosthalo']))/h)

    if len(trial) >= 2 :
        print(i)

        MHI_spherical_group_2 = np.append(MHI_spherical_group_2, np.sum(np.array(lightcone_alfalfa_all[(lightcone_alfalfa_all['id_group_sky'] == group_sky)]['matom_all']))/1.35/h)
        
        Mvir_spherical_group_2 = np.append(Mvir_spherical_group_2, max(np.array(lightcone_alfalfa_all[(lightcone_alfalfa_all['id_group_sky'] == group_sky)]['mvir_hosthalo']))/h)

    if len(trial) > 0:
        print(i)

        MHI_spherical_group_1 = np.append(MHI_spherical_group_1, np.sum(np.array(lightcone_alfalfa_all[(lightcone_alfalfa_all['id_group_sky'] == group_sky)]['matom_all']))/1.35/h)
        
        Mvir_spherical_group_1 = np.append(Mvir_spherical_group_1, max(np.array(lightcone_alfalfa_all[(lightcone_alfalfa_all['id_group_sky'] == group_sky)]['mvir_hosthalo']))/h)



############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
###                                                   Plotting : HI-Halo Scaling Relation [N_g >= 1]
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************



plt = Common_module.load_matplotlib(12,8)
colour_plot = Common_module.colour_scheme(colourmap = plt.cm.Set2)

legendHandles = list()

# Halo_mass_plotting, HI_mass_plotting, HI_mass_low, HI_mass_high = Common_module.halo_value_list(final_massAfunc[(final_massAfunc != 0)], final_HI_mass[(final_massAfunc != 0)], mean=True)					# Mean HI-Halo values for plotting (isolated + group) 

# Error_SAM_1 		= plt.errorbar(Halo_mass_plotting[0:len(HI_mass_plotting)-1], HI_mass_plotting[0:len(HI_mass_plotting)-1], yerr=None,label='$N_{g} \\geq 1$', marker = "o", mfc = 'peachpuff', mec = 'k', c = 'k', elinewidth=2,ls = '-', markersize=15, linewidth=1)  # Plotting Mean HI-Halo values for plotting (isolated + group)


####*********************************************************

Halo_mass_plotting, HI_mass_plotting, HI_mass_low, HI_mass_high = Common_module.halo_value_list(Mvir_spherical_group_1[(Mvir_spherical_group_1 != 0)], MHI_spherical_group_1[(Mvir_spherical_group_1 != 0)], mean=True)                  # Mean HI-Halo values for plotting (isolated + group) 

Error_SAM_1_cor         = plt.errorbar(Halo_mass_plotting[0:len(HI_mass_plotting)-1], HI_mass_plotting[0:len(HI_mass_plotting)-1], yerr=None,label='$N_{g} \\geq 1$', marker = "o", mfc = 'r', mec = 'k', c = 'k', elinewidth=2,ls = '-', markersize=15, linewidth=1)  # Plotting Mean HI-Halo values for plotting (isolated + group)


####*********************************************************


Halo_mass_plotting, HI_mass_plotting, HI_mass_low, HI_mass_high = Common_module.halo_value_list(Mvir_spherical_group_2[(Mvir_spherical_group_2 != 0)], MHI_spherical_group_2[(Mvir_spherical_group_2 != 0)], mean=True)				# Mean HI-Halo values for plotting (isolated + group) 

Error_SAM_2 		= plt.errorbar(Halo_mass_plotting[0:len(HI_mass_plotting)-1], HI_mass_plotting[0:len(HI_mass_plotting)-1], yerr=None,label='$N_{g} \\geq 2$', marker = "^", mfc = 'rebeccapurple', mec = 'k', c = 'k', elinewidth=2,ls = '--', markersize=15, linewidth =1)  # Plotting Mean HI-Halo values for plotting (isolated + group)


####*********************************************************


Halo_mass_plotting, HI_mass_plotting, HI_mass_low, HI_mass_high = Common_module.halo_value_list(Mvir_spherical_group_5[(Mvir_spherical_group_5 != 0)], MHI_spherical_group_5[(Mvir_spherical_group_5 != 0)], mean=True)					# Mean HI-Halo values for plotting (isolated + group) 

Error_SAM_5 		= plt.errorbar(Halo_mass_plotting[0:len(HI_mass_plotting)-1], HI_mass_plotting[0:len(HI_mass_plotting)-1], yerr=None,label='$N_{g} \\geq 5$', marker = "d", mfc = 'goldenrod', mec = 'k', c = 'k', elinewidth=2,ls = ':', markersize=15, linewidth=1)  # Plotting Mean HI-Halo values for plotting (isolated + group)



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



# Error_det_isolated 		= plt.errorbar(Guo_halo_isolated, Guo_HI_isolated, yerr=[Guo_HI_isolated_lower,Guo_HI_isolated_upper],label='$N_{g} >= 1$ (Guo+ 2020)', marker = "X", mfc = 'black', mec = 'black', c = 'black', elinewidth=2,ls = ' ', markersize=15)


# legendHandles.append(Error_SAM_1)
legendHandles.append(Error_SAM_1_cor)
legendHandles.append(Error_SAM_2)
legendHandles.append(Error_SAM_5)

plt.xlabel('$log_{10}(M_{vir}[M_{\odot}])$')
plt.ylabel('$log_{10}(M_{HI}[M_{\odot}])$')
plt.ylim(8,11)

# plt.title('Subhalo')

plt.legend(handles=legendHandles)
# plt.savefig("Plots/GAMA_group_all_GUO_ng_2_z_cut_5_rvir_2.png")
plt.savefig("Intrinsic_values_group_without_mag.png")
# plt.show()
plt.close()
# plt.close()



