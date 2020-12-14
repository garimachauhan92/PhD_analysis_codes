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




############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
###                                                             Reading Data
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************

# path_GAMA     = '/mnt/su3ctm/gchauhan/HI_stacking_paper/GAMA_files_Matias/'

path_GAMA   = '/mnt/su3ctm/gchauhan/HI_stacking_paper/Matias_GAMA_new/'

# path_GAMA     = '/mnt/su3ctm/gchauhan/HI_stacking_paper/Old_GAMA_files/GAMA_files_Matias/'

# path_GAMA     = '/mnt/sshfs/pleiades_gchauhan/HI_stacking_paper/Matias_GAMA_new/'

path        = '/mnt/su3ctm/clagos/SHARK_Out/'

# path      = '/mnt/sshfs/pleiades_gchauhan/SHArk_Out/HI_haloes/'

path_plot   = '/mnt/su3ctm/gchauhan/HI_stacking_paper/Plots/Paper_plots/Distribution_plots/'

shark_runs      = ['Shark-TreeFixed-ReincPSO-kappa0p002','Shark-Lagos18-default-br06','Shark-Lagos18-Kappa-10','Shark-Lagos18-Kappa-0','Shark-Lagos18-Kappa-1','Shark-Lagos18-default-br06-stripping-off']

shark_labels    = ['SHARK-ref','Kappa = 0.02','Kappa = 0','Kappa = 1','Lagos18 (stripping off)']

with open("/home/ghauhan/Parameter_files/redshift_list_medi.txt", 'r') as csv_file:  
# with open("/home/garima/Desktop/redshift_list_medi.txt", 'r') as csv_file:  
    trial = list(csv.reader(csv_file, delimiter=',')) 
    trial = np.array(trial[1:], dtype = np.float) 



simulation      = ['medi-SURFS', 'micro-SURFS']
snapshot_avail  = [x for x in range(100,200,1)]
z_values        = ["%0.2f" %x for x in trial[:,1]]

subvolumes      = 64

############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
###                                                             Reading SHARK-ref File
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************

medi_Kappa_original     = {}

for snapshot in snapshot_avail:

    medi_Kappa_original[snapshot]       = SharkDataReading(path,simulation[0],shark_runs[0],snapshot,subvolumes)


medi_HI, medi_HI_central, medi_HI_satellite,medi_HI_orphan,medi_vir, medi_halo_id = medi_Kappa_original[199].mergeValues_satellite('id_halo','matom_bulge','matom_disk')

medi_stellar, medi_stellar_central, medi_stellar_satellite,medi_stellar_orphan,a, b = medi_Kappa_original[199].mergeValues_satellite('id_halo','mstars_bulge','mstars_disk')


Nsub, a, sub_haloid = medi_Kappa_original[199].mergeValuesNumberSubstructures('id_halo_tree', 'id_subhalo')


############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
###                                                             Reading GAMA Files
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************

GAMA_gals_file      = pd.read_csv(path_GAMA + "Garima_T19-RR14_gals.csv")

GAMA_group_file     = pd.read_csv(path_GAMA + "Garima_T19-RR14_group.csv")

GAMA_censat_file    = pd.read_csv(path_GAMA +"Garima_T19-RR14_abmatch_magerr_censat85.csv" )


# GAMA_gals_file        = pd.read_csv(path_GAMA + "GAMA_T19-RR14_gals.csv")

# GAMA_group_file   = pd.read_csv(path_GAMA + "GAMA_T19-RR14_group.csv")

# GAMA_censat_file  = pd.read_csv(path_GAMA +"GAMA_T19-RR14_abmatch_magerr_censat85.csv" )


############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
###                                                             Merging GAMA Files
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************

grand_gals      = GAMA_gals_file.merge(GAMA_censat_file,left_on='CATAID',right_on='id_galaxy_sky',how='inner')  # Merges Galaxy file with censat (merging CATAID and id_galaxy_sky)

grand_isolated  = grand_gals.loc[(grand_gals['RankIterCen']==-999) & (grand_gals['Z']<=0.06)].copy() # only isolated centrals

grand_group     = grand_gals.loc[(grand_gals['RankIterCen']==1) & (grand_gals['Z']<=0.06)].copy()   # separates the central of groups

grand_group     = grand_group.merge(GAMA_group_file,on='GroupID',how='inner')  # merges group file with galaxy file


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
id_group_sky_all    = [int(k) for k in id_group_sky_all]
new_group_id        = np.zeros(len(snapshot_all))

for i in range(len(snapshot_all)):
    new_group_id[i]        = snapshot_all[i]*10**10 + subvolume_all[i]*10**8 + id_halo_sam_all[i] 

dictionary_df = {'id_halo_sam':id_halo_sam_all, 'id_galaxy_sam':id_galaxy_sam_all, 'id_galaxy_sky':id_galaxy_sky_all, 'type':type_g_all, 'mvir_hosthalo':mvir_hosthalo_all, 'mvir_subhalo':mvir_subhalo_all, 'matom_all':matom_sam_all, 'ra':ra_all, 'dec':dec_all, 'vpec_r':vpec_r_all,'vpec_x':vpec_x_all,'vpec_y':vpec_y_all,'vpec_z':vpec_z_all, 'zcos':zcos_all, 'zobs':zcos_all,'mstars_all':mstars_sam_all, 'id_group_sky':new_group_id}#'id_group_sky':id_group_sky_all} #, 'vvir_hosthalo':vvir_hosthalo_all, 'vvir_subhalo':vvir_subhalo_all


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
id_group_sky_all    = [int(k) for k in id_group_sky_all]
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
###                                                   Listing out halo ids
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************


best_match_dictionary = {}

matching_ID      = np.unique(np.array(grand_group_nfof['GroupID'])) 

for i,j in zip(matching_ID, range(len(matching_ID))):

    # print(i)

    trial               = np.where(grand_gals_lightcone['GroupID'] == i)[0]

    if len(grand_gals_lightcone[(grand_gals_lightcone['GroupID'] == i) & (grand_gals_lightcone['RankIterCen'] == 1)]) != 0:

        unique_elements, counts_elements = np.unique(np.array(grand_gals_lightcone[(grand_gals_lightcone['GroupID'] == i)]['id_group_sky_y']), return_counts=True)

        best_match_dictionary[i] = {'halo_id':np.array(grand_gals_lightcone[(grand_gals_lightcone['GroupID'] == i)]['id_group_sky_y']), 'nFoF':np.array(grand_group_lightcone[(grand_group_lightcone['GroupID'] == i)]['Nfof']), 'halo_id_unique':unique_elements, 'counts':counts_elements, 'MassAfunc':np.array(grand_group_lightcone[(grand_group_lightcone['GroupID'] == i)]['MassAfunc']), 'central_halo_id':np.array(grand_gals_lightcone[(grand_gals_lightcone['GroupID'] == i) & (grand_gals_lightcone['RankIterCen'] == 1)]['id_group_sky_y'])}

    else:

        continue


print('Finished matching GAMA')





###########################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
###                                                   Purity fractions
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************


Purtiy_fraction = {}

# matching_ID      = np.unique(np.array(grand_group_nfof['GroupID'])) 

for i in best_match_dictionary.keys():

    halo_ids    = best_match_dictionary[i]['halo_id_unique']
    counts      = best_match_dictionary[i]['counts']
    nfof_halo   = best_match_dictionary[i]['nFoF'][0]

    add_halo = np.zeros(len(halo_ids))
    add_purity = np.zeros(len(halo_ids))

    for j,k,l in zip(halo_ids,counts, range(len(counts))):

        total_haloes_lightcone = len(lightcone_alfalfa[(lightcone_alfalfa['id_group_sky'] == j)])
        add_halo[l] = j

        add_purity[l]  = k**2/total_haloes_lightcone/nfof_halo

    Purtiy_fraction[i] = {'halo_ids': add_halo[add_purity != np.inf], 'purity_fraction':add_purity[add_purity != np.inf], 'nFoF_group':nfof_halo}



############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
###                                                   Checking Max Purity
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************
yo = np.zeros(len(matching_ID))
yo_halo_id = np.zeros(len(matching_ID))
nfof_compare = np.zeros(len(matching_ID))
best_match_halo = {}


for i,j in zip(Purtiy_fraction.keys(),range(len(Purtiy_fraction.keys()))):

    yo[j] = max(Purtiy_fraction[i]['purity_fraction'])
    prf = Purtiy_fraction[i]['purity_fraction']
    yoyo = np.where(Purtiy_fraction[i]['purity_fraction'] == max(Purtiy_fraction[i]['purity_fraction']))[0][0]
    yo_halo_id[j] = Purtiy_fraction[i]['halo_ids'][yoyo]

    nfof_compare[j] = Purtiy_fraction[i]['nFoF_group']

    ra_group        = np.array(lightcone_alfalfa[(lightcone_alfalfa['id_group_sky'] == yo_halo_id[j])]['ra'])
    zobs_group      = np.array(lightcone_alfalfa[lightcone_alfalfa['id_group_sky'] == yo_halo_id[j]]['zobs'])

    dec_group       = np.array(lightcone_alfalfa[lightcone_alfalfa['id_group_sky'] == yo_halo_id[j]]['dec'])

    best_match_halo[i] = {'halo_id':yo_halo_id[j], 'purity_fraction':yo[j], 'nfof_group':nfof_compare[j], 'ra':ra_group, 'zobs':zobs_group, 'dec':dec_group}








############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
###                                                   Best Match Intrinsic
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************


HI_intrinsic            = []
Mvir_intrinsic          = []
Mvir_GAMA               = []

for i in Purtiy_fraction.keys():

    jj = np.where(grand_group_nfof['GroupID'] == i)[0]
    group_sky = int(best_match_halo[i]['halo_id'])

    HI_intrinsic = np.append(HI_intrinsic, np.sum(np.array(lightcone_alfalfa_all[(lightcone_alfalfa_all['id_group_sky'] == group_sky)]['matom_all']))/1.35/h)
    # Mvir_intrinsic = np.append(Mvir_intrinsic, np.array(grand_group_lightcone_nfof[grand_group_lightcone_nfof['GroupID'] == i]['mvir_hosthalo'])/h)
    Mvir_intrinsic = np.append(Mvir_intrinsic, max(np.array(lightcone_alfalfa_all[(lightcone_alfalfa_all['id_group_sky'] == group_sky)]['mvir_hosthalo']))/h)

    Mvir_GAMA = np.append(Mvir_GAMA, np.array(grand_group_nfof[grand_group_nfof['GroupID'] == i]['MassAfunc']))






############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
###                                                   Contamination wrt ALFALFA
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************


####*********************************************************
### HI in GAMA groups and isolated
####*********************************************************



mvir_group,HI_halo_group_2,HI_central_group_2, cont_HI, cont_star, cont_haloid, type_gal_cont, HI_mass_dictionary, star_mass_dictionary = Common_module.function_stacking(group_cat=True,lightcone_GAMA=grand_group_nfof, lightcone_all=lightcone_alfalfa_all, lightcone_all_GAMA=grand_group_lightcone_nfof, abundance_matching='luminosity',z_cut_halo=700, z_cut_central=700, rvir_cut_central=0.1)    



# def function_stacking_vvir(group_cat,lightcone_GAMA, lightcone_all, lightcone_all_GAMA, vvir_cut_halo=1, rvir_cut_halo=1, vvir_cut_central=1, rvir_cut_central=0.1, abundance_matching=True):

# mvir_group_vvir,HI_halo_group_2_vvir,HI_central_group_2_vvir = Common_module.function_stacking_vvir(group_cat=True,lightcone_GAMA=grand_group_nfof, lightcone_all=lightcone_alfalfa_all, lightcone_all_GAMA=grand_group_lightcone_nfof, abundance_matching='luminosity',vvir_cut_halo=1.5, vvir_cut_central=1.5, rvir_cut_central=0.1, rvir_cut_halo=1)    



############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
###                                                   Reading Simulation Box and Matching up
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************

# HI_cont 			= {}
# HI_other 			= {}
# star_cont 			= {}
halo_mass 			= {}
# fraction  			= {}


# fraction_mass 		= {}

for j in cont_haloid.keys():

	if len(cont_haloid[j]) > 0:

		a = np.where(cont_haloid[j] != best_match_halo[j]['halo_id'])[0]
		b = np.where(cont_haloid[j] == best_match_halo[j]['halo_id'])[0]
		# HI_cont[j] 		= np.log10(np.sum(cont_HI[j][a]))
		# HI_other[j] 	= np.log10(np.sum(cont_HI[j][b]))

		# fraction[j] 		= len(a)/len(cont_haloid[j])
		# fraction_mass[j] 	= float(np.sum(cont_HI[j][a])/HI_mass_dictionary[j])

		# HI_cont[j] 		= np.log10(np.median(cont_HI[j][a]))

		# star_cont[j]	= np.log10(np.sum(cont_star[j][a]))

		halo_mass[j]	= np.log10(max(np.array(lightcone_alfalfa_all[(lightcone_alfalfa_all['id_group_sky'] == best_match_halo[j]['halo_id'])]['mvir_hosthalo']))/h) 		

	else:
		continue


fraction_central 		= {}
median_halo_cont 		= {}
median_satellite_HI 	= {}
median_central_HI 		= {}

median_satellite_star 	= {}
median_central_star 	= {}

halo_value_host 		= {}

value_list 				= {}

for j in cont_haloid.keys():
	print(j)

	if len(cont_haloid[j]) > 0:

		a = np.where(cont_haloid[j] != best_match_halo[j]['halo_id'])[0]

		if len(type_gal_cont[j][a]) > 0:
		
			fraction_central[j] 		= len(np.where(type_gal_cont[j][a] == 0)[0])/len(type_gal_cont[j][a])
			b 							= np.where(type_gal_cont[j][a] == 0)[0]
			new_cont_halo 				= cont_haloid[j][a]
			new_cont_halo 				= new_cont_halo[b]
			new_cont_halo_sat 			= cont_haloid[j][a]
			c 							= np.where(type_gal_cont[j][a] != 0)[0]
			new_cont_halo_sat 			= new_cont_halo_sat[c]


			halo_value = []
			central_HI = []
			central_star = []

			satellite_HI = []
			satellite_star = []

			hosthalo_value = np.zeros(len(new_cont_halo))

			for l in range(len(b)):
				halo_value 		= np.append(halo_value, max(np.array(lightcone_alfalfa_all[(lightcone_alfalfa_all['id_group_sky'] == new_cont_halo[l])]['mvir_hosthalo'])/h))
				central_HI 		= np.append(central_HI, max(np.array(lightcone_alfalfa_all[(lightcone_alfalfa_all['id_group_sky'] == new_cont_halo[l])]['matom_all'])/h/1.35))
				central_star 	= np.append(central_star,max(np.array(lightcone_alfalfa_all[(lightcone_alfalfa_all['id_group_sky'] == new_cont_halo[l])]['mstars_all'])/h))
				hosthalo_value[l] 	= max(np.array(lightcone_alfalfa_all[(lightcone_alfalfa_all['id_group_sky'] == best_match_halo[j]['halo_id'])]['mvir_hosthalo'])/h)

			for l in range(len(new_cont_halo_sat)):
				satellite_HI = np.append(satellite_HI, max(np.array(lightcone_alfalfa_all[(lightcone_alfalfa_all['id_group_sky'] == new_cont_halo_sat[l])]['matom_all'])/h))
				satellite_star = np.append(satellite_star, max(np.array(lightcone_alfalfa_all[(lightcone_alfalfa_all['id_group_sky'] == new_cont_halo_sat[l])]['mstars_all'])/h))



			median_halo_cont[j]			= halo_value		#np.median(halo_value)
			median_central_HI[j]		= central_HI		#np.median(central_HI)
			median_central_star[j]		= central_star		#np.median(central_star)
			halo_value_host[j]			= hosthalo_value
			median_satellite_HI[j]		= satellite_HI		#np.median(satellite_HI)
			median_satellite_star[j]	= satellite_star	#np.median(satellite_star)
			value_list[j] 				= np.median(satellite_star)
	else:
		continue




plottin_halo 			= []
plotting_median_halo 	= []
plottin_HI_sat 			= []
plottin_HI_cen 			= []
plottin_star_sat 		= []
plottin_star_cen		= []
plottin_fraction		= []
plotting_star 			= []
value_list_final		= []
halo_value_list_y 		= []

for i in median_central_HI.keys():
	if len(cont_haloid[i]) > 0:

		plottin_halo 		= np.append(plottin_halo, halo_value_host[i])
		plottin_fraction 	= np.append(plottin_fraction, fraction_central[i])
		plottin_HI_sat 		= np.append(plottin_HI_sat, median_satellite_HI[i])
		plottin_star_sat 	= np.append(plottin_star_sat, median_satellite_star[i])

		plottin_HI_cen 		= np.append(plottin_HI_cen, median_central_HI[i])
		plottin_star_cen 	= np.append(plottin_star_cen, median_central_star[i])

		plotting_median_halo = np.append(plotting_median_halo, median_halo_cont[i])

		# plotting_star 	= np.append(plotting_star, star_cont[i])

		value_list_final	= np.append(value_list_final, value_list[i])
		halo_value_list_y 	= np.append(halo_value_list_y, halo_mass[i]) 



def nanpercentile_lower(arr,q=15.87): 
	arr = arr[~np.isnan(arr)]
	
	if (len(arr) != 0): return (np.nanpercentile(a=arr,q=q))
	else: return None
	

def nanpercentile_upper(arr,q=84.13): 
	arr = arr[~np.isnan(arr)]
	
	if (len(arr) != 0): return (np.nanpercentile(a=arr,q=q))
	else: return None




def halo_value_list(virial_mass,property_plot,mean):
	
	bin_for_disk = np.arange(11,16,0.2)

	halo_mass 		= np.zeros(len(bin_for_disk))
	prop_mass 		= np.zeros(len(bin_for_disk))

	prop_mass_low 		= np.zeros(len(bin_for_disk))
	prop_mass_high		= np.zeros(len(bin_for_disk))
	

	if mean == True:
		for i in range(1,len(bin_for_disk)):
			halo_mass[i-1] 		= np.log10(np.mean(virial_mass[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])]))
			prop_mass[i-1]			= np.log10(np.mean(property_plot[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])]))

			bootarr 			= np.log10(property_plot[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])])
			bootarr = bootarr[bootarr != float('-inf')]
			
			print(len(virial_mass[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])]))

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
			
			if len(bootarr) > 10 :
				if bootarr != []:
					with NumpyRNGContext(1):
						bootresult 			= bootstrap(bootarr,10,bootfunc=np.median)
						bootresult_lower	= bootstrap(bootarr,10,bootfunc=nanpercentile_lower)
						bootresult_upper	= bootstrap(bootarr,10,bootfunc=nanpercentile_upper)

					prop_mass_low[i-1]	= np.mean(bootresult_lower)

					prop_mass_high[i-1]	= np.mean(bootresult_upper)


	return halo_mass, prop_mass, prop_mass_low, prop_mass_high






def halo_value_list_massweighted(virial_mass,property_plot,property_mean):
    
    bin_for_disk = np.arange(11,16,0.2)

    halo_mass       = np.zeros(len(bin_for_disk))
    prop_mass       = np.zeros(len(bin_for_disk))

    prop_mass_low       = np.zeros(len(bin_for_disk))
    prop_mass_high      = np.zeros(len(bin_for_disk))
    

    for i in range(1,len(bin_for_disk)):
        if np.sum((property_mean[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])])) > 0:
            halo_mass[i-1]      = np.log10(np.mean(virial_mass[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])]))
            # prop_mass[i-1]      = np.log10(np.mean(np.sum((property_mean[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])])*(property_plot[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])]))/np.sum(property_plot[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])])))
            
            prop_mass[i-1]          = np.log10(np.average(property_mean[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])], weights=property_plot[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])]))      
                
       

    return halo_mass, prop_mass, prop_mass_low, prop_mass_high






##### Contaminant 

halo_value_contaminant = plotting_median_halo[~np.isnan(plotting_median_halo)]

hosthalo_value_contaminant = plottin_halo[~np.isnan(plotting_median_halo)]

stellar_mass_contaminant = plottin_star_cen[~np.isnan(plottin_star_cen)]

HI_mass_contaminant = plottin_HI_cen[~np.isnan(plottin_HI_cen)]

gf_contaminant      = HI_mass_contaminant/stellar_mass_contaminant

np.average(halo_value_contaminant[(hosthalo_value_contaminant > 10**12.5) & (hosthalo_value_contaminant < 10**13.2)], weights=HI_mass_contaminant[(hosthalo_value_contaminant > 10**12.5) & (hosthalo_value_contaminant < 10**13.2)])


np.average(stellar_mass_contaminant[(hosthalo_value_contaminant > 10**12.5) & (hosthalo_value_contaminant < 10**13.2)], weights=HI_mass_contaminant[(hosthalo_value_contaminant > 10**12.5) & (hosthalo_value_contaminant < 10**13.2)])

np.average(gf_contaminant[(hosthalo_value_contaminant > 10**12.5) & (hosthalo_value_contaminant < 10**13.2)], weights=HI_mass_contaminant[(hosthalo_value_contaminant > 10**12.5) & (hosthalo_value_contaminant < 10**13.2)])

np.mean(gf_contaminant[(stellar_mass_contaminant > 10**10)])#, weights=HI_mass_contaminant[(hosthalo_value_contaminant > 10**12.5) & (hosthalo_value_contaminant < 10**13.2)])


# new_Halo, mstar_cen, a, b = halo_value_list_massweighted(plottin_halo[~np.isnan(plotting_median_halo)], plottin_HI_cen[~np.isnan(plottin_HI_cen)],plottin_star_cen[~np.isnan(plottin_star_cen)] )


# Halo_mass_plotting, median_halo_plotting, HI_mass_low, HI_mass_high = halo_value_list(plottin_halo[~np.isnan(plotting_median_halo)], plotting_median_halo[~np.isnan(plotting_median_halo)], mean=True)


# # Halo_mass_plotting, fraction_plotting, HI_mass_low, HI_mass_high = halo_value_list(plottin_halo[~np.isnan(plottin_fraction)], plottin_fraction[~np.isnan(plottin_fraction)], mean=False)

# Halo_mass_plotting, HI_cen_plotting, HI_mass_low, HI_mass_high = halo_value_list(plottin_halo[~np.isnan(plottin_HI_cen)], plottin_HI_cen[~np.isnan(plottin_HI_cen)], mean=False)

# Halo_mass_plotting, HI_sat_plotting, HI_mass_low, HI_mass_high = halo_value_list(plottin_halo[~np.isnan(plottin_HI_sat)], plottin_HI_sat[~np.isnan(plottin_HI_sat)], mean=False)

# Halo_mass_plotting, star_cen_plotting, HI_mass_low, HI_mass_high = halo_value_list(plottin_halo[~np.isnan(plottin_star_cen)], plottin_star_cen[~np.isnan(plottin_star_cen)], mean=False)

# Halo_mass_plotting, star_sat_plotting, HI_mass_low, HI_mass_high = halo_value_list(plottin_halo[~np.isnan(plottin_star_sat)], plottin_star_sat[~np.isnan(plottin_star_sat)], mean=False)

# Halo_mass_plotting, median_halo_plotting, HI_mass_low, HI_mass_high = halo_value_list(10**halo_value_list_y[~np.isnan(value_list_final)], value_list_final[~np.isnan(value_list_final)], mean=False)


# Halo_mass_plotting, gas_frac_cen, HI_mass_low, HI_mass_high = halo_value_list(plottin_halo[~np.isnan(plottin_star_cen)], plottin_HI_cen[~np.isnan(plottin_star_cen)]/plottin_star_cen[~np.isnan(plottin_star_cen)], mean=False)



# ############################################################################################################################################
# ######--------------------------------------------------------------------------------------------------------------------------------------
# ###                                                  Distributions
# ######--------------------------------------------------------------------------------------------------------------------------------------
# ####*********************************************************


# # hist_HI=plottin_HI_cen[~np.isnan(plottin_HI_cen)]
# # hist_star = plottin_star_cen[~np.isnan(plottin_star_cen)]


# # plt.hist(np.log10(hist_HI[hist_HI > 0]), bins=30, label='HI-cont')
# # plt.hist(np.log10(hist_star[hist_star > 0]), bins=30, label='star-cont')


# scatter_halo        = plottin_halo[(plottin_halo > 10**12.5) & (plottin_halo < 10**13.5)]
# scatter_gasfrac     = plottin_HI_cen[(plottin_halo > 10**12.5) & (plottin_halo < 10**13.5)]/plottin_star_cen[(plottin_halo > 10**12.5) & (plottin_halo < 10**13.5)]


# plt.scatter(np.log10(scatter_halo[scatter_halo > 0]), np.log10(scatter_gasfrac[scatter_halo > 0]), s=1 )
# plt.plot(Halo_mass_plotting, gas_frac_cen, '--k')
# plt.xlabel('gasfrac')
# plt.ylabel('Mvir')
# plt.xlim(12,14)
# plt.savefig('cont_test.png')
# plt.close()


scatter_halo        = plottin_halo[(plottin_halo > 10**12.5) & (plottin_halo < 10**13.5)]
scatter_gasfrac     = plottin_HI_cen[(plottin_halo > 10**12.5) & (plottin_halo < 10**13.5)]/plottin_star_cen[(plottin_halo > 10**12.5) & (plottin_halo < 10**13.5)]


Halo_mass_plotting, gas_frac_cen, HI_mass_low, HI_mass_high = halo_value_list(plottin_halo[~np.isnan(plottin_star_cen)], plottin_HI_cen[~np.isnan(plottin_star_cen)]/plottin_star_cen[~np.isnan(plottin_star_cen)], mean=False)

plt.scatter(np.log10(scatter_halo[scatter_halo > 0]), np.log10(scatter_gasfrac[scatter_halo > 0]), s=1 )
plt.plot(Halo_mass_plotting[(Halo_mass_plotting > 12.4) & (Halo_mass_plotting < 13.6)], gas_frac_cen[(Halo_mass_plotting > 12.4) & (Halo_mass_plotting < 13.6)], '-k', label='median gasfrac')


Halo_mass_plotting, gas_frac_cen, HI_mass_low, HI_mass_high = halo_value_list(plottin_halo[~np.isnan(plottin_star_cen)], plottin_HI_cen[~np.isnan(plottin_star_cen)]/plottin_star_cen[~np.isnan(plottin_star_cen)], mean=True)

plt.plot(Halo_mass_plotting[(Halo_mass_plotting > 12.4) & (Halo_mass_plotting < 13.6)], gas_frac_cen[(Halo_mass_plotting > 12.4) & (Halo_mass_plotting < 13.6)], '--k', label='mean gasfrac')
plt.legend()
plt.ylabel('log(gasfrac)')
plt.xlabel('log(Mvir)')
plt.xlim(12,14)
# plt.ylim(-3,2)
plt.savefig('cont_test_gasfrac.png')
plt.close()



scatter_halo        = plottin_halo[(plottin_halo > 10**12.5) & (plottin_halo < 10**13.5)]
scatter_gasfrac     = plottin_HI_cen[(plottin_halo > 10**12.5) & (plottin_halo < 10**13.5)]#/plottin_star_cen[(plottin_halo > 10**12.5) & (plottin_halo < 10**13)]


Halo_mass_plotting, gas_frac_cen, HI_mass_low, HI_mass_high = halo_value_list(plottin_halo[~np.isnan(plottin_star_cen)], plottin_HI_cen[~np.isnan(plottin_star_cen)], mean=False)

plt.scatter(np.log10(scatter_halo[scatter_halo > 0]), np.log10(scatter_gasfrac[scatter_halo > 0]), s=1 )
plt.plot(Halo_mass_plotting[(Halo_mass_plotting > 12.4) & (Halo_mass_plotting < 13.6)], gas_frac_cen[(Halo_mass_plotting > 12.4) & (Halo_mass_plotting < 13.6)], '-k', label='median MHI')


Halo_mass_plotting, gas_frac_cen, HI_mass_low, HI_mass_high = halo_value_list(plottin_halo[~np.isnan(plottin_star_cen)], plottin_HI_cen[~np.isnan(plottin_star_cen)], mean=True)

plt.plot(Halo_mass_plotting[(Halo_mass_plotting > 12.4) & (Halo_mass_plotting < 13.6)], gas_frac_cen[(Halo_mass_plotting > 12.4) & (Halo_mass_plotting < 13.6)], '--k', label='mean MHI')
plt.legend()
plt.ylabel('log(MHI)')
plt.xlabel('log(Mvir)')
plt.xlim(12,14)
# plt.ylim(6,10)
plt.savefig('cont_test_MHI.png')
plt.close()




scatter_halo        = plottin_halo[(plottin_halo > 10**12.5) & (plottin_halo < 10**13.5)]
scatter_gasfrac     = plottin_star_cen[(plottin_halo > 10**12.5) & (plottin_halo < 10**13.5)]


Halo_mass_plotting, gas_frac_cen, HI_mass_low, HI_mass_high = halo_value_list(plottin_halo[~np.isnan(plottin_star_cen)],plottin_star_cen[~np.isnan(plottin_star_cen)], mean=False)

plt.scatter(np.log10(scatter_halo[scatter_halo > 0]), np.log10(scatter_gasfrac[scatter_halo > 0]), s=1 )
plt.plot(Halo_mass_plotting[(Halo_mass_plotting > 12.4) & (Halo_mass_plotting < 13.6)], gas_frac_cen[(Halo_mass_plotting > 12.4) & (Halo_mass_plotting < 13.6)], '-k', label='median Mstar')


Halo_mass_plotting, gas_frac_cen, HI_mass_low, HI_mass_high = halo_value_list(plottin_halo[~np.isnan(plottin_star_cen)], plottin_star_cen[~np.isnan(plottin_star_cen)], mean=True)

plt.plot(Halo_mass_plotting[(Halo_mass_plotting > 12.4) & (Halo_mass_plotting < 13.6)], gas_frac_cen[(Halo_mass_plotting > 12.4) & (Halo_mass_plotting < 13.6)], '--k', label='mean Mstar')
plt.legend()
plt.ylabel('log(Mstar)')
plt.xlabel('log(Mvir)')
plt.xlim(12,14)
# plt.ylim(6,10)
plt.savefig('cont_test_Mstar.png')
plt.close()




scatter_halo        = plottin_halo[(plottin_halo > 10**12.5) & (plottin_halo < 10**13.5)]
scatter_gasfrac     = plotting_median_halo[(plottin_halo > 10**12.5) & (plottin_halo < 10**13.5)]


Halo_mass_plotting, gas_frac_cen, HI_mass_low, HI_mass_high = halo_value_list(plottin_halo[~np.isnan(plottin_star_cen)],plottin_star_cen[~np.isnan(plottin_star_cen)], mean=False)

plt.scatter(np.log10(scatter_halo[scatter_halo > 0]), np.log10(scatter_gasfrac[scatter_halo > 0]), s=1 )
plt.plot(Halo_mass_plotting[(Halo_mass_plotting > 12.4) & (Halo_mass_plotting < 13.6)], gas_frac_cen[(Halo_mass_plotting > 12.4) & (Halo_mass_plotting < 13.6)], '-k', label='median Mstar')


Halo_mass_plotting, gas_frac_cen, HI_mass_low, HI_mass_high = halo_value_list(plottin_halo[~np.isnan(plottin_star_cen)], plottin_star_cen[~np.isnan(plottin_star_cen)], mean=True)

plt.plot(Halo_mass_plotting[(Halo_mass_plotting > 12.4) & (Halo_mass_plotting < 13.6)], gas_frac_cen[(Halo_mass_plotting > 12.4) & (Halo_mass_plotting < 13.6)], '--k', label='mean Mstar')
plt.legend()
plt.ylabel('log(MHalo)')
plt.xlabel('log(Mvir)')
plt.xlim(12,14)
# plt.ylim(6,10)
plt.savefig('cont_test_MHalo.png')
plt.close()


# indices_sorted 				= medi_vir.argsort()
# sort_medi_vir  				= medi_vir[indices_sorted]
# sort_medi_HI 				= medi_HI[indices_sorted]


# indices_sorted 				= plottin_star_cen[(plottin_halo > 10**13) & (plottin_halo <10**13.2)].argsort()
# sort_star_cen  				= plottin_star_cen[(plottin_halo > 10**13) & (plottin_halo <10**13.2)][indices_sorted]
# sort_HI_cen 				= plottin_HI_cen[(plottin_halo > 10**13) & (plottin_halo <10**13.2)][indices_sorted]


# # trial_cumsum_HI_micro		= (np.cumsum(sort_micro_HI))/max(np.cumsum(sort_micro_HI))*100
# trial_cumsum_HI_medi		= (np.cumsum(sort_HI_cen))


# plt.plot(np.log10(sort_star_cen), np.log10(trial_cumsum_HI_medi), c= 'r')
# plt.savefig('Cumulative_HI_plot_all.png')
# plt.close()