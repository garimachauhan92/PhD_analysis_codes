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
###                                                             Observational Data
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************

GAMA_mock_all = np.array([11.13723819, 11.31436473, 11.50447398, 11.70028291, 11.89776144,12.09912912, 12.29425977, 12.49379795, 12.69379971, 12.89625662,13.09881217, 13.28940974, 13.48448203, 13.69604695, 13.87729364,14.10833794, 14.31740645, 14.46455816, 14.65531803, 14.85977838])

GAMA_all = np.array([11.93301974, 11.90756192, 11.90517985, 12.01001997, 12.14102312,12.2763755 , 12.41596197, 12.59617337, 12.74498655, 12.87574177,12.9996884 , 13.06216291, 13.11885742, 13.02932415, 13.04235032,12.92790232, 12.79370238, 12.91465704, 12.86550605, 12.91032311])

GAMA_all_error_low = np.array([10.5919316 , 10.65319988, 10.74701982, 10.9004293 , 11.04723511,11.16812168, 11.39162266, 11.60002333, 11.86685956, 11.97612241,12.12704723, 12.24277804, 12.26197455, 12.0878462 , 12.03661114,12.01687928, 11.838281  , 11.9858512 , 11.94839637, 12.04957253])

GAMA_all_error_high = np.array([12.85228059, 12.79899834, 12.8199986 , 12.89170886, 12.89739517,12.9704671 , 13.05218794, 13.16828271, 13.30287462, 13.37586121,13.44527718, 13.51056002, 13.56483835, 13.61390678, 13.63589784,13.60490067, 13.46702815, 13.61743053, 13.39241347, 13.43801924])

GAMA_mock_nfof = np.array([11.27317879, 11.57534419,         None, 11.92882668,12.08087199, 12.30446384, 12.51206089, 12.70759161, 12.91623512,13.10414682, 13.30005486, 13.48932392, 13.68842024, 13.88390492,14.09258959, 14.29879005, 14.48625044, 14.65531803, 14.82782021])

GAMA_nfof = np.array([12.8560093 , 12.42401695,         None, 12.95605265,12.6844766 , 12.82830138, 12.85371712, 13.07361222, 13.11033724,13.26311381, 13.37356857, 13.4415795 , 13.58830158, 13.72294376,13.74658415, 13.87735023, 13.6277187 , 13.5209978 , 13.63804144])

GAMA_nfof_error_low = np.array([0.        ,  0.        ,  0.        ,  0.        ,12.34440379, 12.52845834, 12.42970043, 12.78285287, 12.7886987 ,12.93455523, 13.02906963, 13.13158194, 13.18663162, 13.24927595,13.35849567, 13.07144446, 13.02771042, 13.22631099, 13.10823837])

GAMA_nfof_error_high = np.array([ 0.        ,  0.        ,  0.        ,  0.        ,13.13252691, 13.28037244, 13.21944668, 13.43883244, 13.41845543,13.59429299, 13.67464381, 13.76910763, 13.90048615, 14.05433999,14.12069878, 14.27256719, 14.57988424, 14.15417757, 14.34232002])


SDSS_mock_all = np.array([11.14899732, 11.34271645, 11.52234513, 11.70084952, 11.89758148,12.09799611, 12.29480096, 12.49356988, 12.69345429, 12.89444754,13.09826829, 13.28557427, 13.48253937, 13.6956288 , 13.87690685,14.10458854, 14.31740645, 14.48625044, 14.65531803, 14.85977838])

SDSS_all = np.array([11.16680164, 11.43793629, 11.75682367, 11.99572908, 12.25240278,12.41776298, 12.4935869 , 12.55943008, 12.65631063, 12.83780874,12.96343687, 13.16740713, 13.1512884 , 12.95963581, 12.48564557,12.37063684, 12.15347791, 12.34123478, 12.07001388, 12.21796815])

SDSS_all_error_low = np.array([11.11227766, 11.29388507, 11.55552249, 11.77284655, 11.95799399,12.07609672, 12.11791989, 12.1674856 , 12.28849315, 12.43746906,12.45143976, 12.20049441, 11.93676283, 11.91023472, 11.76960131,11.74151135, 11.5778746 , 11.54038243, 11.62348744, 11.76537948])

SDSS_all_error_high = np.array([11.25228475, 11.67494843, 11.93333145, 12.26509961, 12.56633286,12.80741073, 12.93725369, 13.04685231, 13.15113593, 13.26472517,13.37442444, 13.63648081, 13.76025564, 13.85282135, 13.91153792,13.74192238, 13.45491683, 13.30177053, 12.90585649, 12.98989793])

SDSS_mock_nfof = np.array([11.28403203, 11.50669318, 11.72022852, 11.89423901,12.07992682, 12.30446384, 12.51013263, 12.70759161, 12.91350239,13.10400697, 13.30005486, 13.49125485, 13.68809529, 13.87690685,14.08213235, 14.31740645, 14.48625044, 14.64736657, 14.85977838])

SDSS_nfof = np.array([11.24556213, 11.64920877, 12.31424512, 12.37042107,12.53365619, 12.61722738, 12.71625175, 12.79987167, 12.9951709 ,13.12299422, 13.38818978, 13.53584264, 13.68804361, 13.7629678 ,13.7874849 , 14.0553039 , 13.56036237, 12.83444338, 12.85992484])


SDSS_nfof_error_low = np.array([0.        ,  0.        ,  0.        , 12.11268116,12.19709882, 12.28096285, 12.25819728, 12.33492509, 12.5816634 ,12.74019524, 12.92941578, 13.04287483, 13.12145705, 12.85377053,12.8058032 , 12.7989519 , 11.82744073, 12.15206886, 12.28022091])

SDSS_nfof_error_high = np.array([0.        ,  0.        ,  0.        , 12.65354734,12.69333293, 13.1075062 , 13.17615471, 13.52664709, 13.60431385,13.62135704, 13.79946393, 13.90104961, 14.02312992, 14.26173584,14.42239979, 14.71156248, 14.62831594, 14.31555359, 14.43757176])


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
###                                                             Observational Data
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************

GAMA_mock_all = np.array([11.13723819, 11.31436473, 11.50447398, 11.70028291, 11.89776144,12.09912912, 12.29425977, 12.49379795, 12.69379971, 12.89625662,13.09881217, 13.28940974, 13.48448203, 13.69604695, 13.87729364,14.10833794, 14.31740645, 14.46455816, 14.65531803, 14.85977838])

GAMA_all = np.array([11.93301974, 11.90756192, 11.90517985, 12.01001997, 12.14102312,12.2763755 , 12.41596197, 12.59617337, 12.74498655, 12.87574177,12.9996884 , 13.06216291, 13.11885742, 13.02932415, 13.04235032,12.92790232, 12.79370238, 12.91465704, 12.86550605, 12.91032311])

GAMA_all_error_low = np.array([10.5919316 , 10.65319988, 10.74701982, 10.9004293 , 11.04723511,11.16812168, 11.39162266, 11.60002333, 11.86685956, 11.97612241,12.12704723, 12.24277804, 12.26197455, 12.0878462 , 12.03661114,12.01687928, 11.838281  , 11.9858512 , 11.94839637, 12.04957253])

GAMA_all_error_high = np.array([12.85228059, 12.79899834, 12.8199986 , 12.89170886, 12.89739517,12.9704671 , 13.05218794, 13.16828271, 13.30287462, 13.37586121,13.44527718, 13.51056002, 13.56483835, 13.61390678, 13.63589784,13.60490067, 13.46702815, 13.61743053, 13.39241347, 13.43801924])

GAMA_mock_nfof = np.array([11.27317879, 11.57534419,         None, 11.92882668,12.08087199, 12.30446384, 12.51206089, 12.70759161, 12.91623512,13.10414682, 13.30005486, 13.48932392, 13.68842024, 13.88390492,14.09258959, 14.29879005, 14.48625044, 14.65531803, 14.82782021])

GAMA_nfof = np.array([12.8560093 , 12.42401695,         None, 12.95605265,12.6844766 , 12.82830138, 12.85371712, 13.07361222, 13.11033724,13.26311381, 13.37356857, 13.4415795 , 13.58830158, 13.72294376,13.74658415, 13.87735023, 13.6277187 , 13.5209978 , 13.63804144])

GAMA_nfof_error_low = np.array([0.        ,  0.        ,  0.        ,  0.        ,12.34440379, 12.52845834, 12.42970043, 12.78285287, 12.7886987 ,12.93455523, 13.02906963, 13.13158194, 13.18663162, 13.24927595,13.35849567, 13.07144446, 13.02771042, 13.22631099, 13.10823837])

GAMA_nfof_error_high = np.array([ 0.        ,  0.        ,  0.        ,  0.        ,13.13252691, 13.28037244, 13.21944668, 13.43883244, 13.41845543,13.59429299, 13.67464381, 13.76910763, 13.90048615, 14.05433999,14.12069878, 14.27256719, 14.57988424, 14.15417757, 14.34232002])


SDSS_mock_all = np.array([11.14899732, 11.34271645, 11.52234513, 11.70084952, 11.89758148,12.09799611, 12.29480096, 12.49356988, 12.69345429, 12.89444754,13.09826829, 13.28557427, 13.48253937, 13.6956288 , 13.87690685,14.10458854, 14.31740645, 14.48625044, 14.65531803, 14.85977838])

SDSS_all = np.array([11.16680164, 11.43793629, 11.75682367, 11.99572908, 12.25240278,12.41776298, 12.4935869 , 12.55943008, 12.65631063, 12.83780874,12.96343687, 13.16740713, 13.1512884 , 12.95963581, 12.48564557,12.37063684, 12.15347791, 12.34123478, 12.07001388, 12.21796815])

SDSS_all_error_low = np.array([11.11227766, 11.29388507, 11.55552249, 11.77284655, 11.95799399,12.07609672, 12.11791989, 12.1674856 , 12.28849315, 12.43746906,12.45143976, 12.20049441, 11.93676283, 11.91023472, 11.76960131,11.74151135, 11.5778746 , 11.54038243, 11.62348744, 11.76537948])

SDSS_all_error_high = np.array([11.25228475, 11.67494843, 11.93333145, 12.26509961, 12.56633286,12.80741073, 12.93725369, 13.04685231, 13.15113593, 13.26472517,13.37442444, 13.63648081, 13.76025564, 13.85282135, 13.91153792,13.74192238, 13.45491683, 13.30177053, 12.90585649, 12.98989793])

SDSS_mock_nfof = np.array([11.28403203, 11.50669318, 11.72022852, 11.89423901,12.07992682, 12.30446384, 12.51013263, 12.70759161, 12.91350239,13.10400697, 13.30005486, 13.49125485, 13.68809529, 13.87690685,14.08213235, 14.31740645, 14.48625044, 14.64736657, 14.85977838])

SDSS_nfof = np.array([11.24556213, 11.64920877, 12.31424512, 12.37042107,12.53365619, 12.61722738, 12.71625175, 12.79987167, 12.9951709 ,13.12299422, 13.38818978, 13.53584264, 13.68804361, 13.7629678 ,13.7874849 , 14.0553039 , 13.56036237, 12.83444338, 12.85992484])


SDSS_nfof_error_low = np.array([0.        ,  0.        ,  0.        , 12.11268116,12.19709882, 12.28096285, 12.25819728, 12.33492509, 12.5816634 ,12.74019524, 12.92941578, 13.04287483, 13.12145705, 12.85377053,12.8058032 , 12.7989519 , 11.82744073, 12.15206886, 12.28022091])

SDSS_nfof_error_high = np.array([0.        ,  0.        ,  0.        , 12.65354734,12.69333293, 13.1075062 , 13.17615471, 13.52664709, 13.60431385,13.62135704, 13.79946393, 13.90104961, 14.02312992, 14.26173584,14.42239979, 14.71156248, 14.62831594, 14.31555359, 14.43757176])


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



mvir_group,HI_halo_group_2,HI_central_group_2 = Common_module.function_stacking(group_cat=True,lightcone_GAMA=grand_group_nfof, lightcone_all=lightcone_alfalfa_all, lightcone_all_GAMA=grand_group_lightcone_nfof, abundance_matching='luminosity',z_cut_halo=700, z_cut_central=700, rvir_cut_central=0.1)    



# def function_stacking_vvir(group_cat,lightcone_GAMA, lightcone_all, lightcone_all_GAMA, vvir_cut_halo=1, rvir_cut_halo=1, vvir_cut_central=1, rvir_cut_central=0.1, abundance_matching=True):

mvir_group_vvir,HI_halo_group_2_vvir,HI_central_group_2_vvir = Common_module.function_stacking_vvir(group_cat=True,lightcone_GAMA=grand_group_nfof, lightcone_all=lightcone_alfalfa_all, lightcone_all_GAMA=grand_group_lightcone_nfof, abundance_matching='luminosity',vvir_cut_halo=1.5, vvir_cut_central=1.5, rvir_cut_central=0.1, rvir_cut_halo=1)    



############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
###                                                   Reading Simulation Box and Matching up
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************

# def prepare_data(path,simulation,run,snapshot,subvolumes):

#     fields_read     = {'galaxies':('id_halo', 'id_galaxy', 'type', 'mvir_hosthalo','mvir_subhalo', 'matom_bulge', 'matom_disk', 'position_x', 'position_y', 'position_z')}

#     data_reading = SharkDataReading(path,simulation,run,snapshot,subvolumes)

#     data_plotting = data_reading.readIndividualFiles_GAMA(fields=fields_read, snapshot_group=snapshot, subvol_group=subvolumes)

#     return data_plotting



# matched_snapshot        = np.array(grand_group_nfof['snapshot']) 
# matched_subvolume       = np.array(grand_group_nfof['subvolume']) 
# matched_id_galaxy_sam   = np.array(grand_group_nfof['id_galaxy_sam']) 

# Mvir_spherical_group_5 = []
# MHI_spherical_group_5  = []

# for snapshot_run, subvolume_run, i in zip(matched_snapshot, matched_subvolume, range(len(matched_id_galaxy_sam))):

#     (h0,_, id_halo_all_sam, id_galaxy_all_sam, is_central_all_sam, mvir_hosthalo,mvir_subhalo, matom_bulge, matom_disk, position_x_sam, position_y_sam, position_z_sam) = prepare_data(path,simulation[0],shark_runs[0],int(snapshot_run),int(subvolume_run))
    

#     trial           = np.where(id_galaxy_all_sam == int(matched_id_galaxy_sam[i]))[0]
    
#     if len(trial) == 0 :

#         HI_unique_sam = 0
#         Mvir_add    = 0
#         M_HI_sam_mass[i]    = HI_unique_sam
#         Mvir_spherical_group_5 = np.append(Mvir_spherical_group_5, Mvir_add)

#     else:

#         print(i)

#         HI_unique_sam   = (matom_bulge + matom_disk)/h/1.35
#         is_central_sam  = is_central_all_sam[trial]

#         host_halo_id    = id_halo_all_sam[trial]
#         idx_HI           = np.where(id_halo_all_sam == int(host_halo_id))[0]

#         HI_halo         = np.sum(HI_unique_sam[idx_HI])
#         M_vir_unique    = mvir_hosthalo[trial]/h

#         Mvir_spherical_group_5  = np.append(Mvir_spherical_group_5, M_vir_unique)
#         MHI_spherical_group_5   = np.append(MHI_spherical_group_5, HI_halo) 




matched_id_group        = np.unique(np.array(lightcone_alfalfa['id_group_sky'])) 
matched_id_group        = matched_id_group[matched_id_group > 0]


Mvir_spherical_group_5 = []
MHI_spherical_group_5  = []

for group_sky, i in zip(matched_id_group, range(len(matched_id_group))):

    trial           = np.where(lightcone_alfalfa['id_group_sky'] == int(group_sky))[0]
    
    if len(trial) >= 5 :
        print(i)

        MHI_spherical_group_5 = np.append(MHI_spherical_group_5, np.sum(np.array(lightcone_alfalfa_all[(lightcone_alfalfa_all['id_group_sky'] == group_sky)]['matom_all']))/1.35/h)
        
        Mvir_spherical_group_5 = np.append(Mvir_spherical_group_5, max(np.array(lightcone_alfalfa_all[(lightcone_alfalfa_all['id_group_sky'] == group_sky)]['mvir_hosthalo']))/h)




############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
###                                                   Reading Simulation Box and Matching up
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************




def plotting_properties_halo(virial_mass,property_plot,mean,legend_handles,property_name_x='X-Label-math-mode',legend_name='name-of-run', property_name_y='Y-Label-math-mode', xlim_lower=6,xlim_upper=15,ylim_lower=-4,ylim_upper=11,colour_line='r',fill_between=False, first_legend = False, resolution=False, halo_bins=0, alpha=1):
    

    if halo_bins == 0:
        bin_for_disk = np.arange(7,15,0.3)

    elif halo_bins == 1:
        bin_for_disk = [10,11.1,11.2,11.3,11.5,11.7,13.1,15] - np.log10(h)

    else:
        bin_for_disk = [9,11.7,12.4,13.0,14.6] - np.log10(h)


    halo_mass       = np.zeros(len(bin_for_disk))
    prop_mass       = np.zeros(len(bin_for_disk))

    prop_mass_low       = np.zeros(len(bin_for_disk))
    prop_mass_high      = np.zeros(len(bin_for_disk))
    
    halo_mass_plot      = []
    prop_mass_plot      = []
    if mean == True:
        for i in range(1,len(bin_for_disk)):
            halo_mass[i-1]      = np.log10(np.mean(virial_mass[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])]))
            prop_mass[i-1]          = np.log10(np.mean(property_plot[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])]))

            bootarr             = np.log10(property_plot[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])])
            bootarr = bootarr[bootarr != float('-inf')]
            

            if len(bootarr) > 10:
                if bootarr != []:
                    with NumpyRNGContext(1):
                        bootresult          = bootstrap(bootarr,100,bootfunc=np.mean)
                        bootresult_error    = bootstrap(bootarr,100,bootfunc=stats.tstd)#/2
                        # bootresult_error  = bootstrap(bootarr,10,bootfunc=stats.sem)
                        # bootresult_error  = bootstrap(bootarr,10,bootfunc=statistics.median_grouped)
                        # bootresult_error_low  = bootstrap(bootarr,10,bootfunc=statistics.median_low)
                        # bootresult_error_high = bootstrap(bootarr,10,bootfunc=statistics.median_high)
                
                    prop_mass_low[i-1]  = prop_mass[i-1] - np.average(bootresult_error)
                    prop_mass_high[i-1] = np.average(bootresult_error) + prop_mass[i-1]
                    
                    # prop_mass_low[i-1]    = np.average(bootresult_error_low)
                    # prop_mass_high[i-1]   = np.average(bootresult_error_high)
            
    
    else:
        for i in range(1,len(bin_for_disk)):
            halo_mass[i-1]      = np.log10(np.median(virial_mass[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])]))
            prop_mass[i-1]          = np.log10(np.median(property_plot[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])]))
            
            bootarr             = np.log10(property_plot[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])])
            bootarr = bootarr[bootarr != float('-inf')]


            if len(bootarr) > 10 :
                if bootarr != []:
                    with NumpyRNGContext(1):
                        bootresult          = bootstrap(bootarr,100,bootfunc=np.median)
                        bootresult_lower    = bootstrap(bootarr,100,bootfunc=nanpercentile_lower)
                        bootresult_upper    = bootstrap(bootarr,100,bootfunc=nanpercentile_upper)

                    prop_mass_low[i-1]  = np.mean(bootresult_lower)
                    prop_mass_high[i-1] = np.mean(bootresult_upper)
                        
                    

    if resolution == True:
        plt.plot(halo_mass[0:len(halo_mass)-1],prop_mass[0:len(halo_mass)-1],color=colour_line, linestyle='-', alpha=alpha)
        # plt.plot(halo_mass[0:len(halo_mass)-1],prop_mass[0:len(halo_mass)-1],color=colour_line, label='$\\rm %s$' %legend_name)

        if first_legend == True:
            legend_handles.append(mlines.Line2D([],[],color='black',linestyle='-',label='Medi-SURFS'))
            legend_handles.append(mlines.Line2D([],[],color='black',linestyle='-.',label='Micro-SURFS'))
            legend_handles.append(mpatches.Patch(color=colour_line,label=legend_name))

        
        else:
            mpatches.Patch(color=colour_line, label=legend_name,alpha=alpha)
            #legend_handles.append(mpatches.Patch(color=colour_line,label=legend_name))

    else:
        plt.plot(halo_mass[0:len(halo_mass)-1],prop_mass[0:len(halo_mass)-1],color=colour_line, alpha=alpha)#, label='$\\rm %s$' %legend_name)
        legend_handles.append(mpatches.Patch(color=colour_line,label=legend_name))

    if fill_between == True:
        plt.fill_between(halo_mass[(prop_mass_high != 0)],prop_mass_low[ (prop_mass_high != 0)],prop_mass_high[(prop_mass_high != 0)],color=colour_line,alpha=0.1, label='$\\rm %s$' %legend_name)
    
    plt.xlabel('$\\rm %s$ '%property_name_x)
    plt.ylabel('$\\rm %s$ '%property_name_y)

    plt.xlim(xlim_lower,xlim_upper)
    plt.ylim(ylim_lower,ylim_upper)

    #plt.legend(frameon=False)
    # plt.savefig('%s.png'%figure_name)
    # plt.show()





legendHandles = []


Guo_HI_ng_2             = np.array([9.869, 9.914, 10.047, 10.025, 10.073, 10.117, 10.191]) - np.log10(h)
Guo_halo                = np.array([11.9, 12.1, 12.35, 12.6, 12.9, 13.25, 13.75]) - np.log10(h)
Guo_error               = np.array([0.156, 0.094, 0.059, 0.047, 0.042, 0.032, 0.052])


Halo_mass_plotting, HI_mass_plotting, HI_mass_low, HI_mass_high = Common_module.halo_value_list(Mvir_spherical_group_5, MHI_spherical_group_5, mean=True)

plotting_properties_halo(Mvir_spherical_group_5, MHI_spherical_group_5, legend_handles=legendHandles, legend_name='Intrinsic $(N_{g} \\geq 5)$', colour_line='maroon', alpha=0.5, mean=True)


Halo_mass_plotting, HI_mass_plotting, HI_mass_low, HI_mass_high = Common_module.halo_value_list(mvir_group/h, HI_halo_group_2/1.35/h, mean=True)
Error_SAM_2         = plt.errorbar(Halo_mass_plotting[4:len(Halo_mass_plotting)-10], (HI_mass_plotting[4:len(Halo_mass_plotting)-10]), yerr=None,marker = "X", mfc = 'k', mec = 'k', c = 'k', capsize=2,ls = '--', markersize=15, linewidth=1, barsabove=True, capthick=2, label='Mock HI-stacking $(\\Delta v = 700\  km/s)$')



Halo_mass_plotting, HI_mass_plotting, HI_mass_low, HI_mass_high = Common_module.halo_value_list(mvir_group_vvir/h, HI_halo_group_2_vvir/1.35/h, mean=True)
Error_SAM_2_vvir         = plt.errorbar(Halo_mass_plotting[4:len(Halo_mass_plotting)-10], (HI_mass_plotting[4:len(Halo_mass_plotting)-10]), yerr=None,marker = "X", mfc = 'grey', mec = 'grey', c = 'grey', capsize=2,ls = '--', markersize=15, linewidth=1, barsabove=True, capthick=2, label='Mock HI-stacking $(\\Delta v = 1.5 \\times V_{vir})$')


# Halo_intrinsic_mean, HI_intrinsic_mean, a,b = Common_module.halo_value_list(mvir_group/h, HI_intrinsic, mean=True)
Halo_intrinsic_mean, HI_intrinsic_mean, a,b = Common_module.halo_value_list(Mvir_intrinsic, HI_intrinsic, mean=True)
Halo_GAMA_mean, HI_GAMA_mean, a,b = Common_module.halo_value_list(mvir_group/h, HI_halo_group_2, mean=True)


# Error_SAM_best         = plt.errorbar(Halo_GAMA_mean[4:len(Halo_GAMA_mean)-10], (HI_intrinsic_mean[4:len(Halo_GAMA_mean)-10]), yerr=None,marker = "^", mfc = 'goldenrod', mec = 'k', c = 'k', capsize=2,ls = '--', markersize=15, linewidth=1, capthick=2, label='Best Match Halo Intrinsic $(N_{g} \\geq 5)$')

Error_SAM_best         = plt.errorbar(Halo_intrinsic_mean[4:len(Halo_intrinsic_mean)-10], (HI_intrinsic_mean[4:len(Halo_intrinsic_mean)-10]), yerr=None,marker = "^", mfc = 'goldenrod', mec = 'k', c = 'k', capsize=2,ls = '--', markersize=15, linewidth=1, capthick=2, label='Best Match Halo Intrinsic $(N_{g} \\geq 5)$')




Error_Guo_NG         = plt.errorbar(Guo_halo, Guo_HI_ng_2, yerr=Guo_error,marker = "o", mfc = 'r', mec = 'k', c = 'k', capsize=2,ls = ' ', markersize=15, linewidth=1, barsabove=False, capthick=2, label='$N_{g} \\geq 5$ (Guo+2020)')


extra = mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="Group HI stacking ")


legendHandles.append(Error_SAM_2)
legendHandles.append(Error_SAM_2_vvir)
legendHandles.append(Error_SAM_best)
legendHandles.append(Error_Guo_NG)
legendHandles.append(extra)
legendHandles.append(extra)

leg = plt.legend(handles=legendHandles[0:5],loc='upper left', frameon=False)
plt.gca().add_artist(leg)
leg = plt.legend(handles=legendHandles[5:6],loc='lower right', frameon=False)
plt.gca().add_artist(leg)

# plt.legend(handles=legendHandles[5], loc='lower left')


plt.xlabel('$log_{10}(M_{vir} [M_{\odot}])$')
plt.ylabel('$log_{10}(M_{HI} [M_{\odot}])$')
# plt.legend(handles=legendHandles)
plt.xlim(11.5,14)
plt.ylim(9.01,11.5)
# plt.title('SDSS-stacked')
plt.savefig('NG_trial_new_hope.png')
plt.close()




############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
###                                                  Distributions
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************




############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
###                                                  Distributions
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************


# legendHandles = []


# Guo_HI_ng_2             = np.array([9.869, 9.914, 10.047, 10.025, 10.073, 10.117, 10.191]) - np.log10(h)
# Guo_halo                = np.array([11.9, 12.1, 12.35, 12.6, 12.9, 13.25, 13.75]) - np.log10(h)
# Guo_error               = np.array([0.156, 0.094, 0.059, 0.047, 0.042, 0.032, 0.052])


# Halo_mass_plotting, HI_mass_plotting, HI_mass_low, HI_mass_high = Common_module.halo_value_list(Mvir_spherical_group_5, MHI_spherical_group_5, mean=True)

# plotting_properties_halo(Mvir_spherical_group_5, MHI_spherical_group_5, legend_handles=legendHandles, legend_name='Intrinsic $(N_{g} \\geq 5)$', colour_line='maroon', alpha=0.5, mean=True)


# extra = mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="Group HI stacking ")
# legendHandles.append(extra)
# legendHandles.append(extra)


# leg = plt.legend(handles=legendHandles[0:1],loc='upper left', frameon=False)
# plt.gca().add_artist(leg)
# leg = plt.legend(handles=legendHandles[1:2],loc='lower right', frameon=False)
# plt.gca().add_artist(leg)

# plt.xlabel('$log_{10}(M_{vir} [M_{\odot}])$')
# plt.ylabel('$log_{10}(M_{HI} [M_{\odot}])$')
# # plt.legend(handles=legendHandles)
# plt.xlim(11.5,14)
# plt.ylim(9,11.5)

# plt.savefig('presentation_intrinsic.png')
# plt.close()

# ######--------------------------------------------------------------------------------------------------------------------------------------
# ####*********************************************************

# legendHandles = []

# Halo_mass_plotting, HI_mass_plotting, HI_mass_low, HI_mass_high = Common_module.halo_value_list(Mvir_spherical_group_5, MHI_spherical_group_5, mean=True)
# plotting_properties_halo(Mvir_spherical_group_5, MHI_spherical_group_5, legend_handles=legendHandles, legend_name='Intrinsic $(N_{g} \\geq 5)$', colour_line='maroon', alpha=0.5, mean=True)


# Error_Guo_NG         = plt.errorbar(Guo_halo, Guo_HI_ng_2, yerr=Guo_error,marker = "o", mfc = 'r', mec = 'k', c = 'k', capsize=2,ls = ' ', markersize=15, linewidth=1, barsabove=False, capthick=2, label='$N_{g} \\geq 5$ (Guo+2020)')




# legendHandles.append(Error_Guo_NG)

# extra = mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="Group HI stacking ")
# legendHandles.append(extra)
# legendHandles.append(extra)


# leg = plt.legend(handles=legendHandles[0:2],loc='upper left', frameon=False)
# plt.gca().add_artist(leg)
# leg = plt.legend(handles=legendHandles[2:3],loc='lower right', frameon=False)
# plt.gca().add_artist(leg)

# plt.xlabel('$log_{10}(M_{vir} [M_{\odot}])$')
# plt.ylabel('$log_{10}(M_{HI} [M_{\odot}])$')
# # plt.legend(handles=legendHandles)
# plt.xlim(11.5,14)
# plt.ylim(9,11.5)

# plt.savefig('presentation_intrinsic_obs.png')
# plt.close()


# ######--------------------------------------------------------------------------------------------------------------------------------------
# ####*********************************************************

# legendHandles = []

# Halo_mass_plotting, HI_mass_plotting, HI_mass_low, HI_mass_high = Common_module.halo_value_list(Mvir_spherical_group_5, MHI_spherical_group_5, mean=True)
# plotting_properties_halo(Mvir_spherical_group_5, MHI_spherical_group_5, legend_handles=legendHandles, legend_name='Intrinsic $(N_{g} \\geq 5)$', colour_line='maroon', alpha=0.5, mean=True)


# Error_Guo_NG         = plt.errorbar(Guo_halo, Guo_HI_ng_2, yerr=Guo_error,marker = "o", mfc = 'r', mec = 'k', c = 'k', capsize=2,ls = ' ', markersize=15, linewidth=1, barsabove=False, capthick=2, label='$N_{g} \\geq 5$ (Guo+2020)')


# Halo_intrinsic_mean, HI_intrinsic_mean, a,b = Common_module.halo_value_list(Mvir_intrinsic, HI_intrinsic, mean=True)
# Halo_GAMA_mean, HI_GAMA_mean, a,b = Common_module.halo_value_list(mvir_group/h, HI_halo_group_2, mean=True)

# Error_SAM_best         = plt.errorbar(Halo_intrinsic_mean[4:len(Halo_intrinsic_mean)-10], (HI_intrinsic_mean[4:len(Halo_intrinsic_mean)-10]), yerr=None,marker = "^", mfc = 'goldenrod', mec = 'k', c = 'k', capsize=2,ls = '--', markersize=15, linewidth=1, capthick=2, label='Best Match Halo Intrinsic $(N_{g} \\geq 5)$')


# legendHandles.append(Error_Guo_NG)
# legendHandles.append(Error_SAM_best)

# extra = mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="Group HI stacking ")
# legendHandles.append(extra)
# legendHandles.append(extra)


# leg = plt.legend(handles=legendHandles[0:3],loc='upper left', frameon=False)
# plt.gca().add_artist(leg)
# leg = plt.legend(handles=legendHandles[3:4],loc='lower right', frameon=False)
# plt.gca().add_artist(leg)

# plt.xlabel('$log_{10}(M_{vir} [M_{\odot}])$')
# plt.ylabel('$log_{10}(M_{HI} [M_{\odot}])$')
# # plt.legend(handles=legendHandles)
# plt.xlim(11.5,14)
# plt.ylim(9,11.5)

# plt.savefig('presentation_intrinsic_obs_best.png')
# plt.close()



# ######--------------------------------------------------------------------------------------------------------------------------------------
# ####*********************************************************

# legendHandles = []

# Halo_mass_plotting, HI_mass_plotting, HI_mass_low, HI_mass_high = Common_module.halo_value_list(Mvir_spherical_group_5, MHI_spherical_group_5, mean=True)
# plotting_properties_halo(Mvir_spherical_group_5, MHI_spherical_group_5, legend_handles=legendHandles, legend_name='Intrinsic $(N_{g} \\geq 5)$', colour_line='maroon', alpha=0.5, mean=True)


# Error_Guo_NG         = plt.errorbar(Guo_halo, Guo_HI_ng_2, yerr=Guo_error,marker = "o", mfc = 'r', mec = 'k', c = 'k', capsize=2,ls = ' ', markersize=15, linewidth=1, barsabove=False, capthick=2, label='$N_{g} \\geq 5$ (Guo+2020)')


# Halo_intrinsic_mean, HI_intrinsic_mean, a,b = Common_module.halo_value_list(Mvir_intrinsic, HI_intrinsic, mean=True)
# Halo_GAMA_mean, HI_GAMA_mean, a,b = Common_module.halo_value_list(mvir_group/h, HI_halo_group_2, mean=True)

# Error_SAM_best         = plt.errorbar(Halo_intrinsic_mean[4:len(Halo_intrinsic_mean)-10], (HI_intrinsic_mean[4:len(Halo_intrinsic_mean)-10]), yerr=None,marker = "^", mfc = 'goldenrod', mec = 'k', c = 'k', capsize=2,ls = '--', markersize=15, linewidth=1, capthick=2, label='Best Match Halo Intrinsic $(N_{g} \\geq 5)$')

# Halo_mass_plotting, HI_mass_plotting, HI_mass_low, HI_mass_high = Common_module.halo_value_list(mvir_group/h, HI_halo_group_2/1.35/h, mean=True)
# Error_SAM_2         = plt.errorbar(Halo_mass_plotting[4:len(Halo_mass_plotting)-10], (HI_mass_plotting[4:len(Halo_mass_plotting)-10]), yerr=None,marker = "X", mfc = 'k', mec = 'k', c = 'k', capsize=2,ls = '--', markersize=15, linewidth=1, barsabove=True, capthick=2, label='Mock HI-stacking $(\\Delta v = 700\  km/s)$')


# legendHandles.append(Error_Guo_NG)
# legendHandles.append(Error_SAM_best)
# legendHandles.append(Error_SAM_2)

# extra = mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="Group HI stacking ")
# legendHandles.append(extra)
# legendHandles.append(extra)


# leg = plt.legend(handles=legendHandles[0:4],loc='upper left', frameon=False)
# plt.gca().add_artist(leg)
# leg = plt.legend(handles=legendHandles[4:5],loc='lower right', frameon=False)
# plt.gca().add_artist(leg)

# plt.xlabel('$log_{10}(M_{vir} [M_{\odot}])$')
# plt.ylabel('$log_{10}(M_{HI} [M_{\odot}])$')
# # plt.legend(handles=legendHandles)
# plt.xlim(11.5,14)
# plt.ylim(9,11.5)

# plt.savefig('presentation_intrinsic_obs_best_mock.png')
# plt.close()




