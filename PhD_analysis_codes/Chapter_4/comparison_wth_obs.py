import h5py as h5py
import numpy as np
import os
import scipy.stats as scipy
import re
from scipy import stats
import sys
from astropy.stats import bootstrap
import csv as csv
from astropy.utils import NumpyRNGContext
import seaborn as sns
import collections
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from Common_module import SharkDataReading
import Common_module
from scipy.stats import norm


############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
### Defining data_type and constants
######--------------------------------------------------------------------------------------------------------------------------------------
####****************************************************************************************************************************************

df  = np.dtype(float)
G   = 4.301e-9
h   = 0.6751
M_solar_2_g   = 1.99e33
dt = int


############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
### Reading Data
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************


path = '/mnt/su3ctm/gchauhan/SHArk_Out/HI_haloes/' 
# path = '/mnt/sshfs/pleiades_gchauhan/SHArk_Out/HI_haloes/'

shark_runs 		= ['Shark-Lagos18-default-br06','Shark-Lagos18-Kappa-10','Shark-Lagos18-Kappa-0','Shark-Lagos18-Kappa-1','Shark-Lagos18-default-br06-stripping-off']

# shark_runs 		= ['Shark-Lagos18-default-br06','Shark-Lagos18-Kappa-10','Shark-Lagos18-Kappa-0','Shark-Lagos18-Kappa-1','Shark-Lagos18-default-br06-stripping-off']
# shark_labels	= ['Kappa = 0.002','Kappa = 0.02','Kappa = 0','Kappa = 1','Kappa = 0.002 (stripping off)']
shark_labels	= ['SHARK-ref','$\\beta_{disc}$ = 0.5','$\\beta_{disc}$ = 1.5','$\\beta_{disc}$ = 2.5','$\\beta_{disc}$ = 3.5', 'SHARK-no-\n stellar-feedback']




simulation 		= ['medi-SURFS', 'micro-SURFS']
snapshot_avail	= [199]#,174,156]
z_values 		= [0,0.5,1]

subvolumes 		= 64

#--------------------------------------------------------------------------------

# medi_Kappa_original	 	= SharkDataReading(path,simulation[0],shark_runs[0],snapshot_avail[0],subvolumes)

#--------------------------------------------------------------------------------


medi_Kappa_original 	= {}
medi_Kappa_stripping 	= {}

micro_Kappa_original 	= {}
micro_Kappa_stripping 	= {}


for snapshot in snapshot_avail:

	medi_Kappa_original[snapshot]	 	= SharkDataReading(path,simulation[0],shark_runs[0],snapshot,subvolumes)
	
	micro_Kappa_original[snapshot]		= SharkDataReading(path,simulation[1],shark_runs[0],snapshot,subvolumes)
	
#--------------------------------------------------------------------------------

	
	
#--------------------------------------------------------------------------------

def prepare_data(path,simulation,run,snapshot,subvolumes):

	#fields_read = {'galaxies' : ('type','mstars_disk', 'mstars_bulge', 'matom_disk','matom_bulge' , 'mgas_disk', 'mgas_bulge','mvir_hosthalo')}

	fields_read 	= {'halo' : ('age_50', 'age_80', 'lambda','mvir'), 'galaxies':('type', 'm_bh', 'mstars_bulge', 'mstars_disk')}

	data_reading = SharkDataReading(path,simulation,run,snapshot,subvolumes)

	data_plotting = data_reading.readIndividualFiles(fields=fields_read)

	return data_plotting


############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
### Developing Module for Satellite list
######--------------------------------------------------------------------------------------------------------------------------------------
####****************************************************************************************************************************************
plotting_merge_runs 	= [medi_Kappa_original[199]]
plotting_merge_runs_micro = [micro_Kappa_original[199]]


HI_all_1, HI_central_1, HI_satellite_1, Mvir_halo_1 	= medi_Kappa_original[199].prop_mass_rvir('id_halo_tree', 'matom_bulge', 'matom_disk', rvir_times=0)

HI_all_rvir, HI_central_rvir, HI_satellite_rvir, Mvir_halo_rvir 	= medi_Kappa_original[199].prop_mass_rvir('id_halo_tree', 'matom_bulge', 'matom_disk', rvir_times=1)


############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
### Analysis
######--------------------------------------------------------------------------------------------------------------------------------------
####****************************************************************************************************************************************

plt = Common_module.load_matplotlib()
colour_plot = Common_module.colour_scheme()

legendHandles = list()

age_50 			= {}
age_80 			= {}
lambda_spin 	= {}
mvir 			= {}
is_central  	= {}
m_bh 			= {}
mstars_bulge 	= {}
mstars_disk 	= {}

for snapshot in snapshot_avail:

	(h0,_, age_50[snapshot], age_80[snapshot], lambda_spin[snapshot], mvir[snapshot], is_central[snapshot], m_bh[snapshot], mstars_bulge[snapshot], mstars_disk[snapshot]) = prepare_data(path,simulation[0],shark_runs[0],snapshot,subvolumes)
	


plt = Common_module.load_matplotlib()
colour_plot = Common_module.colour_scheme()

legendHandles = list()

age_50_micro 		= {}
age_80_micro 		= {}
lambda_spin_micro 	= {}
mvir_micro 			= {}
is_central_micro  	= {}
m_bh_micro 			= {}
mstars_bulge_micro 	= {}
mstars_disk_micro 	= {}

for snapshot in snapshot_avail:

	(h0,_, age_50_micro[snapshot], age_80_micro[snapshot], lambda_spin_micro[snapshot], mvir_micro[snapshot], is_central_micro[snapshot], m_bh_micro[snapshot], mstars_bulge_micro[snapshot], mstars_disk_micro[snapshot]) = prepare_data(path,simulation[1],shark_runs[0],snapshot,subvolumes)





####****************************************************************************************************************************************



virial_mass_1 	= Mvir_halo_1[HI_all_1 !=0]

property_plot_1 = HI_all_1[HI_all_1 !=0]


####****************************************************************************************************************************************

############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
### Observation Data
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************


Martindale_halo_group 	= np.array([10.352, 12.039,12.681,13.764]) - np.log10(h)
Martindale_HI_group   	= np.array([9.36, 10.09, 10.04, 9.64]) - 2*np.log10(h)
err_group_y 			= np.array([0.14,0.05,0.05,0.39])
err_group_x_lower 		= Martindale_halo_group - np.array([9.0,11.7,12.4,13.0]) + np.log10(h)
err_group_x_upper 		= np.array([11.7,12.4,13.0,14.6]) -  Martindale_halo_group - np.log10(h)

Martindale_halo_isolated 	= np.array([10.865, 11.146, 11.267, 11.395, 11.583, 12.041]) - np.log10(h)
Martindale_HI_isolated 		= np.array([8.36, 8.94, 9.10, 9.24, 9.40, 9.80]) - 2*np.log10(h)
err_isolated_y 				= np.array([0.15,0.12,0.09,0.08,0.08,0.06]) 
err_isolated_x_lower 		= Martindale_halo_isolated - np.array([10.0,11.1,11.2,11.3,11.5,11.7]) + np.log10(h)
err_isolated_x_upper		= np.array([11.1,11.2,11.3,11.5,11.7,13.1]) - Martindale_halo_isolated - np.log10(h)

Obuljen_Halo 					= np.array([12.633921,13.003603,13.385534,13.758411,14.131233,14.498241,14.871365]) - np.log10(h)
Obuljen_HI 						= np.array([10.342192,10.328132,10.432698,10.571177,10.658807,10.973563,11.33916]) - np.log10(h)
yerr_obuljen_upper 				= np.array([10.535413,10.433216,10.493719,10.611862,10.784234,11.102389,11.51205]) - Obuljen_HI - np.log10(h)
yerr_obuljen_lower 				= Obuljen_HI - np.array([10.142192,10.223039,10.371694,10.506769,10.557124,10.851525,11.152715])  + np.log10(h)

data_ECO = []
with open('datafile1-ECO_new.csv', newline='') as csvfile:
	reader = csv.reader(csvfile, delimiter=' ')
	for row in reader:
		data_ECO.append(row)


data_RES = []
with open('datafile1-RES.txt', newline='') as csvfile:
	reader = csv.reader(csvfile,delimiter=' ')
	for row in reader:
		data_RES.append(row)





M_halo_ECO = np.zeros(len(data_ECO))
M_HI_ECO = np.zeros(len(data_ECO))

for i in range(len(data_ECO)):
	M_halo_ECO[i] = np.float(data_ECO[i][6]) 
	M_HI_ECO[i] = np.log10((10**np.float(data_ECO[i][12]) - 10**np.float(data_ECO[i][11]))/1.4)


M_halo_RES = np.zeros(len(data_RES))
M_HI_RES = np.zeros(len(data_RES))

for i in range(len(data_RES)):
	M_halo_RES[i] = np.float(data_RES[i][6]) 
	M_HI_RES[i] = np.log10((10**np.float(data_RES[i][12]) - 10**np.float(data_RES[i][11]))/1.4)


legendHandles = []


Eco_halo, Eco_HI, Eco_HI_lower, Eco_HI_higher = Common_module.halo_value_list(10**M_halo_ECO,10**(M_HI_ECO),mean=False)

# Common_module.plotting_properties_halo(10**M_halo_ECO,10**(M_HI_ECO),mean=False,legend_handles=legendHandles,colour_line='rebeccapurple',fill_between=True,xlim_lower=10,xlim_upper=14.5,ylim_lower=8,ylim_upper=14,legend_name='yo', property_name_x='log_{10}(M_{vir}[M_{\odot}])', property_name_y='log_{10}(M_{HI} [M_{\odot}])', resolution = False, first_legend=False)

# Common_module.plotting_properties_halo(10**M_halo_RES,10**(M_HI_RES),mean=False,legend_handles=legendHandles,colour_line='tab:blue',fill_between=False,xlim_lower=10,xlim_upper=14.5,ylim_lower=8,ylim_upper=14,legend_name='yo', property_name_x='log_{10}(M_{vir}[M_{\odot}])', property_name_y='log_{10}(M_{HI} [M_{\odot}])', resolution = False, first_legend=False)


####****************************************************************************************************************************************



plt = Common_module.load_matplotlib(12,8)
colour_plot = Common_module.colour_scheme()

legendHandles = list()

extra = mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="z = 0")
legendHandles.append(extra)


medi_HI, medi_HI_central, medi_HI_satellite,medi_HI_orphan,medi_vir = medi_Kappa_original[199].mergeValues_satellite('id_halo','matom_bulge','matom_disk')

medi_stellar, medi_stellar_central, medi_stellar_satellite,medi_stellar_orphan,a = medi_Kappa_original[199].mergeValues_satellite('id_halo','mstars_bulge','mstars_disk')

HI_all 			= medi_HI#[medi_vir >= 10**11.15]
Stellar_all 	= medi_vir#[medi_vir >= 10**11.15]

property_plot 	= HI_all#/medi_vir[medi_vir >= 10**11.15]





Common_module.plotting_properties_halo(Stellar_all,property_plot,mean=True,legend_handles=legendHandles,colour_line='r',fill_between=False,xlim_lower=11,xlim_upper=14,ylim_lower=-5,ylim_upper=-0.5,legend_name=shark_labels[0], property_name_x='log_{10}(M_{vir}[M_{\odot}])', property_name_y='log_{10}(M_{HI} [M_{\odot}])', resolution = False, first_legend=False)



####------------------------------------------------    GUO - POINTS ---------------------------------------------------------------------------------#####

Guo_halo_isolated = np.array([11.118972,11.383127,11.616222,11.888279,12.125619,12.384735,12.630501,12.885231,13.258829,13.750053]) - np.log10(h)

Guo_HI_isolated = np.array([8.896476,9.197137,9.401432,9.436123,9.586453,9.678965,9.732929,9.856277,10.068282,10.09141]) - np.log10(h)

Guo_halo_group = np.array([11.375638,11.635152,11.894217,12.144291,12.372295,12.635426,12.872547,13.258932,13.754722]) - np.log10(h)

Guo_HI_group = np.array([9.601872,9.806168,9.902533,9.925661,9.906387,9.906387,9.991189,10.099119,10.153084]) - np.log10(h)


Guo_HI_isolated_upper = np.array([8.925696,9.208094,9.41009,9.435954,9.595778,9.694416,9.743239,9.883966,10.062517,10.1294775]) - np.log10(h) - Guo_HI_isolated

Guo_HI_isolated_lower = Guo_HI_isolated - np.array([8.887403,9.188963,9.390961,9.432108,9.572819,9.663798,9.708758,9.834168,10.031865,10.022272]) + np.log10(h)

Guo_HI_group_upper = np.array([9.705927,9.877222,9.9605255,9.990219,9.924178,9.938503,10.010336,10.116128,10.2214155]) - np.log10(h) - Guo_HI_group

Guo_HI_group_lower = Guo_HI_group - np.array([9.491466,9.739415,9.853371,9.9059725,9.85147,9.869625,9.945203,10.070175,10.114194]) + np.log10(h)

Guo_halo_central = np.array([11.128378,11.385135,11.621622,11.864865,12.121622,12.364865,12.594595,12.8918915,13.25,13.756757]) - np.log10(h)

Guo_HI_central = np.array([8.887856,9.171226,9.37035,9.420132,9.531181,9.580963,9.623085,9.695843,9.695843,9.607768]) - np.log10(h)

Guo_HI_central_err = np.array([0.020,0.010,0.008,0.007,0.010,0.013,0.017,0.022,0.024,0.050])

Guo_halo_satellite = np.array([12.140271,12.377828,12.608598,12.880091,13.239819,13.755656]) - np.log10(h)

Guo_HI_satellite = np.array([8.677083,8.964912,9.076206,9.337171,9.805373,9.8936405]) - np.log10(h)

Guo_HI_satellite_err = np.array([ 0.01414214,0.01838478, 0.02404163, 0.0311127 , 0.03394113, 0.07071068])


####------------------------------------------------    ADAM Points ---------------------------------------------------------------------------------#####

M200_TNG = np.array([1.9789e+11, 3.1461e+11, 4.9416e+11, 7.9089e+11, 1.2426e+12, 1.9790e+12, 3.1261e+12, 4.9794e+12, 7.8911e+12, 1.2709e+13, 1.9385e+13, 3.1304e+13, 5.1680e+13, 7.6882e+13, 2.1259e+14])

HI_upper_bound_TNG = np.array([3.7923e+09, 4.6992e+09, 6.8354e+09, 9.9620e+09, 1.3600e+10, 1.5544e+10, 1.6748e+10, 2.3139e+10, 2.8092e+10 ,2.8907e+10, 3.6333e+10, 3.4057e+10, 3.3870e+10, 3.9220e+10 ,3.3098e+10])

HI_value_TNG = np.array([1.1283e+09, 1.5561e+09, 2.3782e+09, 3.5064e+09, 4.7196e+09, 5.3960e+09, 4.8214e+09, 7.0548e+09, 1.1344e+10, 1.4431e+10, 1.5097e+10, 1.4698e+10, 1.6366e+10, 1.4520e+10, 1.9091e+10])

HI_lower_bound_TNG = np.array([2.6005e+08, 4.4938e+08, 6.1690e+08, 7.5648e+08, 3.5354e+08, 3.3198e+08, 6.2118e+08, 1.1635e+09, 3.2063e+09, 3.2798e+09, 5.9244e+09, 6.1435e+09, 4.0487e+09, 7.3216e+09, 7.4010e+09])


####------------------------------------------------    GUO - POINTS ---------------------------------------------------------------------------------#####

plt.axvline(x=11.2, linestyle=':', color = 'k')


Error_det_Obuljen 		= plt.errorbar(Obuljen_Halo, Obuljen_HI, yerr=[yerr_obuljen_lower,yerr_obuljen_upper], xerr=None,label='Obuljen+ 2019', marker = "X", mfc = 'green', mec = 'green', c = 'green', elinewidth=2,ls = '--',lw=1, markersize=10)

Error_det_isolated 		= plt.errorbar(Guo_halo_isolated, Guo_HI_isolated, yerr=Guo_HI_central_err,label='$N_{g} >= 1$ (Guo+ 2020)', marker = "p", mfc = 'maroon', mec = 'maroon', c = 'maroon', elinewidth=2,ls = 'dashdot', lw=1,markersize=10)

# Error_det_ECO = plt.errorbar(Eco_halo, Eco_HI, yerr=[Eco_HI - Eco_HI_lower, Eco_HI_higher-Eco_HI], label = 'ECO (Eckert+ 2017)', marker = "o", mfc = 'rebeccapurple', mec='rebeccapurple', c='rebeccapurple', elinewidth=2, ls = ":", lw=1, markersize=10)

# legendHandles.append(Error_det_group)
legendHandles.append(mpatches.Patch(color='r',label=shark_labels[0]))
# legendHandles.append(mpatches.Patch(color='goldenrod',label='SHARK-ref (Guo criteria)'))
legendHandles.append(Error_det_isolated)
legendHandles.append(Error_det_Obuljen)
# legendHandles.append(Error_det_ECO)
# legendHandles.append(mpatches.Patch(color='rebeccapurple',label='ECO (Eckert+ 2017)'))
# legendHandles.append(mpatches.Patch(color='tab:blue',label= 'RESOLVE (Eckert+ 2017)'))
# legendHandles.append(Error_det_Adam)
# legendHandles.append(Error_det_group_martin)
# legendHandles.append(Error_det_isolated_martin)

# legendHandles.append(Error_Baugh)


plt.axvline(x=11.2, linestyle=':', color = 'k')

plt.legend(handles=legendHandles)

virial_mass_1 	= Mvir_halo_rvir[HI_all_rvir !=0]

property_plot_1 = HI_all_rvir[HI_all_rvir !=0]



Common_module.plotting_properties_halo(virial_mass_1,property_plot_1,mean=True,legend_handles=legendHandles,colour_line='goldenrod',fill_between=False,xlim_lower=11,xlim_upper=14,ylim_lower=8,ylim_upper=11,legend_name='Lagos18 (Rvir = 1)', property_name_x='log_{10}(M_{vir}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])', resolution = False, first_legend=True)


# plt.savefig("obs_plot_rvir_Guo_obuljen.png")
plt.show()





####****************************************************************************************************************************************

plt = Common_module.load_matplotlib(12,8)
colour_plot = Common_module.colour_scheme()

legendHandles = list()

extra = mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="z = 0")
legendHandles.append(extra)

plt.axvline(x=11.2, linestyle=':', color = 'k')

medi_HI, medi_HI_central, medi_HI_satellite,a,medi_vir = plotting_merge_runs[0].mergeValues_satellite('id_halo','matom_bulge','matom_disk')

# medi_stellar, medi_stellar_central, medi_stellar_satellite,a,a = plotting_merge_runs[0].mergeValues_satellite('id_halo','mstars_bulge','mstars_disk')


virial_mass 	= {'all' : Mvir_halo_1[Mvir_halo_1 >= 10**11.15], 'central' : Mvir_halo_1[Mvir_halo_1 >= 10**11.15], 'satellite' : Mvir_halo_1[Mvir_halo_1 >= 10**11.15]}
# stellar_property = {'all' : medi_stellar[medi_vir >= 10**11.15], 'central' : medi_stellar_central[medi_vir >= 10**11.15], 'satellite' : medi_stellar_satellite[medi_vir >= 10**11.15]}


property_plot = {'all' : HI_all_1[Mvir_halo_1 >= 10**11.15], 'central' : HI_all_1[Mvir_halo_1 >= 10**11.15], 'satellite' : (HI_all_rvir[Mvir_halo_rvir>= 10**11.15] - HI_all_1[[Mvir_halo_1 >= 10**11.15]])}

Common_module.plotting_properties_separate_merged(virial_mass,stellar_property=virial_mass,property_plot = property_plot,mean=True,legend_handles=legendHandles,property_name_x='log_{10}(M_{vir}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=11,xlim_upper=14,ylim_lower=8,ylim_upper=11, first_legend=False,colour_line='indianred',legend_name=shark_labels[0], resolution=False)

virial_mass 	= {'all' : medi_vir[medi_vir >= 10**11.15], 'central' : medi_vir[medi_vir >= 10**11.15], 'satellite' : medi_vir[medi_vir >= 10**11.15]}

property_plot = {'all' : medi_HI[medi_vir >= 10**11.15], 'central' : medi_HI_central[medi_vir >= 10**11.15], 'satellite' : medi_HI_satellite[medi_vir >= 10**11.15]}

Common_module.plotting_properties_separate_merged(virial_mass,stellar_property=virial_mass,property_plot = property_plot,mean=True,legend_handles=legendHandles,property_name_x='log_{10}(M_{vir}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=11,xlim_upper=14,ylim_lower=8,ylim_upper=11, first_legend=False,colour_line='brown',legend_name=shark_labels[0], resolution=False)

Error_det_isolated_central 		= plt.errorbar(Guo_halo_central, Guo_HI_central, label='$N_{g} >= 1$ Central (Guo+ 2020)', yerr = Guo_HI_central_err,marker = "X", mfc = 'indigo', mec = 'indigo', c = 'indigo', elinewidth=2,ls = ':', markersize=10)

Error_det_isolated_satellite 		= plt.errorbar(Guo_halo_satellite, Guo_HI_satellite,label='$N_{g} >= 1$ Satellite (Guo+ 2020)', yerr =Guo_HI_satellite_err  ,marker = "o", mfc = 'goldenrod', mec = 'goldenrod', c = 'goldenrod', elinewidth=2,ls = ':', markersize=10)


# legendHandles.append(mlines.Line2D([],[],color='grey',linestyle='--',label='Centrals (Guo criteria)'))
# legendHandles.append(mlines.Line2D([],[],color='grey',linestyle='-.',label='Satellites (Guo criteria)'))
legendHandles.append(mlines.Line2D([],[],color='k',linestyle='--',label='Centrals'))
legendHandles.append(mlines.Line2D([],[],color='k',linestyle='-.',label='Satellites'))
legendHandles.append(mpatches.Patch(color='brown',label=shark_labels[0]))
legendHandles.append(mpatches.Patch(color='indianred',label='SHARK-ref (Guo+ 2020 criteria)'))

legendHandles.append(Error_det_isolated_central)
legendHandles.append(Error_det_isolated_satellite)

plt.legend(handles=legendHandles)
# plt.savefig("obs_plot_rvir_GUO_separate_criteria.png")
plt.show()







############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
### Combined Plot
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************


plt = Common_module.load_matplotlib(12,8)
colour_plot = Common_module.colour_scheme()

legendHandles = list()

extra = mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="z = 0")
legendHandles.append(extra)


medi_HI, medi_HI_central, medi_HI_satellite,medi_HI_orphan,medi_vir = medi_Kappa_original[199].mergeValues_satellite('id_halo','matom_bulge','matom_disk')

medi_stellar, medi_stellar_central, medi_stellar_satellite,medi_stellar_orphan,a = medi_Kappa_original[199].mergeValues_satellite('id_halo','mstars_bulge','mstars_disk')

HI_all 			= medi_HI#[medi_vir >= 10**11.15]
Stellar_all 	= medi_vir#[medi_vir >= 10**11.15]

property_plot 	= HI_all#/medi_vir[medi_vir >= 10**11.15]

fig8 = plt.figure()
gs1  = fig8.add_gridspec(nrows=2,ncols=3,hspace=0)
f8_ax1 = fig8.add_subplot(gs1[:-1,:])

Common_module.plotting_properties_halo(Stellar_all,property_plot,mean=True,legend_handles=legendHandles,colour_line='r',fill_between=False,xlim_lower=11,xlim_upper=14,ylim_lower=-5,ylim_upper=-0.5,legend_name=shark_labels[0], property_name_x='log_{10}(M_{vir}[M_{\odot}])', property_name_y='log_{10}(M_{HI} [M_{\odot}])', resolution = True, first_legend=False)



####------------------------------------------------    GUO - POINTS ---------------------------------------------------------------------------------#####

Guo_halo_isolated = np.array([11.118972,11.383127,11.616222,11.888279,12.125619,12.384735,12.630501,12.885231,13.258829,13.750053]) - np.log10(h)

Guo_HI_isolated = np.array([8.896476,9.197137,9.401432,9.436123,9.586453,9.678965,9.732929,9.856277,10.068282,10.09141]) - np.log10(h)

Guo_halo_group = np.array([11.375638,11.635152,11.894217,12.144291,12.372295,12.635426,12.872547,13.258932,13.754722]) - np.log10(h)

Guo_HI_group = np.array([9.601872,9.806168,9.902533,9.925661,9.906387,9.906387,9.991189,10.099119,10.153084]) - np.log10(h)


Guo_HI_isolated_upper = np.array([8.925696,9.208094,9.41009,9.435954,9.595778,9.694416,9.743239,9.883966,10.062517,10.1294775]) - np.log10(h) - Guo_HI_isolated

Guo_HI_isolated_lower = Guo_HI_isolated - np.array([8.887403,9.188963,9.390961,9.432108,9.572819,9.663798,9.708758,9.834168,10.031865,10.022272]) + np.log10(h)

Guo_HI_group_upper = np.array([9.705927,9.877222,9.9605255,9.990219,9.924178,9.938503,10.010336,10.116128,10.2214155]) - np.log10(h) - Guo_HI_group

Guo_HI_group_lower = Guo_HI_group - np.array([9.491466,9.739415,9.853371,9.9059725,9.85147,9.869625,9.945203,10.070175,10.114194]) + np.log10(h)

Guo_halo_central = np.array([11.128378,11.385135,11.621622,11.864865,12.121622,12.364865,12.594595,12.8918915,13.25,13.756757]) - np.log10(h)

Guo_HI_central = np.array([8.887856,9.171226,9.37035,9.420132,9.531181,9.580963,9.623085,9.695843,9.695843,9.607768]) - np.log10(h)

Guo_HI_central_err = np.array([0.020,0.010,0.008,0.007,0.010,0.013,0.017,0.022,0.024,0.050])

Guo_halo_satellite = np.array([12.140271,12.377828,12.608598,12.880091,13.239819,13.755656]) - np.log10(h)

Guo_HI_satellite = np.array([8.677083,8.964912,9.076206,9.337171,9.805373,9.8936405]) - np.log10(h)

Guo_HI_satellite_err = np.array([ 0.01414214,0.01838478, 0.02404163, 0.0311127 , 0.03394113, 0.07071068])


####------------------------------------------------    GUO - POINTS ---------------------------------------------------------------------------------#####



Error_det_Obuljen 		= plt.errorbar(Obuljen_Halo, Obuljen_HI, yerr=[yerr_obuljen_lower,yerr_obuljen_upper], xerr=None,label='Obuljen+ 2019', marker = "X", mfc = 'green', mec = 'green', c = 'green', elinewidth=2,ls = '--',lw=1, markersize=10)

Error_det_isolated 		= plt.errorbar(Guo_halo_isolated, Guo_HI_isolated, yerr=Guo_HI_central_err,label='$N_{g} >= 1$ (Guo+ 2020)', marker = "p", mfc = 'maroon', mec = 'maroon', c = 'maroon', elinewidth=2,ls = 'dashdot', lw=1,markersize=10)
legendHandles.append(mpatches.Patch(color='r',label=shark_labels[0]))
legendHandles.append(Error_det_isolated)
legendHandles.append(Error_det_Obuljen)


plt.legend(handles=legendHandles, loc='lower right')

virial_mass_1 	= Mvir_halo_rvir[HI_all_rvir !=0]

property_plot_1 = HI_all_rvir[HI_all_rvir !=0]

plt.axvline(x=11.2, linestyle=':', color = 'k')


Common_module.plotting_properties_halo(virial_mass_1,property_plot_1,mean=True,legend_handles=legendHandles,colour_line='goldenrod',fill_between=False,xlim_lower=11,xlim_upper=14,ylim_lower=8.01,ylim_upper=11,legend_name='Lagos18 (Rvir = 1)', property_name_x='log_{10}(M_{vir}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])', resolution = False, first_legend=True)

f8_ax1.set_xticklabels([])
####------------------------------------------------    GUO - POINTS ---------------------------------------------------------------------------------#####

f8_ax2 = fig8.add_subplot(gs1[-1, :])

legendHandles = list()


medi_HI, medi_HI_central, medi_HI_satellite,a,medi_vir = plotting_merge_runs[0].mergeValues_satellite('id_halo','matom_bulge','matom_disk')


virial_mass 	= {'all' : medi_vir[medi_vir >= 10**11.15], 'central' : medi_vir[medi_vir >= 10**11.15], 'satellite' : medi_vir[medi_vir >= 10**11.15]}

property_plot = {'all' : medi_HI[medi_vir >= 10**11.15], 'central' : medi_HI_central[medi_vir >= 10**11.15], 'satellite' : medi_HI_satellite[medi_vir >= 10**11.15]}

Common_module.plotting_properties_separate_merged(virial_mass,stellar_property=virial_mass,property_plot = property_plot,mean=True,legend_handles=legendHandles,property_name_x='log_{10}(M_{vir}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=11,xlim_upper=14,ylim_lower=8.01,ylim_upper=10.9, first_legend=False,colour_line='red',legend_name=shark_labels[0], resolution=False)

Error_det_isolated_central 		= plt.errorbar(Guo_halo_central, Guo_HI_central, label='$N_{g} >= 1$ Central (Guo+ 2020)', yerr = Guo_HI_central_err,marker = "X", mfc = 'indigo', mec = 'indigo', c = 'indigo', elinewidth=2,ls = ':', markersize=10)

Error_det_isolated_satellite 		= plt.errorbar(Guo_halo_satellite, Guo_HI_satellite,label='$N_{g} >= 1$ Satellite (Guo+ 2020)', yerr =Guo_HI_satellite_err  ,marker = "o", mfc = 'goldenrod', mec = 'goldenrod', c = 'goldenrod', elinewidth=2,ls = ':', markersize=10)

legendHandles.append(mlines.Line2D([],[],color='r',linestyle='--',label='Centrals'))
legendHandles.append(mlines.Line2D([],[],color='r',linestyle='-.',label='Satellites'))
# legendHandles.append(mpatches.Patch(color='brown',label=shark_labels[0]))
# legendHandles.append(mpatches.Patch(color='indianred',label='SHARK-ref (Guo+ 2020 criteria)'))

legendHandles.append(Error_det_isolated_central)
legendHandles.append(Error_det_isolated_satellite)

plt.axvline(x=11.2, linestyle=':', color = 'k')

leg = plt.legend(handles=legendHandles[2:5],loc='lower right', frameon=False)
plt.gca().add_artist(leg)

plt.legend(handles=legendHandles[0:2], loc='upper left')

# plt.legend(handles=legendHandles, loc='lower right')
plt.savefig("obs_plot_rvir_GUO_combined.png")
plt.show()


