import h5py as h5py
import numpy as np
import os
import scipy.stats as scipy
import re
from scipy import stats
from astropy.stats import bootstrap
from astropy.utils import NumpyRNGContext
import seaborn as sns
import collections
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from Common_module import SharkDataReading
import Common_module
from matplotlib.legend import Legend
############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
### Defining data_type and constants
######--------------------------------------------------------------------------------------------------------------------------------------
####****************************************************************************************************************************************

dt  = np.dtype(float)
G   = 4.301e-9
h   = 0.6751
# h = 0.7
M_solar_2_g   = 1.99e33
dt = int


############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
### Reading Data
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************




#--------------------------------------------------------------------------------


# path = '/mnt/su3ctm/gchauhan/SHArk_Out/HI_haloes/' 
path = '/mnt/sshfs/pleiades_gchauhan/SHArk_Out/HI_haloes/'

# shark_runs = ['Shark-Lagos18-default-br06','Shark-Lagos18-Kappa-10','Shark-Lagos18-Kappa-0','Shark-Lagos18-Kappa-1', 'Shark-Lagos18-default-gd14']#,]#,'Lagos18-Kappa-original-stripping-off','Test-default-chauhan-stripping-off-br06']

# shark_runs	= ['Shark-Lagos18-default-br06','Shark-Lagos18-default-beta-0p5','Shark-Lagos18-default-beta-1p5','Shark-Lagos18-default-beta-2p5', 'Shark-Lagos18-default-no-stellar']

# shark_runs	= ['Shark-Lagos18-default-br06', 'Shark-Lagos18-default-no-stellar','Shark-Lagos18-default-eps-3','Shark-Lagos18-default-eps-5','Shark-Lagos18-default-eps-7','Shark-Lagos18-default-eps-10']

shark_runs	= ['Shark-Lagos18-default-br06', 'Shark-Lagos18-default-vcut-20', 'Shark-Lagos18-default-vcut-30', 'Shark-Lagos18-default-vcut-40', 'Shark-Lagos18-default-vcut-50' ]
# shark_runs 		= ['Shark-Lagos18-default-br06','Shark-Lagos18-Kappa-10','Shark-Lagos18-Kappa-1','Shark-Lagos18-Kappa-0','Test-default-chauhan-stripping-off-br06','Shark-Lagos18-Kappa-original']#'Shark-Lagos18-default-br06-stripping-off'] #'Test-default-chauhan-stripping-off-br06'Shark-Lagos18-Kappa-original

# shark_labels	= ['$\kappa_{agn}$ = 0.002 (Lagos18)','$\kappa_{agn}$ = 0.02','$\kappa_{agn}$ = 0','$\kappa_{agn}$ = 1','Kappa = 0.002 (stripping off)']
# shark_labels	= ['SHARK-ref','$\kappa_{agn}$ = 0.02','$\kappa_{agn}$ = 1', 'SHARK-no-AGN','SHARK-stripping-off']
# shark_labels	= ['SHARK-ref','Kappa = 0.02','Kappa = 0','Kappa = 1','SHARK-GD14']
# shark_labels	= ['SHARK-ref','$\\beta_{disc}$ = 0.5','$\\beta_{disc}$ = 1.5','$\\beta_{disc}$ = 2.5', 'SHARK-no-\n stellar-feedback']
# shark_labels	= ['SHARK-ref','$\\epsilon_{disc}$ = 0','$\\epsilon_{disc}$ = 3','$\\epsilon_{disc}$ = 5','$\\epsilon_{disc}$ = 7', '$\\epsilon_{disc}$ = 10']
shark_labels	= ['SHARK-ref','$v_{cut}$ = 20 km/s','$v_{cut}$ = 30 km/s','$v_{cut}$ = 40 km/s' ,'$v_{cut}$ = 50 km/s','Beta = 5']


simulation = ['medi-SURFS', 'micro-SURFS','L105_N2048']

snapshot = [199,174,156,131,113,100,189]

subvolumes = [64,128]



medi_Kappa_original	 	= SharkDataReading(path,simulation[0],shark_runs[0],snapshot[0],subvolumes[0])
medi_Kappa_10	 		= SharkDataReading(path,simulation[0],shark_runs[1],snapshot[0],subvolumes[0])
medi_Kappa_0	 		= SharkDataReading(path,simulation[0],shark_runs[2],snapshot[0],subvolumes[0])
medi_Kappa_1	 		= SharkDataReading(path,simulation[0],shark_runs[3],snapshot[0],subvolumes[0])
medi_Kappa_stripping	= SharkDataReading(path,simulation[0],shark_runs[4],snapshot[0],subvolumes[0])
# medi_Kappa_beta_5 		= SharkDataReading(path,simulation[0],shark_runs[5],snapshot[0],subvolumes[0])
# medi_Kappa_beta_0 		= SharkDataReading(path,simulation[0],shark_runs[6],snapshot[0],subvolumes[0])


micro_Kappa_original	= SharkDataReading(path,simulation[1],shark_runs[0],snapshot[0],subvolumes[0])
micro_Kappa_10	 		= SharkDataReading(path,simulation[1],shark_runs[1],snapshot[0],subvolumes[0])
micro_Kappa_0	 		= SharkDataReading(path,simulation[1],shark_runs[2],snapshot[0],subvolumes[0])
micro_Kappa_1	 		= SharkDataReading(path,simulation[1],shark_runs[3],snapshot[0],subvolumes[0])
micro_Kappa_stripping	= SharkDataReading(path,simulation[1],shark_runs[4],snapshot[0],subvolumes[0])
# micro_Kappa_beta_5	    = SharkDataReading(path,simulation[1],shark_runs[5],snapshot[0],subvolumes[0])
# micro_Kappa_beta_0 		= SharkDataReading(path,simulation[1],shark_runs[6],snapshot[0],subvolumes[0])


# genesis_Kappa_original	= SharkDataReading(path,simulation[2],shark_runs[5],snapshot[6],subvolumes[1])





#------------------------------------------------------------------------------------------------------------

############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
### Analysis - MEDI
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************


plt = Common_module.load_matplotlib(12,8)
colour_plot = Common_module.colour_scheme(colourmap = plt.cm.Set2)

legendHandles = list()


####*********************************************************


fig8 = plt.figure()
gs1  = fig8.add_gridspec(nrows=3,ncols=3,hspace=0)
f8_ax1 = fig8.add_subplot(gs1[:-1,:])

####*********************************************************


plotting_merge_runs 	= [medi_Kappa_original, medi_Kappa_10, medi_Kappa_0, medi_Kappa_1, medi_Kappa_stripping]#, medi_Kappa_beta_5,medi_Kappa_beta_0]
plotting_merge_runs_micro = [micro_Kappa_original, micro_Kappa_10, micro_Kappa_0, micro_Kappa_1, micro_Kappa_stripping]#, micro_Kappa_beta_5, micro_Kappa_beta_0]


extra = mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="z = 0")
legendHandles.append(extra)


medi_HI, medi_HI_central, medi_HI_satellite,medi_HI_orphan,medi_vir = plotting_merge_runs[0].mergeValues_satellite('id_halo','matom_bulge','matom_disk')

medi_stellar, medi_stellar_central, medi_stellar_satellite,medi_stellar_orphan,a = plotting_merge_runs[0].mergeValues_satellite('id_halo','mstars_bulge','mstars_disk')

HI_all 			= medi_HI[medi_vir >= 10**11.15]
Stellar_all 	= medi_vir[medi_vir >= 10**11.15]

property_plot 	= HI_all/medi_vir[medi_vir >= 10**11.15]

Common_module.plotting_properties_halo(Stellar_all, property_plot,mean=False,legend_handles=legendHandles,colour_line='r',fill_between=True,xlim_lower=10,xlim_upper=14.5,ylim_lower=-5,ylim_upper=-0.5,legend_name=shark_labels[0], property_name_x='log_{10}(M_{vir}[M_{\odot}])', property_name_y='log_{10}(M_{HI} [M_{\odot}])', resolution = True, first_legend=False)


medi_halo, medi_prop, medi_low, medi_high = Common_module.halo_value_list(Stellar_all, property_plot, mean=True)

micro_HI, micro_HI_central, micro_HI_satellite, micro_HI_orphan, micro_vir = plotting_merge_runs_micro[0].mergeValues_satellite('id_halo','matom_bulge','matom_disk')

micro_stellar, micro_stellar_central, micro_stellar_satellite, micro_stellar_orphan, a = plotting_merge_runs_micro[0].mergeValues_satellite('id_halo','mstars_bulge','mstars_disk')

HI_all 			= micro_HI[micro_vir <= 10**11.3]
Stellar_all 	= micro_vir[micro_vir <= 10**11.3]

property_plot 	= HI_all/micro_vir[micro_vir <= 10**11.3]


Common_module.plotting_properties_halo(Stellar_all,property_plot,mean=False,legend_handles=legendHandles,colour_line='r',fill_between=True,xlim_lower=10.2,xlim_upper=14.8,ylim_lower=-5,ylim_upper=-1.2,legend_name=shark_labels[0], property_name_x='log_{10}(M_{vir}[M_{\odot}])', property_name_y='log_{10}(M_{HI}/M_{vir})', resolution = False, first_legend=False)


micro_halo, micro_prop, micro_low, micro_high = Common_module.halo_value_list(Stellar_all, property_plot, mean=True)


for i,j in zip(range(1,len(plotting_merge_runs)),range(1,len(plotting_merge_runs))):
	medi_HI, medi_HI_central, medi_HI_satellite,medi_HI_orphan,medi_vir = plotting_merge_runs[i].mergeValues_satellite('id_halo','matom_bulge','matom_disk')

	medi_stellar, medi_stellar_central, medi_stellar_satellite,medi_stellar_orphan,a = plotting_merge_runs[i].mergeValues_satellite('id_halo','mstars_bulge','mstars_disk')

	HI_all 			= medi_HI[medi_vir >= 10**11.15]
	Stellar_all 	= medi_vir[medi_vir >= 10**11.15]

	property_plot 	= HI_all/medi_vir[medi_vir >= 10**11.15]

	Common_module.plotting_properties_halo(Stellar_all,property_plot,mean=False,legend_handles=legendHandles,colour_line=colour_plot[i],fill_between=False,xlim_lower=10,xlim_upper=14.5,ylim_lower=-4,ylim_upper=12,legend_name=shark_labels[i], property_name_y='log_{10}(M_{HI}/M_{vir})',property_name_x='log_{10}(M_{vir}[M_{\odot}])', resolution = False, first_legend=True)


	micro_HI, micro_HI_central, micro_HI_satellite, micro_HI_orphan, micro_vir = plotting_merge_runs_micro[i].mergeValues_satellite('id_halo','matom_bulge','matom_disk')

	micro_stellar, micro_stellar_central, micro_stellar_satellite, micro_stellar_orphan, a = plotting_merge_runs_micro[i].mergeValues_satellite('id_halo','mstars_bulge','mstars_disk')

	HI_all 			= micro_HI[micro_vir <= 10**11.3]
	Stellar_all 	= micro_vir[micro_vir <= 10**11.3]

	property_plot 	= HI_all/micro_vir[micro_vir <= 10**11.3]


	Common_module.plotting_properties_halo(Stellar_all,property_plot,mean=False,legend_handles=legendHandles,colour_line=colour_plot[i],fill_between=False,xlim_lower=9.5,xlim_upper=14.8,ylim_lower=-5,ylim_upper=-1.5,legend_name=shark_labels[i], property_name_x='log_{10}(M_{vir}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])', resolution = True, first_legend=False)

legendHandles.append(mlines.Line2D([],[],color='black',linestyle='-',label='All'))
legendHandles.append(mlines.Line2D([],[],color='black',linestyle='--',label='Centrals'))
legendHandles.append(mlines.Line2D([],[],color='black',linestyle='-.',label='Satellites'))
		
plt.axvline(x=11.2, linestyle=':', color = 'k')
leg = plt.legend(handles=legendHandles[4:6],loc='upper right', frameon=False)
plt.gca().add_artist(leg)
leg = plt.legend(handles=legendHandles[6:9],loc='lower right', frameon=False)
plt.gca().add_artist(leg)

plt.legend(handles=legendHandles[0:4], loc='upper left')

############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
### Separate Plot
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************


####*********************************************************

f8_ax2 = fig8.add_subplot(gs1[-1, :],sharex=f8_ax1)

####*********************************************************


''' stellar_property = {'all' : stellar_all, 'centrals': stellar_central, 'satellites': stellar_satellites}   '''
medi_HI, medi_HI_central, medi_HI_satellite,a,medi_vir = plotting_merge_runs[0].mergeValues_satellite('id_halo','matom_bulge','matom_disk')

medi_stellar, medi_stellar_central, medi_stellar_satellite,a,a = plotting_merge_runs[0].mergeValues_satellite('id_halo','mstars_bulge','mstars_disk')


virial_mass 	= {'all' : medi_vir[medi_vir >= 10**11.15], 'central' : medi_vir[medi_vir >= 10**11.15], 'satellite' : medi_vir[medi_vir >= 10**11.15]}

stellar_property = {'all' : medi_stellar[medi_vir >= 10**11.15], 'central' : medi_stellar_central[medi_vir >= 10**11.15], 'satellite' : medi_stellar_satellite[medi_vir >= 10**11.15]}

property_plot = {'all' : medi_HI[medi_vir >= 10**11.15]/medi_vir[medi_vir >= 10**11.15], 'central' : medi_HI_central[medi_vir >= 10**11.15]/medi_vir[medi_vir >= 10**11.15], 'satellite' : medi_HI_satellite[medi_vir >= 10**11.15]/medi_vir[medi_vir >= 10**11.15]}

# property_plot = {'all' : medi_HI[medi_vir >= 10**11.15], 'central' : medi_HI_central[medi_vir >= 10**11.15], 'satellite' : medi_HI_satellite[medi_vir >= 10**11.15]}


Common_module.plotting_properties_separate_merged(virial_mass,stellar_property=virial_mass,property_plot = property_plot,mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{vir}[M_{\odot}])', property_name_y='log_{10}(M_{HI}/M_{vir})',xlim_lower=10,xlim_upper=15,ylim_lower=-5,ylim_upper=-0.5, first_legend=False,colour_line='r',legend_name=shark_labels[0], resolution=True)


micro_HI, micro_HI_central, micro_HI_satellite, micro_HI_orphan, micro_vir = micro_Kappa_original.mergeValues_satellite('id_halo','matom_bulge','matom_disk')

micro_stellar, micro_stellar_central, micro_stellar_satellite, micro_stellar_orphan, a = micro_Kappa_original.mergeValues_satellite('id_halo','mstars_bulge','mstars_disk')

virial_mass 	= {'all' : micro_vir[micro_vir <= 10**11.3], 'central' : micro_vir[micro_vir <= 10**11.3], 'satellite' : micro_vir[micro_vir <= 10**11.3]}

stellar_property = {'all' : micro_stellar[micro_vir <= 10**11.3], 'central' : micro_stellar_central[micro_vir <= 10**11.3], 'satellite' : micro_stellar_satellite[micro_vir <= 10**11.3]}

property_plot = {'all' : micro_HI[micro_vir <= 10**11.3]/micro_vir[micro_vir <= 10**11.3], 'central' : micro_HI_central[micro_vir <= 10**11.3]/micro_vir[micro_vir <= 10**11.3], 'satellite' : micro_HI_satellite[micro_vir <= 10**11.3]/micro_vir[micro_vir <= 10**11.3]}


# property_plot = {'all' : micro_HI[micro_vir <= 10**11.3], 'central' : micro_HI_central[micro_vir <= 10**11.3], 'satellite' : micro_HI_satellite[micro_vir <= 10**11.3]}


Common_module.plotting_properties_separate_merged(virial_mass,stellar_property=virial_mass,property_plot = property_plot,mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{vir}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=6,ylim_upper=11.9, first_legend=False,colour_line='r',legend_name=shark_labels[0])


medi_HI, medi_HI_central, medi_HI_satellite,medi_HI_orphan,medi_vir = plotting_merge_runs[4].mergeValues_satellite('id_halo','matom_bulge','matom_disk')

medi_stellar, medi_stellar_central, medi_stellar_satellite,medi_stellar_orphan,a = plotting_merge_runs[4].mergeValues_satellite('id_halo','mstars_bulge','mstars_disk')

virial_mass 	= {'all' : medi_vir[medi_vir >= 10**11.15], 'central' : medi_vir[medi_vir >= 10**11.15], 'satellite' : medi_vir[medi_vir >= 10**11.15]}

# stellar_property = {'all' : medi_stellar[medi_vir >= 10**11.15], 'central' : medi_stellar_central[medi_vir >= 10**11.15], 'satellite' : medi_stellar_satellite[medi_vir >= 10**11.15]}

property_plot = {'all' : medi_HI[medi_vir >= 10**11.15]/medi_vir[medi_vir >= 10**11.15], 'central' : medi_HI_central[medi_vir >= 10**11.15]/medi_vir[medi_vir >= 10**11.15], 'satellite' : medi_HI_satellite[medi_vir >= 10**11.15]/medi_vir[medi_vir >= 10**11.15]}

# property_plot = {'all' : medi_HI[medi_vir >= 10**11.15], 'central' : medi_HI_central[medi_vir >= 10**11.15], 'satellite' : medi_HI_satellite[medi_vir >= 10**11.15]}


Common_module.plotting_properties_separate_merged(virial_mass,stellar_property=virial_mass,property_plot = property_plot,mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{vir}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=-5,ylim_upper=-0.5, first_legend=False,colour_line=colour_plot[4],legend_name=shark_labels[4], resolution=True)

micro_HI, micro_HI_central, micro_HI_satellite, micro_HI_orphan, micro_vir = plotting_merge_runs_micro[4].mergeValues_satellite('id_halo','matom_bulge','matom_disk')

micro_stellar, micro_stellar_central, micro_stellar_satellite, micro_stellar_orphan, a = plotting_merge_runs_micro[4].mergeValues_satellite('id_halo','mstars_bulge','mstars_disk')


virial_mass 	= {'all' : micro_vir[micro_vir <= 10**11.3], 'central' : micro_vir[micro_vir <= 10**11.3], 'satellite' : micro_vir[micro_vir <= 10**11.3]}

stellar_property = {'all' : micro_stellar[micro_vir <= 10**11.3], 'central' : micro_stellar_central[micro_vir <= 10**11.3], 'satellite' : micro_stellar_satellite[micro_vir <= 10**11.3]}

property_plot = {'all' : micro_HI[micro_vir <= 10**11.3]/micro_vir[micro_vir <= 10**11.3], 'central' : micro_HI_central[micro_vir <= 10**11.3]/micro_vir[micro_vir <= 10**11.3], 'satellite' : micro_HI_satellite[micro_vir <= 10**11.3]/micro_vir[micro_vir <= 10**11.3]}

# property_plot = {'all' : micro_HI[micro_vir <= 10**11.3], 'central' : micro_HI_central[micro_vir <= 10**11.3], 'satellite' : micro_HI_satellite[micro_vir <= 10**11.3]}

Common_module.plotting_properties_separate_merged(virial_mass,stellar_property=virial_mass,property_plot = property_plot,mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{vir}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=9.5,xlim_upper=14.8,ylim_lower=-5,ylim_upper=-1.5, first_legend=True,colour_line=colour_plot[4],legend_name=shark_labels[0])


plt.axvline(x=11.2, linestyle=':', color = 'k')
# plt.fill_between([0,12], 6, 12, color='b', alpha=0.1,hatch='o')
# plt.fill_between([12,13],6,12,color='g',alpha=0.1, hatch='*')
# plt.fill_between([13,15],6,12,color='r',alpha=0.1, hatch='x')


plt.savefig("Referee_photoionisation.png")
plt.show()




