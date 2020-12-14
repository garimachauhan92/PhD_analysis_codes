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
shark_labels	= ['Lagos18 (Default)','Kappa = 0.02','Kappa = 0','Kappa = 1','Lagos18 (stripping off)']




simulation 		= ['medi-SURFS', 'micro-SURFS']
snapshot_avail	= [199,174,156,131]
z_values 		= [0,0.5,1,2]

subvolumes 		= 64

colour_plot 	= ['maroon', 'mediumorchid', 'dodgerblue', 'limegreen']
colour_plot 	= Common_module.colour_scheme(n = 20,colourmap=plt.cm.tab20c)
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
legend_handles = []
for snapshot,i,j in zip(snapshot_avail,range(12,16), range(4)):
	micro_HI, micro_HI_central, micro_HI_satellite, micro_HI_orphan, micro_vir = micro_Kappa_original[snapshot].mergeValues_satellite('id_halo','matom_bulge','matom_disk')


	medi_HI, medi_HI_central, medi_HI_satellite, medi_HI_orphan, medi_vir = medi_Kappa_original[snapshot].mergeValues_satellite('id_halo','matom_bulge','matom_disk')


	indices_sorted 				= medi_vir.argsort()
	sort_medi_vir  				= medi_vir[indices_sorted]
	sort_medi_HI 				= medi_HI[indices_sorted]

	indices_sorted_micro 		= micro_vir.argsort()
	sort_micro_vir 				= micro_vir[indices_sorted_micro]
	sort_micro_HI 				= micro_HI[indices_sorted_micro]

	append_array_micro 			= np.where(sort_micro_vir[sort_micro_vir <= 10**11.2])[0]
	append_array_medi 			= np.where(sort_medi_vir[sort_medi_vir > 10**11.2])[0]

	# final_cumulative_HI 		= []
	# final_cumulative_vir 		= []

	# final_cumulative_HI 		= np.append(final_cumulative_HI,sort_micro_HI[append_array_micro])
	# final_cumulative_HI 		= np.append(final_cumulative_HI,sort_medi_HI[append_array_medi])

	# final_cumulative_vir 		= np.append(final_cumulative_vir,sort_micro_vir[append_array_micro])
	# final_cumulative_vir 		= np.append(final_cumulative_vir,sort_medi_vir[append_array_medi])

	# final_cumulative_HI 		= np.append(sort_micro_HI[append_array_micro], sort_medi_HI[append_array_medi], axis=0)
	# final_cumulative_vir 		= np.append(sort_micro_vir[append_array_micro], sort_medi_vir[append_array_medi], axis=0)


	trial_cumsum_HI_micro		= (np.cumsum(sort_micro_HI))/max(np.cumsum(sort_micro_HI))*100
	trial_cumsum_HI_medi		= (np.cumsum(sort_medi_HI))/max(np.cumsum(sort_medi_HI))*100


	plt.plot(np.log10(sort_micro_vir[sort_micro_vir < 10**11.2]), trial_cumsum_HI_micro[sort_micro_vir < 10**11.2], c= colour_plot[i])
	plt.plot(np.log10(sort_medi_vir[sort_medi_vir > 10**11.2]), trial_cumsum_HI_medi[sort_medi_vir > 10**11.2], c = colour_plot[i])
	# plt.fill_between(np.log10(sort_medi_vir[sort_medi_vir > 10**11.2]),trial_cumsum_HI_medi[sort_medi_vir > 10**11.2], alpha = 0.2, color = 'red')
	legend_handles.append(mpatches.Patch(color=colour_plot[i],label='z = %s'%z_values[j]))


plt.axvline(x=11.2, linestyle=':', color = 'k')
plt.plot([12,12], [00,70], linestyle = '-.', color = 'r')
plt.plot([00,12], [70,70], linestyle = '-.', color = 'r')

plt.plot([13,13], [0,88.25], linestyle = '-.', color = 'g')
plt.plot([00,13], [88.25,88.25], linestyle = '-.', color = 'g')

plt.xlabel('$log_{10}(M_{vir}[M_{\odot}])$')
plt.ylabel('Percentage HI')
plt.xlim(10.2,15)
plt.ylim(0,105)
plt.legend(handles = legend_handles)

plt.savefig('new_plots_corrected/Cumulative_HI_plot_all.png')
plt.show()