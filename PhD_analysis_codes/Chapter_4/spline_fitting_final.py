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
import matplotlib.colors as colors
from Common_module import SharkDataReading
import Common_module
from scipy.stats import norm
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit



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

with open("/home/ghauhan/Parameter_files/redshift_list_medi.txt", 'r') as csv_file:  
	# with open("/home/garima/Desktop/redshift_list_medi.txt", 'r') as csv_file:  
	trial = list(csv.reader(csv_file, delimiter=',')) 
	trial = np.array(trial[1:], dtype = np.float) 



simulation 		= ['medi-SURFS', 'micro-SURFS']
snapshot_avail	= [x for x in range(100,200,1)]
z_values 		= ["%0.2f" %x for x in trial[:,1]]

subvolumes 		= 64

simulation 		= ['medi-SURFS', 'micro-SURFS']
snapshot_avail	= [199,174,156,131]
z_values 		= [0,0.5,1,2]

subvolumes 		= 64


#--------------------------------------------------------------------------------


medi_Kappa_original 	= {}
medi_Kappa_stripping 	= {}

micro_Kappa_original 	= {}
micro_Kappa_stripping 	= {}


for snapshot in snapshot_avail:

	medi_Kappa_original[snapshot]	 	= SharkDataReading(path,simulation[0],shark_runs[0],snapshot,subvolumes)
	
	# micro_Kappa_original[snapshot]		= SharkDataReading(path,simulation[1],shark_runs[0],snapshot,subvolumes)
	
#--------------------------------------------------------------------------------



###################################################
######--------------------------------------------------------------------------------------------------------------------------------------
### Preparing Data
######--------------------------------------------------------------------------------------------------------------------------------------
####****************************************************************************************************************************************






def plotting_properties_halo_model(virial_mass,property_plot,sigma,mean,legend_handles,property_name_x='X-Label-math-mode',legend_name='name-of-run', property_name_y='Y-Label-math-mode', xlim_lower=6,xlim_upper=15,ylim_lower=-4,ylim_upper=11,colour_line='r',fill_between=False, first_legend = False, resolution=False):
	
	bin_for_disk = np.arange(7,15,0.2)

	# bin_for_disk =  [9.0,10.5,11.15,11.25,11.4,11.6,11.7,12.4,13.0,14.6,15]

	halo_mass 		= np.zeros(len(bin_for_disk))
	prop_mass 		= np.zeros(len(bin_for_disk))

	prop_mass_low 		= np.zeros(len(bin_for_disk))
	prop_mass_high		= np.zeros(len(bin_for_disk))
	

	if mean == True:
		for i in range(1,len(bin_for_disk)):
			halo_mass[i-1] 		= np.log10(np.mean(virial_mass[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])]))
			prop_mass[i-1]			= np.log10(np.mean(property_plot[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])]))

		
			prop_mass_low[i-1]	= prop_mass[i-1] - np.mean(sigma[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])])
			prop_mass_high[i-1]	= np.mean(sigma[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])]) + prop_mass[i-1]
			
	
	else:
		for i in range(1,len(bin_for_disk)):
			halo_mass[i-1] 		= np.log10(np.median(virial_mass[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])]))
			prop_mass[i-1]			= np.log10(np.median(property_plot[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])]))
			
			prop_mass_low[i-1]	= prop_mass[i-1] - np.mean(sigma[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])])
			prop_mass_high[i-1]	= np.mean(sigma[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])]) + prop_mass[i-1]
			

	if resolution == True:
		plt.plot(halo_mass[0:len(halo_mass)-1],prop_mass[0:len(halo_mass)-1],color=colour_line, linestyle='-')
		# plt.plot(halo_mass[0:len(halo_mass)-1],prop_mass[0:len(halo_mass)-1],color=colour_line, label='$\\rm %s$' %legend_name)

		if first_legend == True:
			legend_handles.append(mlines.Line2D([],[],color='black',linestyle='-',label='Medi-SURFS'))
			legend_handles.append(mlines.Line2D([],[],color='black',linestyle='-.',label='Micro-SURFS'))
			legend_handles.append(mpatches.Patch(color=colour_line,label=legend_name))

		
		else:
			mpatches.Patch(color=colour_line, label=legend_name)
			legend_handles.append(mpatches.Patch(color=colour_line,label=legend_name))

	else:
		plt.plot(halo_mass[0:len(halo_mass)-1],prop_mass[0:len(halo_mass)-1],color=colour_line, label='$\\rm %s$' %legend_name)


	if fill_between == True:
		plt.fill_between(halo_mass[0:len(halo_mass)-5],prop_mass_low[0:len(halo_mass)-5],prop_mass_high[0:len(halo_mass)-5],color=colour_line,alpha=0.1)
	
	plt.xlabel('$\\rm %s$ '%property_name_x)
	plt.ylabel('$\\rm %s$ '%property_name_y)

	plt.xlim(xlim_lower,xlim_upper)
	plt.ylim(ylim_lower,ylim_upper)

	

	#plt.legend(frameon=False)
	# plt.savefig('%s.png'%figure_name)
	# plt.show()







def prepare_data(path,simulation,run,snapshot,subvolumes):

	#fields_read = {'galaxies' : ('type','mstars_disk', 'mstars_bulge', 'matom_disk','matom_bulge' , 'mgas_disk', 'mgas_bulge','mvir_hosthalo')}

	fields_read 	= {'halo' : ('age_50', 'age_80', 'lambda','mvir'), 'galaxies':('type', 'm_bh', 'mstars_bulge', 'mstars_disk')}

	data_reading = SharkDataReading(path,simulation,run,snapshot,subvolumes)

	data_plotting = data_reading.readIndividualFiles(fields=fields_read)

	return data_plotting




######------------------------------------------------------------------------------------------------
### Micro-SURFS - 
######------------------------------------------------------------------------------------------------


# micro_Kappa_original_HI 				= {}
# micro_Kappa_original_HI_central 		= {}
# micro_Kappa_original_HI_satellite 		= {}
# micro_Kappa_original_HI_orphan  		= {}
# micro_Kappa_original_vir 				= {}

# micro_Kappa_original_stellar 			= {}
# micro_Kappa_original_stellar_central 	= {}
# micro_Kappa_original_stellar_satellite 	= {}
# micro_Kappa_original_stellar_orphan		= {}

# micro_substructure_original				= {}


# for k in snapshot_avail[97:99]:
# 	micro_Kappa_original_HI[k], micro_Kappa_original_HI_central[k], micro_Kappa_original_HI_satellite[k],micro_Kappa_original_HI_orphan[k],micro_Kappa_original_vir[k] = micro_Kappa_original[k].mergeValues_satellite('id_halo','matom_bulge','matom_disk')

# 	# micro_Kappa_original_stellar[k], micro_Kappa_original_stellar_central[k], micro_Kappa_original_stellar_satellite[k],micro_Kappa_original_HI_orphan[k],a = micro_Kappa_original[k].mergeValues_satellite('id_halo','mstars_bulge','mstars_disk')

# 	micro_substructure_original[k],a = micro_Kappa_original[k].mergeValuesNumberSubstructures('id_halo','id_subhalo')

	


# #####------------------------------------------------------------------------------------------------
# ## Medi-SURFS 
# #####------------------------------------------------------------------------------------------------

medi_Kappa_original_HI 					= {}
medi_Kappa_original_HI_central 			= {}
medi_Kappa_original_HI_satellite 		= {}
medi_Kappa_original_HI_orphan 			= {}
medi_Kappa_original_vir 				= {}

medi_Kappa_original_stellar 			= {}
medi_Kappa_original_stellar_central 	= {}
medi_Kappa_original_stellar_satellite 	= {}
medi_Kappa_original_stellar_orphan 		= {}
medi_substructure_original				= {}

medi_bh 								= {}
medi_bh_central 						= {}
medi_bh_satellite  						= {}

medi_subhalo  							= {}
medi_subhalo_central  					= {}
medi_subhalo_satellite 					= {}



for k in snapshot_avail:
	medi_Kappa_original_HI[k], medi_Kappa_original_HI_central[k], medi_Kappa_original_HI_satellite[k],medi_Kappa_original_HI_orphan[k],medi_Kappa_original_vir[k] = medi_Kappa_original[k].mergeValues_satellite('id_halo','matom_bulge','matom_disk')

	# medi_Kappa_original_stellar[k], medi_Kappa_original_stellar_central[k], medi_Kappa_original_stellar_satellite[k],medi_Kappa_original_stellar_orphan[k],a = medi_Kappa_original[k].mergeValues_satellite('id_halo','mstars_bulge','mstars_disk')

	# medi_bh[k], medi_bh_central[k], medi_bh_satellite[k],a,medi_vir = medi_Kappa_original[k].mergeValues_satellite('id_halo','m_bh','matom_disk', baryon_property=False)

	medi_subhalo[k], medi_subhalo_central[k], medi_subhalo_satellite[k],a,medi_vir = medi_Kappa_original[k].mergeValues_satellite('id_halo','mvir_subhalo','matom_disk', baryon_property=False)



############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
### Analysis
######--------------------------------------------------------------------------------------------------------------------------------------
####****************************************************************************************************************************************

# plotting_merge_runs 	= [medi_Kappa_original[k],medi_Kappa_stripping[k] ]
# plotting_merge_runs_micro = [micro_Kappa_original, micro_Kappa_10, micro_Kappa_0, micro_Kappa_1, micro_Kappa_stripping ]


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
	


# plt = Common_module.load_matplotlib()
# colour_plot = Common_module.colour_scheme()

# legendHandles = list()

# age_50_micro 		= {}
# age_80_micro 		= {}
# lambda_spin_micro 	= {}
# mvir_micro 			= {}
# is_central_micro  	= {}
# m_bh_micro 			= {}
# mstars_bulge_micro 	= {}
# mstars_disk_micro 	= {}

# for snapshot in snapshot_avail:

# 	(h0,_, age_50_micro[snapshot], age_80_micro[snapshot], lambda_spin_micro[snapshot], mvir_micro[snapshot], is_central_micro[snapshot], m_bh_micro[snapshot], mstars_bulge_micro[snapshot], mstars_disk_micro[snapshot]) = prepare_data(path,simulation[1],shark_runs[0],snapshot,subvolumes)




####****************************************************************************************************************************************
####****************************************************************************************************************************************
####****************************************************************************************************************************************

parameter_list_low 		= {}
parameter_list_high	 	= {}
parameter_list_med	 	= {}


for k,j in zip(snapshot_avail,z_values):

	print(k)

	HI_all 		=	np.log10(medi_Kappa_original_HI[k][(medi_Kappa_original_vir[k] >= 10**10) & (medi_Kappa_original_vir[k] < 10**11.9) & (medi_Kappa_original_HI_central[k] >= 10**6) ])
	M_vir_plotting = np.log10(medi_Kappa_original_vir[k][(medi_Kappa_original_vir[k] >= 10**10) & (medi_Kappa_original_vir[k] < 10**11.9) & (medi_Kappa_original_HI_central[k] >= 10**6) ])

	lambda_spin_plotting = np.log10(lambda_spin[k][(medi_Kappa_original_vir[k] >= 10**10) & (medi_Kappa_original_vir[k] < 10**11.9) & (medi_Kappa_original_HI_central[k] >= 10**6) ])

	genesis_halo, genesis_prop, genesis_low, genesis_high = Common_module.halo_value_list(10**M_vir_plotting, 10**HI_all, mean=False)

	# genesis_prop = genesis_prop[~np.isnan(genesis_halo)]
	# genesis_halo = genesis_halo[~np.isnan(genesis_halo)]

	legendHandles = []


	def func(x,a,b,c):
		return a + b*x + c*x**2

	popt, pcov	=	curve_fit(func, genesis_halo[3:10], genesis_prop[3:10])
	# popt, pcov	=	curve_fit(func, genesis_halo[0:10], genesis_prop[0:10])

	parameter_list_low[k] = popt

	trial_HI_2  = func(M_vir_plotting, *popt)
	# trial_HI 	= func(genesis_halo[25:35], *popt)

	
	# sigma_all = 0.25*np.ones(len(trial_HI))
	sigma_all_2 = 0.25*np.ones(len(trial_HI_2))
	
	
	
	plotting_properties_halo_model(10**M_vir_plotting,10**trial_HI_2,mean=False,sigma=sigma_all_2,legend_handles=legendHandles,colour_line='green',fill_between=True,xlim_lower=10,xlim_upper=15,ylim_lower=7,ylim_upper=11,legend_name='Model Fit', property_name_x='log_{10}(M_{vir}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])', resolution = True, first_legend=False)


	Common_module.plotting_properties_halo(10**M_vir_plotting,10**HI_all,mean=False,legend_handles=legendHandles,colour_line=colour_plot[0],fill_between=True,xlim_lower=10.2,xlim_upper=14.5,ylim_lower=7,ylim_upper=11.2,legend_name=shark_labels[0], property_name_x='log_{10}(M_{HI}[M_{\odot}])', property_name_y='log_{10}(M_{vir}[M_{\odot}])', resolution = True, first_legend=False)


	residual_HI 	= HI_all - trial_HI_2 


	
	np.savetxt("/mnt/su3ctm/gchauhan/Environmental_trial/parameter_files/csv_files_parameters/lambda_spin_residual_%s.csv" %k, lambda_spin_plotting, delimiter=",")
	np.savetxt("/mnt/su3ctm/gchauhan/Environmental_trial/parameter_files/csv_files_parameters/HI_low_residual_%s.csv" %k, HI_all, delimiter=",")
	np.savetxt("/mnt/su3ctm/gchauhan/Environmental_trial/parameter_files/csv_files_parameters/M_vir_low_residual_%s.csv" %k, M_vir_plotting, delimiter=",")
	np.savetxt("/mnt/su3ctm/gchauhan/Environmental_trial/parameter_files/csv_files_parameters/residual_HI_low_%s.csv" %k, residual_HI, delimiter=",")

	


	HI_all 		=	np.log10(medi_Kappa_original_HI[k][(medi_Kappa_original_vir[k] >= 10**11.8) & (medi_Kappa_original_vir[k] < 10**13.1) & (medi_Kappa_original_HI_central[k] >= 10**6) ])
	M_vir_plotting = np.log10(medi_Kappa_original_vir[k][(medi_Kappa_original_vir[k] >= 10**11.8) & (medi_Kappa_original_vir[k] < 10**13.1) & (medi_Kappa_original_HI_central[k] >= 10**6) ])

	fraction_vir = (medi_subhalo_satellite[k][(medi_Kappa_original_vir[k] >= 10**11.8) & (medi_Kappa_original_vir[k] < 10**13.1) & (medi_Kappa_original_HI_central[k] >= 10**6) ]/medi_Kappa_original_vir[k][(medi_Kappa_original_vir[k] >= 10**11.8) & (medi_Kappa_original_vir[k] < 10**13.1) & (medi_Kappa_original_HI_central[k] >= 10**6)])

	lambda_spin_plotting = np.log10(lambda_spin[k][(medi_Kappa_original_vir[k] >= 10**11.8) & (medi_Kappa_original_vir[k] < 10**13.1) & (medi_Kappa_original_HI_central[k] >= 10**6) ])

	genesis_halo, genesis_prop, genesis_low, genesis_high = Common_module.halo_value_list(10**M_vir_plotting[fraction_vir > 0], 10**HI_all[fraction_vir > 0], mean=False)

	# genesis_prop = genesis_prop[~np.isnan(genesis_halo)]
	# genesis_halo = genesis_halo[~np.isnan(genesis_halo)]

	

	HI_all 					= HI_all[fraction_vir > 0]
	M_vir_plotting			= M_vir_plotting[fraction_vir > 0]
	lambda_spin_plotting	= lambda_spin_plotting[fraction_vir > 0]


	fraction_vir_plotting	= np.log10(fraction_vir[fraction_vir > 0])


	legendHandles = []


	def func(x,a,b,c,d,e,f):
		return a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5

	# def func(x,a,b):
	# 	return a + b*x 


	popt, pcov	=	curve_fit(func, genesis_halo[9:16], genesis_prop[9:16])

	parameter_list_med[k] = popt

	trial_HI_2  = func(M_vir_plotting, *popt)
	# trial_HI 	= func(genesis_halo[34:41], *popt)

	residual_HI 	= HI_all - trial_HI_2 


	# sigma_all = 0.25*np.ones(len(trial_HI))
	sigma_all_2 = 0.25*np.ones(len(trial_HI_2))
	
	
	plotting_properties_halo_model(10**M_vir_plotting,10**trial_HI_2,mean=False,sigma=sigma_all_2,legend_handles=legendHandles,colour_line='green',fill_between=True,xlim_lower=10,xlim_upper=15,ylim_lower=7,ylim_upper=11,legend_name='Model Fit', property_name_x='log_{10}(M_{vir}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])', resolution = True, first_legend=False)


	Common_module.plotting_properties_halo(10**M_vir_plotting,10**HI_all,mean=False,legend_handles=legendHandles,colour_line=colour_plot[0],fill_between=True,xlim_lower=10.2,xlim_upper=14.5,ylim_lower=7,ylim_upper=11.2,legend_name=shark_labels[0], property_name_x='log_{10}(M_{HI}[M_{\odot}])', property_name_y='log_{10}(M_{vir}[M_{\odot}])', resolution = True, first_legend=False)


	# extra = mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="z = 0"  )
	# #legendHandles.append(extra)
	# plt.title('z = 0')


	# residual_HI 	= HI_all - trial_HI_2 
	# bounds = np.logspace(np.log10(0.01), np.log10(0.5), 100)
	# norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
		
	# plt.scatter(M_vir_plotting, residual_HI, c = (lambda_spin_plotting), s = 0.1,norm=norm, cmap = 'Spectral')
	# cbar = plt.colorbar()
	# plt.show()

	# plt.scatter(M_vir_plotting, residual_HI, c = (fraction_vir_plotting), s = 0.1, norm=norm, cmap = 'Spectral')
	# cbar = plt.colorbar()
	# plt.show()


	np.savetxt("/mnt/su3ctm/gchauhan/Environmental_trial/parameter_files/csv_files_parameters/lambda_spin_transition_residual_%s.csv" %k, lambda_spin_plotting, delimiter=",")
	np.savetxt("/mnt/su3ctm/gchauhan/Environmental_trial/parameter_files/csv_files_parameters/HI_transition_residual_%s.csv" %k, HI_all, delimiter=",")
	np.savetxt("/mnt/su3ctm/gchauhan/Environmental_trial/parameter_files/csv_files_parameters/M_vir_transition_residual_%s.csv" %k, M_vir_plotting, delimiter=",")
	np.savetxt("/mnt/su3ctm/gchauhan/Environmental_trial/parameter_files/csv_files_parameters/residual_HI_transition_%s.csv" %k, residual_HI, delimiter=",")
	np.savetxt("/mnt/su3ctm/gchauhan/Environmental_trial/parameter_files/csv_files_parameters/fraction_vir_transition_%s.csv" %k, fraction_vir_plotting, delimiter=",")

	HI_all 		=	np.log10(medi_Kappa_original_HI[k][(medi_Kappa_original_vir[k] >= 10**12.9) & (medi_Kappa_original_vir[k] < 10**15) & (medi_Kappa_original_HI_central[k] >= 10**6) ])
	M_vir_plotting = np.log10(medi_Kappa_original_vir[k][(medi_Kappa_original_vir[k] >= 10**12.9) & (medi_Kappa_original_vir[k] < 10**15) & (medi_Kappa_original_HI_central[k] >= 10**6) ])

	fraction_vir = (medi_subhalo_satellite[k][(medi_Kappa_original_vir[k] >= 10**12.9) & (medi_Kappa_original_vir[k] < 10**15) & (medi_Kappa_original_HI_central[k] >= 10**6) ]/medi_Kappa_original_vir[k][(medi_Kappa_original_vir[k] >= 10**12.9) & (medi_Kappa_original_vir[k] < 10**15) & (medi_Kappa_original_HI_central[k] >= 10**6)])

	lambda_spin_plotting = np.log10(lambda_spin[k][(medi_Kappa_original_vir[k] >= 10**12.9) & (medi_Kappa_original_vir[k] < 10**15) & (medi_Kappa_original_HI_central[k] >= 10**6) ])

	genesis_halo, genesis_prop, genesis_low, genesis_high = Common_module.halo_value_list(10**M_vir_plotting[fraction_vir > 0], 10**HI_all[fraction_vir > 0], mean=False)

	# genesis_prop = genesis_prop[~np.isnan(genesis_halo)]
	# genesis_halo = genesis_halo[~np.isnan(genesis_halo)]

	HI_all 					= HI_all[fraction_vir > 0]
	M_vir_plotting			= M_vir_plotting[fraction_vir > 0]
	lambda_spin_plotting	= lambda_spin_plotting[fraction_vir > 0]


	fraction_vir_plotting	= np.log10(fraction_vir[fraction_vir > 0])


	legendHandles = []


	# def func(x,a,b,c,d):
	# 	return a + b*x + c*x**2 + d*x**3

	def func(x,a,b):
		return a + b*x 


	popt, pcov	=	curve_fit(func, genesis_halo[15:21], genesis_prop[15:21])

	parameter_list_high[k] = popt

	trial_HI_2  = func(M_vir_plotting, *popt)
	# trial_HI 	= func(genesis_halo[40:49], *popt)

	residual_HI 	= HI_all - trial_HI_2 


	# sigma_all = 0.25*np.ones(len(trial_HI))
	sigma_all_2 = 0.25*np.ones(len(trial_HI_2))
	
	
	plotting_properties_halo_model(10**M_vir_plotting,10**trial_HI_2,mean=False,sigma=sigma_all_2,legend_handles=legendHandles,colour_line='green',fill_between=True,xlim_lower=10,xlim_upper=15,ylim_lower=7,ylim_upper=11,legend_name='Model Fit', property_name_x='log_{10}(M_{vir}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])', resolution = True, first_legend=False)


	Common_module.plotting_properties_halo(10**M_vir_plotting,10**HI_all,mean=False,legend_handles=legendHandles,colour_line=colour_plot[0],fill_between=True,xlim_lower=10.2,xlim_upper=14.5,ylim_lower=7,ylim_upper=11.2,legend_name=shark_labels[0], property_name_x='log_{10}(M_{HI}[M_{\odot}])', property_name_y='log_{10}(M_{vir}[M_{\odot}])', resolution = True, first_legend=False)





	np.savetxt("/mnt/su3ctm/gchauhan/Environmental_trial/parameter_files/csv_files_parameters/HI_high_residual_%s.csv" %k, HI_all, delimiter=",")
	np.savetxt("/mnt/su3ctm/gchauhan/Environmental_trial/parameter_files/csv_files_parameters/M_vir_high_residual_%s.csv" %k, M_vir_plotting, delimiter=",")
	np.savetxt("/mnt/su3ctm/gchauhan/Environmental_trial/parameter_files/csv_files_parameters/residual_HI_high_%s.csv" %k, residual_HI, delimiter=",")
	np.savetxt("/mnt/su3ctm/gchauhan/Environmental_trial/parameter_files/csv_files_parameters/fraction_vir_high_%s.csv" %k, fraction_vir_plotting, delimiter=",")



# path_save = '/mnt/sshfs/pleiades_gchauhan/Environmental_trial/parameter_files/'
# path_save = '/mnt/su3ctm/gchauhan/Environmental_trial/parameter_files/'

# np.save(path_save + "parameter_list_high.npy", parameter_list_high)
# np.save(path_save + "parameter_list_med.npy", parameter_list_med)
# np.save(path_save + "parameter_list_low.npy", parameter_list_low)



####****************************************************************************************************************************************
####****************************************************************************************************************************************
####****************************************************************************************************************************************
