import h5py as h5py
import numpy as np
import os
import scipy.stats as scipy
import re
from scipy import stats
from astropy.stats import bootstrap
import csv as csv
from astropy.utils import NumpyRNGContext
import seaborn as sns
import collections
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from Common_module import SharkDataReading
import Common_module
import matplotlib as matplotlib



############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
### Defining data_type and constants
######--------------------------------------------------------------------------------------------------------------------------------------
####****************************************************************************************************************************************

dt  = np.dtype(float)
G   = 4.301e-9
h   = 0.6751
M_solar_2_g   = 1.99e33
dt = int


############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
### Reading Data
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************


# class FormatScalarFormatter(matplotlib.ticker.ScalarFormatter):
# 	def __init__(self, fformat="%.2f", offset=True, mathText=True):
# 		self.fformat = fformat
# 		matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,
#                                                     useMathText=mathText)
# 	def _set_format(self, vmin, vmax):
# 		self.format = self.fformat
# 		if self._useMathText:
# 			self.format = '$%s$' % matplotlib.ticker._mathdefault(self.format)




path = '/mnt/su3ctm/gchauhan/SHArk_Out/HI_haloes/' 
# path = ' /mnt/su3ctm/clagos/SHARK_Out/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/'
# path = '/mnt/sshfs/pleiades_gchauhan/SHArk_Out/HI_haloes/'



shark_runs 		= ['Shark-Lagos18-default-br06','Shark-Lagos18-Kappa-10','Shark-Lagos18-Kappa-0','Shark-Lagos18-Kappa-1','Shark-Lagos18-default-br06-stripping-off']

# shark_runs 		= ['Shark-Lagos18-default-br06','Shark-Lagos18-Kappa-10','Shark-Lagos18-Kappa-0','Shark-Lagos18-Kappa-1','Shark-Lagos18-default-br06-stripping-off']
# shark_labels	= ['Kappa = 0.002','Kappa = 0.02','Kappa = 0','Kappa = 1','Kappa = 0.002 (stripping off)']
shark_labels	= ['SHARK-ref','Kappa = 0.02','Kappa = 0','Kappa = 1','Lagos18 (stripping off)']




simulation 		= ['medi-SURFS', 'micro-SURFS']
snapshot_avail	= [199]#,174,156,131]
z_values 		= [0]#,0.5,1,2]

subvolumes 		= 64

#--------------------------------------------------------------------------------


medi_Kappa_original 	= {}
medi_Kappa_stripping 	= {}

micro_Kappa_original 	= {}
micro_Kappa_stripping 	= {}


for snapshot in snapshot_avail:

	medi_Kappa_original[snapshot]	 	= SharkDataReading(path,simulation[0],shark_runs[0],snapshot,subvolumes)
	
	micro_Kappa_original[snapshot]		= SharkDataReading(path,simulation[1],shark_runs[0],snapshot,subvolumes)
	
#--------------------------------------------------------------------------------



###################################################
######--------------------------------------------------------------------------------------------------------------------------------------
### Preparing Data
######--------------------------------------------------------------------------------------------------------------------------------------
####****************************************************************************************************************************************

def prepare_data(path,simulation,run,snapshot,subvolumes):
	
	#fields_read = {'galaxies' : ('type','mstars_disk', 'mstars_bulge', 'matom_disk','matom_bulge' , 'mgas_disk', 'mgas_bulge','mvir_hosthalo')}

	fields_read 	= {'halo' : ('age_50', 'age_80', 'lambda','mvir'), 'galaxies':('mvir_subhalo','lambda_subhalo','type', 'm_bh', 'mstars_bulge', 'mstars_disk', 'matom_bulge', 'matom_disk','id_subhalo_tree')}
	
	data_reading = SharkDataReading(path,simulation,run,snapshot,subvolumes)

	data_plotting = data_reading.readIndividualFiles(fields=fields_read)

	return data_plotting




######------------------------------------------------------------------------------------------------
### Micro-SURFS - 
######------------------------------------------------------------------------------------------------


micro_Kappa_original_HI 				= {}
micro_Kappa_original_HI_central 		= {}
micro_Kappa_original_HI_satellite 		= {}
micro_Kappa_original_HI_orphan  		= {}
micro_Kappa_original_vir 				= {}

micro_Kappa_original_stellar 			= {}
micro_Kappa_original_stellar_central 	= {}
micro_Kappa_original_stellar_satellite 	= {}
micro_Kappa_original_stellar_orphan		= {}

micro_substructure_original				= {}


for k in snapshot_avail:
	micro_Kappa_original_HI[k], micro_Kappa_original_HI_central[k], micro_Kappa_original_HI_satellite[k],micro_Kappa_original_HI_orphan[k],micro_Kappa_original_vir[k] = micro_Kappa_original[k].mergeValues_satellite('id_halo','matom_bulge','matom_disk')

	micro_substructure_original[k],a = micro_Kappa_original[k].mergeValuesNumberSubstructures('id_halo','id_subhalo')

	


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


for k in snapshot_avail:
	medi_Kappa_original_HI[k], medi_Kappa_original_HI_central[k], medi_Kappa_original_HI_satellite[k],medi_Kappa_original_HI_orphan[k],medi_Kappa_original_vir[k] = medi_Kappa_original[k].mergeValues_satellite('id_halo','matom_bulge','matom_disk')

	medi_substructure_original[k],a = medi_Kappa_original[k].mergeValuesNumberSubstructures('id_halo','id_subhalo')

	



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
mvir_subhalo    = {}
lambda_subhalo  = {}
matom_bulge     = {}
matom_disk      = {}
id_subhalo_tree = {}

for snapshot in snapshot_avail:

	(h0,_, age_50[snapshot], age_80[snapshot], lambda_spin[snapshot], mvir[snapshot], mvir_subhalo[snapshot], lambda_subhalo[snapshot], is_central[snapshot], m_bh[snapshot], mstars_bulge[snapshot], mstars_disk[snapshot], matom_bulge[snapshot], matom_disk[snapshot], id_subhalo_tree[snapshot]) = prepare_data(path,simulation[0],shark_runs[0],snapshot,subvolumes)
	

subhalo_infall_satellite  = np.zeros(0)
subhalo_infall_central  = np.zeros(0)
subhalo_infall_all  = np.zeros(0)
subhalo_id 		= np.zeros(0).astype(int)
is_central_subhalo = np.zeros(0).astype(int)
# id_subhalo_tree = np.zeros(0).astype(int)
lambda_infall_all = np.zeros(0).astype(int)
mvir_infall_all = np.zeros(0).astype(int)
lambda_infall_satellite = np.zeros(0).astype(int)
mvir_infall_satellite = np.zeros(0).astype(int)
lambda_infall_central = np.zeros(0).astype(int)
mvir_infall_central = np.zeros(0).astype(int)


for k in range(0,subvolumes):
	print(k)
	galaxy_prop 	= h5py.File(path  + simulation[0] + "/" +  shark_runs[0] + "/" + "199" + "/%s/"%k + "galaxies.hdf5", 'r')
	
	a = galaxy_prop['subhalo/infall_time_subhalo'][:]
	
	b = galaxy_prop['subhalo/id'][:]
	subhalo_id	=	np.append(subhalo_id, b)

	c = galaxy_prop['subhalo/main_progenitor'][:]
	d = galaxy_prop['galaxies/id_subhalo_tree'][:]

	lambda_yo = galaxy_prop['galaxies/lambda_subhalo'][:]
	mvir_yo   = galaxy_prop['galaxies/mvir_subhalo'][:]


	is_central_yo = galaxy_prop['galaxies/type'][:]

	ind = np.where(is_central_yo == 1)
	idsubh = b
	idsubhalos_types1 = d[ind]
	zinfall = a
	zinfall_galaxies = np.zeros(shape = (len(idsubhalos_types1)))
	# mvir_galaxies = np.zeros(shape = (len(idsubhalos_types1)))
	# lambda_galaxies = np.zeros(shape = (len(idsubhalos_types1)))

	for i,g in enumerate(idsubhalos_types1):
		if i%1000==0: print(i) 
		match = np.where(idsubh == g)
		zinfall_galaxies[i] 	= zinfall[match]
		# mvir_galaxies[i] 		= mvir_subhalo[match]
		# lambda_galaxies[i] 		= lambda_yo[match]

	subhalo_infall_satellite 	= np.append(subhalo_infall_satellite, zinfall_galaxies)
	# lambda_infall_satellite 	= np.append(lambda_infall_satellite, lambda_galaxies)
	# mvir_infall_satellite 		= np.append(mvir_infall_satellite, mvir_galaxies)
 


	# ind = np.where(is_central_yo == 0)
	# idsubh = b
	# idsubhalos_types1 = d[ind]
	# zinfall = a
	# zinfall_galaxies = np.zeros(shape = (len(idsubhalos_types1)))
	# # mvir_galaxies = np.zeros(shape = (len(idsubhalos_types1)))
	# # lambda_galaxies = np.zeros(shape = (len(idsubhalos_types1)))

	
	# for i,g in enumerate(idsubhalos_types1):
	# 	if i%1000==0: print(i) 
	# 	match = np.where(idsubh == g)
	# 	zinfall_galaxies[i] = zinfall[match]
	# 	# mvir_galaxies[i] 		= mvir_subhalo[match]
	# 	# lambda_galaxies[i] 		= lambda_yo[match]

	# subhalo_infall_central = np.append(subhalo_infall_central, zinfall_galaxies)
	# # lambda_infall_central 	= np.append(lambda_infall_central, lambda_galaxies)
	# # mvir_infall_central 		= np.append(mvir_infall_central, mvir_galaxies)
 
	# ind = np.where(is_central_yo != 2)
	# idsubh = b
	# idsubhalos_types1 = d[ind]
	# zinfall = a
	# zinfall_galaxies = np.zeros(shape = (len(idsubhalos_types1)))
	# mvir_galaxies = np.zeros(shape = (len(idsubhalos_types1)))
	# lambda_galaxies = np.zeros(shape = (len(idsubhalos_types1)))

	# for i,g in enumerate(idsubhalos_types1):
	# 	if i%1000==0: print(i) 
	# 	match = np.where(idsubh == g)
	# 	zinfall_galaxies[i] = zinfall[match]
	# 	# mvir_galaxies[i] 		= mvir_subhalo[match]
	# 	# lambda_galaxies[i] 		= lambda_yo[match]


	# subhalo_infall_all = np.append(subhalo_infall_all, zinfall_galaxies)
	# # lambda_infall_all 	= np.append(lambda_infall_all, lambda_galaxies)
	# # mvir_infall_all 		= np.append(mvir_infall_all, mvir_galaxies)


######--------------------------------------------------------------------------------------------------------------------------------------
### Analysis
######--------------------------------------------------------------------------------------------------------------------------------------


age_50_micro 			= {}
age_80_micro 			= {}
lambda_spin_micro 		= {}
mvir_micro 				= {}
is_central_micro  		= {}
m_bh_micro 				= {}
mstars_bulge_micro 		= {}
mstars_disk_micro 		= {}
mvir_subhalo_micro 		= {}
lambda_subhalo_micro 	= {}
matom_bulge_micro 		= {}
matom_disk_micro 		= {}
subhalo_infall_micro 	= {}
id_subhalo_tree_micro 	= {}

for snapshot in snapshot_avail:

	(h0,_, age_50_micro[snapshot], age_80_micro[snapshot], lambda_spin_micro[snapshot], mvir_micro[snapshot], mvir_subhalo_micro[snapshot], lambda_subhalo_micro[snapshot], is_central_micro[snapshot], m_bh_micro[snapshot], mstars_bulge_micro[snapshot], mstars_disk_micro[snapshot], matom_bulge_micro[snapshot], matom_disk_micro[snapshot], id_subhalo_tree_micro[snapshot]) = prepare_data(path,simulation[1],shark_runs[0],snapshot,subvolumes)



# subhalo_infall_micro  = np.zeros(0)
subhalo_infall_satellite_micro  = np.zeros(0)
subhalo_infall_central_micro  = np.zeros(0)
subhalo_infall_all_micro  = np.zeros(0)
subhalo_id_micro 		= np.zeros(0).astype(int)
is_central_subhalo_micro = np.zeros(0).astype(int)

lambda_infall_all_micro = np.zeros(0).astype(int)
mvir_infall_all_micro = np.zeros(0).astype(int)
lambda_infall_satellite_micro = np.zeros(0).astype(int)
mvir_infall_satellite_micro = np.zeros(0).astype(int)
lambda_infall_central_micro = np.zeros(0).astype(int)
mvir_infall_central_micro = np.zeros(0).astype(int)


for k in range(0,subvolumes):
	print(k)
	galaxy_prop 	= h5py.File(path  + simulation[1] + "/" +  shark_runs[0] + "/" + "199" + "/%s/"%k + "galaxies.hdf5", 'r')
	
	a = galaxy_prop['subhalo/infall_time_subhalo'][:]
	b = galaxy_prop['subhalo/id'][:]
	subhalo_id_micro	=	np.append(subhalo_id_micro, b)

	c = galaxy_prop['subhalo/main_progenitor'][:]
	d = galaxy_prop['galaxies/id_subhalo_tree'][:]

	is_central_yo = galaxy_prop['galaxies/type'][:]


	lambda_yo = galaxy_prop['galaxies/lambda_subhalo'][:]
	mvir_yo   = galaxy_prop['galaxies/mvir_subhalo'][:]


	ind = np.where(is_central_yo == 1)
	idsubh = b
	idsubhalos_types1 = d[ind]
	zinfall = a
	zinfall_galaxies = np.zeros(shape = (len(idsubhalos_types1)))
	# mvir_galaxies = np.zeros(shape = (len(idsubhalos_types1)))
	# lambda_galaxies = np.zeros(shape = (len(idsubhalos_types1)))

	for i,g in enumerate(idsubhalos_types1):
		if i%10000==0: print(i) 
		match = np.where(idsubh == g)
		zinfall_galaxies[i] = zinfall[match]
		# mvir_galaxies[i] 		= mvir_subhalo[match]
		# lambda_galaxies[i] 		= lambda_yo[match]

	subhalo_infall_satellite_micro = np.append(subhalo_infall_satellite_micro, zinfall_galaxies)

	# lambda_infall_satellite_micro 	= np.append(lambda_infall_satellite_micro, lambda_galaxies)
	# mvir_infall_satellite_micro 		= np.append(mvir_infall_satellite_micro, mvir_galaxies)



	# ind = np.where(is_central_yo == 0)
	# idsubh = b
	# idsubhalos_types1 = d[ind]
	# zinfall = a
	# zinfall_galaxies = np.zeros(shape = (len(idsubhalos_types1)))
	# # mvir_galaxies = np.zeros(shape = (len(idsubhalos_types1)))
	# # lambda_galaxies = np.zeros(shape = (len(idsubhalos_types1)))

	# for i,g in enumerate(idsubhalos_types1):
	# 	if i%1000==0: print(i) 
	# 	match = np.where(idsubh == g)
	# 	zinfall_galaxies[i] = zinfall[match]
	# 	# mvir_galaxies[i] 		= mvir_subhalo[match]
	# 	# lambda_galaxies[i] 		= lambda_yo[match]


	# subhalo_infall_central_micro = np.append(subhalo_infall_central_micro, zinfall_galaxies)
	# # lambda_infall_central_micro 	= np.append(lambda_infall_central_micro, lambda_galaxies)
	# # mvir_infall_central_micro 		= np.append(mvir_infall_central_micro, mvir_galaxies)


	# ind = np.where(is_central_yo != 2)
	# idsubh = b
	# idsubhalos_types1 = d[ind]
	# zinfall = a
	# zinfall_galaxies = np.zeros(shape = (len(idsubhalos_types1)))
	# # mvir_galaxies = np.zeros(shape = (len(idsubhalos_types1)))
	# # lambda_galaxies = np.zeros(shape = (len(idsubhalos_types1)))

	# for i,g in enumerate(idsubhalos_types1):
	# 	if i%1000==0: print(i) 
	# 	match = np.where(idsubh == g)
	# 	zinfall_galaxies[i] = zinfall[match]
	# 	# mvir_galaxies[i] 		= mvir_subhalo[match]
	# 	# lambda_galaxies[i] 		= lambda_yo[match]
	


	# subhalo_infall_all_micro = np.append(subhalo_infall_all_micro, zinfall_galaxies)
	# # lambda_infall_all_micro 	= np.append(lambda_infall_all_micro, lambda_galaxies)
	# # mvir_infall_all_micro 		= np.append(mvir_infall_all_micro, mvir_galaxies)




##############################################################################
#### Subhalo matching - medi
##############################################################################

HI_micro 				= (matom_bulge_micro[199] + matom_disk_micro[199])/1.4
HI_medi 				= (matom_bulge[199] + matom_disk[199])/1.4

Stellar_medi 			= mstars_disk[199] + mstars_bulge[199]
Stellar_micro 			= mstars_disk_micro[199] + mstars_bulge_micro[199]


mvir_subhalo 			= mvir_subhalo[199]
mvir_subhalo_micro 		= mvir_subhalo_micro[199]
	
is_central 				= is_central[199]
is_central_micro 		= is_central_micro[199]

lambda_subhalo 			= lambda_subhalo[199]
lambda_subhalo_micro 	= lambda_subhalo_micro[199]

m_bh 					= m_bh[199]
m_bh_micro 				= m_bh_micro[199]


############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
### Plot
######--------------------------------------------------------------------------------------------------------------------------------------
####****************************************************************************************************************************************



plt = Common_module.load_matplotlib(12,8)
colour_plot = Common_module.colour_scheme()


def plotting_scatter(virial_mass, property_plot, weights_property,legend_handles,mean=True, property_name_x='Property_x',property_name_y='Property_y',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=12,colour_map='cool_r', legend_name='name-of-run',colour_bar_title='property of scatter',norm_bound_lower=0.1,norm_bound_upper=1,median_values=False,n_min=10,colour_bar_log=False):
	
	import matplotlib.colors as colors
	import splotch as spl

	bin_for_disk = np.arange(5,13.6,0.2)

	stellar_mass 	= np.zeros(len(bin_for_disk))
	prop_mass 		= np.zeros(len(bin_for_disk))


	stellar_mass_central 	= np.zeros(len(bin_for_disk))
	prop_mass_central 		= np.zeros(len(bin_for_disk))

	stellar_mass_satellite 	= np.zeros(len(bin_for_disk))
	prop_mass_satellite 	= np.zeros(len(bin_for_disk))
	
	prop_mass_low 			= np.zeros(len(bin_for_disk))
	prop_mass_high			= np.zeros(len(bin_for_disk))
	
	
	(property_all, property_central, property_satellite) = property_plot.items()
	property_all = np.array(property_all[1])
	# print(property_all)
	property_central = np.array(property_central[1])
	property_satellite = np.array(property_satellite[1])

	(halo_mass, halo_mass_central, halo_mass_satellite) = virial_mass.items()
	halo_mass = np.array(halo_mass[1])
	halo_mass_central = np.array(halo_mass_central[1])
	halo_mass_satellite = np.array(halo_mass_satellite[1])

	bounds = np.logspace(np.log10(norm_bound_lower), np.log10(norm_bound_upper), 100)
	norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
	
	if median_values == False:
		scatter_yo = plt.scatter(np.log10(halo_mass),np.log10(property_all), c=weights_property,cmap=colour_map, s= 1, alpha=0.8, label=legend_name, norm=mcolors.LogNorm(vmin=0.01,vmax=0.1))
		if colour_bar_title != None:
			cbar = plt.colorbar(scatter_yo)
			cbar.set_label(colour_bar_title, rotation=270, labelpad = 20)
			cbar.ax.tick_params(labelsize=16)		
		plt.xlabel('$\\rm %s$ '%property_name_x)
		plt.ylabel('$\\rm %s$ '%property_name_y)

	else:
		scatter_yo=spl.hist2D(np.log10(halo_mass),np.log10(property_all),c=weights_property,cstat='median',bins=bin_for_disk,output=True,clabel=colour_bar_title,cmap=colour_map,nmin=n_min,clog=colour_bar_log)
		
		
		
	# legend_handles.append(scatter_yo)
	# contour_plot = plt.contourf(X,Y,A,10,cmap='bone_r',vmin=100) 
	# plt.contourf(X,Y,B,10,cmap='inferno_r',vmin=100) 
	
	plt.xlim(xlim_lower,xlim_upper)
	plt.ylim(ylim_lower,ylim_upper)





def plotting_properties_halo(virial_mass,property_plot,mean,legend_handles,property_name_x='X-Label-math-mode',legend_name='name-of-run', property_name_y='Y-Label-math-mode', xlim_lower=6,xlim_upper=15,ylim_lower=-4,ylim_upper=11,colour_line='r', halo_bins=0, linestyle_plot = '--'):
	

	if halo_bins == 0:
		bin_for_disk = np.arange(7,13.6,0.1)

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
	
	else:
		for i in range(1,len(bin_for_disk)):
			if len(property_plot[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])]) > 2:
				halo_mass[i-1] 		= np.log10(np.median(virial_mass[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])]))
				prop_mass[i-1]			= np.log10(np.median(property_plot[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])]))
			
			else:
				continue		

	plt.plot(halo_mass[halo_mass != 0],prop_mass[halo_mass != 0],color=colour_line, linestyle=linestyle_plot)
	print(halo_mass)
	plt.xlabel('$\\rm %s$ '%property_name_x)
	plt.ylabel('$\\rm %s$ '%property_name_y)

	plt.xlim(xlim_lower,xlim_upper)
	plt.ylim(ylim_lower,ylim_upper)

	#plt.legend(frameon=False)
	# plt.savefig('%s.png'%figure_name)
	# plt.show()





############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
### Plot
######--------------------------------------------------------------------------------------------------------------------------------------
####****************************************************************************************************************************************


# for snapshot in range(len(snapshot_avail)):

# 	legendHandles = []
# 	virial_mass_scatter 	= {'all' : mvir_subhalo_micro[mvir_subhalo_micro <= 10**11.3], 'central' : mvir_subhalo_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 0], 'satellite' : mvir_subhalo_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 1]}

# 	property_plot_scatter   = {'all' : HI_micro[mvir_subhalo_micro <= 10**11.3], 'central' : HI_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 0], 'satellite' : HI_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 1]}

# 	# Common_module.plotting_properties_halo(virial_mass['all'],property_plot['all'],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{vir}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13, first_legend=True,colour_line='k',legend_name='Kappa - 0.002')

# 	plotting_scatter(virial_mass_scatter,property_plot = property_plot_scatter ,weights_property=lambda_subhalo_micro[mvir_subhalo_micro <= 10**11.3],mean=True,legend_handles=legendHandles,property_name_x='log_{10}(M_{vir}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=9,xlim_upper=14,ylim_lower=5,ylim_upper=12,colour_map='Spectral_r',legend_name='Individual haloes', colour_bar_title=None,norm_bound_upper=0.5, median_values=True, colour_bar_log = True, n_min=10)

# 	##############################################################################################################

# 	virial_mass 	= {'all' : mvir_subhalo, 'central' : mvir_subhalo[is_central == 0], 'satellite' :mvir_subhalo[is_central == 1]}

# 	property_plot   = {'all' : HI_medi, 'central' : HI_medi[is_central == 0], 'satellite' : HI_medi[is_central == 1]}



# 	virial_mass_scatter 	= {'all' : mvir_subhalo[mvir_subhalo >= 10**11.15], 'central' : mvir_subhalo[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 0], 'satellite' : mvir_subhalo[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 1]}

# 	property_plot_scatter   = {'all' : HI_medi[mvir_subhalo >= 10**11.15], 'central' : HI_medi[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 0], 'satellite' : HI_medi[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 1]}

	

# 	Common_module.plotting_properties_separate_merged(virial_mass,virial_mass,property_plot,mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13, first_legend=False,colour_line='k',legend_name='Kappa - 0.002')

# 	yo = plotting_scatter(virial_mass_scatter,property_plot = property_plot_scatter,weights_property=lambda_subhalo[mvir_subhalo >= 10**11.15],mean=True,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10.2,xlim_upper=14,ylim_lower=6,ylim_upper=12,colour_map='Spectral_r',legend_name='Individual haloes', colour_bar_title=None,norm_bound_upper=0.5, median_values=True, colour_bar_log = True, n_min=20)

# 	plt.axvline(x=11.2, linestyle=':', color = 'k')

# 	cbar = plt.colorbar(yo, format='%.2f')
# 	cbar.ax.minorticks_on()
# 	cbar.update_ticks()
	
# 	cbar.set_label('$\lambda_{subhalo}$', rotation=270, labelpad = 20)


# 	extra = mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="z = %s" %z_values[snapshot] )
# 	legendHandles.append(extra)

# 	legendHandles.append(mlines.Line2D([],[],color='black',linestyle='--',label='Centrals'))
# 	legendHandles.append(mlines.Line2D([],[],color='black',linestyle='-.',label='Satellites'))

	
# 	plt.legend(handles=legendHandles, markerscale=10)
# 	plt.savefig("new_plots_corrected/Spin_parameter_final_subhalo_all.png" )

# 	plt.show()







# for snapshot in range(len(snapshot_avail)):

# 	legendHandles = []
# 	virial_mass_scatter 	= {'all' : mvir_subhalo_micro[mvir_subhalo_micro <= 10**11.3], 'central' : mvir_subhalo_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 0], 'satellite' : mvir_subhalo_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 1]}

# 	property_plot_scatter   = {'all' : HI_micro[mvir_subhalo_micro <= 10**11.3], 'central' : HI_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 0], 'satellite' : HI_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 1]}

# 	# Common_module.plotting_properties_halo(virial_mass['all'],property_plot['all'],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{vir}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13, first_legend=True,colour_line='k',legend_name='Kappa - 0.002')

# 	plotting_scatter(virial_mass_scatter,property_plot = property_plot_scatter ,weights_property=m_bh_micro[mvir_subhalo_micro <= 10**11.3]/Stellar_micro[mvir_subhalo_micro <= 10**11.3],mean=True,legend_handles=legendHandles,property_name_x='log_{10}(M_{vir}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=9,xlim_upper=14,ylim_lower=5,ylim_upper=12,colour_map='Spectral_r',legend_name='Individual haloes', colour_bar_title=None,norm_bound_upper=0.5, median_values=True, colour_bar_log = True, n_min=20)

# 	##############################################################################################################

# 	virial_mass 	= {'all' : mvir_subhalo, 'central' : mvir_subhalo[is_central == 0], 'satellite' :mvir_subhalo[is_central == 1]}

# 	property_plot   = {'all' : HI_medi, 'central' : HI_medi[is_central == 0], 'satellite' : HI_medi[is_central == 1]}



# 	virial_mass_scatter 	= {'all' : mvir_subhalo[mvir_subhalo >= 10**11.15], 'central' : mvir_subhalo[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 0], 'satellite' : mvir_subhalo[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 1]}

# 	property_plot_scatter   = {'all' : HI_medi[mvir_subhalo >= 10**11.15], 'central' : HI_medi[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 0], 'satellite' : HI_medi[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 1]}

	

# 	Common_module.plotting_properties_separate_merged(virial_mass,virial_mass,property_plot,mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13, first_legend=False,colour_line='k',legend_name='Kappa - 0.002')

# 	yo = plotting_scatter(virial_mass_scatter,property_plot = property_plot_scatter,weights_property=m_bh[mvir_subhalo >= 10**11.15]/Stellar_medi[mvir_subhalo >= 10**11.15],mean=True,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10.2,xlim_upper=14,ylim_lower=6,ylim_upper=12,colour_map='Spectral_r',legend_name='Individual haloes', colour_bar_title=None,norm_bound_lower=0.0002,norm_bound_upper=0.006, median_values=True, colour_bar_log = True, n_min=20)

# 	plt.axvline(x=11.2, linestyle=':', color = 'k')
# 	fmt = FormatScalarFormatter("%.2f")
# 	fmt.set_powerlimits((0.0002, 0.006))
# 	cbar = plt.colorbar(yo, ticks=[0.0002,0.0005,0.001,0.002,0.003],format=fmt)
# 	# cbar.formatter.set_scientific(True)
# 	# cbar.formatter.set_powerlimits((0,0))
# 	cbar.ax.minorticks_on()
# 	cbar.update_ticks()

# 	cbar.set_label('$M_{BH}/M_{*}$', rotation=270, labelpad = 30)


# 	extra = mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="z = %s" %z_values[snapshot] )
# 	legendHandles.append(extra)

# 	legendHandles.append(mlines.Line2D([],[],color='black',linestyle='--',label='Centrals'))
# 	legendHandles.append(mlines.Line2D([],[],color='black',linestyle='-.',label='Satellites'))

	
# 	plt.legend(handles=legendHandles, markerscale=10)
# 	plt.savefig("new_plots_corrected/BH_fraction_subhalo_all.png" )

# 	plt.show()








# for snapshot in range(len(snapshot_avail)):

# 	legendHandles = []
# 	virial_mass_scatter 	= {'all' : mvir_subhalo_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 1], 'central' : mvir_subhalo_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 0], 'satellite' : mvir_subhalo_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 1]}

# 	property_plot_scatter   = {'all' : HI_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 1], 'central' : HI_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 0], 'satellite' : HI_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 1]}

# 	# Common_module.plotting_properties_halo(virial_mass['all'],property_plot['all'],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{vir}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13, first_legend=True,colour_line='k',legend_name='Kappa - 0.002')

# 	plotting_scatter(virial_mass_scatter,property_plot = property_plot_scatter ,weights_property=m_bh_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 1]/Stellar_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 1],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{vir}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=9,xlim_upper=14,ylim_lower=5,ylim_upper=12,colour_map='Spectral_r',legend_name='Individual haloes', colour_bar_title=None,norm_bound_upper=0.5, median_values=True, colour_bar_log = True, n_min=20)

# 	##############################################################################################################

# 	virial_mass 	= {'all' : mvir_subhalo, 'central' : mvir_subhalo[is_central == 0], 'satellite' :mvir_subhalo[is_central == 1]}

# 	property_plot   = {'all' : HI_medi, 'central' : HI_medi[is_central == 0], 'satellite' : HI_medi[is_central == 1]}



# 	virial_mass_scatter 	= {'all' : mvir_subhalo[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 1], 'central' : mvir_subhalo[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 0], 'satellite' : mvir_subhalo[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 1]}

# 	property_plot_scatter   = {'all' : HI_medi[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 1], 'central' : HI_medi[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 0], 'satellite' : HI_medi[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 1]}

	

# 	# Common_module.plotting_properties_separate_merged(virial_mass,virial_mass,property_plot,mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13, first_legend=False,colour_line='k',legend_name='Kappa - 0.002')



# 	plotting_properties_halo(mvir_subhalo[is_central == 1],HI_medi[is_central == 1],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13,colour_line='k',legend_name='Kappa - 0.002',linestyle_plot='-')


# 	# plotting_properties_halo(mvir_subhalo[is_central == 1][(subhalo_infall_satellite > 0) & (subhalo_infall_satellite <= 0.1)],HI_medi[is_central == 1][(subhalo_infall_satellite > 0) & (subhalo_infall_satellite <= 0.1)],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13,colour_line='k',legend_name='Kappa - 0.002',linestyle_plot='--')



# 	# plotting_properties_halo(mvir_subhalo[is_central == 1][(subhalo_infall_satellite > 0.1) & (subhalo_infall_satellite <= 0.2)],HI_medi[is_central == 1][(subhalo_infall_satellite > 0.1) & (subhalo_infall_satellite <= 0.2)],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13,colour_line='k',legend_name='Kappa - 0.002',linestyle_plot='-.')


# 	# plotting_properties_halo(mvir_subhalo[is_central == 1][(subhalo_infall_satellite > 0.2) & (subhalo_infall_satellite <= 0.4)],HI_medi[is_central == 1][(subhalo_infall_satellite > 0.2) & (subhalo_infall_satellite <= 0.4)],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13,colour_line='k',legend_name='Kappa - 0.002',linestyle_plot=(0, (3, 2, 1, 2, 1, 2)))

# 	# plotting_properties_halo(mvir_subhalo[is_central == 1][(subhalo_infall_satellite > 0.6)],HI_medi[is_central == 1][(subhalo_infall_satellite > 0.6)],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13,colour_line='k',legend_name='Kappa - 0.002',linestyle_plot=':')



# 	yo = plotting_scatter(virial_mass_scatter,property_plot = property_plot_scatter,weights_property=m_bh[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 1]/Stellar_medi[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 1],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=11,xlim_upper=13,ylim_lower=6,ylim_upper=11,colour_map='Spectral_r',legend_name='Individual haloes', colour_bar_title=None,norm_bound_lower=0.0002,norm_bound_upper=0.006, median_values=True, colour_bar_log = True, n_min=20)

# 	plt.axvline(x=11.2, linestyle=':', color = 'k')
# 	fmt = FormatScalarFormatter("%.2f")
# 	fmt.set_powerlimits((0.0002, 0.006))
# 	cbar = plt.colorbar(yo, ticks=[0.0002,0.0005,0.001,0.002,0.003],format=fmt)
# 	# cbar.formatter.set_scientific(True)
# 	# cbar.formatter.set_powerlimits((0,0))
# 	cbar.ax.minorticks_on()
# 	cbar.update_ticks()

# 	cbar.set_label('$M_{BH}/M_{*}$', rotation=270, labelpad = 30)


# 	extra = mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="z = %s" %z_values[snapshot] )
# 	legendHandles.append(extra)

# 	legendHandles.append(mlines.Line2D([],[],color='black',linestyle='-',label='All Satellites'))
# 	# legendHandles.append(mlines.Line2D([],[],color='black',linestyle='--',label='0 < $z_{infall}$ $\leq$ 0.1'))
# 	# legendHandles.append(mlines.Line2D([],[],color='black',linestyle='-.',label='0.2 < $z_{infall}$ $\leq$ 0.4'))
# 	# legendHandles.append(mlines.Line2D([],[],color='black',linestyle=(0, (3, 2, 1, 2, 1, 2)),label='0.4 < $z_{infall}$ $\leq$ 0.6'))

# 	# legendHandles.append(mlines.Line2D([],[],color='black',linestyle=':',label='$z_{infall}$ $\geq$ 0.6'))

	
# 	plt.legend(handles=legendHandles, markerscale=10)
# 	plt.savefig("new_plots_corrected/BH_fraction_subhalo_satellite.png" )

# 	plt.show()









# for snapshot in range(len(snapshot_avail)):

# 	legendHandles = []
# 	virial_mass_scatter 	= {'all' : mvir_subhalo_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 0], 'central' : mvir_subhalo_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 0], 'satellite' : mvir_subhalo_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 1]}

# 	property_plot_scatter   = {'all' : HI_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 0], 'central' : HI_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 0], 'satellite' : HI_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 1]}

# 	# Common_module.plotting_properties_halo(virial_mass['all'],property_plot['all'],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{vir}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13, first_legend=True,colour_line='k',legend_name='Kappa - 0.002')

# 	plotting_scatter(virial_mass_scatter,property_plot = property_plot_scatter ,weights_property=m_bh_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 0]/Stellar_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 0],mean=True,legend_handles=legendHandles,property_name_x='log_{10}(M_{vir}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=9,xlim_upper=14,ylim_lower=5,ylim_upper=12,colour_map='Spectral_r',legend_name='Individual haloes', colour_bar_title=None,norm_bound_upper=0.5, median_values=True, colour_bar_log = True, n_min=20)

# 	##############################################################################################################

# 	virial_mass 	= {'all' : mvir_subhalo, 'central' : mvir_subhalo[is_central == 0], 'satellite' :mvir_subhalo[is_central == 1]}

# 	property_plot   = {'all' : HI_medi, 'central' : HI_medi[is_central == 0], 'satellite' : HI_medi[is_central == 1]}



# 	virial_mass_scatter 	= {'all' : mvir_subhalo[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 0], 'central' : mvir_subhalo[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 0], 'satellite' : mvir_subhalo[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 1]}

# 	property_plot_scatter   = {'all' : HI_medi[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 0], 'central' : HI_medi[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 0], 'satellite' : HI_medi[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 1]}

	

# 	plotting_properties_halo(mvir_subhalo[is_central == 0],HI_medi[is_central == 0],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13,colour_line='k',legend_name='Kappa - 0.002',linestyle_plot='-')

# 	# Common_module.plotting_properties_separate_merged(virial_mass,virial_mass,property_plot,mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13, first_legend=False,colour_line='k',legend_name='Kappa - 0.002')

# 	yo = plotting_scatter(virial_mass_scatter,property_plot = property_plot_scatter,weights_property=m_bh[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 0]/Stellar_medi[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 0],mean=True,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10.2,xlim_upper=14,ylim_lower=6,ylim_upper=12,colour_map='Spectral_r',legend_name='Individual haloes', colour_bar_title=None,norm_bound_lower=0.0002,norm_bound_upper=0.006, median_values=True, colour_bar_log = True, n_min=50)

# 	plt.axvline(x=11.2, linestyle=':', color = 'k')
# 	fmt = FormatScalarFormatter("%.2f")
# 	fmt.set_powerlimits((0.0002, 0.006))
# 	cbar = plt.colorbar(yo, ticks=[0.0002,0.0005,0.001,0.002,0.003],format=fmt)
# 	# cbar.formatter.set_scientific(True)
# 	# cbar.formatter.set_powerlimits((0,0))
# 	cbar.ax.minorticks_on()
# 	cbar.update_ticks()

# 	cbar.set_label('$M_{BH}/M_{*}$', rotation=270, labelpad = 30)


# 	extra = mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="z = %s" %z_values[snapshot] )
# 	legendHandles.append(extra)

# 	legendHandles.append(mlines.Line2D([],[],color='black',linestyle='-',label='All Centrals'))
# 	# legendHandles.append(mlines.Line2D([],[],color='black',linestyle='-.',label='Satellites'))

	
# 	plt.legend(handles=legendHandles, markerscale=10)
# 	plt.savefig("new_plots_corrected/BH_fraction_subhalo_central.png" )

# 	plt.show()



# for snapshot in range(len(snapshot_avail)):

# 	legendHandles = []
# 	virial_mass_scatter 	= {'all' : mvir_subhalo_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 1], 'central' : mvir_subhalo_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 0], 'satellite' : mvir_subhalo_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 1]}

# 	property_plot_scatter   = {'all' : HI_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 1], 'central' : HI_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 0], 'satellite' : HI_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 1]}

# 	# Common_module.plotting_properties_halo(virial_mass['all'],property_plot['all'],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{vir}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13, first_legend=True,colour_line='k',legend_name='Kappa - 0.002')

# 	plotting_scatter(virial_mass_scatter,property_plot = property_plot_scatter ,weights_property=lambda_subhalo_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 1],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{vir}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=9,xlim_upper=14,ylim_lower=5,ylim_upper=12,colour_map='Spectral_r',legend_name='Individual haloes', colour_bar_title=None,norm_bound_upper=0.5, median_values=True, colour_bar_log = True, n_min=10)

# 	##############################################################################################################

# 	virial_mass 	= {'all' : mvir_subhalo, 'central' : mvir_subhalo[is_central == 0], 'satellite' :mvir_subhalo[is_central == 1]}

# 	property_plot   = {'all' : HI_medi, 'central' : HI_medi[is_central == 0], 'satellite' : HI_medi[is_central == 1]}



# 	virial_mass_scatter 	= {'all' : mvir_subhalo[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 1], 'central' : mvir_subhalo[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 0], 'satellite' : mvir_subhalo[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 1]}

# 	property_plot_scatter   = {'all' : HI_medi[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 1], 'central' : HI_medi[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 0], 'satellite' : HI_medi[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 1]}

	



# 	plotting_properties_halo(mvir_subhalo[is_central == 1],HI_medi[is_central == 1],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13,colour_line='k',legend_name='Kappa - 0.002',linestyle_plot='-')


# 	plotting_properties_halo(mvir_subhalo[is_central == 1][(subhalo_infall_satellite > 0) & (subhalo_infall_satellite <= 0.1)],HI_medi[is_central == 1][(subhalo_infall_satellite > 0) & (subhalo_infall_satellite <= 0.1)],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13,colour_line='k',legend_name='Kappa - 0.002',linestyle_plot='--')



# 	plotting_properties_halo(mvir_subhalo[is_central == 1][(subhalo_infall_satellite > 0.1) & (subhalo_infall_satellite <= 0.2)],HI_medi[is_central == 1][(subhalo_infall_satellite > 0.1) & (subhalo_infall_satellite <= 0.2)],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13,colour_line='k',legend_name='Kappa - 0.002',linestyle_plot='-.')


# 	plotting_properties_halo(mvir_subhalo[is_central == 1][(subhalo_infall_satellite > 0.2) & (subhalo_infall_satellite <= 0.4)],HI_medi[is_central == 1][(subhalo_infall_satellite > 0.2) & (subhalo_infall_satellite <= 0.4)],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13,colour_line='k',legend_name='Kappa - 0.002',linestyle_plot=(0, (3, 2, 1, 2, 1, 2)))

# 	plotting_properties_halo(mvir_subhalo[is_central == 1][(subhalo_infall_satellite > 0.6)],HI_medi[is_central == 1][(subhalo_infall_satellite > 0.6)],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13,colour_line='k',legend_name='Kappa - 0.002',linestyle_plot=':')



# 	# Common_module.plotting_properties_separate_merged(virial_mass,virial_mass,property_plot,mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13, first_legend=False,colour_line='k',legend_name='Kappa - 0.002')

# 	yo = plotting_scatter(virial_mass_scatter,property_plot = property_plot_scatter,weights_property=lambda_subhalo[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 1],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10.2,xlim_upper=14,ylim_lower=6,ylim_upper=12,colour_map='Spectral_r',legend_name='Individual haloes', colour_bar_title=None,norm_bound_upper=0.05, median_values=True, colour_bar_log = True, n_min=20)

# 	plt.axvline(x=11.2, linestyle=':', color = 'k')

# 	# fmt = FormatScalarFormatter("%.2f")
# 	# fmt.set_powerlimits((0.001, 0.01))
# 	# cbar = plt.colorbar(yo, ticks=[0.001,0.003,0.005,0.007,0.009,0.01] ,format=fmt)
# 	# # cbar.formatter.set_scientific(True)
# 	# # cbar.formatter.set_powerlimits((0,0))
# 	# # cbar.ax.minorticks_on()
# 	# cbar.update_ticks()

# 	cbar = plt.colorbar(yo, format='%.2f')
# 	cbar.ax.minorticks_on()
# 	cbar.update_ticks()
	

# 	cbar.set_label('$\lambda_{subhalo}$', rotation=270, labelpad = 30)


# 	extra = mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="z = %s" %z_values[snapshot] )
# 	legendHandles.append(extra)

# 	legendHandles.append(mlines.Line2D([],[],color='black',linestyle='-',label='All Satellites'))
# 	legendHandles.append(mlines.Line2D([],[],color='black',linestyle='--',label='0 < $z_{infall}$ $\leq$ 0.1'))
# 	legendHandles.append(mlines.Line2D([],[],color='black',linestyle='-.',label='0.2 < $z_{infall}$ $\leq$ 0.4'))
# 	legendHandles.append(mlines.Line2D([],[],color='black',linestyle=(0, (3, 2, 1, 2, 1, 2)),label='0.4 < $z_{infall}$ $\leq$ 0.6'))

# 	legendHandles.append(mlines.Line2D([],[],color='black',linestyle=':',label='$z_{infall}$ $\geq$ 0.6'))

# 	plt.legend(handles=legendHandles, markerscale=10)
# 	plt.savefig("new_plots_corrected/Spin_parameter_final_subhalo_satellite.png" )

# 	plt.show()






# for snapshot in range(len(snapshot_avail)):

# 	legendHandles = []
# 	virial_mass_scatter 	= {'all' : mvir_subhalo_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 0], 'central' : mvir_subhalo_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 0], 'satellite' : mvir_subhalo_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 1]}

# 	property_plot_scatter   = {'all' : HI_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 0], 'central' : HI_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 0], 'satellite' : HI_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 1]}

# 	# Common_module.plotting_properties_halo(virial_mass['all'],property_plot['all'],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{vir}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13, first_legend=True,colour_line='k',legend_name='Kappa - 0.002')

# 	plotting_scatter(virial_mass_scatter,property_plot = property_plot_scatter ,weights_property=lambda_subhalo_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 0],mean=True,legend_handles=legendHandles,property_name_x='log_{10}(M_{vir}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=9,xlim_upper=14,ylim_lower=5,ylim_upper=12,colour_map='Spectral_r',legend_name='Individual haloes', colour_bar_title=None,norm_bound_upper=0.5, median_values=True, colour_bar_log = True, n_min=10)

# 	##############################################################################################################

# 	virial_mass 	= {'all' : mvir_subhalo, 'central' : mvir_subhalo[is_central == 0], 'satellite' :mvir_subhalo[is_central == 1]}

# 	property_plot   = {'all' : HI_medi, 'central' : HI_medi[is_central == 0], 'satellite' : HI_medi[is_central == 1]}



# 	virial_mass_scatter 	= {'all' : mvir_subhalo[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 0], 'central' : mvir_subhalo[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 0], 'satellite' : mvir_subhalo[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 1]}

# 	property_plot_scatter   = {'all' : HI_medi[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 0], 'central' : HI_medi[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 0], 'satellite' : HI_medi[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 1]}

	



# 	plotting_properties_halo(mvir_subhalo[is_central == 0],HI_medi[is_central == 0],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13,colour_line='k',legend_name='Kappa - 0.002',linestyle_plot='-')


# 	plotting_properties_halo(mvir_subhalo[is_central == 0][(subhalo_infall_central > 0) & (subhalo_infall_central <= 0.1)],HI_medi[is_central == 0][(subhalo_infall_central > 0) & (subhalo_infall_central <= 0.1)],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13,colour_line='k',legend_name='Kappa - 0.002',linestyle_plot='--')



# 	plotting_properties_halo(mvir_subhalo[is_central == 0][(subhalo_infall_central > 0.1) & (subhalo_infall_central <= 0.2)],HI_medi[is_central == 0][(subhalo_infall_central > 0.1) & (subhalo_infall_central <= 0.2)],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13,colour_line='k',legend_name='Kappa - 0.002',linestyle_plot='-.')


# 	plotting_properties_halo(mvir_subhalo[is_central == 0][(subhalo_infall_central > 0.2) & (subhalo_infall_central <= 0.4)],HI_medi[is_central == 0][(subhalo_infall_central > 0.2) & (subhalo_infall_central <= 0.4)],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13,colour_line='k',legend_name='Kappa - 0.002',linestyle_plot=(0, (3, 2, 1, 2, 1, 2)))




# 	# Common_module.plotting_properties_separate_merged(virial_mass,virial_mass,property_plot,mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13, first_legend=False,colour_line='k',legend_name='Kappa - 0.002')

# 	yo = plotting_scatter(virial_mass_scatter,property_plot = property_plot_scatter,weights_property=lambda_subhalo[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 0],mean=True,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10.2,xlim_upper=14,ylim_lower=6,ylim_upper=12,colour_map='Spectral_r',legend_name='Individual haloes', colour_bar_title=None,norm_bound_upper=0.5, median_values=True, colour_bar_log = True, n_min=50)

# 	plt.axvline(x=11.2, linestyle=':', color = 'k')

# 	cbar = plt.colorbar(yo, format='%.2f')
# 	cbar.ax.minorticks_on()
# 	cbar.update_ticks()
	
# 	cbar.set_label('$\lambda_{subhalo}$', rotation=270, labelpad = 20)


# 	extra = mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="z = %s" %z_values[snapshot] )
# 	legendHandles.append(extra)

# 	legendHandles.append(mlines.Line2D([],[],color='black',linestyle='-',label='All Central'))
# 	# legendHandles.append(mlines.Line2D([],[],color='black',linestyle='--',label='0 < $z_{infall}$ $\leq$ 0.1'))
# 	# legendHandles.append(mlines.Line2D([],[],color='black',linestyle='-.',label='0.2 < $z_{infall}$ $\leq$ 0.4'))
# 	# legendHandles.append(mlines.Line2D([],[],color='black',linestyle=(0, (3, 2, 1, 2, 1, 2)),label='0.4 < $z_{infall}$ $\leq$ 0.6'))

	
# 	plt.legend(handles=legendHandles, markerscale=10)
# 	plt.savefig("new_plots_corrected/Spin_parameter_final_subhalo_central.png" )

# 	plt.show()







# ############################################################################################################################################
# ######--------------------------------------------------------------------------------------------------------------------------------------
# ### Plot
# ######--------------------------------------------------------------------------------------------------------------------------------------
# ####****************************************************************************************************************************************


# for snapshot in range(len(snapshot_avail)):

# 	legendHandles = []

# 	virial_mass_scatter 	= {'all' : mvir_subhalo_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 1][(subhalo_infall_satellite_micro[mvir_subhalo_micro[is_central_micro == 1] <= 10**11.3] > 0) & (subhalo_infall_satellite_micro[mvir_subhalo_micro[is_central_micro == 1] <= 10**11.3] <= 0.1)], 'central' : mvir_subhalo_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 0], 'satellite' : mvir_subhalo_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 1]}

# 	property_plot_scatter   = {'all' : HI_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 1][(subhalo_infall_satellite_micro[mvir_subhalo_micro[is_central_micro == 1] <= 10**11.3] > 0) & (subhalo_infall_satellite_micro[mvir_subhalo_micro[is_central_micro == 1] <= 10**11.3] <= 0.1)], 'central' : HI_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 0], 'satellite' : HI_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 1]}

# 	# Common_module.plotting_properties_halo(virial_mass['all'],property_plot['all'],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{vir}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13, first_legend=True,colour_line='k',legend_name='Kappa - 0.002')

# 	plotting_scatter(virial_mass_scatter,property_plot = property_plot_scatter ,weights_property=lambda_subhalo_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 1][(subhalo_infall_satellite_micro[mvir_subhalo_micro[is_central_micro == 1] <= 10**11.3] > 0) & (subhalo_infall_satellite_micro[mvir_subhalo_micro[is_central_micro == 1] <= 10**11.3] <= 0.1)],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{vir}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=9,xlim_upper=14,ylim_lower=5,ylim_upper=12,colour_map='Spectral_r',legend_name='Individual haloes', colour_bar_title=None,norm_bound_upper=0.5, median_values=True, colour_bar_log = True, n_min=10)

# 	#############################################################################################################

# 	virial_mass 	= {'all' : mvir_subhalo, 'central' : mvir_subhalo[is_central == 0], 'satellite' :mvir_subhalo[is_central == 1]}

# 	property_plot   = {'all' : HI_medi, 'central' : HI_medi[is_central == 0], 'satellite' : HI_medi[is_central == 1]}



# 	virial_mass_scatter 	= {'all' : mvir_subhalo[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 1][(subhalo_infall_satellite[mvir_subhalo[is_central == 1] >= 10**11.15] > 0.0) & (subhalo_infall_satellite[mvir_subhalo[is_central == 1] >= 10**11.15] <= 0.1)], 'central' : mvir_subhalo[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 0], 'satellite' : mvir_subhalo[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 1]}

# 	property_plot_scatter   = {'all' : HI_medi[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 1][(subhalo_infall_satellite[mvir_subhalo[is_central == 1] >= 10**11.15] > 0.0) & (subhalo_infall_satellite[mvir_subhalo[is_central == 1] >= 10**11.15] <= 0.1)], 'central' : HI_medi[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 0], 'satellite' : HI_medi[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 1]}

	
# 	# virial_mass_scatter 	= {'all' : mvir_subhalo[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 1][(subhalo_infall_satellite[mvir_subhalo[is_central == 1] >= 10**11.15] > 0.6)], 'central' : mvir_subhalo[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 0], 'satellite' : mvir_subhalo[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 1]}

# 	# property_plot_scatter   = {'all' : HI_medi[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 1][(subhalo_infall_satellite[mvir_subhalo[is_central == 1] >= 10**11.15] > 0.6)], 'central' : HI_medi[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 0], 'satellite' : HI_medi[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 1]}




# 	# plotting_properties_halo(mvir_subhalo[is_central == 1],HI_medi[is_central == 1],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13,colour_line='k',legend_name='Kappa - 0.002',linestyle_plot='-')


# 	plotting_properties_halo(mvir_subhalo[is_central == 1][(subhalo_infall_satellite > 0) & (subhalo_infall_satellite <= 0.1)],HI_medi[is_central == 1][(subhalo_infall_satellite > 0) & (subhalo_infall_satellite <= 0.1)],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13,colour_line='k',legend_name='Kappa - 0.002',linestyle_plot='--')



# 	# plotting_properties_halo(mvir_subhalo[is_central == 1][(subhalo_infall_satellite > 0.1) & (subhalo_infall_satellite <= 0.2)],HI_medi[is_central == 1][(subhalo_infall_satellite > 0.1) & (subhalo_infall_satellite <= 0.2)],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13,colour_line='k',legend_name='Kappa - 0.002',linestyle_plot='-.')


# 	# plotting_properties_halo(mvir_subhalo[is_central == 1][(subhalo_infall_satellite > 0.4) & (subhalo_infall_satellite <= 0.6)],HI_medi[is_central == 1][(subhalo_infall_satellite > 0.4) & (subhalo_infall_satellite <= 0.6)],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13,colour_line='k',legend_name='Kappa - 0.002',linestyle_plot=(0, (3, 2, 1, 2, 1, 2)))

# 	# plotting_properties_halo(mvir_subhalo[is_central == 1][(subhalo_infall_satellite > 0.6)],HI_medi[is_central == 1][(subhalo_infall_satellite > 0.6)],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13,colour_line='k',legend_name='Kappa - 0.002',linestyle_plot=':')



# 	# Common_module.plotting_properties_separate_merged(virial_mass,virial_mass,property_plot,mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13, first_legend=False,colour_line='k',legend_name='Kappa - 0.002')

# 	yo = plotting_scatter(virial_mass_scatter,property_plot = property_plot_scatter,weights_property=lambda_subhalo[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 1][(subhalo_infall_satellite[mvir_subhalo[is_central == 1] >= 10**11.15] > 0.0) & (subhalo_infall_satellite[mvir_subhalo[is_central == 1] >= 10**11.15] <= 0.1)],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10.2,xlim_upper=14,ylim_lower=6,ylim_upper=12,colour_map='Spectral_r',legend_name='Individual haloes', colour_bar_title=None,norm_bound_upper=0.05, median_values=True, colour_bar_log = True, n_min=20)


# 	# yo = plotting_scatter(virial_mass_scatter,property_plot = property_plot_scatter,weights_property=lambda_subhalo[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 1][(subhalo_infall_satellite[mvir_subhalo[is_central == 1] >= 10**11.15] > 0.6)],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10.2,xlim_upper=14,ylim_lower=6,ylim_upper=12,colour_map='Spectral_r',legend_name='Individual haloes', colour_bar_title=None,norm_bound_upper=0.05, median_values=True, colour_bar_log = True, n_min=20)


# 	plt.axvline(x=11.2, linestyle=':', color = 'k')

# 	# fmt = FormatScalarFormatter("%.2f")
# 	# fmt.set_powerlimits((0.001, 0.01))
# 	# cbar = plt.colorbar(yo, ticks=[0.001,0.003,0.005,0.007,0.009,0.01] ,format=fmt)
# 	# # cbar.formatter.set_scientific(True)
# 	# # cbar.formatter.set_powerlimits((0,0))
# 	# # cbar.ax.minorticks_on()
# 	# cbar.update_ticks()

# 	cbar = plt.colorbar(yo, format='%.2f')
# 	# cbar.ax.minorticks_on()
# 	# cbar.update_ticks()
	

# 	cbar.set_label('$\lambda_{subhalo}$', rotation=270, labelpad = 30)


# 	extra = mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="z = %s" %z_values[snapshot] )
# 	legendHandles.append(extra)

# 	# legendHandles.append(mlines.Line2D([],[],color='black',linestyle='-',label='All Satellites'))
# 	legendHandles.append(mlines.Line2D([],[],color='black',linestyle='--',label='0 < $z_{infall}$ $\leq$ 0.1'))
# 	# legendHandles.append(mlines.Line2D([],[],color='black',linestyle='-.',label='0.1 < $z_{infall}$ $\leq$ 0.2'))
# 	# legendHandles.append(mlines.Line2D([],[],color='black',linestyle='-.',label='0.2 < $z_{infall}$ $\leq$ 0.4'))
# 	# legendHandles.append(mlines.Line2D([],[],color='black',linestyle=(0, (3, 2, 1, 2, 1, 2)),label='0.4 < $z_{infall}$ $\leq$ 0.6'))

# 	# legendHandles.append(mlines.Line2D([],[],color='black',linestyle=':',label='$z_{infall}$ $\geq$ 0.6'))

# 	plt.legend(handles=legendHandles, markerscale=10)
# 	plt.savefig("new_plots_corrected/Spin_parameter_subhalo_satellite_z_0_0p1.png" )

# 	plt.show()








# for snapshot in range(len(snapshot_avail)):

# 	legendHandles = []

# 	virial_mass_scatter 	= {'all' : mvir_subhalo_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 1][(subhalo_infall_satellite_micro[mvir_subhalo_micro[is_central_micro == 1] <= 10**11.3] > 0.0) & (subhalo_infall_satellite_micro[mvir_subhalo_micro[is_central_micro == 1] <= 10**11.3] <= 0.1)], 'central' : mvir_subhalo_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 0], 'satellite' : mvir_subhalo_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 1]}

# 	property_plot_scatter   = {'all' : HI_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 1][(subhalo_infall_satellite_micro[mvir_subhalo_micro[is_central_micro == 1] <= 10**11.3] > 0.0) & (subhalo_infall_satellite_micro[mvir_subhalo_micro[is_central_micro == 1] <= 10**11.3] <= 0.1)], 'central' : HI_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 0], 'satellite' : HI_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 1]}

# 	# Common_module.plotting_properties_halo(virial_mass['all'],property_plot['all'],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{vir}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13, first_legend=True,colour_line='k',legend_name='Kappa - 0.002')

# 	fraction_BH = m_bh_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 1][(subhalo_infall_satellite_micro[mvir_subhalo_micro[is_central_micro == 1] <= 10**11.3] > 0.0) & (subhalo_infall_satellite_micro[mvir_subhalo_micro[is_central_micro == 1] <= 10**11.3] <= 0.1)]/Stellar_micro[mvir_subhalo_micro <= 10**11.3][is_central_micro[mvir_subhalo_micro <= 10**11.3] == 1][(subhalo_infall_satellite_micro[mvir_subhalo_micro[is_central_micro == 1] <= 10**11.3] > 0.0) & (subhalo_infall_satellite_micro[mvir_subhalo_micro[is_central_micro == 1] <= 10**11.3] <= 0.1)]



# 	plotting_scatter(virial_mass_scatter,property_plot = property_plot_scatter ,weights_property=fraction_BH,mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{vir}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=9,xlim_upper=14,ylim_lower=5,ylim_upper=12,colour_map='Spectral_r',legend_name='Individual haloes', colour_bar_title=None,norm_bound_upper=0.5, median_values=True, colour_bar_log = True, n_min=10)

# 	#############################################################################################################

# 	virial_mass 	= {'all' : mvir_subhalo, 'central' : mvir_subhalo[is_central == 0], 'satellite' :mvir_subhalo[is_central == 1]}

# 	property_plot   = {'all' : HI_medi, 'central' : HI_medi[is_central == 0], 'satellite' : HI_medi[is_central == 1]}



# 	virial_mass_scatter 	= {'all' : mvir_subhalo[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 1][(subhalo_infall_satellite[mvir_subhalo[is_central == 1] >= 10**11.15] > 0.0) & (subhalo_infall_satellite[mvir_subhalo[is_central == 1] >= 10**11.15] <= 0.1)], 'central' : mvir_subhalo[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 0], 'satellite' : mvir_subhalo[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 1]}

# 	property_plot_scatter   = {'all' : HI_medi[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 1][(subhalo_infall_satellite[mvir_subhalo[is_central == 1] >= 10**11.15] > 0.0) & (subhalo_infall_satellite[mvir_subhalo[is_central == 1] >= 10**11.15] <= 0.1)], 'central' : HI_medi[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 0], 'satellite' : HI_medi[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 1]}

	
# 	# virial_mass_scatter 	= {'all' : mvir_subhalo[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 1][(subhalo_infall_satellite[mvir_subhalo[is_central == 1] >= 10**11.15] > 0.6)], 'central' : mvir_subhalo[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 0], 'satellite' : mvir_subhalo[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 1]}

# 	# property_plot_scatter   = {'all' : HI_medi[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 1][(subhalo_infall_satellite[mvir_subhalo[is_central == 1] >= 10**11.15] > 0.6)], 'central' : HI_medi[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 0], 'satellite' : HI_medi[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 1]}


# 	fraction_BH = m_bh[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 1][(subhalo_infall_satellite[mvir_subhalo[is_central == 1] >= 10**11.15] > 0.0) & (subhalo_infall_satellite[mvir_subhalo[is_central == 1] >= 10**11.15] <= 0.1)]/Stellar_medi[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 1][(subhalo_infall_satellite[mvir_subhalo[is_central == 1] >= 10**11.15] > 0.0) & (subhalo_infall_satellite[mvir_subhalo[is_central == 1] >= 10**11.15] <= 0.1)]




# 	# plotting_properties_halo(mvir_subhalo[is_central == 1],HI_medi[is_central == 1],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13,colour_line='k',legend_name='Kappa - 0.002',linestyle_plot='-')


# 	plotting_properties_halo(mvir_subhalo[is_central == 1][(subhalo_infall_satellite > 0) & (subhalo_infall_satellite <= 0.1)],HI_medi[is_central == 1][(subhalo_infall_satellite > 0) & (subhalo_infall_satellite <= 0.1)],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13,colour_line='k',legend_name='Kappa - 0.002',linestyle_plot='--')



# 	# plotting_properties_halo(mvir_subhalo[is_central == 1][(subhalo_infall_satellite > 0.1) & (subhalo_infall_satellite <= 0.2)],HI_medi[is_central == 1][(subhalo_infall_satellite > 0.1) & (subhalo_infall_satellite <= 0.2)],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13,colour_line='k',legend_name='Kappa - 0.002',linestyle_plot='-.')


# 	# plotting_properties_halo(mvir_subhalo[is_central == 1][(subhalo_infall_satellite > 0.4) & (subhalo_infall_satellite <= 0.6)],HI_medi[is_central == 1][(subhalo_infall_satellite > 0.4) & (subhalo_infall_satellite <= 0.6)],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13,colour_line='k',legend_name='Kappa - 0.002',linestyle_plot=(0, (3, 2, 1, 2, 1, 2)))

# 	# plotting_properties_halo(mvir_subhalo[is_central == 1][(subhalo_infall_satellite > 0.6)],HI_medi[is_central == 1][(subhalo_infall_satellite > 0.6)],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13,colour_line='k',legend_name='Kappa - 0.002',linestyle_plot=':')



# 	# Common_module.plotting_properties_separate_merged(virial_mass,virial_mass,property_plot,mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13, first_legend=False,colour_line='k',legend_name='Kappa - 0.002')

# 	yo = plotting_scatter(virial_mass_scatter,property_plot = property_plot_scatter,weights_property=fraction_BH,mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10.2,xlim_upper=14,ylim_lower=6,ylim_upper=12,colour_map='Spectral_r',legend_name='Individual haloes', colour_bar_title=None,norm_bound_upper=0.05, median_values=True, colour_bar_log = True, n_min=20)


# 	# yo = plotting_scatter(virial_mass_scatter,property_plot = property_plot_scatter,weights_property=lambda_subhalo[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 1][(subhalo_infall_satellite[mvir_subhalo[is_central == 1] >= 10**11.15] > 0.6)],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10.2,xlim_upper=14,ylim_lower=6,ylim_upper=12,colour_map='Spectral_r',legend_name='Individual haloes', colour_bar_title=None,norm_bound_upper=0.05, median_values=True, colour_bar_log = True, n_min=20)


# 	plt.axvline(x=11.2, linestyle=':', color = 'k')

# 	fmt = FormatScalarFormatter("%.2f")
# 	fmt.set_powerlimits((0.0002, 0.006))
# 	# cbar = plt.colorbar(yo, ticks=[0.0002,0.0005,0.001,0.002,0.003],format=fmt)
# 	cbar = plt.colorbar(yo,format=fmt)
# 	# cbar.formatter.set_scientific(True)
# 	# cbar.formatter.set_powerlimits((0,0))
# 	cbar.ax.minorticks_on()
# 	cbar.update_ticks()

# 	cbar.set_label('$M_{BH}/M_{*}$', rotation=270, labelpad = 30)


# 	extra = mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="z = %s" %z_values[snapshot] )
# 	legendHandles.append(extra)

# 	# legendHandles.append(mlines.Line2D([],[],color='black',linestyle='-',label='All Satellites'))
# 	legendHandles.append(mlines.Line2D([],[],color='black',linestyle='--',label='0 < $z_{infall}$ $\leq$ 0.1'))
# 	# legendHandles.append(mlines.Line2D([],[],color='black',linestyle='-.',label='0.1 < $z_{infall}$ $\leq$ 0.2'))
# 	# legendHandles.append(mlines.Line2D([],[],color='black',linestyle='-.',label='0.2 < $z_{infall}$ $\leq$ 0.4'))
# 	# legendHandles.append(mlines.Line2D([],[],color='black',linestyle=(0, (3, 2, 1, 2, 1, 2)),label='0.4 < $z_{infall}$ $\leq$ 0.6'))

# 	# legendHandles.append(mlines.Line2D([],[],color='black',linestyle=':',label='$z_{infall}$ $\geq$ 0.6'))

# 	plt.legend(handles=legendHandles, markerscale=10)
# 	plt.savefig("new_plots_corrected/BH_fraction_subhalo_satellite_z_0_0p1.png" )

# 	plt.show()







############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
### Plot - zinfall coloured
######--------------------------------------------------------------------------------------------------------------------------------------
####****************************************************************************************************************************************



for snapshot in range(len(snapshot_avail)):

	# fig, ax = plt.subplots(figsize=(1,6))



	legendHandles = []
	virial_mass 	= {'all' : mvir_subhalo, 'central' : mvir_subhalo[is_central == 0], 'satellite' :mvir_subhalo[is_central == 1]}

	property_plot   = {'all' : HI_medi, 'central' : HI_medi[is_central == 0], 'satellite' : HI_medi[is_central == 1]}



	virial_mass_scatter 	= {'all' : mvir_subhalo[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 1], 'central' : mvir_subhalo[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 0], 'satellite' : mvir_subhalo[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 1]}

	property_plot_scatter   = {'all' : HI_medi[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 1], 'central' : HI_medi[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 0], 'satellite' : HI_medi[mvir_subhalo >= 10**11.15][is_central[mvir_subhalo >= 10**11.15] == 1]}

	

	virial_mass_scatter 	= {'all' : mvir_subhalo[is_central == 1], 'central' : mvir_subhalo[is_central == 0], 'satellite' : mvir_subhalo[is_central== 1]}

	property_plot_scatter   = {'all' : HI_medi[is_central == 1], 'central' : HI_medi[is_central == 0], 'satellite' : HI_medi[is_central == 1]}

	


	plotting_properties_halo(mvir_subhalo[is_central == 1],HI_medi[is_central == 1],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13,colour_line='k',legend_name='Kappa - 0.002',linestyle_plot='-')


	
	# plotting_properties_halo(mvir_subhalo[is_central == 1][(subhalo_infall_satellite > 0) & (subhalo_infall_satellite <= 0.1)],HI_medi[is_central == 1][(subhalo_infall_satellite > 0) & (subhalo_infall_satellite <= 0.1)],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13,colour_line='k',legend_name='Kappa - 0.002',linestyle_plot='--')



	# plotting_properties_halo(mvir_subhalo[is_central == 1][(subhalo_infall_satellite > 0.1) & (subhalo_infall_satellite <= 0.2)],HI_medi[is_central == 1][(subhalo_infall_satellite > 0.1) & (subhalo_infall_satellite <= 0.2)],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13,colour_line='k',legend_name='Kappa - 0.002',linestyle_plot='-.')


	# plotting_properties_halo(mvir_subhalo[is_central == 1][(subhalo_infall_satellite > 0.2) & (subhalo_infall_satellite <= 0.4)],HI_medi[is_central == 1][(subhalo_infall_satellite > 0.2) & (subhalo_infall_satellite <= 0.4)],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13,colour_line='k',legend_name='Kappa - 0.002',linestyle_plot=(0, (3, 2, 1, 2, 1, 2)))

	# plotting_properties_halo(mvir_subhalo[is_central == 1][(subhalo_infall_satellite > 0.6)],HI_medi[is_central == 1][(subhalo_infall_satellite > 0.6)],mean=False,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=13,colour_line='k',legend_name='Kappa - 0.002',linestyle_plot=':')




	# yo = plotting_scatter(virial_mass_scatter,property_plot = property_plot_scatter,weights_property=subhalo_infall_satellite[mvir_subhalo[is_central == 1] >= 10**11.15],mean=True,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=11,xlim_upper=13,ylim_lower=6,ylim_upper=11,colour_map='Spectral_r',legend_name='Individual haloes', colour_bar_title=None, median_values=True, colour_bar_log = True, n_min=20)

	yo = plotting_scatter(virial_mass_scatter,property_plot = property_plot_scatter,weights_property=subhalo_infall_satellite,mean=True,legend_handles=legendHandles,property_name_x='log_{10}(M_{subhalo}[M_{\odot}])', property_name_y='log_{10}(M_{HI}[M_{\odot}])',xlim_lower=10.5,xlim_upper=13,ylim_lower=6.01,ylim_upper=11,colour_map='Spectral_r',legend_name='Individual haloes', colour_bar_title=None, median_values=True, colour_bar_log = True, n_min=20)



	# plt.axvline(x=11.2, linestyle=':', color = 'k')
	
	# cbar = plt.colorbar(yo,ticks=[0,0.1,0.2,0.3,0.4,0.5],format='%.1f')#, format='%.2f')
	# cbar.ax.minorticks_on()
	# cbar.update_ticks()

	N = 1000
	cmap = plt.get_cmap('Spectral_r',N)

	norm = matplotlib.colors.LogNorm(vmin=0.01,vmax=0.5)
	sm 	= plt.cm.ScalarMappable(cmap=cmap, norm=norm)
	sm.set_array([])
	cbar = plt.colorbar(sm, ticks = [0.01,0.03,0.05,0.07,0.1,0.2,0.3,0.4], format='%.2f')#, ticks=np.logspace(0.01,0.5,N))

	cbar.set_label('$z_{\\rm infall}$', rotation=270, labelpad = 30)
	# cbar.set_yticklabels([np.logspace(0.01,0.5,N)])

	extra = mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="z = %s" %z_values[snapshot] )
	legendHandles.append(extra)

	legendHandles.append(mlines.Line2D([],[],color='black',linestyle='-',label='All Satellites'))
		


	# # 	legendHandles.append(mlines.Line2D([],[],color='black',linestyle='-',label='All Satellites'))
	# legendHandles.append(mlines.Line2D([],[],color='black',linestyle='--',label='0 < $z_{infall}$ $\leq$ 0.1'))
	# legendHandles.append(mlines.Line2D([],[],color='black',linestyle='-.',label='0.2 < $z_{infall}$ $\leq$ 0.4'))
	# legendHandles.append(mlines.Line2D([],[],color='black',linestyle=(0, (3, 2, 1, 2, 1, 2)),label='0.4 < $z_{infall}$ $\leq$ 0.6'))

	# legendHandles.append(mlines.Line2D([],[],color='black',linestyle=':',label='$z_{infall}$ $\geq$ 0.6'))

	plt.legend(handles=legendHandles, markerscale=10)
	plt.savefig("new_plots_corrected/z_infall_subhalo_satellite_lower_limit.png" )

	plt.show()






