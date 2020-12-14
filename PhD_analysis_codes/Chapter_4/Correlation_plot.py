from __future__ import print_function
import h5py as h5py
import numpy as np
import os
import scipy.stats as scipy
from scipy import stats
import re
import seaborn as sns
import collections
from astropy.stats import bootstrap
from astropy.utils import NumpyRNGContext
import Common_module
from Common_module import SharkDataReading
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import Corrfunc
from Corrfunc.theory import wp

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
c = 299792


###################################################
######--------------------------------------------------------------------------------------------------------------------------------------
### Preparing Data
######--------------------------------------------------------------------------------------------------------------------------------------
####****************************************************************************************************************************************

def prepare_data(path,simulation,run,snapshot,subvolumes):

	fields_read = {'galaxies' : ('type','mstars_disk', 'mstars_bulge', 'matom_disk','matom_bulge' , 'mgas_disk', 'mgas_bulge', 'position_x', 'position_y', 'position_z', 'm_bh'), 'halo' : ('age_50', 'age_80', 'lambda','mvir')}

	data_reading = SharkDataReading(path,simulation,run,snapshot,subvolumes)

	data_plotting = data_reading.readIndividualFiles(fields=fields_read)

	return data_plotting


############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
### Observations
######--------------------------------------------------------------------------------------------------------------------------------------
####****************************************************************************************************************************************

######--------------------------------------------------------------------------------------------------------------------------------------
### GUO 2017
######--------------------------------------------------------------------------------------------------------------------------------------



Guo_MHI_10_8_rp 	= np.array([0.1227819,0.20853947,0.32952935,0.5084339,0.78537744,1.2412754,1.9656094,3.1216574,4.83789])
Guo_MHI_10_8_wp		= np.array([60.944553,58.500393,56.123882,52.639153,43.149254,40.477413,30.33595,16.235306,9.086236])

Guo_MHI_10_9_rp 	= np.array([0.1265169,0.20435749,0.32277223,0.53441894,0.8431906,1.2685914,1.9996135,3.0793297,5.4764323,7.64559,13.576287,20.929209])
Guo_MHI_10_9_wp 	= np.array([70.810234,47.505707,46.390217,36.32053,27.820642,24.33245,14.9459305,10.95471,7.346773,3.541283,1.6682141,1.5588177]) 


Guo_MHI_10_10_rp 	= np.array([0.12439757,0.19722882,0.30520678,0.5204231,0.8053421,1.276847,2.1250377,3.536669,4.8478026,8.068127,13.105869,22.896173])
Guo_MHI_10_10_wp 	= np.array([204.5215,97.8008,53.48103,53.48103,35.763332,31.981073,22.869581,16.35398,12.788646,7.8203616,5.0008874,3.1979182]) 



######--------------------------------------------------------------------------------------------------------------------------------------
### OBLUJEN 2019 - ALFALFA 100
######--------------------------------------------------------------------------------------------------------------------------------------

Oblujen_rp 			= np.array([0.113884285,0.1496366,0.19407314,0.25170568,0.330725,0.43455127,0.57097226,0.73573166,0.96670383,1.2701863,1.6261053,2.1505318,2.7891603,3.6174378,4.722282,6.204773,8.152671,10.437131,13.713715,17.786182,23.218468,29.918373])

Oblujen_wp 			= np.array([163.46979,147.0648,110.36846,51.854027,58.515713,52.64338,48.081306,37.190754,32.9568,35.00983,26.274004,22.251152,17.473202,15.719682,13.312819,10.143025,8.853525,6.446589,5.139386,3.2174551,1.7581766,0.61992085])


new_Meyer_rp 		= np.array([-0.5602612,-0.40914178,-0.2630597,-0.10690299,0.03414179,0.18526119,0.3414179,0.4875,0.6386194,0.78973883,0.92574626,1.0869403,1.2330223,1.3841418])

new_Meyer_wp 		= np.array([2.3615818,2.2372882,1.8531073,1.7514124,1.5480226,1.3220339,1.0621469,0.88135594,0.6666667,0.39548022,0.056497175,-0.15819208,-0.40677965,-1.0282485])

new_Meyer_wp_upper 	= np.array([2.4858756,2.338983,1.9661016,1.8644068,1.661017,1.4124293,1.1977401,0.9830508,0.79096043,0.5536723,0.19209039,-0.11299435,-0.27118644,-0.79096043])

new_Meyer_wp_lower 	= np.array([2.2259886,2.1468925,1.7062147,1.6045197,1.40113,1.1638418,0.91525424,0.7457627,0.49717513,0.19209039,-0.18079096,-0.15819208,-0.6214689,-1.6158192])


######--------------------------------------------------------------------------------------------------------------------------------------
### Martin Ann - ALFALFA 40
######--------------------------------------------------------------------------------------------------------------------------------------

Martin_rp 		= np.array([0.14093524,0.17503998,0.2115876,0.26757836,0.32638147,0.409039,0.50802183,0.6140946,0.7765977,0.9645255,1.2087958,1.4878153,1.8312393,2.2743785,2.7245457,3.476774,4.2792983,5.2197185,6.600971,8.051603,10.090707])

Martin_wp 		= np.array([71.899345,82.56729,47.47617,47.98411,36.00058,48.240116,43.834126,40.471474,34.869373,29.566957,32.88701,27.298784,23.898525,20.481197,17.646172,18.025778,15.530633,14.187441,12.420294,10.701065,7.8595104])

Martin_wp_upper = np.array([103.24412,107.16218,64.98574,63.956604,45.49761,57.806747,54.231033,48.240116,41.342106,35.05541,37.366783,31.684595,27.298784,22.90256,20.15685,20.049881,17.740316,15.949372,14.11215,12.0300255,8.977754])

Martin_wp_lower = np.array([40.687397,57.806747,30.364145,32.53888,27.153913,37.566143,34.684326,32.19444,28.944302,24.805462,28.034819,23.147589,20.372507,17.740316,15.042632,15.613492,13.2392235,12.223602,10.701065,9.318455,6.6999083])


######--------------------------------------------------------------------------------------------------------------------------------------
### Meyer - HIPASS
######--------------------------------------------------------------------------------------------------------------------------------------

Meyer_rp 				= np.array([0.10661723,0.1638753,0.25126123,0.37335518,0.5795552,0.8798334,1.3355689,2.028852,3.0796144,4.6739345,7.091362,10.41607,15.988701,25.030848])

Meyer_wp_high 			= np.array([186.31184,213.09978,70.34887,54.42879,35.01734,20.084059,11.000997,8.707919,4.661236,2.3286607,0.9902716,0.5813547,0.3411629,0.0619031])
Meyer_wp_high_dn_err	= np.array([96.0188,148.0266,56.213356,35.37823,24.979733,13.387654,8.426408,6.0861754,3.261829,1.4886686,0.55248857,0.35585833,0.19071895,0.008362377])
Meyer_wp_high_up_err	= np.array([316.26978,307.92377,101.66205,78.64069,47.23745,24.705645,14.499678,10.962119,6.145427,3.2131956,1.5693314,0.859351,0.4816649,0.14175321])

Meyer_wp_low 			= np.array([253.91382,152.5396,83.50012,46.78896,25.613415,18.966255,10.382589,5.684194,3.2598577,2.004326,1.0718666,0.67459434,0.3138452,0.09609779])
Meyer_wp_low_dn_err 	= np.array([97.81656,88.87401,44.403934,32.800514,15.649414,12.974565,6.480594,3.3897378,1.8148301,0.971362,0.52005696,0.36680844,0.16331783,0.010052651])
Meyer_wp_low_up_err 	= np.array([331.16574,218.03957,94.89873,59.6709,32.678345,23.048805,12.344811,6.918586,4.5560694,2.7354054,1.5330439,0.9213008,0.41958833,0.19122127])


######--------------------------------------------------------------------------------------------------------------------------------------
### Papastergis - ALFALFA
######--------------------------------------------------------------------------------------------------------------------------------------


Papastergis_rp 			= np.array([-0.80708957,-0.69123137,-0.5854478,-0.48470148,-0.38899255,-0.2983209,-0.20261194,-0.09682836,0.0039179106,0.11473881,0.20037313,0.31119403,0.4119403,
0.50261194,0.61847013,0.6990672,0.79477614,0.9005597,1.0063432,1.0970149,1.2027985,1.2985075,1.4143656])

Papastergis_wp 			= np.array([2.9717515,2.3276837,2.5310733,2.2146893,2.0112994,1.9322034,1.841808,1.559322,1.5141243,1.40113,1.1525424,1.0508474,0.8361582,0.65536726,0.5762712,0.39548022,0.20338982,0.06779661,-0.14689265,-0.37288135,-0.6101695,-1.039548,-1.2316384])

Papastergis_wp_low 		= np.array([2.7344632,1.7740113,2.2711864,2.0903955,1.8644068,1.841808,1.7288135,1.480226,1.4124293,1.3333334,1.1186441,0.9491525,0.8135593,0.63276833,0.4858757,
0.3502825,0.13559322,0.011299435,-0.2259887,-0.46327683,-0.72316384,-1.1977401,-1.5028249])


Papastergis_wp_up 		= np.array([3.1638417,2.6553671,2.6553671,2.3841808,2.1694915,2.0451977,1.9548023,1.6836158,1.6271186,1.4915254,1.2542373,1.1299435,0.91525424,0.7344633,0.6440678,0.49717513,0.27118644,0.14689265,-0.07909604,-0.27118644,-0.519774,-0.8926554,-1.0508474])




############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
### Analysis
######--------------------------------------------------------------------------------------------------------------------------------------
####****************************************************************************************************************************************


path = '/mnt/su3ctm/gchauhan/SHArk_Out/HI_haloes/' 
# path = '/mnt/sshfs/pleiades_gchauhan/SHArk_Out/HI_haloes/'

shark_runs 		= ['Shark-Lagos18-default-br06','Shark-Lagos18-Kappa-10','Shark-Lagos18-Kappa-0','Shark-Lagos18-Kappa-1','Shark-Lagos18-default-br06-stripping-off']

shark_labels	= ['Lagos18 (Default)','Kappa = 0.02','Kappa = 0','Kappa = 1','Lagos18 (stripping off)']




simulation 		= ['medi-SURFS', 'micro-SURFS']
snapshot_avail	= [199,174,156]
z_values 		= [0,0.5,1]

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

	medi_Kappa_original_stellar[k], medi_Kappa_original_stellar_central[k], medi_Kappa_original_stellar_satellite[k],medi_Kappa_original_stellar_orphan[k],a = medi_Kappa_original[k].mergeValues_satellite('id_halo','mstars_bulge','mstars_disk')

	medi_substructure_original[k],a = medi_Kappa_original[k].mergeValuesNumberSubstructures('id_halo','id_subhalo')

	


# #####------------------------------------------------------------------------------------------------
# ## Micro-SURFS 
# #####------------------------------------------------------------------------------------------------

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

	micro_Kappa_original_stellar[k], micro_Kappa_original_stellar_central[k], micro_Kappa_original_stellar_satellite[k],micro_Kappa_original_stellar_orphan[k],a = micro_Kappa_original[k].mergeValues_satellite('id_halo','mstars_bulge','mstars_disk')

	micro_substructure_original[k],a = micro_Kappa_original[k].mergeValuesNumberSubstructures('id_halo','id_subhalo')

	


############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
### Correlation Function
######--------------------------------------------------------------------------------------------------------------------------------------
####****************************************************************************************************************************************


age_50 				= {}
age_80 				= {}
lambda_spin 		= {}
mvir 				= {}
is_central  		= {}
m_bh 				= {}
mstars_bulge 		= {}
mstars_disk 		= {}
matom_disk 			= {}
matom_bulge 		= {}
mgas_disk 			= {}
mgas_bulge 			= {}
position_x 			= {}
position_y 			= {}
position_z 			= {}



for snapshot in [199]:#snapshot_avail:

	boxsize = 210

	(h0,_, is_central[snapshot], mstars_disk[snapshot], mstars_bulge[snapshot], matom_disk[snapshot], matom_bulge[snapshot], mgas_disk[snapshot],mgas_bulge[snapshot], position_x[snapshot], position_y[snapshot], position_z[snapshot],m_bh[snapshot], age_50[snapshot], age_80[snapshot], lambda_spin[snapshot], mvir[snapshot]) = prepare_data(path,simulation[0],shark_runs[0],snapshot,subvolumes)

	for i in range(len(position_x[snapshot])):

		if i%1000 == 0:
			print(i)

		# print(i)
		if position_x[snapshot][i] < 0:
			position_x[snapshot][i] = boxsize + position_x[snapshot][i]
		
		if position_y[snapshot][i] < 0:
			position_y[snapshot][i] = boxsize + position_y[snapshot][i]
		
		if position_z[snapshot][i] < 0:
			position_z[snapshot][i] = boxsize + position_z[snapshot][i]
				
		if position_x[snapshot][i] > boxsize:
			position_x[snapshot][i] = position_x[snapshot][i] - boxsize 
		
		if position_y[snapshot][i] > boxsize:
			position_y[snapshot][i] =  position_y[snapshot][i] - boxsize 
		
		if position_z[snapshot][i] > boxsize:
			position_z[snapshot][i] = position_z[snapshot][i] - boxsize 

print('Done Medi-SURFS')		


age_50_micro 			= {}
age_80_micro 			= {}
lambda_spin_micro 		= {}
mvir_micro 				= {}
is_central_micro  		= {}
m_bh_micro 				= {}
mstars_bulge_micro 		= {}
mstars_disk_micro 		= {}
matom_disk_micro 		= {}
matom_bulge_micro 		= {}
mgas_disk_micro 		= {}
mgas_bulge_micro 		= {}
position_x_micro 		= {}
position_y_micro 		= {}
position_z_micro 		= {}



for snapshot in [199]:#snapshot_avail:

	boxsize = 40

	(h0,_, is_central_micro[snapshot], mstars_disk_micro[snapshot], mstars_bulge_micro[snapshot], matom_disk_micro[snapshot], matom_bulge_micro[snapshot], mgas_disk_micro[snapshot],mgas_bulge_micro[snapshot], position_x_micro[snapshot], position_y_micro[snapshot], position_z_micro[snapshot],m_bh_micro[snapshot], age_50_micro[snapshot], age_80_micro[snapshot], lambda_spin_micro[snapshot], mvir_micro[snapshot]) = prepare_data(path,simulation[1],shark_runs[0],snapshot,subvolumes)
	
	for i in range(len(position_x_micro[snapshot])):

		if i%1000 == 0:
			print(i)

		if position_x_micro[snapshot][i] < 0:
			position_x_micro[snapshot][i] = boxsize + position_x_micro[snapshot][i]
		
		if position_y_micro[snapshot][i] < 0:
			position_y_micro[snapshot][i] = boxsize + position_y_micro[snapshot][i]
		
		if position_z_micro[snapshot][i] < 0:
			position_z_micro[snapshot][i] = boxsize + position_z_micro[snapshot][i]
				
		if position_x_micro[snapshot][i] > boxsize:
			position_x_micro[snapshot][i] = position_x_micro[snapshot][i] - boxsize 
		
		if position_y_micro[snapshot][i] > boxsize:
			position_y_micro[snapshot][i] =  position_y_micro[snapshot][i] - boxsize 
		
		if position_z_micro[snapshot][i] > boxsize:
			position_z_micro[snapshot][i] = position_z_micro[snapshot][i] - boxsize 
		

print('Done Micro-SURFS')		


####****************************************************************************************************************************************
###### CORRFUNC - doing wp - that is the projected correlation function
####****************************************************************************************************************************************


# for snapshot in [0]:#range(len(snapshot_avail)):



# 	# -----------------------------------------------------------------------------
# 	# # Micro!!!!!
# 	# -----------------------------------------------------------------------------


# 	boxsize = 40
# 	pimax   = 10  ## Maximum separation along z axis
# 	nthreads = 4

# 	#----------------------------------------------------------------------------------------
# 	## Setting up bins
# 	#----------------------------------------------------------------------------------------
# 	rmin = 0.1
# 	rmax = 10

# 	nbins = 20

# 	#----------------------------------------------------------------------------------------
# 	## Halo_properties
# 	#----------------------------------------------------------------------------------------

# 	virial_mass 	= {'all' : micro_Kappa_original_vir[snapshot_avail[snapshot]], 'central' : micro_Kappa_original_vir[snapshot_avail[snapshot]], 'satellite' : micro_Kappa_original_vir[snapshot_avail[snapshot]]}

# 	stellar_property = {'all' : micro_Kappa_original_stellar[snapshot_avail[snapshot]], 'central' : micro_Kappa_original_stellar_central[snapshot_avail[snapshot]], 'satellite' : micro_Kappa_original_stellar_central[snapshot_avail[snapshot]]}

# 	property_plot = {'all' : micro_Kappa_original_HI[snapshot_avail[snapshot]], 'central' : micro_Kappa_original_HI_central[snapshot_avail[snapshot]], 'satellite' : micro_Kappa_original_HI_satellite[snapshot_avail[snapshot]]}

# 	HI_all 		=	(matom_disk_micro[snapshot_avail[snapshot]] + matom_bulge_micro[snapshot_avail[snapshot]])/1.35

# 	# HI_all 		= 	property_plot['all']
	
# 	#------------------------------------------------------------------------------
# 	## Creating the bins
# 	#------------------------------------------------------------------------------

# 	rbins = np.logspace(np.log10(rmin), np.log10(rmax), nbins  + 1)


# 	#------------------------------------------------------------------------------
# 	## calling wp
# 	#-----------------------------------------------------------------------------

# 	wp_results_10_8 = wp(boxsize, pimax, nthreads, rbins, position_x_micro[snapshot_avail[snapshot]][HI_all >= 10**9.25], position_y_micro[snapshot_avail[snapshot]][HI_all >= 10**9.25], position_z_micro[snapshot_avail[snapshot]][HI_all >= 10**9.25], verbose=True, output_rpavg=True)

# 	wp_results_10_8_5 = wp(boxsize, pimax, nthreads, rbins, position_x_micro[snapshot_avail[snapshot]][HI_all >= 10**8.5], position_y_micro[snapshot_avail[snapshot]][HI_all >= 10**8.5], position_z_micro[snapshot_avail[snapshot]][HI_all >= 10**8.5], verbose=True, output_rpavg=True)

# 	# wp_results_10_8 = wp(boxsize, pimax, nthreads, rbins, position_x_micro[snapshot_avail[snapshot]][is_central_micro[snapshot_avail[snapshot]] ==0][property_plot['all'] >= 10**8], position_y_micro[snapshot_avail[snapshot]][is_central_micro[snapshot_avail[snapshot]] ==0][property_plot['all'] >= 10**8], position_z_micro[snapshot_avail[snapshot]][is_central_micro[snapshot_avail[snapshot]] ==0][property_plot['all'] >= 10**8], verbose=True, output_rpavg=True)  

# 	# wp_results_10_8_5 = wp(boxsize, pimax, nthreads, rbins, position_x_micro[snapshot_avail[snapshot]][is_central_micro[snapshot_avail[snapshot]] ==0][property_plot['all'] >= 10**8.5], position_y_micro[snapshot_avail[snapshot]][is_central_micro[snapshot_avail[snapshot]] ==0][property_plot['all'] >= 10**8.5], position_z_micro[snapshot_avail[snapshot]][is_central_micro[snapshot_avail[snapshot]] ==0][property_plot['all'] >= 10**8.5], verbose=True, output_rpavg=True)  
	

# 	wp_results 	= [wp_results_10_8]#,wp_results_10_8_5]

# 	correlation_string = [8,8.5]

# 	legendHandles = list()

# 	mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="z = %s" %z_values[snapshot] )
# 	legendHandles.append(mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="z = %s" %z_values[snapshot] ))


# 	for wp_result_yo, j in zip(wp_results, range(len(wp_results))):
		

# 		wp_plot = np.zeros(len(rbins) -2)
# 		wp_ravg = np.zeros(len(rbins) -2)

# 		for i in range(len(wp_plot)):
# 			wp_plot[i] = wp_result_yo[i][3]
# 			wp_ravg[i] = wp_result_yo[i][2]


	

# 		plt.plot(wp_ravg,wp_plot,marker='*',color=colour_plot[j], linestyle='--',label = '$M_{HI} > 10^{%s}\ (Micro)$' %correlation_string[j])
		

# 		plt.xscale('log')
# 		plt.yscale('log')
# 		plt.xlim(0,20)
# 		plt.ylim(0,500)
		
# 		plt.ylabel('$\\rm w_p(r_p)[h^{-1} Mpc]$')
# 		plt.xlabel('$\\rm r_p[h^{-1} Mpc]$')
		
# 		legendHandles.append(mpatches.Patch(color=colour_plot[j],label='$M_{HI} > 10^{%s}\ (Micro)$' %correlation_string[j]))

	





	
# 	boxsize = 210
# 	pimax   = 10  ## Maximum separation along z axis
# 	nthreads = 4

# 	#----------------------------------------------------------------------------------------
# 	## Setting up bins
# 	#----------------------------------------------------------------------------------------
# 	rmin = 0.1
# 	rmax = 50

# 	nbins = 30

# 	#----------------------------------------------------------------------------------------
# 	## Halo_properties
# 	#----------------------------------------------------------------------------------------

# 	virial_mass 	= {'all' : medi_Kappa_original_vir[snapshot_avail[snapshot]], 'central' : medi_Kappa_original_vir[snapshot_avail[snapshot]], 'satellite' : medi_Kappa_original_vir[snapshot_avail[snapshot]]}

# 	stellar_property = {'all' : medi_Kappa_original_stellar[snapshot_avail[snapshot]], 'central' : medi_Kappa_original_stellar_central[snapshot_avail[snapshot]], 'satellite' : medi_Kappa_original_stellar_central[snapshot_avail[snapshot]]}

# 	property_plot = {'all' : medi_Kappa_original_HI[snapshot_avail[snapshot]], 'central' : medi_Kappa_original_HI_central[snapshot_avail[snapshot]], 'satellite' : medi_Kappa_original_HI_satellite[snapshot_avail[snapshot]]}

# 	HI_all 		=	(matom_disk[snapshot_avail[snapshot]] + matom_bulge[snapshot_avail[snapshot]])/1.35
	
# 	# HI_all 		= 	property_plot['all']

# 	#------------------------------------------------------------------------------
# 	## Creating the bins
# 	#------------------------------------------------------------------------------

# 	rbins = np.logspace(np.log10(rmin), np.log10(rmax), nbins  + 1)


# 	#------------------------------------------------------------------------------
# 	## calling wp
# 	#-----------------------------------------------------------------------------

	
# 	wp_results_10_9 = wp(boxsize, pimax, nthreads, rbins, position_x[snapshot_avail[snapshot]][HI_all >= 10**9], position_y[snapshot_avail[snapshot]][HI_all >= 10**9], position_z[snapshot_avail[snapshot]][HI_all >= 10**9], verbose=True, output_rpavg=True)

# 	wp_results_10_9_5 = wp(boxsize, pimax, nthreads, rbins, position_x[snapshot_avail[snapshot]][HI_all >= 10**9.5], position_y[snapshot_avail[snapshot]][HI_all >= 10**9.5],position_z[snapshot_avail[snapshot]][HI_all >= 10**9.5], verbose=True, output_rpavg=True)

# 	wp_results_10_10 = wp(boxsize, pimax, nthreads, rbins, position_x[snapshot_avail[snapshot]][HI_all >= 10**10], position_y[snapshot_avail[snapshot]][HI_all >= 10**10], position_z[snapshot_avail[snapshot]][HI_all >= 10**10], verbose=True, output_rpavg=True)

# 	wp_results_10_10_5 = wp(boxsize, pimax, nthreads, rbins, position_x[snapshot_avail[snapshot]][HI_all >= 10**10.5], position_y[snapshot_avail[snapshot]][HI_all >= 10**10.5], position_z[snapshot_avail[snapshot]][HI_all >= 10**10.5], verbose=True, output_rpavg=True)

	
# 	wp_results 	= [wp_results_10_9,wp_results_10_10]#wp_results_10_9_5,wp_results_10_10]#,wp_results_10_10_5,wp_results_10_11]

# 	correlation_string = [8,8.5,9,10,10.5,11]
	

# 	#-----------------------------------------------------------------------------
# 	## Assigning values
# 	#-----------------------------------------------------------------------------


	
# 	for wp_result_yo, j in zip(wp_results, [2,3]):
		

# 		wp_plot = np.zeros(len(rbins) -2)
# 		wp_ravg = np.zeros(len(rbins) -2)

# 		for i in range(len(wp_plot)):
# 			wp_plot[i] = wp_result_yo[i][3]
# 			wp_ravg[i] = wp_result_yo[i][2]


# 		plt.plot(wp_ravg,wp_plot,marker='*',color=colour_plot[j], label = '$M_{HI} > 10^{%s}$' %correlation_string[j])
		

# 		plt.xscale('log')
# 		plt.yscale('log')
# 		plt.xlim(0.1,50)
# 		plt.ylim(1,300)
		
# 		plt.ylabel('$\\rm w_p(r_p)[h^{-1} Mpc]$')
# 		plt.xlabel('$\\rm r_p[h^{-1} Mpc]$')
		
# 		legendHandles.append(mpatches.Patch(color=colour_plot[j],label='$M_{HI} > 10^{%s}$' %correlation_string[j]))

	

# 	# guo_10_8 = plt.plot(Guo_MHI_10_8_rp,Guo_MHI_10_8_wp, linestyle=':',color='k', label='$M_{HI} > 10^{8}$ (Guo+2017)')
# 	# guo_10_9 = plt.plot(Guo_MHI_10_9_rp,Guo_MHI_10_9_wp, linestyle=':',color='grey', label='$M_{HI} > 10^{9}$ (Guo+2017)')
# 	# guo_10_10 = plt.plot(Guo_MHI_10_10_rp,Guo_MHI_10_10_wp, linestyle=':',color='lightgrey', label='$M_{HI} > 10^{10}$ (Guo+2017)')
	
# 	# legendHandles.append(mpatches.Patch(color='k',label='$M_{HI} > 10^{8}$ (Guo+2017)'))
# 	# legendHandles.append(mpatches.Patch(color='grey',label='$M_{HI} > 10^{9}$ (Guo+2017)'))
# 	# legendHandles.append(mpatches.Patch(color='lightgrey',label='$M_{HI} > 10^{10}$ (Guo+2017)'))
	
# 	# legendHandles.append(guo_10_8)
# 	# legendHandles.append(guo_10_9)
# 	# legendHandles.append(guo_10_10)
	


# 	martin_plot_low = plt.errorbar(Meyer_rp,Meyer_wp_low, yerr=[Meyer_wp_low - Meyer_wp_low_dn_err, Meyer_wp_low_up_err - Meyer_wp_low],marker='X',color='k', label= 'HIPASS $M_{HI} < 10^{9.25}$ (Meyer+2007)')
	
# 	martin_plot_high = plt.errorbar(Meyer_rp,Meyer_wp_high, yerr=[Meyer_wp_high - Meyer_wp_high_dn_err, Meyer_wp_high_up_err - Meyer_wp_high],marker='X',color='grey', label= 'HIPASS $M_{HI} >= 10^{9.25}$ (Meyer+2007)')
	
# 	# legendHandles.append(mpatches.Patch(color='k',label='$M_{HI} > 10^{8}$ (Guo+2017)'))
# 	# legendHandles.append(mpatches.Patch(color='grey',label='$M_{HI} > 10^{9}$ (Guo+2017)'))
# 	# legendHandles.append(mpatches.Patch(color='lightgrey',label='$M_{HI} > 10^{10}$ (Guo+2017)'))
	
# 	legendHandles.append(martin_plot_low)
# 	legendHandles.append(martin_plot_high)

# 	plt.legend(handles=legendHandles)
# 	# plt.savefig('Plot/Paper_plots/Correlation_galaxy_Meyer_' + str(z_values[snapshot]) + '.png')
# 	plt.show()

# 	# plt.legend(handles=legendHandles)
# 	# # plt.savefig('Plot/Paper_plots/Correlation_galaxy_GUO_' + str(z_values[snapshot]) + '.png')
# 	# plt.show()




# for snapshot in [0]:#range(len(snapshot_avail)):


# 	# -----------------------------------------------------------------------------
# 	# # Micro!!!!!
# 	# -----------------------------------------------------------------------------


# 	boxsize = 45
# 	pimax   = 2  ## Maximum separation along z axis
# 	nthreads = 4

# 	#----------------------------------------------------------------------------------------
# 	## Setting up bins
# 	#----------------------------------------------------------------------------------------
# 	rmin = 0.1
# 	rmax = 15

# 	nbins = 20

# 	#----------------------------------------------------------------------------------------
# 	## Halo_properties
# 	#----------------------------------------------------------------------------------------

# 	virial_mass 	= {'all' : micro_Kappa_original_vir[snapshot_avail[snapshot]], 'central' : micro_Kappa_original_vir[snapshot_avail[snapshot]], 'satellite' : micro_Kappa_original_vir[snapshot_avail[snapshot]]}

# 	stellar_property = {'all' : micro_Kappa_original_stellar[snapshot_avail[snapshot]], 'central' : micro_Kappa_original_stellar_central[snapshot_avail[snapshot]], 'satellite' : micro_Kappa_original_stellar_central[snapshot_avail[snapshot]]}

# 	property_plot = {'all' : micro_Kappa_original_HI[snapshot_avail[snapshot]], 'central' : micro_Kappa_original_HI_central[snapshot_avail[snapshot]], 'satellite' : micro_Kappa_original_HI_satellite[snapshot_avail[snapshot]]}

# 	HI_all 		=	(matom_disk_micro[snapshot_avail[snapshot]] + matom_bulge_micro[snapshot_avail[snapshot]])/1.35

# 	# HI_all 		= 	property_plot['all']
	
# 	#------------------------------------------------------------------------------
# 	## Creating the bins
# 	#------------------------------------------------------------------------------

# 	rbins = np.logspace(np.log10(rmin), np.log10(rmax), nbins  + 1)


# 	#------------------------------------------------------------------------------
# 	## calling wp
# 	#-----------------------------------------------------------------------------

# 	wp_results_10_8 = wp(boxsize, pimax, nthreads, rbins, position_x_micro[snapshot_avail[snapshot]][(HI_all < 10**9.25) & (HI_all > 10**8)], position_y_micro[snapshot_avail[snapshot]][(HI_all < 10**9.25) & (HI_all > 10**8)], position_z_micro[snapshot_avail[snapshot]][(HI_all < 10**9.25) & (HI_all > 10**8)], verbose=True, output_rpavg=True)





# 	wp_results 	= [wp_results_10_8]#,wp_results_10_8_5]

# 	correlation_string = ['M_{HI} < 10^{9.25}',8.5]

# 	legendHandles = list()

# 	mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="z = %s" %z_values[snapshot] )
# 	legendHandles.append(mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="z = %s" %z_values[snapshot] ))


# 	for wp_result_yo, j in zip(wp_results, range(len(wp_results))):
		

# 		wp_plot = np.zeros(len(rbins) -2)
# 		wp_ravg = np.zeros(len(rbins) -2)

# 		for i in range(len(wp_plot)):
# 			wp_plot[i] = wp_result_yo[i][3]
# 			wp_ravg[i] = wp_result_yo[i][2]


	

# 		plt.plot(wp_ravg,wp_plot,marker='*',color=colour_plot[j], linestyle='--',label = '$%s$' %correlation_string[j])
		

# 		plt.xscale('log')
# 		plt.yscale('log')
# 		plt.xlim(0,20)
# 		plt.ylim(0,500)
		
# 		plt.ylabel('$\\rm w_p(r_p)[h^{-1} Mpc]$')
# 		plt.xlabel('$\\rm r_p[h^{-1} Mpc]$')
		
# 		legendHandles.append(mpatches.Patch(color=colour_plot[j],label='$%s$' %correlation_string[j]))

	







# 	# -----------------------------------------------------------------------------
# 	# # Micro!!!!!
# 	# -----------------------------------------------------------------------------


# 	boxsize = 210
# 	pimax   = 15  ## Maximum separation along z axis
# 	nthreads = 4

# 	#----------------------------------------------------------------------------------------
# 	## Setting up bins
# 	#----------------------------------------------------------------------------------------
# 	rmin = 0.1
# 	rmax = 20

# 	nbins = 15

# 	#----------------------------------------------------------------------------------------
# 	## Halo_properties
# 	#----------------------------------------------------------------------------------------

# 	virial_mass 	= {'all' : medi_Kappa_original_vir[snapshot_avail[snapshot]], 'central' : medi_Kappa_original_vir[snapshot_avail[snapshot]], 'satellite' : medi_Kappa_original_vir[snapshot_avail[snapshot]]}

# 	stellar_property = {'all' : medi_Kappa_original_stellar[snapshot_avail[snapshot]], 'central' : medi_Kappa_original_stellar_central[snapshot_avail[snapshot]], 'satellite' : medi_Kappa_original_stellar_central[snapshot_avail[snapshot]]}

# 	property_plot = {'all' : medi_Kappa_original_HI[snapshot_avail[snapshot]], 'central' : medi_Kappa_original_HI_central[snapshot_avail[snapshot]], 'satellite' : medi_Kappa_original_HI_satellite[snapshot_avail[snapshot]]}

# 	HI_all 		=	(matom_disk[snapshot_avail[snapshot]] + matom_bulge[snapshot_avail[snapshot]])/1.35

# 	# HI_all 		= 	property_plot['all']
	
# 	#------------------------------------------------------------------------------
# 	## Creating the bins
# 	#------------------------------------------------------------------------------

# 	rbins = np.logspace(np.log10(rmin), np.log10(rmax), nbins  + 1)


# 	#------------------------------------------------------------------------------
# 	## calling wp
# 	#-----------------------------------------------------------------------------

# 	# wp_results_10_9 = wp(boxsize, pimax, nthreads, rbins, position_x[snapshot_avail[snapshot]][(HI_all < 10**9.25) & (HI_all > 10**7)], position_y[snapshot_avail[snapshot]][(HI_all < 10**9.25) & (HI_all > 10**7)], position_z[snapshot_avail[snapshot]][(HI_all < 10**9.25) & (HI_all > 10**7)], verbose=True, output_rpavg=True)

# 	wp_results_10_10 = wp(boxsize, pimax, nthreads, rbins, position_x[snapshot_avail[snapshot]][HI_all >= 10**9.25], position_y[snapshot_avail[snapshot]][HI_all >= 10**9.25], position_z[snapshot_avail[snapshot]][HI_all >= 10**9.25], verbose=True, output_rpavg=True)

# 	# wp_results_10_9 = wp(boxsize, pimax, nthreads, rbins, position_x_micro[snapshot_avail[snapshot]][(HI_all < 10**9)], position_y_micro[snapshot_avail[snapshot]][(HI_all < 10**9)], position_z_micro[snapshot_avail[snapshot]][(HI_all < 10**9)], verbose=True, output_rpavg=True)

# 	# wp_results_10_10 = wp(boxsize, pimax, nthreads, rbins, position_x_micro[snapshot_avail[snapshot]][HI_all >= 10**9], position_y_micro[snapshot_avail[snapshot]][HI_all >= 10**9], position_z_micro[snapshot_avail[snapshot]][HI_all >= 10**9], verbose=True, output_rpavg=True)


# 	wp_results 	= [wp_results_10_10, wp_results_10_10]#,wp_results_10_8_5]

# 	correlation_string = ['M_{HI} >= 10^{9.25}']


# 	# mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="z = %s" %z_values[snapshot] )
# 	# legendHandles.append(mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="z = %s" %z_values[snapshot] ))


# 	for wp_result_yo, j in zip(wp_results, range(1,len(wp_results))):
		

# 		wp_plot = np.zeros(len(rbins) -2)
# 		wp_ravg = np.zeros(len(rbins) -2)

# 		for i in range(len(wp_plot)):
# 			wp_plot[i] = wp_result_yo[i][3]
# 			wp_ravg[i] = wp_result_yo[i][2]


	

# 		plt.plot(wp_ravg,wp_plot,marker='*',color=colour_plot[j], linestyle='--',label = '$%s$' %correlation_string[j-1])
		

# 		plt.xscale('log')
# 		plt.yscale('log')
# 		plt.xlim(0.1,20)
# 		plt.ylim(0.1,500)
		
# 		plt.ylabel('$\\rm w_p(r_p)[h^{-1} Mpc]$')
# 		plt.xlabel('$\\rm r_p[h^{-1} Mpc]$')
		
# 		legendHandles.append(mpatches.Patch(color=colour_plot[j],label='$%s$' %correlation_string[j-1]))

	
# 	martin_plot_low = plt.errorbar(Meyer_rp,Meyer_wp_low, yerr=[Meyer_wp_low - Meyer_wp_low_dn_err, Meyer_wp_low_up_err - Meyer_wp_low],marker='X',color='k', label= 'HIPASS $M_{HI} < 10^{9.25}$ (Meyer+2007)')
	
# 	martin_plot_high = plt.errorbar(Meyer_rp,Meyer_wp_high, yerr=[Meyer_wp_high - Meyer_wp_high_dn_err, Meyer_wp_high_up_err - Meyer_wp_high],marker='X',color='grey', label= 'HIPASS $M_{HI} >= 10^{9.25}$ (Meyer+2007)')
	
# 	# legendHandles.append(mpatches.Patch(color='k',label='$M_{HI} > 10^{8}$ (Guo+2017)'))
# 	# legendHandles.append(mpatches.Patch(color='grey',label='$M_{HI} > 10^{9}$ (Guo+2017)'))
# 	# legendHandles.append(mpatches.Patch(color='lightgrey',label='$M_{HI} > 10^{10}$ (Guo+2017)'))
	
# 	legendHandles.append(martin_plot_low)
# 	legendHandles.append(martin_plot_high)

# 	plt.legend(handles=legendHandles)
# 	# plt.savefig('Plot/Paper_plots/Correlation_galaxy_Meyer_' + str(z_values[snapshot]) + '.png')
# 	# plt.show()







# for snapshot in [0]:#range(len(snapshot_avail)):


# 	# -----------------------------------------------------------------------------
# 	# # Micro!!!!!
# 	# -----------------------------------------------------------------------------


# 	boxsize = 45
# 	pimax   = 2  ## Maximum separation along z axis
# 	nthreads = 4

# 	#----------------------------------------------------------------------------------------
# 	## Setting up bins
# 	#----------------------------------------------------------------------------------------
# 	rmin = 0.1
# 	rmax = 15

# 	nbins = 20

# 	#----------------------------------------------------------------------------------------
# 	## Halo_properties
# 	#----------------------------------------------------------------------------------------

# 	virial_mass 	= {'all' : micro_Kappa_original_vir[snapshot_avail[snapshot]], 'central' : micro_Kappa_original_vir[snapshot_avail[snapshot]], 'satellite' : micro_Kappa_original_vir[snapshot_avail[snapshot]]}

# 	stellar_property = {'all' : micro_Kappa_original_stellar[snapshot_avail[snapshot]], 'central' : micro_Kappa_original_stellar_central[snapshot_avail[snapshot]], 'satellite' : micro_Kappa_original_stellar_central[snapshot_avail[snapshot]]}

# 	property_plot = {'all' : micro_Kappa_original_HI[snapshot_avail[snapshot]], 'central' : micro_Kappa_original_HI_central[snapshot_avail[snapshot]], 'satellite' : micro_Kappa_original_HI_satellite[snapshot_avail[snapshot]]}

# 	HI_all 		=	(matom_disk_micro[snapshot_avail[snapshot]] + matom_bulge_micro[snapshot_avail[snapshot]])/1.35

# 	# HI_all 		= 	property_plot['all']
	
# 	#------------------------------------------------------------------------------
# 	## Creating the bins
# 	#------------------------------------------------------------------------------

# 	rbins = np.logspace(np.log10(rmin), np.log10(rmax), nbins  + 1)


# 	#------------------------------------------------------------------------------
# 	## calling wp
# 	#-----------------------------------------------------------------------------

# 	wp_results_10_8 = wp(boxsize, pimax, nthreads, rbins, position_x_micro[snapshot_avail[snapshot]][(HI_all < 10**9.25) & (HI_all > 10**8)], position_y_micro[snapshot_avail[snapshot]][(HI_all < 10**9.25) & (HI_all > 10**8)], position_z_micro[snapshot_avail[snapshot]][(HI_all < 10**9.25) & (HI_all > 10**8)], verbose=True, output_rpavg=True)





# 	wp_results 	= [wp_results_10_8]#,wp_results_10_8_5]

# 	correlation_string = ['M_{HI} < 10^{9.25}',8.5]

# 	legendHandles = list()

# 	mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="z = %s" %z_values[snapshot] )
# 	legendHandles.append(mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="z = %s" %z_values[snapshot] ))


# 	for wp_result_yo, j in zip(wp_results, range(len(wp_results))):
		

# 		wp_plot = np.zeros(len(rbins) -2)
# 		wp_ravg = np.zeros(len(rbins) -2)

# 		for i in range(len(wp_plot)):
# 			wp_plot[i] = wp_result_yo[i][3]
# 			wp_ravg[i] = wp_result_yo[i][2]


	

# 		plt.plot(wp_ravg,wp_plot,marker='*',color=colour_plot[j], linestyle='--',label = '$%s$' %correlation_string[j])
		

# 		plt.xscale('log')
# 		plt.yscale('log')
# 		plt.xlim(0,20)
# 		plt.ylim(0,500)
		
# 		plt.ylabel('$\\rm w_p(r_p)[h^{-1} Mpc]$')
# 		plt.xlabel('$\\rm r_p[h^{-1} Mpc]$')
		
# 		legendHandles.append(mpatches.Patch(color=colour_plot[j],label='$%s$' %correlation_string[j]))

	







# 	# -----------------------------------------------------------------------------
# 	# # Micro!!!!!
# 	# -----------------------------------------------------------------------------


# 	boxsize = 210
# 	pimax   = 30  ## Maximum separation along z axis
# 	nthreads = 4

# 	#----------------------------------------------------------------------------------------
# 	## Setting up bins
# 	#----------------------------------------------------------------------------------------
# 	rmin = 0.1
# 	rmax = 20

# 	nbins = 15

# 	#----------------------------------------------------------------------------------------
# 	## Halo_properties
# 	#----------------------------------------------------------------------------------------

# 	virial_mass 	= {'all' : medi_Kappa_original_vir[snapshot_avail[snapshot]], 'central' : medi_Kappa_original_vir[snapshot_avail[snapshot]], 'satellite' : medi_Kappa_original_vir[snapshot_avail[snapshot]]}

# 	stellar_property = {'all' : medi_Kappa_original_stellar[snapshot_avail[snapshot]], 'central' : medi_Kappa_original_stellar_central[snapshot_avail[snapshot]], 'satellite' : medi_Kappa_original_stellar_central[snapshot_avail[snapshot]]}

# 	property_plot = {'all' : medi_Kappa_original_HI[snapshot_avail[snapshot]], 'central' : medi_Kappa_original_HI_central[snapshot_avail[snapshot]], 'satellite' : medi_Kappa_original_HI_satellite[snapshot_avail[snapshot]]}

# 	HI_all 		=	(matom_disk[snapshot_avail[snapshot]] + matom_bulge[snapshot_avail[snapshot]])/1.35

# 	# HI_all 		= 	property_plot['all']
	
# 	#------------------------------------------------------------------------------
# 	## Creating the bins
# 	#------------------------------------------------------------------------------

# 	rbins = np.logspace(np.log10(rmin), np.log10(rmax), nbins  + 1)


# 	#------------------------------------------------------------------------------
# 	## calling wp
# 	#-----------------------------------------------------------------------------

# 	# wp_results_10_9 = wp(boxsize, pimax, nthreads, rbins, position_x[snapshot_avail[snapshot]][(HI_all < 10**9.25) & (HI_all > 10**7)], position_y[snapshot_avail[snapshot]][(HI_all < 10**9.25) & (HI_all > 10**7)], position_z[snapshot_avail[snapshot]][(HI_all < 10**9.25) & (HI_all > 10**7)], verbose=True, output_rpavg=True)

# 	wp_results_10_10 = wp(boxsize, pimax, nthreads, rbins, position_x[snapshot_avail[snapshot]][HI_all >= 10**9.25], position_y[snapshot_avail[snapshot]][HI_all >= 10**9.25], position_z[snapshot_avail[snapshot]][HI_all >= 10**9.25], verbose=True, output_rpavg=True)

# 	# wp_results_10_9 = wp(boxsize, pimax, nthreads, rbins, position_x_micro[snapshot_avail[snapshot]][(HI_all < 10**9)], position_y_micro[snapshot_avail[snapshot]][(HI_all < 10**9)], position_z_micro[snapshot_avail[snapshot]][(HI_all < 10**9)], verbose=True, output_rpavg=True)

# 	# wp_results_10_10 = wp(boxsize, pimax, nthreads, rbins, position_x_micro[snapshot_avail[snapshot]][HI_all >= 10**9], position_y_micro[snapshot_avail[snapshot]][HI_all >= 10**9], position_z_micro[snapshot_avail[snapshot]][HI_all >= 10**9], verbose=True, output_rpavg=True)


# 	wp_results 	= [wp_results_10_10, wp_results_10_10]#,wp_results_10_8_5]

# 	correlation_string = ['M_{HI} >= 10^{9.25}']


# 	# mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="z = %s" %z_values[snapshot] )
# 	# legendHandles.append(mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="z = %s" %z_values[snapshot] ))


# 	for wp_result_yo, j in zip(wp_results, range(1,len(wp_results))):
		

# 		wp_plot = np.zeros(len(rbins) -2)
# 		wp_ravg = np.zeros(len(rbins) -2)

# 		for i in range(len(wp_plot)):
# 			wp_plot[i] = wp_result_yo[i][3]
# 			wp_ravg[i] = wp_result_yo[i][2]


	

# 		plt.plot(wp_ravg,wp_plot,marker='*',color=colour_plot[j], linestyle='--',label = '$%s$' %correlation_string[j-1])
		

# 		plt.xscale('log')
# 		plt.yscale('log')
# 		plt.xlim(0.1,20)
# 		plt.ylim(0.1,500)
		
# 		plt.ylabel('$\\rm w_p(r_p)[h^{-1} Mpc]$')
# 		plt.xlabel('$\\rm r_p[h^{-1} Mpc]$')
		
# 		legendHandles.append(mpatches.Patch(color=colour_plot[j],label='$%s$' %correlation_string[j-1]))

	
# 	martin_plot_low = plt.errorbar(Meyer_rp,Meyer_wp_low, yerr=[Meyer_wp_low - Meyer_wp_low_dn_err, Meyer_wp_low_up_err - Meyer_wp_low],marker='X',color='k', label= 'HIPASS $M_{HI} < 10^{9.25}$ (Meyer+2007)')
	
# 	martin_plot_high = plt.errorbar(Meyer_rp,Meyer_wp_high, yerr=[Meyer_wp_high - Meyer_wp_high_dn_err, Meyer_wp_high_up_err - Meyer_wp_high],marker='X',color='grey', label= 'HIPASS $M_{HI} >= 10^{9.25}$ (Meyer+2007)')
	
# 	# legendHandles.append(mpatches.Patch(color='k',label='$M_{HI} > 10^{8}$ (Guo+2017)'))
# 	# legendHandles.append(mpatches.Patch(color='grey',label='$M_{HI} > 10^{9}$ (Guo+2017)'))
# 	# legendHandles.append(mpatches.Patch(color='lightgrey',label='$M_{HI} > 10^{10}$ (Guo+2017)'))
	
# 	legendHandles.append(martin_plot_low)
# 	legendHandles.append(martin_plot_high)

# 	plt.legend(handles=legendHandles)
# 	# plt.savefig('Plot/Paper_plots/Correlation_galaxy_Meyer_' + str(z_values[snapshot]) + '.png')
# 	plt.show()






######--------------------------------------------------------------------------------------------------------------------------------------
### OBLUJEN 2019 - ALFALFA 100
######--------------------------------------------------------------------------------------------------------------------------------------

Oblujen_rp 			= np.array([0.113884285,0.1496366,0.19407314,0.25170568,0.330725,0.43455127,0.57097226,0.73573166,0.96670383,1.2701863,1.6261053,2.1505318,2.7891603,3.6174378,4.722282,6.204773,8.152671,10.437131,13.713715,17.786182,23.218468,29.918373])

Oblujen_wp 			= np.array([163.46979,147.0648,110.36846,51.854027,58.515713,52.64338,48.081306,37.190754,32.9568,35.00983,26.274004,22.251152,17.473202,15.719682,13.312819,10.143025,8.853525,6.446589,5.139386,3.2174551,1.7581766,0.61992085])

Obuljen_wp_upper 	= np.array([206.0939,180.89703,145.26,69.50549,77.28397,62.98758,62.220417,49.235367,41.945744,43.96044,31.829893,27.93321,22.431704,19.68435,16.523716,14.078141,11.819868,8.815234,6.9755445,4.553331,2.9723997,1.4867996])

Obuljen_wp_lower 	= np.array([134.20477,123.126816,86.552124,42.66049,50.32287,46.85696,46.286263,34.518066,31.666803,34.184284,26.653036,23.045252,18.236965,15.539746,13.236526,10.320345,8.410713,6.654209,5.1123376,3.0540779,1.3372389,0.39867625])


# for snapshot in [0]:#range(len(snapshot_avail)):



# 	# -----------------------------------------------------------------------------
# 	# # Micro!!!!!
# 	# -----------------------------------------------------------------------------


# 	boxsize = 210
# 	pimax   = 15  ## Maximum separation along z axis
# 	nthreads = 4

# 	#----------------------------------------------------------------------------------------
# 	## Setting up bins
# 	#----------------------------------------------------------------------------------------
# 	rmin = 0.1
# 	rmax = 50

# 	nbins = 15

# 	#----------------------------------------------------------------------------------------
# 	## Halo_properties
# 	#----------------------------------------------------------------------------------------

# 	virial_mass 	= {'all' : medi_Kappa_original_vir[snapshot_avail[snapshot]], 'central' : micro_Kappa_original_vir[snapshot_avail[snapshot]], 'satellite' : medi_Kappa_original_vir[snapshot_avail[snapshot]]}

# 	stellar_property = {'all' : medi_Kappa_original_stellar[snapshot_avail[snapshot]], 'central' : medi_Kappa_original_stellar_central[snapshot_avail[snapshot]], 'satellite' : medi_Kappa_original_stellar_central[snapshot_avail[snapshot]]}

# 	property_plot = {'all' : medi_Kappa_original_HI[snapshot_avail[snapshot]], 'central' : medi_Kappa_original_HI_central[snapshot_avail[snapshot]], 'satellite' : medi_Kappa_original_HI_satellite[snapshot_avail[snapshot]]}

# 	HI_all 		=	(matom_disk[snapshot_avail[snapshot]] + matom_bulge[snapshot_avail[snapshot]])
# 	# HI_all 		= 	property_plot['all']
	
# 	#------------------------------------------------------------------------------
# 	## Creating the bins
# 	#------------------------------------------------------------------------------

# 	rbins = np.logspace(np.log10(rmin), np.log10(rmax), nbins  + 1)


# 	#------------------------------------------------------------------------------
# 	## calling wp
# 	#-----------------------------------------------------------------------------

	
# 	# wp_results_10_9 = wp(boxsize, pimax, nthreads, rbins, position_x[snapshot_avail[snapshot]][(HI_all < 10**9.25)], position_y[snapshot_avail[snapshot]][(HI_all < 10**9.25)], position_z[snapshot_avail[snapshot]][(HI_all < 10**9.25)], verbose=True, output_rpavg=True)

# 	wp_results_10_10 = wp(boxsize, pimax, nthreads, rbins, position_x[snapshot_avail[snapshot]][HI_all >= 10**9], position_y[snapshot_avail[snapshot]][HI_all >= 10**9], position_z[snapshot_avail[snapshot]][HI_all >= 10**9], verbose=True, output_rpavg=True)


# 	wp_results 	= [wp_results_10_10]#, wp_results_10_10]#,wp_results_10_8_5]

# 	correlation_string = ['Lagos18 (Medi-SURFS)']

# 	legendHandles = list()

# 	mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="z = %s" %z_values[snapshot] )
# 	legendHandles.append(mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="z = %s" %z_values[snapshot] ))


# 	for wp_result_yo, j in zip(wp_results, range(len(wp_results))):
		

# 		wp_plot = np.zeros(len(rbins) -2)
# 		wp_ravg = np.zeros(len(rbins) -2)

# 		for i in range(len(wp_plot)):
# 			wp_plot[i] = wp_result_yo[i][3]
# 			wp_ravg[i] = wp_result_yo[i][2]


	

# 		plt.plot(wp_ravg,wp_plot,marker='*',color=colour_plot[j], linestyle='--',label = '%s' %correlation_string[j])
		

# 		plt.xscale('log')
# 		plt.yscale('log')
# 		plt.xlim(0.1,20)
# 		plt.ylim(0.1,500)
		
# 		plt.ylabel('$\\rm w_p(r_p)[h^{-1} Mpc]$')
# 		plt.xlabel('$\\rm r_p[h^{-1} Mpc]$')
		
# 		legendHandles.append(mpatches.Patch(color=colour_plot[j],label='%s' %correlation_string[j]))

	



# 	# -----------------------------------------------------------------------------
# 	# # Micro!!!!!
# 	# -----------------------------------------------------------------------------


# 	boxsize = 40
# 	pimax   = 10  ## Maximum separation along z axis
# 	nthreads = 4

# 	#----------------------------------------------------------------------------------------
# 	## Setting up bins
# 	#----------------------------------------------------------------------------------------
# 	rmin = 0.1
# 	rmax = 1

# 	nbins = 10

# 	#----------------------------------------------------------------------------------------
# 	## Halo_properties
# 	#----------------------------------------------------------------------------------------

# 	virial_mass 	= {'all' : micro_Kappa_original_vir[snapshot_avail[snapshot]], 'central' : micro_Kappa_original_vir[snapshot_avail[snapshot]], 'satellite' : medi_Kappa_original_vir[snapshot_avail[snapshot]]}

# 	stellar_property = {'all' : micro_Kappa_original_stellar[snapshot_avail[snapshot]], 'central' : medi_Kappa_original_stellar_central[snapshot_avail[snapshot]], 'satellite' : medi_Kappa_original_stellar_central[snapshot_avail[snapshot]]}

# 	property_plot = {'all' : micro_Kappa_original_HI[snapshot_avail[snapshot]], 'central' : medi_Kappa_original_HI_central[snapshot_avail[snapshot]], 'satellite' : medi_Kappa_original_HI_satellite[snapshot_avail[snapshot]]}

# 	HI_all 		=	(matom_disk_micro[snapshot_avail[snapshot]] + matom_bulge_micro[snapshot_avail[snapshot]])/1.35
# 	# HI_all 		= 	property_plot['all']
	
# 	#------------------------------------------------------------------------------
# 	## Creating the bins
# 	#------------------------------------------------------------------------------

# 	rbins = np.logspace(np.log10(rmin), np.log10(rmax), nbins  + 1)


# 	#------------------------------------------------------------------------------
# 	## calling wp
# 	#-----------------------------------------------------------------------------

	
# 	# wp_results_10_9 = wp(boxsize, pimax, nthreads, rbins, position_x[snapshot_avail[snapshot]][(HI_all < 10**9.25)], position_y[snapshot_avail[snapshot]][(HI_all < 10**9.25)], position_z[snapshot_avail[snapshot]][(HI_all < 10**9.25)], verbose=True, output_rpavg=True)

# 	wp_results_10_10 = wp(boxsize, pimax, nthreads, rbins, position_x_micro[snapshot_avail[snapshot]][HI_all >= 10**8], position_y_micro[snapshot_avail[snapshot]][HI_all >= 10**8], position_z_micro[snapshot_avail[snapshot]][HI_all >= 10**8], verbose=True, output_rpavg=True)


# 	wp_results 	= [wp_results_10_10]#, wp_results_10_10]#,wp_results_10_8_5]

# 	correlation_string = ['Lagos18']

# 	# legendHandles = list()

# 	# mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="z = %s" %z_values[snapshot] )
# 	# legendHandles.append(mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="z = %s" %z_values[snapshot] ))


# 	for wp_result_yo, j in zip(wp_results, range(len(wp_results))):
		

# 		wp_plot = np.zeros(len(rbins) -2)
# 		wp_ravg = np.zeros(len(rbins) -2)

# 		for i in range(len(wp_plot)):
# 			wp_plot[i] = wp_result_yo[i][3]
# 			wp_ravg[i] = wp_result_yo[i][2]


	

# 		plt.plot(wp_ravg,wp_plot,marker='*',color=colour_plot[2], linestyle='--',label = '%s' %correlation_string[j])
		

# 		plt.xscale('log')
# 		plt.yscale('log')
# 		plt.xlim(0.1,20)
# 		plt.ylim(0.1,500)
		
# 		plt.ylabel('$\\rm w_p(r_p)[h^{-1} Mpc]$')
# 		plt.xlabel('$\\rm r_p[h^{-1} Mpc]$')
		
# 		legendHandles.append(mpatches.Patch(color=colour_plot[2],label='%s (Micro-SURFS)' %correlation_string[j]))

	
# 	# martin_plot_low = plt.errorbar(Meyer_rp,Meyer_wp_low, yerr=[Meyer_wp_low - Meyer_wp_low_dn_err, Meyer_wp_low_up_err - Meyer_wp_low],marker='X',color='k', label= 'HIPASS $M_{HI} < 10^{9.25}$ (Meyer+2007)')
	
# 	# martin_plot_high = plt.errorbar(Meyer_rp,Meyer_wp_high, yerr=[Meyer_wp_high - Meyer_wp_high_dn_err, Meyer_wp_high_up_err - Meyer_wp_high],marker='X',color='grey', label= 'HIPASS $M_{HI} >= 10^{9.25}$ (Meyer+2007)')
	
# 	# legendHandles.append(mpatches.Patch(color='k',label='$M_{HI} > 10^{8}$ (Guo+2017)'))
# 	# legendHandles.append(mpatches.Patch(color='grey',label='$M_{HI} > 10^{9}$ (Guo+2017)'))
# 	# legendHandles.append(mpatches.Patch(color='lightgrey',label='$M_{HI} > 10^{10}$ (Guo+2017)'))
	
# 	# Obuljen_plot = plt.errorbar(Oblujen_rp,Oblujen_wp, yerr=None ,marker='X',color='k', label= 'ALFALFA 100% (Oblujen+2019)')
# 	# martin_plot = plt.errorbar(Martin_rp,Martin_wp,yerr=[Martin_wp-Martin_wp_lower,Martin_wp_upper-Martin_wp], marker='X',color='grey', label= 'ALFALFA 40% (Martin+2012)')

	
# 	Obuljen_plot = plt.fill_between(Oblujen_rp,Obuljen_wp_lower,Obuljen_wp_upper,color='k',alpha=0.1,label='ALFALFA 100% (Oblujen+2019)')

# 	martin_plot = plt.fill_between(Martin_rp,Martin_wp_lower,Martin_wp_upper,color='grey',alpha=0.1,label='ALFALFA 40% (Martin+2019)')		


# 	# legendHandles.append(mpatches.Patch(color='k',label='$M_{HI} > 10^{8}$ (Guo+2017)'))
# 	# legendHandles.append(mpatches.Patch(color='grey',label='$M_{HI} > 10^{9}$ (Guo+2017)'))
# 	# legendHandles.append(mpatches.Patch(color='lightgrey',label='$M_{HI} > 10^{10}$ (Guo+2017)'))
	
# 	# legendHandles.append(martin_plot_low)
# 	# legendHandles.append(martin_plot_high)


# 	legendHandles.append(Obuljen_plot)
# 	legendHandles.append(martin_plot)
# 	plt.legend(handles=legendHandles)
	
# 	# plt.savefig('new_plots_corrected/Correlation_galaxy_alfalfa_' + str(z_values[snapshot]) + '.png')
# 	# plt.show()










# for snapshot in [0]:#range(len(snapshot_avail)):



# 	# -----------------------------------------------------------------------------
# 	# # Micro!!!!!
# 	# -----------------------------------------------------------------------------


# 	boxsize = 210
# 	pimax   = 30  ## Maximum separation along z axis
# 	nthreads = 4

# 	#----------------------------------------------------------------------------------------
# 	## Setting up bins
# 	#----------------------------------------------------------------------------------------
# 	rmin = 0.1
# 	rmax = 50

# 	nbins = 15

# 	#----------------------------------------------------------------------------------------
# 	## Halo_properties
# 	#----------------------------------------------------------------------------------------

# 	virial_mass 	= {'all' : medi_Kappa_original_vir[snapshot_avail[snapshot]], 'central' : micro_Kappa_original_vir[snapshot_avail[snapshot]], 'satellite' : medi_Kappa_original_vir[snapshot_avail[snapshot]]}

# 	stellar_property = {'all' : medi_Kappa_original_stellar[snapshot_avail[snapshot]], 'central' : medi_Kappa_original_stellar_central[snapshot_avail[snapshot]], 'satellite' : medi_Kappa_original_stellar_central[snapshot_avail[snapshot]]}

# 	property_plot = {'all' : medi_Kappa_original_HI[snapshot_avail[snapshot]], 'central' : medi_Kappa_original_HI_central[snapshot_avail[snapshot]], 'satellite' : medi_Kappa_original_HI_satellite[snapshot_avail[snapshot]]}

# 	HI_all 		=	(matom_disk[snapshot_avail[snapshot]] + matom_bulge[snapshot_avail[snapshot]])
# 	# HI_all 		= 	property_plot['all']
	
# 	#------------------------------------------------------------------------------
# 	## Creating the bins
# 	#------------------------------------------------------------------------------

# 	rbins = np.logspace(np.log10(rmin), np.log10(rmax), nbins  + 1)


# 	#------------------------------------------------------------------------------
# 	## calling wp
# 	#-----------------------------------------------------------------------------

	
# 	# wp_results_10_9 = wp(boxsize, pimax, nthreads, rbins, position_x[snapshot_avail[snapshot]][(HI_all < 10**9.25)], position_y[snapshot_avail[snapshot]][(HI_all < 10**9.25)], position_z[snapshot_avail[snapshot]][(HI_all < 10**9.25)], verbose=True, output_rpavg=True)

# 	wp_results_10_10 = wp(boxsize, pimax, nthreads, rbins, position_x[snapshot_avail[snapshot]][HI_all >= 10**9], position_y[snapshot_avail[snapshot]][HI_all >= 10**9], position_z[snapshot_avail[snapshot]][HI_all >= 10**9], verbose=True, output_rpavg=True)


# 	wp_results 	= [wp_results_10_10]#, wp_results_10_10]#,wp_results_10_8_5]

# 	correlation_string = ['Lagos18 (Medi-SURFS)']

# 	legendHandles = list()

# 	mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="z = %s" %z_values[snapshot] )
# 	legendHandles.append(mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="z = %s" %z_values[snapshot] ))


# 	for wp_result_yo, j in zip(wp_results, range(len(wp_results))):
		

# 		wp_plot = np.zeros(len(rbins) -2)
# 		wp_ravg = np.zeros(len(rbins) -2)

# 		for i in range(len(wp_plot)):
# 			wp_plot[i] = wp_result_yo[i][3]
# 			wp_ravg[i] = wp_result_yo[i][2]


	

# 		plt.plot(wp_ravg,wp_plot,marker='*',color=colour_plot[j], linestyle='--',label = '%s' %correlation_string[j])
		

# 		plt.xscale('log')
# 		plt.yscale('log')
# 		plt.xlim(0.1,20)
# 		plt.ylim(0.1,500)
		
# 		plt.ylabel('$\\rm w_p(r_p)[h^{-1} Mpc]$')
# 		plt.xlabel('$\\rm r_p[h^{-1} Mpc]$')
		
# 		legendHandles.append(mpatches.Patch(color=colour_plot[j],label='%s' %correlation_string[j]))

	



# 	# -----------------------------------------------------------------------------
# 	# # Micro!!!!!
# 	# -----------------------------------------------------------------------------


# 	boxsize = 40
# 	pimax   = 10  ## Maximum separation along z axis
# 	nthreads = 4

# 	#----------------------------------------------------------------------------------------
# 	## Setting up bins
# 	#----------------------------------------------------------------------------------------
# 	rmin = 0.1
# 	rmax = 1

# 	nbins = 10

# 	#----------------------------------------------------------------------------------------
# 	## Halo_properties
# 	#----------------------------------------------------------------------------------------

# 	virial_mass 	= {'all' : micro_Kappa_original_vir[snapshot_avail[snapshot]], 'central' : micro_Kappa_original_vir[snapshot_avail[snapshot]], 'satellite' : medi_Kappa_original_vir[snapshot_avail[snapshot]]}

# 	stellar_property = {'all' : micro_Kappa_original_stellar[snapshot_avail[snapshot]], 'central' : medi_Kappa_original_stellar_central[snapshot_avail[snapshot]], 'satellite' : medi_Kappa_original_stellar_central[snapshot_avail[snapshot]]}

# 	property_plot = {'all' : micro_Kappa_original_HI[snapshot_avail[snapshot]], 'central' : medi_Kappa_original_HI_central[snapshot_avail[snapshot]], 'satellite' : medi_Kappa_original_HI_satellite[snapshot_avail[snapshot]]}

# 	HI_all 		=	(matom_disk_micro[snapshot_avail[snapshot]] + matom_bulge_micro[snapshot_avail[snapshot]])/1.35
# 	# HI_all 		= 	property_plot['all']
	
# 	#------------------------------------------------------------------------------
# 	## Creating the bins
# 	#------------------------------------------------------------------------------

# 	rbins = np.logspace(np.log10(rmin), np.log10(rmax), nbins  + 1)


# 	#------------------------------------------------------------------------------
# 	## calling wp
# 	#-----------------------------------------------------------------------------

	
# 	# wp_results_10_9 = wp(boxsize, pimax, nthreads, rbins, position_x[snapshot_avail[snapshot]][(HI_all < 10**9.25)], position_y[snapshot_avail[snapshot]][(HI_all < 10**9.25)], position_z[snapshot_avail[snapshot]][(HI_all < 10**9.25)], verbose=True, output_rpavg=True)

# 	wp_results_10_10 = wp(boxsize, pimax, nthreads, rbins, position_x_micro[snapshot_avail[snapshot]][HI_all >= 10**8], position_y_micro[snapshot_avail[snapshot]][HI_all >= 10**8], position_z_micro[snapshot_avail[snapshot]][HI_all >= 10**8], verbose=True, output_rpavg=True)


# 	wp_results 	= [wp_results_10_10]#, wp_results_10_10]#,wp_results_10_8_5]

# 	correlation_string = ['Lagos18']

# 	# legendHandles = list()

# 	# mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="z = %s" %z_values[snapshot] )
# 	# legendHandles.append(mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="z = %s" %z_values[snapshot] ))


# 	for wp_result_yo, j in zip(wp_results, range(len(wp_results))):
		

# 		wp_plot = np.zeros(len(rbins) -2)
# 		wp_ravg = np.zeros(len(rbins) -2)

# 		for i in range(len(wp_plot)):
# 			wp_plot[i] = wp_result_yo[i][3]
# 			wp_ravg[i] = wp_result_yo[i][2]


	

# 		plt.plot(wp_ravg,wp_plot,marker='*',color=colour_plot[2], linestyle='--',label = '%s' %correlation_string[j])
		

# 		plt.xscale('log')
# 		plt.yscale('log')
# 		plt.xlim(0.1,20)
# 		plt.ylim(0.1,500)
		
# 		plt.ylabel('$\\rm w_p(r_p)[h^{-1} Mpc]$')
# 		plt.xlabel('$\\rm r_p[h^{-1} Mpc]$')
		
# 		legendHandles.append(mpatches.Patch(color=colour_plot[2],label='%s (Micro-SURFS)' %correlation_string[j]))

	
# 	# martin_plot_low = plt.errorbar(Meyer_rp,Meyer_wp_low, yerr=[Meyer_wp_low - Meyer_wp_low_dn_err, Meyer_wp_low_up_err - Meyer_wp_low],marker='X',color='k', label= 'HIPASS $M_{HI} < 10^{9.25}$ (Meyer+2007)')
	
# 	# martin_plot_high = plt.errorbar(Meyer_rp,Meyer_wp_high, yerr=[Meyer_wp_high - Meyer_wp_high_dn_err, Meyer_wp_high_up_err - Meyer_wp_high],marker='X',color='grey', label= 'HIPASS $M_{HI} >= 10^{9.25}$ (Meyer+2007)')
	
# 	# legendHandles.append(mpatches.Patch(color='k',label='$M_{HI} > 10^{8}$ (Guo+2017)'))
# 	# legendHandles.append(mpatches.Patch(color='grey',label='$M_{HI} > 10^{9}$ (Guo+2017)'))
# 	# legendHandles.append(mpatches.Patch(color='lightgrey',label='$M_{HI} > 10^{10}$ (Guo+2017)'))
	
# 	# Obuljen_plot = plt.errorbar(Oblujen_rp,Oblujen_wp, yerr=None ,marker='X',color='k', label= 'ALFALFA 100% (Oblujen+2019)')
# 	# martin_plot = plt.errorbar(Martin_rp,Martin_wp,yerr=[Martin_wp-Martin_wp_lower,Martin_wp_upper-Martin_wp], marker='X',color='grey', label= 'ALFALFA 40% (Martin+2012)')

	
# 	Obuljen_plot = plt.fill_between(Oblujen_rp,Obuljen_wp_lower,Obuljen_wp_upper,color='k',alpha=0.1,label='ALFALFA 100% (Oblujen+2019)')

# 	martin_plot = plt.fill_between(Martin_rp,Martin_wp_lower,Martin_wp_upper,color='grey',alpha=0.1,label='ALFALFA 40% (Martin+2019)')		


# 	# legendHandles.append(mpatches.Patch(color='k',label='$M_{HI} > 10^{8}$ (Guo+2017)'))
# 	# legendHandles.append(mpatches.Patch(color='grey',label='$M_{HI} > 10^{9}$ (Guo+2017)'))
# 	# legendHandles.append(mpatches.Patch(color='lightgrey',label='$M_{HI} > 10^{10}$ (Guo+2017)'))
	
# 	# legendHandles.append(martin_plot_low)
# 	# legendHandles.append(martin_plot_high)


# 	legendHandles.append(Obuljen_plot)
# 	legendHandles.append(martin_plot)
# 	plt.legend(handles=legendHandles)
	
# 	plt.savefig('new_plots_corrected/Correlation_galaxy_alfalfa_' + str(z_values[snapshot]) + '.png')
# 	plt.show()




plt = Common_module.load_matplotlib(12,8)
colour_plot = Common_module.colour_scheme()

legendHandles = list()






for snapshot in [0]:#range(len(snapshot_avail)):




	# -----------------------------------------------------------------------------
	# # Micro!!!!!
	# -----------------------------------------------------------------------------

	# legendHandles = []

	boxsize = 210
	pimax   = 30  ## Maximum separation along z axis
	nthreads = 4

	#----------------------------------------------------------------------------------------
	## Setting up bins
	#----------------------------------------------------------------------------------------
	rmin = 0.1
	rmax = 50

	nbins = 15

	#----------------------------------------------------------------------------------------
	## Halo_properties
	#----------------------------------------------------------------------------------------

	virial_mass 	= {'all' : medi_Kappa_original_vir[snapshot_avail[snapshot]], 'central' : medi_Kappa_original_vir[snapshot_avail[snapshot]], 'satellite' : medi_Kappa_original_vir[snapshot_avail[snapshot]]}

	stellar_property = {'all' : medi_Kappa_original_stellar[snapshot_avail[snapshot]], 'central' : medi_Kappa_original_stellar_central[snapshot_avail[snapshot]], 'satellite' : medi_Kappa_original_stellar_central[snapshot_avail[snapshot]]}

	property_plot = {'all' : medi_Kappa_original_HI[snapshot_avail[snapshot]], 'central' : medi_Kappa_original_HI_central[snapshot_avail[snapshot]], 'satellite' : medi_Kappa_original_HI_satellite[snapshot_avail[snapshot]]}

	HI_all 		=	(matom_disk[snapshot_avail[snapshot]] + matom_bulge[snapshot_avail[snapshot]])

	# HI_all 		= 	property_plot['all']
	
	#------------------------------------------------------------------------------
	## Creating the bins
	#------------------------------------------------------------------------------

	rbins = np.logspace(np.log10(rmin), np.log10(rmax), nbins  + 1)


	#------------------------------------------------------------------------------
	## calling wp
	#-----------------------------------------------------------------------------

	# wp_results_10_9 = wp(boxsize, pimax, nthreads, rbins, position_x[snapshot_avail[snapshot]][(HI_all < 10**9.25) & (HI_all > 10**7)], position_y[snapshot_avail[snapshot]][(HI_all < 10**9.25) & (HI_all > 10**7)], position_z[snapshot_avail[snapshot]][(HI_all < 10**9.25) & (HI_all > 10**7)], verbose=True, output_rpavg=True)

	wp_results_10_10 = wp(boxsize, pimax, nthreads, rbins, position_x[snapshot_avail[snapshot]][HI_all >= 10**9.25], position_y[snapshot_avail[snapshot]][HI_all >= 10**9.25], position_z[snapshot_avail[snapshot]][HI_all >= 10**9.25], verbose=True, output_rpavg=True)

	# wp_results_10_10 = wp(boxsize, pimax, nthreads, rbins, position_x[snapshot_avail[snapshot]][(HI_all < 10**9.25) & (HI_all > 10**8)], position_y[snapshot_avail[snapshot]][(HI_all < 10**9.25) & (HI_all > 10**8)], position_z[snapshot_avail[snapshot]][(HI_all < 10**9.25) & (HI_all > 10**8)], verbose=True, output_rpavg=True)



	# wp_results_10_9 = wp(boxsize, pimax, nthreads, rbins, position_x_micro[snapshot_avail[snapshot]][(HI_all < 10**9)], position_y_micro[snapshot_avail[snapshot]][(HI_all < 10**9)], position_z_micro[snapshot_avail[snapshot]][(HI_all < 10**9)], verbose=True, output_rpavg=True)

	# wp_results_10_10 = wp(boxsize, pimax, nthreads, rbins, position_x_micro[snapshot_avail[snapshot]][HI_all >= 10**9], position_y_micro[snapshot_avail[snapshot]][HI_all >= 10**9], position_z_micro[snapshot_avail[snapshot]][HI_all >= 10**9], verbose=True, output_rpavg=True)


	wp_results 	= [wp_results_10_10, wp_results_10_10]#,wp_results_10_8_5]

	correlation_string = ['SHARK-ref (medi-SURFS)']


	# mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="z = %s" %z_values[snapshot] )
	# legendHandles.append(mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="z = %s" %z_values[snapshot] ))


	for wp_result_yo, j in zip(wp_results, range(1,len(wp_results))):
		

		wp_plot = np.zeros(len(rbins) -2)
		wp_ravg = np.zeros(len(rbins) -2)

		for i in range(len(wp_plot)):
			wp_plot[i] = wp_result_yo[i][3]
			wp_ravg[i] = wp_result_yo[i][2]


	

		plt.plot(np.log10(wp_ravg/h),np.log10((wp_plot/wp_ravg)),marker='*',color='green', linestyle='--',label = '$%s$' %correlation_string[j-1])
		

		# plt.xscale('log')
		# plt.yscale('log')
		# plt.xlim(0.1,20)
		# plt.ylim(0.1,500)
		
		plt.ylabel('$\\rm w_p(r_p)/(r_p)$')
		plt.xlabel('$\\rm r_p[h^{-1} Mpc]$')
		
		legendHandles.append(mpatches.Patch(color='green',label='%s' %correlation_string[j-1]))


	# -----------------------------------------------------------------------------
	# # Micro!!!!!
	# -----------------------------------------------------------------------------

	# legendHandles = []

	boxsize = 40
	pimax   = 10  ## Maximum separation along z axis
	nthreads = 4

	#----------------------------------------------------------------------------------------
	## Setting up bins
	#----------------------------------------------------------------------------------------
	rmin = 0.1
	rmax = 3.5

	nbins = 10

	#----------------------------------------------------------------------------------------
	## Halo_properties
	#----------------------------------------------------------------------------------------

	virial_mass 	= {'all' : micro_Kappa_original_vir[snapshot_avail[snapshot]], 'central' : medi_Kappa_original_vir[snapshot_avail[snapshot]], 'satellite' : medi_Kappa_original_vir[snapshot_avail[snapshot]]}

	stellar_property = {'all' : micro_Kappa_original_stellar[snapshot_avail[snapshot]], 'central' : medi_Kappa_original_stellar_central[snapshot_avail[snapshot]], 'satellite' : medi_Kappa_original_stellar_central[snapshot_avail[snapshot]]}

	property_plot = {'all' : micro_Kappa_original_HI[snapshot_avail[snapshot]], 'central' : medi_Kappa_original_HI_central[snapshot_avail[snapshot]], 'satellite' : medi_Kappa_original_HI_satellite[snapshot_avail[snapshot]]}

	HI_all 		=	(matom_disk_micro[snapshot_avail[snapshot]] + matom_bulge_micro[snapshot_avail[snapshot]])

	# HI_all 		= 	property_plot['all']
	
	#------------------------------------------------------------------------------
	## Creating the bins
	#------------------------------------------------------------------------------

	rbins = np.logspace(np.log10(rmin), np.log10(rmax), nbins  + 1)


	#------------------------------------------------------------------------------
	## calling wp
	#-----------------------------------------------------------------------------

	# wp_results_10_9 = wp(boxsize, pimax, nthreads, rbins, position_x[snapshot_avail[snapshot]][(HI_all < 10**9.25) & (HI_all > 10**7)], position_y[snapshot_avail[snapshot]][(HI_all < 10**9.25) & (HI_all > 10**7)], position_z[snapshot_avail[snapshot]][(HI_all < 10**9.25) & (HI_all > 10**7)], verbose=True, output_rpavg=True)

	wp_results_10_10 = wp(boxsize, pimax, nthreads, rbins, position_x_micro[snapshot_avail[snapshot]][HI_all >= 10**7], position_y_micro[snapshot_avail[snapshot]][HI_all >= 10**7], position_z_micro[snapshot_avail[snapshot]][HI_all >= 10**7], verbose=True, output_rpavg=True)

	# wp_results_10_10 = wp(boxsize, pimax, nthreads, rbins, position_x[snapshot_avail[snapshot]][(HI_all < 10**9.25) & (HI_all > 10**8)], position_y[snapshot_avail[snapshot]][(HI_all < 10**9.25) & (HI_all > 10**8)], position_z[snapshot_avail[snapshot]][(HI_all < 10**9.25) & (HI_all > 10**8)], verbose=True, output_rpavg=True)



	# wp_results_10_9 = wp(boxsize, pimax, nthreads, rbins, position_x_micro[snapshot_avail[snapshot]][(HI_all < 10**9)], position_y_micro[snapshot_avail[snapshot]][(HI_all < 10**9)], position_z_micro[snapshot_avail[snapshot]][(HI_all < 10**9)], verbose=True, output_rpavg=True)

	# wp_results_10_10 = wp(boxsize, pimax, nthreads, rbins, position_x_micro[snapshot_avail[snapshot]][HI_all >= 10**9], position_y_micro[snapshot_avail[snapshot]][HI_all >= 10**9], position_z_micro[snapshot_avail[snapshot]][HI_all >= 10**9], verbose=True, output_rpavg=True)


	wp_results 	= [wp_results_10_10, wp_results_10_10]#,wp_results_10_8_5]

	correlation_string = ['SHARK-ref (micro-SURFS)']


	# mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="z = %s" %z_values[snapshot] )
	# legendHandles.append(mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label ="z = %s" %z_values[snapshot] ))


	for wp_result_yo, j in zip(wp_results, range(1,len(wp_results))):
		

		wp_plot = np.zeros(len(rbins) -2)
		wp_ravg = np.zeros(len(rbins) -2)

		for i in range(len(wp_plot)):
			wp_plot[i] = wp_result_yo[i][3]
			wp_ravg[i] = wp_result_yo[i][2]


	

		plt.plot(np.log10(wp_ravg/h),np.log10((wp_plot/wp_ravg)),marker='*',color='red', linestyle='--',label = '$%s$' %correlation_string[j-1])
		

		# plt.xscale('log')
		# plt.yscale('log')
		# plt.xlim(0.1,20)
		# plt.ylim(0.1,500)
		
		plt.ylabel('$\\rm w_p(r_p)/(r_p)$')
		plt.xlabel('$\\rm r_p[h^{-1} Mpc]$')
		
		legendHandles.append(mpatches.Patch(color='red',label='%s' %correlation_string[j-1]))

	
	
	# martin_plot_low = plt.errorbar(new_Meyer_rp + 0.17,new_Meyer_wp, yerr=[new_Meyer_wp - new_Meyer_wp_lower, new_Meyer_wp_upper - new_Meyer_wp],marker='X',color='k', label= 'HIPASS $M_{HI} < 10^{9.25}$ (Meyer+2007)')
	
	
	martin_plot_low = plt.fill_between(new_Meyer_rp + 0.17,new_Meyer_wp_lower,new_Meyer_wp_upper,color='k',hatch='*',label='HIPASS (Meyer+2007) ', alpha = 0.1)
	papastergis_plot = plt.fill_between(Papastergis_rp + 0.17,Papastergis_wp_low,Papastergis_wp_up,color='brown',hatch='o',label='ALFALFA (Papastergis+2013) ', alpha = 0.1)

	# martin_plot_high = plt.errorbar(np.log10(Meyer_rp),np.log10(Meyer_wp_high), yerr=[Meyer_wp_high - Meyer_wp_high_dn_err, Meyer_wp_high_up_err - Meyer_wp_high],marker='X',color='grey', label= 'HIPASS $M_{HI} >= 10^{9.25}$ (Meyer+2007)')
	
	# legendHandles.append(mpatches.Patch(color='k',label='$M_{HI} > 10^{8}$ (Guo+2017)'))
	# legendHandles.append(mpatches.Patch(color='grey',label='$M_{HI} > 10^{9}$ (Guo+2017)'))
	# legendHandles.append(mpatches.Patch(color='lightgrey',label='$M_{HI} > 10^{10}$ (Guo+2017)'))
	
	legendHandles.append(martin_plot_low)
	# legendHandles.append(martin_plot_high)
	legendHandles.append(papastergis_plot)
	
	plt.legend(handles=legendHandles)
	plt.savefig('new_plots_corrected/Correlation_galaxy_final.png')
	plt.show()


