import h5py as h5py
import numpy as np
import os
import scipy.stats as scipy
import re
from scipy import stats
from astropy.stats import bootstrap
from astropy.utils import NumpyRNGContext
from astropy.cosmology import Planck15
# from astropy.cosmology.FLRW import angular_diameter_distance
import time 
import collections
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.colors as colors
import splotch as spl
import statistics as statistics
import csv as csv


import pandas as pd
import splotch as spl

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck15

############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
### Plot Formatting
######--------------------------------------------------------------------------------------------------------------------------------------
####****************************************************************************************************************************************

def load_matplotlib(fig_size_a=12,figsize_b=8):
	import matplotlib.pyplot as mpl

	figSize = (fig_size_a,figsize_b)
	labelFS = 24 #30 #24
	tickFS =  18 #24 #18
	titleFS = 24 #30 #24
	textFS =  16 #24 #16
	legendFS = 18 #24 #18
	linewidth_plot = 3
	markersize_lines_plot = 15

	# Adjust axes
	mpl.rcParams['axes.linewidth'] = 1
	mpl.rcParams['axes.axisbelow'] = 'line'

	# Adjust Fonts
	mpl.rcParams['font.family'] = "sans-serif"
	mpl.rcParams['font.sans-serif'] = "Computer Modern Sans Serif"
	mpl.rcParams['font.style'] = "normal"
	mpl.rcParams['mathtext.fontset'] = "custom"
	mpl.rcParams['mathtext.rm'] = "Computer Modern Sans Serif"
	mpl.rcParams['mathtext.it'] = "Computer Modern Sans Serif"
	mpl.rcParams['mathtext.bf'] = "Computer Modern Sans Serif"
	

	mpl.rcParams['axes.titlesize'] = labelFS
	mpl.rcParams['axes.labelsize'] = labelFS
	mpl.rcParams['xtick.labelsize'] = tickFS
	mpl.rcParams['ytick.labelsize'] = tickFS
	mpl.rcParams['legend.fontsize'] = legendFS

	# Adjust line-widths
	mpl.rcParams['lines.linewidth'] = linewidth_plot
	mpl.rcParams['lines.markersize'] = markersize_lines_plot

	#Adjust Legend
	mpl.rcParams['legend.markerscale'] = 1
	mpl.rcParams['legend.fancybox'] = False
	mpl.rcParams['legend.frameon'] = False

	# Adjust ticks
	for a in ['x','y']:
		mpl.rcParams['{0}tick.major.size'.format(a)] = 5
		mpl.rcParams['{0}tick.minor.size'.format(a)] = 2.5
		
		mpl.rcParams['{0}tick.major.width'.format(a)] = 1
		mpl.rcParams['{0}tick.minor.width'.format(a)] = 1
		
		mpl.rcParams['{0}tick.direction'.format(a)] = 'in'
		mpl.rcParams['{0}tick.minor.visible'.format(a)] = True

	mpl.rcParams['xtick.top'] = True
	mpl.rcParams['ytick.right'] = True

	# Adjust figure and subplots
	mpl.rcParams['figure.figsize'] = figSize
	mpl.rcParams['figure.subplot.left'] = 0.1
	mpl.rcParams['figure.subplot.right'] = 0.96
	mpl.rcParams['figure.subplot.top'] = 0.96
	mpl.rcParams['savefig.bbox'] = 'tight'

	# Error-bars
	mpl.rcParams['errorbar.capsize'] = 3
	# mpl.rcParams['errorbar.capthick'] = 2
	# mpl.rcParams['errorbar.linewidth'] = 2	
	# mpl.rcParams['errorbar.ecolor'] = 'k'

	return mpl 

plt = load_matplotlib(12,8)


def nanpercentile_lower(arr,q=15.87): 
	arr = arr[~np.isnan(arr)]
	
	if (len(arr) != 0): return (np.nanpercentile(a=arr,q=q))
	else: return None
	

def nanpercentile_upper(arr,q=84.13): 
	arr = arr[~np.isnan(arr)]
	
	if (len(arr) != 0): return (np.nanpercentile(a=arr,q=q))
	else: return None



def plotting_properties_halo(virial_mass,property_plot,mean,legend_handles,property_name_x='X-Label-math-mode',legend_name='name-of-run', property_name_y='Y-Label-math-mode', xlim_lower=6,xlim_upper=15,ylim_lower=-4,ylim_upper=11,colour_line='r',fill_between=False, first_legend = False, resolution=False, halo_bins=0, alpha=1):
	

	if halo_bins == 0:
		bin_for_disk = np.arange(7,15,0.1)

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



def plotting_properties_halo_fill_between(virial_mass,property_plot,mean,legend_handles,property_name_x='X-Label-math-mode',legend_name='name-of-run', property_name_y='Y-Label-math-mode', xlim_lower=6,xlim_upper=15,ylim_lower=-4,ylim_upper=11,colour_line='r',fill_between=False, first_legend = False, resolution=False, halo_bins=0):
	

	if halo_bins == 0:
		bin_for_disk = np.arange(7,15,0.1)

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




def plotting_properties_halo_spline_testing(virial_mass,property_plot,mean,legend_handles,property_name_x='X-Label-math-mode',legend_name='name-of-run', property_name_y='Y-Label-math-mode', xlim_lower=6,xlim_upper=15,ylim_lower=-4,ylim_upper=11,colour_line='r',fill_between=False, first_legend = False, resolution=False, halo_bins=0):
	

	if halo_bins == 0:
		bin_for_disk = np.arange(-4,15,0.1)

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
						
					

	if resolution == True:
		plt.plot((halo_mass[0:len(halo_mass)-1] - halo_mass[0]),(prop_mass[0:len(halo_mass)-1] - prop_mass[0]),color=colour_line, linestyle='-')
		# plt.plot(halo_mass[0:len(halo_mass)-1],prop_mass[0:len(halo_mass)-1],color=colour_line, label='$\\rm %s$' %legend_name)

		if first_legend == True:
			legend_handles.append(mlines.Line2D([],[],color='black',linestyle='-',label='Medi-SURFS'))
			legend_handles.append(mlines.Line2D([],[],color='black',linestyle='-.',label='Micro-SURFS'))
			legend_handles.append(mpatches.Patch(color=colour_line,label=legend_name))

		
		else:
			mpatches.Patch(color=colour_line, label=legend_name)
			# legend_handles.append(mpatches.Patch(color=colour_line,label=legend_name))

	else:
		plt.plot((halo_mass[0:len(halo_mass)-1] - min(halo_mass[0:len(halo_mass)-1][prop_mass != 0])),(prop_mass[0:len(halo_mass)-1] - min(prop_mass[0:len(halo_mass)-1][prop_mass != 0])),color=colour_line, label='$\\rm %s$' %legend_name)
		# legend_handles.append(mpatches.Patch(color=colour_line,label=legend_name))

	if fill_between == True:
		plt.fill_between((halo_mass[(prop_mass_high != 0)] - halo_mass[0]),(prop_mass_low[(prop_mass_high != 0)] - prop_mass_low[0]),(prop_mass_high[(prop_mass_high != 0)] - prop_mass_high[0]),color=colour_line,alpha=0.1, label='$\\rm %s$' %legend_name)
	
	plt.xlabel('$\\rm %s$ '%property_name_x)
	plt.ylabel('$\\rm %s$ '%property_name_y)

	plt.xlim(xlim_lower,xlim_upper)
	plt.ylim(ylim_lower,ylim_upper)
#plt.legend(frameon=False)
	# plt.savefig('%s.png'%figure_name)
	# plt.show()




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

			# if len(bootarr) > 10:
			# 	if bootarr != []:
			# 		with NumpyRNGContext(1):
			# 			bootresult 			= bootstrap(bootarr,10,bootfunc=np.mean)
			# 			bootresult_error	= bootstrap(bootarr,10,bootfunc=stats.tstd)/2

			# 		prop_mass_low[i-1]	= prop_mass[i-1] - np.average(bootresult_error)
			# 		prop_mass_high[i-1]	= np.average(bootresult_error) + prop_mass[i-1]
					
			
	
	else:
		for i in range(1,len(bin_for_disk)):
			halo_mass[i-1] 		= np.log10(np.median(virial_mass[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])]))
			prop_mass[i-1]			= np.log10(np.median(property_plot[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])]))
			
			bootarr 			= np.log10(property_plot[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])])
			bootarr = bootarr[bootarr != float('-inf')]

			print(len(virial_mass[(virial_mass < 10**bin_for_disk[i]) & (virial_mass >= 10**bin_for_disk[i-1])]))
			
			# if len(bootarr) > 10 :
			# 	if bootarr != []:
			# 		with NumpyRNGContext(1):
			# 			bootresult 			= bootstrap(bootarr,10,bootfunc=np.median)
			# 			bootresult_lower	= bootstrap(bootarr,10,bootfunc=nanpercentile_lower)
			# 			bootresult_upper	= bootstrap(bootarr,10,bootfunc=nanpercentile_upper)

			# 		prop_mass_low[i-1]	= np.mean(bootresult_lower)

			# 		prop_mass_high[i-1]	= np.mean(bootresult_upper)


	return halo_mass, prop_mass, prop_mass_low, prop_mass_high





def colour_scheme(n=10,colourmap=plt.cm.Set1):
	colours_plot = colourmap(np.linspace(0,1,n))
	return colours_plot
	


def plotting_scatter(virial_mass, property_plot, weights_property,legend_handles,mean=True, property_name_x='Property_x',property_name_y='Property_y',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=12,colour_map='cool_r', legend_name='name-of-run',colour_bar_title='property of scatter',norm_bound_lower=0,norm_bound_upper=1,median_values=False,n_min=10,colour_bar_log=False):
	
	(property_all, property_central, property_satellite) = property_plot.items()
	property_all = np.array(property_all[1])
	property_central = np.array(property_central[1])
	property_satellite = np.array(property_satellite[1])

	(halo_mass, halo_mass_central, halo_mass_satellite) = virial_mass.items()
	halo_mass = np.array(halo_mass[1])
	halo_mass_central = np.array(halo_mass_central[1])
	halo_mass_satellite = np.array(halo_mass_satellite[1])


	bin_for_disk = np.arange(5,16,0.2)

	stellar_mass 	= np.zeros(len(bin_for_disk))
	prop_mass 		= np.zeros(len(bin_for_disk))


	stellar_mass_central 	= np.zeros(len(bin_for_disk))
	prop_mass_central 		= np.zeros(len(bin_for_disk))

	stellar_mass_satellite 	= np.zeros(len(bin_for_disk))
	prop_mass_satellite 	= np.zeros(len(bin_for_disk))
	
	prop_mass_low 			= np.zeros(len(bin_for_disk))
	prop_mass_high			= np.zeros(len(bin_for_disk))
	



	if colour_bar_log == False:
		bounds = np.linspace(norm_bound_lower, norm_bound_upper, 1000)
		norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
		
		
		if median_values == False:
			scatter_yo = plt.scatter(np.log10(halo_mass),np.log10(property_all), c=weights_property,cmap=colour_map, s= 0.1, alpha=0.8, label=legend_name, norm=norm)
			cbar = plt.colorbar(scatter_yo)
			cbar.set_label(colour_bar_title, rotation=270, labelpad = 20)
			cbar.ax.tick_params(labelsize=16)		
			plt.xlabel('$\\rm %s$ '%property_name_x)
			plt.ylabel('$\\rm %s$ '%property_name_y)

		else:
		
			scatter_yo=spl.hist2D(np.log10(halo_mass),np.log10(property_all),c=weights_property,cstat='median',bins=bin_for_disk,output=True,norm=norm,clabel=colour_bar_title,cmap=colour_map,nmin=n_min)
			plt.xlabel('$\\rm %s$ '%property_name_x)
			plt.ylabel('$\\rm %s$ '%property_name_y)

	else:

		bounds = np.logspace(np.log10(norm_bound_lower), np.log10(norm_bound_upper), 100)
		norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256) #colors.LogNorm(bounds)#
	
		if median_values == False:
			scatter_yo = plt.scatter(np.log10(halo_mass),np.log10(property_all), c=weights_property,cmap=colour_map, s= 0.1, alpha=0.8, label=legend_name, norm=norm)
			cbar = plt.colorbar(scatter_yo)
			cbar.ax.set_yticklabels(['{.1f}'.format(x) for x in np.arange(norm_bound_lower, norm_bound_upper)])
			cbar.set_label(colour_bar_title, rotation=270, labelpad = 20)
			cbar.ax.tick_params(labelsize=16)		
			plt.xlabel('$\\rm %s$ '%property_name_x)
			plt.ylabel('$\\rm %s$ '%property_name_y)

		else:
		
			scatter_yo=spl.hist2D(np.log10(halo_mass),np.log10(property_all),c=weights_property,cstat='median',bins=bin_for_disk,output=True,clabel=colour_bar_title,norm=norm,cmap=colour_map,nmin=n_min)
			# cbar = plt.colorbar(scatter_yo)
			# cbar.set_label(colour_bar_title, rotation=270, labelpad = 20)
			# cbar.ax.set_yticklabels(['{.1f}'.format(x) for x in np.arange(norm_bound_lower, norm_bound_upper)])
			# cbar.ax.tick_params(labelsize=16)		
			plt.xlabel('$\\rm %s$ '%property_name_x)
			plt.ylabel('$\\rm %s$ '%property_name_y)

		
		
	# legend_handles.append(scatter_yo)
	# contour_plot = plt.contourf(X,Y,A,10,cmap='bone_r',vmin=100) 
	# plt.contourf(X,Y,B,10,cmap='inferno_r',vmin=100) 
	
	plt.xlim(xlim_lower,xlim_upper)
	plt.ylim(ylim_lower,ylim_upper)



def plotting_scatter_without_log(virial_mass, property_plot, weights_property,legend_handles,mean=True, property_name_x='Property_x',property_name_y='Property_y',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=12,colour_map='cool_r', legend_name='name-of-run',colour_bar_title='property of scatter',norm_bound_lower=0,norm_bound_upper=1,median_values=False,n_min=10,colour_bar_log=False):
	

	bin_for_disk = np.arange(-5,5,0.1)

	

	(property_all, property_central, property_satellite) = property_plot.items()
	property_all = np.array(property_all[1])
	property_central = np.array(property_central[1])
	property_satellite = np.array(property_satellite[1])

	(halo_mass, halo_mass_central, halo_mass_satellite) = virial_mass.items()
	halo_mass = np.array(halo_mass[1])
	halo_mass_central = np.array(halo_mass_central[1])
	halo_mass_satellite = np.array(halo_mass_satellite[1])


	


	if colour_bar_log == False:
		bounds = np.linspace(norm_bound_lower, norm_bound_upper, 1000)
		norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
		
		
		if median_values == False:
			scatter_yo = plt.scatter(halo_mass,property_all, c=weights_property,cmap=colour_map, s= 0.1, alpha=0.8, label=legend_name, norm=norm)
			cbar = plt.colorbar(scatter_yo)
			cbar.set_label(colour_bar_title, rotation=270, labelpad = 20)
			cbar.ax.tick_params(labelsize=16)		
			plt.xlabel('$\\rm %s$ '%property_name_x)
			plt.ylabel('$\\rm %s$ '%property_name_y)

		else:
		
			scatter_yo=spl.hist2D(halo_mass,property_all,c=weights_property,cstat='median',bins=bin_for_disk,output=True,norm=norm,clabel=colour_bar_title,cmap=colour_map,nmin=n_min)
	
	else:

		bounds = np.logspace(np.log10(norm_bound_lower), np.log10(norm_bound_upper), 1000)
		norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
	
		if median_values == False:
			scatter_yo = plt.scatter(halo_mass,property_all, c=weights_property,cmap=colour_map, s= 0.1, alpha=0.8, label=legend_name, norm=norm)
			cbar = plt.colorbar(scatter_yo)
			cbar.set_label(colour_bar_title, rotation=270, labelpad = 20)
			cbar.ax.tick_params(labelsize=16)		
			plt.xlabel('$\\rm %s$ '%property_name_x)
			plt.ylabel('$\\rm %s$ '%property_name_y)

		else:
		
			scatter_yo=spl.hist2D(halo_mass,property_all,c=weights_property,cstat='median',bins=bin_for_disk,output=True,clabel=colour_bar_title,norm=norm,cmap=colour_map,nmin=n_min)
			plt.xlabel('$\\rm %s$ '%property_name_x)
			plt.ylabel('$\\rm %s$ '%property_name_y)

		
		
	# legend_handles.append(scatter_yo)
	# contour_plot = plt.contourf(X,Y,A,10,cmap='bone_r',vmin=100) 
	# plt.contourf(X,Y,B,10,cmap='inferno_r',vmin=100) 
	
	plt.xlim(xlim_lower,xlim_upper)
	plt.ylim(ylim_lower,ylim_upper)



## Making the dictionary

''' stellar_property = {'all' : stellar_all, 'centrals': stellar_central, 'satellites': stellar_satellites}   '''



def plotting_properties_separate_merged(virial_mass,stellar_property,property_plot,mean,legend_handles,property_name_x='X-Label-math-mode',legend_name='name-of-run', property_name_y='Y-Label-math-mode', xlim_lower=6,xlim_upper=15,ylim_lower=-4,ylim_upper=11, first_legend=False, halo_bins= False,colour_line='r', resolution = False, alpha=1):

	if halo_bins== True:
		colour_line = colour_scheme()

		bins_for_halo 	= [1,12,13,14,18]
		legend_name 	= ["a", "$\\rm M_{halo} < 10^{12}\ M_{\odot}$", "$\\rm 10^{12} < M_{halo} < 10^{13}\ M_{\odot}$", "$\\rm 10^{13} < M_{halo} < 10^{14}\ M_{\odot}$","$\\rm M_{halo} > 10^{14}\ M_{\odot}$", "b"]


	bin_for_disk = np.arange(5,16,0.2)
	# bin_for_disk =  [9.0,10.5,11.15,11.25,11.4,11.6,11.7,12.4,13.0,14.6,15,16]

	stellar_mass_all 	 	= np.zeros(len(bin_for_disk))
	prop_mass_all 	 		= np.zeros(len(bin_for_disk))

	stellar_mass_central 	= np.zeros(len(bin_for_disk))
	prop_mass_central 		= np.zeros(len(bin_for_disk))

	stellar_mass_satellite 	= np.zeros(len(bin_for_disk))
	prop_mass_satellite 	= np.zeros(len(bin_for_disk))
	
	prop_mass_low 			= np.zeros(len(bin_for_disk))
	prop_mass_high			= np.zeros(len(bin_for_disk))
	
	(stellar_all, stellar_central, stellar_satellite) = stellar_property.items()
	stellar_all = np.array(stellar_all[1])
	stellar_central = np.array(stellar_central[1])
	stellar_satellite = np.array(stellar_satellite[1])

	(property_all, property_central, property_satellite) = property_plot.items()
	property_all = np.array(property_all[1])
	property_central = np.array(property_central[1])
	property_satellite = np.array(property_satellite[1])

	(halo_mass, halo_mass_central, halo_mass_satellite) = virial_mass.items()
	halo_mass = np.array(halo_mass[1])
	halo_mass_central = np.array(halo_mass_central[1])
	halo_mass_satellite = np.array(halo_mass_satellite[1])





	if halo_bins == True:

		if mean == True:

			for j in range(1,len(bins_for_halo)):

				for i in range(1,len(bin_for_disk)):

					stellar_mass_all[i-1] 		= np.log10(np.mean(stellar_all[(stellar_all < 10**bin_for_disk[i]) & (stellar_all >= 10**bin_for_disk[i-1]) & (halo_mass_all >= 10**bins_for_halo[j-1])& (halo_mass_all <= 10**bins_for_halo[j])]))
					

					prop_mass_central[i-1] = np.log10(np.mean(property_central[(stellar_central < 10**bin_for_disk[i]) & (stellar_central >= 10**bin_for_disk[i-1]) & (halo_mass_central >= 10**bins_for_halo[j-1])& (halo_mass_all <= 10**bins_for_halo[j])]))


					stellar_mass_central[i-1] 		= np.log10(np.mean(stellar_central[(stellar_central < 10**bin_for_disk[i]) & (stellar_central >= 10**bin_for_disk[i-1]) & (halo_mass_central >= 10**bins_for_halo[j-1])& (halo_mass_central <= 10**bins_for_halo[j])]))
					

					prop_mass_central[i-1] = np.log10(np.mean(property_central[(stellar_central < 10**bin_for_disk[i]) & (stellar_central >= 10**bin_for_disk[i-1]) & (halo_mass_central >= 10**bins_for_halo[j-1])& (halo_mass_central <= 10**bins_for_halo[j])]))

					stellar_mass_satellite[i-1] 		= np.log10(np.mean(stellar_satellite[(stellar_satellite < 10**bin_for_disk[i]) & (stellar_satellite >= 10**bin_for_disk[i-1]) & (halo_mass_satellite >= 10**bins_for_halo[j-1])& (halo_mass_satellite <= 10**bins_for_halo[j])]))
					

					prop_mass_satellite[i-1] = np.log10(np.mean(property_satellite[(stellar_satellite < 10**bin_for_disk[i]) & (stellar_satellite >= 10**bin_for_disk[i-1]) & (halo_mass_satellite >= 10**bins_for_halo[j-1])& (halo_mass_satellite <= 10**bins_for_halo[j])]))
					
					bootarr  	= np.log10(property_central[(stellar_central < 10**bin_for_disk[i]) & (stellar_central >= 10**bin_for_disk[i-1])& (halo_mass_central >= 10**bins_for_halo[j-1])& (halo_mass_central <= 10**bins_for_halo[j])])
					bootarr = bootarr[bootarr != float('-inf')]
					
					if len(bootarr) > 10:
						prop_mass_central[i-1] = prop_mass_central[i-1]
						prop_mass_satellite[i-1] = prop_mass_satellite[i-1]

					else:
						prop_mass_central[i-1] = None
						prop_mass_satellite[i-1] = None
					

				# plt.plot(stellar_mass_all[0:len(stellar_mass_all)-1],prop_mass_all[0:len(prop_mass_all)-1],color=colour_line[j], linestyle='-')
				plt.plot(stellar_mass_central[0:len(stellar_mass_central)-1],prop_mass_central[0:len(prop_mass_central)-1],color=colour_line[j], linestyle='--', alpha=alpha)
				plt.plot(stellar_mass_satellite[0:len(stellar_mass_satellite)-1],prop_mass_satellite[0:len(prop_mass_satellite)-1],color=colour_line[j], linestyle='-.', alpha=alpha)

				legend_handles.append(mpatches.Patch(color=colour_line[j],label=legend_name[j]))


		else:
			for j in range(1,len(bins_for_halo)):

				for i in range(1,len(bin_for_disk)):

					stellar_mass_all[i-1] 		= np.log10(np.median(stellar_all[(stellar_all < 10**bin_for_disk[i]) & (stellar_all >= 10**bin_for_disk[i-1]) & (halo_mass_all >= 10**bins_for_halo[j-1])& (halo_mass_all <= 10**bins_for_halo[j])]))
					

					prop_mass_all[i-1] = np.log10(np.median(property_all[(stellar_all < 10**bin_for_disk[i]) & (stellar_all >= 10**bin_for_disk[i-1]) & (halo_mass_all >= 10**bins_for_halo[j-1])& (halo_mass_all <= 10**bins_for_halo[j])]))


					stellar_mass_central[i-1] 		= np.log10(np.median(stellar_central[(stellar_central < 10**bin_for_disk[i]) & (stellar_central >= 10**bin_for_disk[i-1]) & (halo_mass_central >= 10**bins_for_halo[j-1])& (halo_mass_central <= 10**bins_for_halo[j])]))
					

					prop_mass_central[i-1] = np.log10(np.median(property_central[(stellar_central < 10**bin_for_disk[i]) & (stellar_central >= 10**bin_for_disk[i-1]) & (halo_mass_central >= 10**bins_for_halo[j-1])& (halo_mass_central <= 10**bins_for_halo[j])]))

					stellar_mass_satellite[i-1] 		= np.log10(np.median(stellar_satellite[(stellar_satellite < 10**bin_for_disk[i]) & (stellar_satellite >= 10**bin_for_disk[i-1]) & (halo_mass_satellite >= 10**bins_for_halo[j-1])& (halo_mass_satellite <= 10**bins_for_halo[j])]))
					

					prop_mass_satellite[i-1] = np.log10(np.median(property_satellite[(stellar_satellite < 10**bin_for_disk[i]) & (stellar_satellite >= 10**bin_for_disk[i-1]) & (halo_mass_satellite >= 10**bins_for_halo[j-1])& (halo_mass_satellite <= 10**bins_for_halo[j])]))
					
					bootarr  	= np.log10(property_central[(stellar_central < 10**bin_for_disk[i]) & (stellar_central >= 10**bin_for_disk[i-1])& (halo_mass_central >= 10**bins_for_halo[j-1])& (halo_mass_central <= 10**bins_for_halo[j])])
					bootarr = bootarr[bootarr != float('-inf')]
					

					if len(bootarr) > 10:

						prop_mass_central[i-1] = prop_mass_central[i-1]
						prop_mass_satellite[i-1] = prop_mass_satellite[i-1]

					else:
						prop_mass_central[i-1] = None
						prop_mass_satellite[i-1] = None
					

				# plt.plot(stellar_mass_all[0:len(stellar_mass_all)-1],prop_mass_all[0:len(prop_mass_all)-1],color=colour_line[j], linestyle='-')
				plt.plot(stellar_mass_central[0:len(stellar_mass_central)-1],prop_mass_central[0:len(prop_mass_central)-1],color=colour_line[j], linestyle='--', alpha=alpha)
				plt.plot(stellar_mass_satellite[0:len(stellar_mass_satellite)-1],prop_mass_satellite[0:len(prop_mass_satellite)-1],color=colour_line[j], linestyle='-.', alpha=alpha)

				legend_handles.append(mpatches.Patch(color=colour_line[j],label=legend_name[j]))



	else:
		if mean == True:

		

			for i in range(1,len(bin_for_disk)):

				stellar_mass_all[i-1] 		= np.log10(np.mean(stellar_all[(stellar_all < 10**bin_for_disk[i]) & (stellar_all >= 10**bin_for_disk[i-1])]))
				
				prop_mass_all[i-1] = np.log10(np.mean(property_all[(stellar_all < 10**bin_for_disk[i]) & (stellar_all >= 10**bin_for_disk[i-1]) ]))

				stellar_mass_central[i-1] 		= np.log10(np.mean(stellar_central[(stellar_central < 10**bin_for_disk[i]) & (stellar_central >= 10**bin_for_disk[i-1])]))
				

				prop_mass_central[i-1] = np.log10(np.mean(property_central[(stellar_central < 10**bin_for_disk[i]) & (stellar_central >= 10**bin_for_disk[i-1]) ]))

				stellar_mass_satellite[i-1] 		= np.log10(np.mean(stellar_satellite[(stellar_satellite < 10**bin_for_disk[i]) & (stellar_satellite >= 10**bin_for_disk[i-1]) ]))
				

				prop_mass_satellite[i-1] = np.log10(np.mean(property_satellite[(stellar_satellite < 10**bin_for_disk[i]) & (stellar_satellite >= 10**bin_for_disk[i-1]) ]))
				
				bootarr  	= np.log10(property_satellite[(stellar_satellite < 10**bin_for_disk[i]) & (stellar_satellite >= 10**bin_for_disk[i-1])])
				bootarr = bootarr[bootarr != float('-inf')]
				

				if len(bootarr) > 10:

					prop_mass_central[i-1] = prop_mass_central[i-1]
					prop_mass_satellite[i-1] = prop_mass_satellite[i-1]

				else:
					prop_mass_central[i-1] = None
					prop_mass_satellite[i-1] = None
				
			
			# plt.plot(stellar_mass_all[0:len(stellar_mass_all)-1],prop_mass_all[0:len(prop_mass_all)-1],color=colour_line, linestyle='-')
			plt.plot(stellar_mass_central[0:len(stellar_mass_central)-1],prop_mass_central[0:len(prop_mass_central)-1],color=colour_line, linestyle='--', alpha=alpha)
			plt.plot(stellar_mass_satellite[0:len(stellar_mass_satellite)-1],prop_mass_satellite[0:len(prop_mass_satellite)-1],color=colour_line, linestyle='-.', alpha=alpha)

			if resolution == True:
				legend_handles.append(mpatches.Patch(color=colour_line,label=legend_name, alpha=alpha))


		else:
		

			for i in range(1,len(bin_for_disk)):

				stellar_mass_all[i-1] 		= np.log10(np.median(stellar_all[(stellar_all < 10**bin_for_disk[i]) & (stellar_all >= 10**bin_for_disk[i-1]) ]))
				

				prop_mass_all[i-1] = np.log10(np.median(property_all[(stellar_all < 10**bin_for_disk[i]) & (stellar_all >= 10**bin_for_disk[i-1])]))

				stellar_mass_central[i-1] 		= np.log10(np.median(stellar_central[(stellar_central < 10**bin_for_disk[i]) & (stellar_central >= 10**bin_for_disk[i-1]) ]))
				

				prop_mass_central[i-1] = np.log10(np.median(property_central[(stellar_central < 10**bin_for_disk[i]) & (stellar_central >= 10**bin_for_disk[i-1])]))

				stellar_mass_satellite[i-1] 		= np.log10(np.median(stellar_satellite[(stellar_satellite < 10**bin_for_disk[i]) & (stellar_satellite >= 10**bin_for_disk[i-1])]))
				

				prop_mass_satellite[i-1] = np.log10(np.median(property_satellite[(stellar_satellite < 10**bin_for_disk[i]) & (stellar_satellite >= 10**bin_for_disk[i-1]) ]))
				
				bootarr  	= np.log10(property_satellite[(stellar_satellite < 10**bin_for_disk[i]) & (stellar_satellite >= 10**bin_for_disk[i-1])])
				bootarr = bootarr[bootarr != float('-inf')]
				

				if len(bootarr) > 3:

					prop_mass_central[i-1] = prop_mass_central[i-1]
					prop_mass_satellite[i-1] = prop_mass_satellite[i-1]

				else:
					prop_mass_central[i-1] = None
					prop_mass_satellite[i-1] = None
				

			# plt.plot(stellar_mass_all[0:len(stellar_mass_all)-1],prop_mass_all[0:len(prop_mass_all)-1],color=colour_line, linestyle='-')
			plt.plot(stellar_mass_central[0:len(stellar_mass_central)-1],prop_mass_central[0:len(prop_mass_central)-1],color=colour_line, linestyle='--', alpha=alpha)
			plt.plot(stellar_mass_satellite[0:len(stellar_mass_satellite)-1],prop_mass_satellite[0:len(prop_mass_satellite)-1],color=colour_line, linestyle='-.', alpha=alpha)

			if resolution == True:
				legend_handles.append(mpatches.Patch(color=colour_line,label=legend_name))

	
	
	if first_legend == True:
		# legend_handles.append(mlines.Line2D([],[],color='black',linestyle='-',label='All'))
		legend_handles.append(mlines.Line2D([],[],color='black',linestyle='--',label='Centrals'))
		legend_handles.append(mlines.Line2D([],[],color='black',linestyle='-.',label='Satellites'))
		
		
	
	
	plt.xlabel('$\\rm %s$ '%property_name_x)
	plt.ylabel('$\\rm %s$ '%property_name_y)

	plt.xlim(xlim_lower,xlim_upper)
	plt.ylim(ylim_lower,ylim_upper)




def plotting_2d_hist(virial_mass, property_plot,legend_handles,mean=True, property_name_x='Property_x',property_name_y='Property_y',xlim_lower=10,xlim_upper=15,ylim_lower=4,ylim_upper=12,colour_map='cool_r', legend_name='name-of-run'):
	
	bin_for_disk = np.arange(5,16,0.2)

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
	property_central = np.array(property_central[1])
	property_satellite = np.array(property_satellite[1])

	(halo_mass, halo_mass_central, halo_mass_satellite) = virial_mass.items()
	halo_mass = np.array(halo_mass[1])
	halo_mass_central = np.array(halo_mass_central[1])
	halo_mass_satellite = np.array(halo_mass_satellite[1])

	A,x,y = np.histogram2d(np.log10(halo_mass[property_central != 0]),np.log10(property_central[property_central != 0]),bins=100)
	A = A.T

	B,x,y = np.histogram2d(np.log10(halo_mass[property_central != 0]),np.log10(property_central[property_central != 0]),bins=100)
	B = B.T

	X,Y = np.meshgrid(x[0:100],y[0:100])

	
	contour_plot = plt.contour(X,Y,A,levels=10, vmin=100) 
	# cbar = plt.colorbar(scatter_yo)
	# cbar.set_label(colour_bar_title, rotation=270, labelpad = 20)
	# cbar.ax.tick_params(labelsize=16)		
	plt.xlabel('$\\rm %s$ '%property_name_x)
	plt.ylabel('$\\rm %s$ '%property_name_y)

	plt.xlim(xlim_lower,xlim_upper)
	plt.ylim(ylim_lower,ylim_upper)

	# plt.legend()
	# plt.savefig('%s.png'%figure_name)
	# plt.show()




def correlationFunctionGalaxies(gal_x,gal_y,gal_z,max_R,boxsize):

	indicesSorted = np.argsort(gal_x)
	count_gal = 0


	random_x = np.random.uniform(0,boxsize,len(gal_x))
	random_y = np.random.uniform(0,boxsize,len(gal_y))
	random_z = np.random.uniform(0,boxsize,len(gal_z))

	indicesSorted_random = np.argsort(random_x)
	count_random = 0

	for n, ii in enumerate(indicesSorted):

		for jj in indicesSorted[n:]:
			if (ii != jj):
				if ((np.sqrt((gal_x[jj] - gal_x[ii])**2 + (gal_y[jj] - gal_y[ii])**2 + (gal_z[jj] - gal_z[ii])**2)) > max_R):

					continue

				else:
					count_gal = count_gal + 1

					if count_gal%1000==0:
						print(count_gal)


	for n, ii in enumerate(indicesSorted_random):

		for jj in indicesSorted_random[n:]:
			if (ii != jj):
				if ((np.sqrt((random_x[jj] - random_x[ii])**2 + (random_y[jj] - random_y[ii])**2 + (random_z[jj] - random_z[ii])**2)) <= max_R):

					count_random = count_random + 1

					if count_random%1000==0:
						print(count_random)


	

	return count_gal, count_random


'''
	*** My correlation function is the 3d distance one - which takes everything and not the projected distance - for projected distance I might have to change the 3d coordinates to the projected one and use the same function ****

	bins = np.logspace(np.log10(0.1), np.log10(10), 20)

	count_gal = np.zeros(len(bins))
	count_random = np.zeros(len(bins))

	for i in range(len(bins)):
	count_gal[i], count_random[i] = correlationFunctionGalaxies(position_x[HI_all >= 10**9], position_y[HI_all >= 10**9], position_z[HI_all >= 10**9],bins[i],40)


wp = count_gal/count_random -1

plt.scatter(bins,wp)
'''




############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
### Reading Data - SHARK
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************


dt  = np.dtype(float)
G   = 4.301e-9
h   = 0.6751
M_solar_2_g   = 1.99e33
# dt = int


class SharkDataReading:

	def __init__(self,filepath,simulation,model,snapshot,subvolumes):

		self.filepath = filepath
		self.simulation = simulation
		self.model = model
		self.snapshot = snapshot
		self.subvolumes = subvolumes




	def readHDF5Data(self):

		galaxy_prop = h5py.File(self.filepath  + self.simulation + "/" + self.model + str(self.snapshot) + "/1/" + "galaxies.hdf5", 'r')
		self.d_data = {}

		for key in galaxy_prop.id:
			if isinstance(galaxy_prop[key].id, h5py.h5g.GroupID):
				keytemp = key.decode('utf-8')
				if keytemp in ['cosmology','run_info']:
					self.d_data[keytemp] = {} #If group, new dictionary is made
					for key2 in galaxy_prop[key].id: #loping over ids in the group
						keytemp2 = key2.decode('utf-8')
						self.d_data[keytemp][keytemp2] = galaxy_prop[key][key2].value
				else:
					self.d_data[keytemp] = {} #If group, new dictionary is made
					for key2 in galaxy_prop[key].id: #loping over ids in the group
						keytemp2 = key2.decode('utf-8')
						self.d_data[keytemp][keytemp2] = galaxy_prop[key][key2][:]

		
		for l in range(1,self.subvolumes):
			galaxy_prop = h5py.File(self.filepath  + self.simulation + "/" + self.model + str(self.snapshot) + "/%s/"%l + "galaxies.hdf5", 'r')
			for key in galaxy_prop.id:
				if isinstance(galaxy_prop[key].id, h5py.h5g.GroupID):
					keytemp = key.decode('utf-8')
					if keytemp in ['cosmology','run_info']:
						for key2 in galaxy_prop[key].id: #loping over ids in the group
							keytemp2 = key2.decode('utf-8')
							self.d_data[keytemp][keytemp2] = np.append(self.d_data[keytemp][keytemp2],galaxy_prop[key][key2].value)
					else:
						for key2 in galaxy_prop[key].id: #loping over ids in the group
							keytemp2 = key2.decode('utf-8')
							self.d_data[keytemp][keytemp2] = np.append(self.d_data[keytemp][keytemp2],galaxy_prop[key][key2][:])




		return self.d_data




	def mergeValues(self,haloID, propMerge_bulge, propMerge_disk, baryon_property=True):

		self.haloID 			= haloID
		self.propMerge_bulge	= propMerge_bulge
		self.propMerge_disk		= propMerge_disk

		prop_final_mass 		= []
		prop_central_mass		= []
		prop_satellite_mass 	= []
		prop_orphan_mass 	 	= []

		final_vir 				= []
		is_central_final 		= []


		if baryon_property == True:
			for k in range(0,64):

				galaxy_prop 	= h5py.File(self.filepath  + self.simulation + "/" +  self.model + "/" + str(self.snapshot) + "/%s/"%k + "galaxies.hdf5", 'r')

				halo_ID 		= np.array(galaxy_prop['galaxies/%s' %self.haloID],dtype = dt)
				
				prop_merge		= np.array(galaxy_prop['galaxies/%s' %self.propMerge_bulge], dtype= dt)/h + np.array(galaxy_prop['galaxies/%s' %self.propMerge_disk], dtype=dt)/h 

				M_vir			= np.array(galaxy_prop['galaxies/mvir_hosthalo'], dtype=dt)/h

				is_central		= np.array(galaxy_prop['galaxies/type'], dtype=dt)
				
				galaxy_prop.close()

				(unique_elements, indices_elements, counts_elements) = np.unique(halo_ID, return_index=True,return_counts=True)

				idmax = max(unique_elements)
				print('number of haloes:', len(unique_elements))

				
				Unique_HaloIDs 	= np.unique(halo_ID)

				
				prop_Halo	 	= np.histogram(halo_ID,bins=np.append(Unique_HaloIDs,Unique_HaloIDs[-1]+1), weights=prop_merge)[0]
					
				central_mass 	= np.histogram(halo_ID[is_central ==0],bins=np.append(Unique_HaloIDs,Unique_HaloIDs[-1]+1), weights=prop_merge[is_central == 0])[0]

				satellite_mass 	= np.histogram(halo_ID[is_central == 1],bins=np.append(Unique_HaloIDs,Unique_HaloIDs[-1]+1), weights=prop_merge[is_central == 1])[0]
				
				orphan_mass 	= np.histogram(halo_ID[is_central == 2],bins=np.append(Unique_HaloIDs,Unique_HaloIDs[-1]+1), weights=prop_merge[is_central == 2])[0]

				virial_Mass 	= M_vir[indices_elements]

				prop_final_mass 	= np.append(prop_final_mass,prop_Halo)
				prop_central_mass	= np.append(prop_central_mass, central_mass)
				prop_satellite_mass	= np.append(prop_satellite_mass,satellite_mass)
				prop_orphan_mass 	= np.append(prop_orphan_mass,orphan_mass)
				final_vir 			= np.append(final_vir,virial_Mass)
				is_central_final 	= np.append(is_central_final, is_central)

			

		else:
			for k in range(0,64):

				galaxy_prop 	= h5py.File(self.filepath  + self.simulation + "/" +  self.model + "/" + str(self.snapshot) + "/%s/"%k + "galaxies.hdf5", 'r')

				halo_ID 		= np.array(galaxy_prop['galaxies/%s' %self.haloID],dtype = dt)
				
				prop_merge		= np.array(galaxy_prop['galaxies/%s' %self.propMerge_bulge], dtype= dt)/h  

				M_vir			= np.array(galaxy_prop['galaxies/mvir_hosthalo'], dtype=dt)/h

				is_central		= np.array(galaxy_prop['galaxies/type'], dtype=dt)
				
				galaxy_prop.close()

				(unique_elements, indices_elements, counts_elements) = np.unique(halo_ID, return_index=True,return_counts=True)

				idmax = max(unique_elements)
				print('number of haloes:', len(unique_elements))

				
				Unique_HaloIDs 	= np.unique(halo_ID)

				
				prop_Halo	 	= np.histogram(halo_ID,bins=np.append(Unique_HaloIDs,Unique_HaloIDs[-1]+1), weights=prop_merge)[0]
					
				central_mass 	= np.histogram(halo_ID[is_central ==0],bins=np.append(Unique_HaloIDs,Unique_HaloIDs[-1]+1), weights=prop_merge[is_central == 0])[0]

				satellite_mass 	= np.histogram(halo_ID[is_central ==1],bins=np.append(Unique_HaloIDs,Unique_HaloIDs[-1]+1), weights=prop_merge[is_central == 1])[0]
				
				orphan_mass 	= np.histogram(halo_ID[is_central == 2],bins=np.append(Unique_HaloIDs,Unique_HaloIDs[-1]+1), weights=prop_merge[is_central == 2])[0]
				

				virial_Mass 	= M_vir[indices_elements]

				prop_final_mass 	= np.append(prop_final_mass,prop_Halo)
				prop_central_mass	= np.append(prop_central_mass, central_mass)
				prop_satellite_mass	= np.append(prop_satellite_mass,satellite_mass)
				prop_orphan_mass	= np.append(prop_orphan_mass,orphan_mass)
				final_vir 			= np.append(final_vir,virial_Mass)
				is_central_final 	= np.append(is_central_final, is_central)

				#print(indices_elements)
			


		return prop_final_mass[0:len(prop_final_mass)], prop_central_mass[0:len(prop_final_mass)], prop_satellite_mass[0:len(prop_final_mass)], prop_orphan_mass[0:len(prop_orphan_mass)] ,final_vir[0:len(prop_final_mass)]




	def mergeValues_satellite(self,haloID, propMerge_bulge, propMerge_disk,subvol_number=64, baryon_property=True):

		self.haloID 			= haloID
		self.propMerge_bulge	= propMerge_bulge
		self.propMerge_disk		= propMerge_disk

		prop_final_mass 		= []
		prop_central_mass		= []
		prop_satellite_mass 	= []
		prop_orphan_mass 	 	= []

		final_vir 				= []
		is_central_final 		= []
		final_halo_id 			= []

		if baryon_property == True:
			for k in range(0,subvol_number):

				galaxy_prop 	= h5py.File(self.filepath  + self.simulation + "/" +  self.model + "/" + str(self.snapshot) + "/%s/"%k + "galaxies.hdf5", 'r')

				halo_ID 		= np.array(galaxy_prop['galaxies/%s' %self.haloID],dtype = dt)
				
				prop_merge		= (np.array(galaxy_prop['galaxies/%s' %self.propMerge_bulge], dtype= dt)/h + np.array(galaxy_prop['galaxies/%s' %self.propMerge_disk], dtype=dt)/h)/1.35 

				M_vir			= np.array(galaxy_prop['galaxies/mvir_hosthalo'], dtype=dt)/h

				is_central		= np.array(galaxy_prop['galaxies/type'], dtype=dt)
				
				galaxy_prop.close()

				(unique_elements, indices_elements, counts_elements) = np.unique(halo_ID, return_index=True,return_counts=True)

				idmax = max(unique_elements)
				print('number of haloes:', len(unique_elements))

				
				Unique_HaloIDs 	= np.unique(halo_ID)

				
				prop_Halo	 	= np.histogram(halo_ID,bins=np.append(Unique_HaloIDs,Unique_HaloIDs[-1]+1), weights=prop_merge)[0]
					
				central_mass 	= np.histogram(halo_ID[is_central ==0],bins=np.append(Unique_HaloIDs,Unique_HaloIDs[-1]+1), weights=prop_merge[is_central == 0])[0]

				satellite_mass 	= np.histogram(halo_ID[is_central > 0],bins=np.append(Unique_HaloIDs,Unique_HaloIDs[-1]+1), weights=prop_merge[is_central > 0])[0]
				
				orphan_mass 	= np.histogram(halo_ID[is_central == 2],bins=np.append(Unique_HaloIDs,Unique_HaloIDs[-1]+1), weights=prop_merge[is_central == 2])[0]

				virial_Mass 	= M_vir[indices_elements]
				host_halo_add 	= halo_ID[indices_elements]

				prop_final_mass 	= np.append(prop_final_mass,prop_Halo)
				prop_central_mass	= np.append(prop_central_mass, central_mass)
				prop_satellite_mass	= np.append(prop_satellite_mass,satellite_mass)
				prop_orphan_mass 	= np.append(prop_orphan_mass,orphan_mass)
				final_vir 			= np.append(final_vir,virial_Mass)
				is_central_final 	= np.append(is_central_final, is_central)
				final_halo_id 		= np.append(final_halo_id, host_halo_add)
			

		else:
			for k in range(0,subvol_number):

				galaxy_prop 	= h5py.File(self.filepath  + self.simulation + "/" +  self.model + "/" + str(self.snapshot) + "/%s/"%k + "galaxies.hdf5", 'r')

				halo_ID 		= np.array(galaxy_prop['galaxies/%s' %self.haloID],dtype = dt)
				
				prop_merge		= np.array(galaxy_prop['galaxies/%s' %self.propMerge_bulge], dtype= dt)/h  

				M_vir			= np.array(galaxy_prop['galaxies/mvir_hosthalo'], dtype=dt)/h

				is_central		= np.array(galaxy_prop['galaxies/type'], dtype=dt)
				
				galaxy_prop.close()

				(unique_elements, indices_elements, counts_elements) = np.unique(halo_ID, return_index=True,return_counts=True)

				idmax = max(unique_elements)
				print('number of haloes:', len(unique_elements))

				
				Unique_HaloIDs 	= np.unique(halo_ID)

				
				prop_Halo	 	= np.histogram(halo_ID,bins=np.append(Unique_HaloIDs,Unique_HaloIDs[-1]+1), weights=prop_merge)[0]
					
				central_mass 	= np.histogram(halo_ID[is_central ==0],bins=np.append(Unique_HaloIDs,Unique_HaloIDs[-1]+1), weights=prop_merge[is_central == 0])[0]

				satellite_mass 	= np.histogram(halo_ID[is_central > 0],bins=np.append(Unique_HaloIDs,Unique_HaloIDs[-1]+1), weights=prop_merge[is_central > 0])[0]
				
				orphan_mass 	= np.histogram(halo_ID[is_central == 2],bins=np.append(Unique_HaloIDs,Unique_HaloIDs[-1]+1), weights=prop_merge[is_central == 2])[0]
				

				virial_Mass 	= M_vir[indices_elements]
				host_halo_add 	= halo_ID[indices_elements]

				prop_final_mass 	= np.append(prop_final_mass,prop_Halo)
				prop_central_mass	= np.append(prop_central_mass, central_mass)
				prop_satellite_mass	= np.append(prop_satellite_mass,satellite_mass)
				prop_orphan_mass	= np.append(prop_orphan_mass,orphan_mass)
				final_vir 			= np.append(final_vir,virial_Mass)
				is_central_final 	= np.append(is_central_final, is_central)
				final_halo_id 		= np.append(final_halo_id, host_halo_add)
				#print(indices_elements)
			


		return prop_final_mass[0:len(prop_final_mass)], prop_central_mass[0:len(prop_final_mass)], prop_satellite_mass[0:len(prop_final_mass)], prop_orphan_mass[0:len(prop_orphan_mass)] ,final_vir[0:len(prop_final_mass)], final_halo_id[0:len(prop_final_mass)]




	def mergeValues_satellite_virial_velocity(self,haloID, propMerge_bulge, propMerge_disk,subvol_number=64, baryon_property=True):

		self.haloID 			= haloID
		self.propMerge_bulge	= propMerge_bulge
		self.propMerge_disk		= propMerge_disk

		prop_final_mass 		= []
		prop_central_mass		= []
		prop_satellite_mass 	= []
		prop_orphan_mass 	 	= []

		final_vir 				= []
		is_central_final 		= []
		

		if baryon_property == True:
			for k in range(0,subvol_number):

				galaxy_prop 	= h5py.File(self.filepath  + self.simulation + "/" +  self.model + "/" + str(self.snapshot) + "/%s/"%k + "galaxies.hdf5", 'r')

				halo_ID 		= np.array(galaxy_prop['galaxies/%s' %self.haloID],dtype = dt)
				
				prop_merge		= (np.array(galaxy_prop['galaxies/%s' %self.propMerge_bulge], dtype= dt)/h + np.array(galaxy_prop['galaxies/%s' %self.propMerge_disk], dtype=dt)/h)/1.35 

				M_vir			= np.array(galaxy_prop['galaxies/vvir_hosthalo'], dtype=dt)

				is_central		= np.array(galaxy_prop['galaxies/type'], dtype=dt)
				
				galaxy_prop.close()

				(unique_elements, indices_elements, counts_elements) = np.unique(halo_ID, return_index=True,return_counts=True)

				idmax = max(unique_elements)
				print('number of haloes:', len(unique_elements))

				
				Unique_HaloIDs 	= np.unique(halo_ID)

				
				prop_Halo	 	= np.histogram(halo_ID,bins=np.append(Unique_HaloIDs,Unique_HaloIDs[-1]+1), weights=prop_merge)[0]
					
				central_mass 	= np.histogram(halo_ID[is_central ==0],bins=np.append(Unique_HaloIDs,Unique_HaloIDs[-1]+1), weights=prop_merge[is_central == 0])[0]

				satellite_mass 	= np.histogram(halo_ID[is_central > 0],bins=np.append(Unique_HaloIDs,Unique_HaloIDs[-1]+1), weights=prop_merge[is_central > 0])[0]
				
				orphan_mass 	= np.histogram(halo_ID[is_central == 2],bins=np.append(Unique_HaloIDs,Unique_HaloIDs[-1]+1), weights=prop_merge[is_central == 2])[0]

				virial_Mass 	= M_vir[indices_elements]

				prop_final_mass 	= np.append(prop_final_mass,prop_Halo)
				prop_central_mass	= np.append(prop_central_mass, central_mass)
				prop_satellite_mass	= np.append(prop_satellite_mass,satellite_mass)
				prop_orphan_mass 	= np.append(prop_orphan_mass,orphan_mass)
				final_vir 			= np.append(final_vir,virial_Mass)
				is_central_final 	= np.append(is_central_final, is_central)

			

		else:
			for k in range(0,subvol_number):

				galaxy_prop 	= h5py.File(self.filepath  + self.simulation + "/" +  self.model + "/" + str(self.snapshot) + "/%s/"%k + "galaxies.hdf5", 'r')

				halo_ID 		= np.array(galaxy_prop['galaxies/%s' %self.haloID],dtype = dt)
				
				prop_merge		= np.array(galaxy_prop['galaxies/%s' %self.propMerge_bulge], dtype= dt)/h  

				M_vir			= np.array(galaxy_prop['galaxies/vvir_hosthalo'], dtype=dt)

				is_central		= np.array(galaxy_prop['galaxies/type'], dtype=dt)
				
				galaxy_prop.close()

				(unique_elements, indices_elements, counts_elements) = np.unique(halo_ID, return_index=True,return_counts=True)

				idmax = max(unique_elements)
				print('number of haloes:', len(unique_elements))

				
				Unique_HaloIDs 	= np.unique(halo_ID)

				
				prop_Halo	 	= np.histogram(halo_ID,bins=np.append(Unique_HaloIDs,Unique_HaloIDs[-1]+1), weights=prop_merge)[0]
					
				central_mass 	= np.histogram(halo_ID[is_central ==0],bins=np.append(Unique_HaloIDs,Unique_HaloIDs[-1]+1), weights=prop_merge[is_central == 0])[0]

				satellite_mass 	= np.histogram(halo_ID[is_central > 0],bins=np.append(Unique_HaloIDs,Unique_HaloIDs[-1]+1), weights=prop_merge[is_central > 0])[0]
				
				orphan_mass 	= np.histogram(halo_ID[is_central == 2],bins=np.append(Unique_HaloIDs,Unique_HaloIDs[-1]+1), weights=prop_merge[is_central == 2])[0]
				

				virial_Mass 	= M_vir[indices_elements]

				prop_final_mass 	= np.append(prop_final_mass,prop_Halo)
				prop_central_mass	= np.append(prop_central_mass, central_mass)
				prop_satellite_mass	= np.append(prop_satellite_mass,satellite_mass)
				prop_orphan_mass	= np.append(prop_orphan_mass,orphan_mass)
				final_vir 			= np.append(final_vir,virial_Mass)
				is_central_final 	= np.append(is_central_final, is_central)

				#print(indices_elements)
			


		return prop_final_mass[0:len(prop_final_mass)], prop_central_mass[0:len(prop_final_mass)], prop_satellite_mass[0:len(prop_final_mass)], prop_orphan_mass[0:len(prop_orphan_mass)] ,final_vir[0:len(prop_final_mass)]





	def mergeValuesNumberSubstructures(self,haloID, subhaloID,subvol_number=64):

		self.haloID 			= haloID
		self.subhaloID			= subhaloID
		
		substructure_count		= []
		final_vir 				= []
		final_halo_id 			= []

		for k in range(0,subvol_number):

			galaxy_prop 	= h5py.File(self.filepath  + self.simulation + "/" +  self.model + "/" + str(self.snapshot) + "/%s/"%k + "galaxies.hdf5", 'r')

			halo_ID 		= np.array(galaxy_prop['galaxies/%s' %self.haloID],dtype = dt)
			
			# prop_merge		= np.array(galaxy_prop['galaxies/%s' %self.propMerge_bulge], dtype= dt)/h + np.array(galaxy_prop['galaxies/%s' %self.propMerge_disk], dtype=dt)/h 

			M_vir			= np.array(galaxy_prop['galaxies/mvir_hosthalo'], dtype=dt)/h

			
			galaxy_prop.close()

			(unique_elements, indices_elements, counts_elements) = np.unique(halo_ID, return_index=True,return_counts=True,)

			idmax = max(unique_elements)
			print('number of haloes: ', len(unique_elements))

			
			Unique_HaloIDs 	= np.unique(halo_ID)

			
			count_Halo	 	= np.histogram(halo_ID,bins=np.append(Unique_HaloIDs,Unique_HaloIDs[-1]+1))[0]
				
			#central_mass 	= np.histogram(halo_ID[is_central ==0],bins=np.append(Unique_HaloIDs,Unique_HaloIDs[-1]+1), weights=prop_merge[is_central == 0])[0]

			#satellite_mass 	= np.histogram(halo_ID[is_central >0],bins=np.append(Unique_HaloIDs,Unique_HaloIDs[-1]+1), weights=prop_merge[is_central > 0])[0]
				
			virial_Mass 	= M_vir[indices_elements]
			host_halo_add 	= halo_ID[indices_elements]

			substructure_count 	= np.append(substructure_count,count_Halo)
			final_vir 			= np.append(final_vir,virial_Mass)
			final_halo_id 		= np.append(final_halo_id, host_halo_add)

			#print(indices_elements)
		return substructure_count[0:len(substructure_count)], final_vir[0:len(substructure_count)], final_halo_id[0:len(substructure_count)]




	def readIndividualFiles(self,fields,include_h0_volh=True):

		data = collections.OrderedDict()
		self.fields = fields 
		subvolumes_range = range(0,self.subvolumes)

		for idx,subv in enumerate(subvolumes_range):

			fname = os.path.join(self.filepath, self.simulation, self.model, str(self.snapshot),str(subv),'galaxies.hdf5')
			print('Reading data from %s'%fname)

			with h5py.File(fname,'r') as f:
				if idx == 0 and include_h0_volh:
					data['h0'] = f['cosmology/h'].value
					data['vol'] = f['run_info/effective_volume'].value * len(subvolumes_range)

				for gname, dsname in self.fields.items():
					print(gname)
					group = f[gname]

					for dsname in dsname:
						full_name = '%s/%s' % (gname, dsname)
						print(full_name)
						l = data.get(full_name, None)

						if l is None:
							l = group[dsname].value

						else:
							l = np.concatenate([l,group[dsname].value])

						data[full_name] = l

		return list(data.values())





	def readIndividualFiles_GAMA(self, fields, snapshot_group, subvol_group, include_h0_volh=True):

		data = collections.OrderedDict()
		self.fields = fields 
		
		idx = subvol_group
		subv = subvol_group

		fname = os.path.join(self.filepath, self.simulation, self.model, str(snapshot_group),str(subv),'galaxies.hdf5')
		print('Reading data from %s'%fname)

		with h5py.File(fname,'r') as f:
			if idx == subvol_group and include_h0_volh:
				data['h0'] = f['cosmology/h'].value
				data['vol'] = f['run_info/effective_volume'].value * 1

			for gname, dsname in self.fields.items():
				print(gname)
				group = f[gname]

				for dsname in dsname:
					full_name = '%s/%s' % (gname, dsname)

					l = data.get(full_name, None)

					if l is None:
						l = group[dsname].value

					else:
						l = np.concatenate([l,group[dsname].value])

					data[full_name] = l

		return list(data.values())






	def satelliteList_mvir(self,haloID, propMerge_bulge, propMerge_disk, subvol_number=64):

		self.haloID 			= haloID
		self.propMerge_bulge	= propMerge_bulge
		self.propMerge_disk		= propMerge_disk

		
		mvir_12_4 				= []
		mvir_12_6				= []
		mvir_12_8 				= []
		mvir_13 				= []
		mvir_13_2 				= []
		mvir_13_4 				= []
		mvir_13_6				= []
		mvir_13_8 				= []
		mvir_14 				= []
		mvir_14_5 				= []
		

		for k in range(0,subvol_number):

			galaxy_prop 	= h5py.File(self.filepath  + self.simulation + "/" +  self.model + "/" + str(self.snapshot) + "/%s/"%k + "galaxies.hdf5", 'r')

			halo_ID 		= np.array(galaxy_prop['galaxies/%s' %self.haloID],dtype = dt)
			
			prop_merge		= np.array(galaxy_prop['galaxies/%s' %self.propMerge_bulge], dtype= dt)/h + np.array(galaxy_prop['galaxies/%s' %self.propMerge_disk], dtype=dt)/h 

			M_vir			= np.array(galaxy_prop['galaxies/mvir_hosthalo'], dtype=dt)/h

			is_central		= np.array(galaxy_prop['galaxies/type'], dtype=dt)
			
			galaxy_prop.close()

			(unique_elements, indices_elements, counts_elements) = np.unique(halo_ID, return_index=True,return_counts=True)

			idmax = max(unique_elements)
			print('number of haloes:', len(unique_elements))

			
			Unique_HaloIDs 	= np.unique(halo_ID)

			for i in range(len(Unique_HaloIDs)):

				a = np.where(halo_ID == Unique_HaloIDs[i])[0]

				if (M_vir[a[0]] > 10**12.4) & (M_vir[a[0]] <= 10**12.6):
					b = np.where(is_central[a] == 1)[0]
					mvir_12_4 		= np.append(mvir_12_4,prop_merge[b]) 
				
				elif (M_vir[a[0]] > 10**12.6) & (M_vir[a[0]] <= 10**12.8):
					b = np.where(is_central[a] == 1)[0]
					mvir_12_6 		= np.append(mvir_12_6,prop_merge[b]) 
				
				elif (M_vir[a[0]] > 10**12.8) & (M_vir[a[0]] <= 10**13):
					b = np.where(is_central[a] == 1)[0]
					mvir_12_8 		= np.append(mvir_12_8,prop_merge[b]) 
				
				elif (M_vir[a[0]] > 10**13) & (M_vir[a[0]] <= 10**13.2):
					b = np.where(is_central[a] == 1)[0]
					mvir_13 		= np.append(mvir_13,prop_merge[b]) 
				
				elif (M_vir[a[0]] > 10**13.2) & (M_vir[a[0]] <= 10**13.4):
					b = np.where(is_central[a] == 1)[0]
					mvir_13_2 		= np.append(mvir_13_2,prop_merge[b]) 
				
				elif (M_vir[a[0]] > 10**13.4) & (M_vir[a[0]] <= 10**13.6):
					b = np.where(is_central[a] == 1)[0]
					mvir_13_4 		= np.append(mvir_13_4,prop_merge[b]) 

				elif (M_vir[a[0]] > 10**13.6) & (M_vir[a[0]] <= 10**13.8):
					b = np.where(is_central[a] == 1)[0]
					mvir_13_6 		= np.append(mvir_13_6,prop_merge[b]) 

				elif (M_vir[a[0]] > 10**13.8) & (M_vir[a[0]] <= 10**14):
					b = np.where(is_central[a] == 1)[0]
					mvir_13_8 		= np.append(mvir_13_8,prop_merge[b]) 

				elif (M_vir[a[0]] > 10**14) & (M_vir[a[0]] <= 10**14.5):
					b = np.where(is_central[a] == 1)[0]
					mvir_14 		= np.append(mvir_14,prop_merge[b]) 

				elif (M_vir[a[0]] > 10**14.5):
					b = np.where(is_central[a] == 1)[0]
					mvir_14_5 		= np.append(mvir_14_5,prop_merge[b]) 



		return mvir_12_4, mvir_12_6, mvir_12_8, mvir_13, mvir_13_2, mvir_13_4, mvir_13_6, mvir_13_8, mvir_14, mvir_14_5


	def satelliteList_mvir_micro(self,haloID, propMerge_bulge, propMerge_disk, subvol_number=64):

		self.haloID 			= haloID
		self.propMerge_bulge	= propMerge_bulge
		self.propMerge_disk		= propMerge_disk

		
		mvir_12_4 				= []
		mvir_13 				= []
		mvir_14 				= []
		

		for k in range(0,subvol_number):

			galaxy_prop 	= h5py.File(self.filepath  + self.simulation + "/" +  self.model + "/" + str(self.snapshot) + "/%s/"%k + "galaxies.hdf5", 'r')

			halo_ID 		= np.array(galaxy_prop['galaxies/%s' %self.haloID],dtype = dt)
			
			prop_merge		= np.array(galaxy_prop['galaxies/%s' %self.propMerge_bulge], dtype= dt)/h + np.array(galaxy_prop['galaxies/%s' %self.propMerge_disk], dtype=dt)/h 

			M_vir			= np.array(galaxy_prop['galaxies/mvir_hosthalo'], dtype=dt)/h

			is_central		= np.array(galaxy_prop['galaxies/type'], dtype=dt)
			
			galaxy_prop.close()

			(unique_elements, indices_elements, counts_elements) = np.unique(halo_ID, return_index=True,return_counts=True)

			idmax = max(unique_elements)
			print('number of haloes:', len(unique_elements))

			
			Unique_HaloIDs 	= np.unique(halo_ID)

			for i in range(len(Unique_HaloIDs)):

				a = np.where(halo_ID == Unique_HaloIDs[i])[0]

				if (M_vir[a[0]] > 10**12.4) & (M_vir[a[0]] <= 10**13):
					b = np.where(is_central[a] == 1)[0]
					mvir_12_4 		= np.append(mvir_12_4,prop_merge[b]) 
				
				elif (M_vir[a[0]] > 10**13) & (M_vir[a[0]] <= 10**14):
					b = np.where(is_central[a] == 1)[0]
					mvir_13 		= np.append(mvir_13,prop_merge[b]) 
								
				elif (M_vir[a[0]] > 10**14) & (M_vir[a[0]] <= 10**15):
					b = np.where(is_central[a] == 1)[0]
					mvir_14 		= np.append(mvir_14,prop_merge[b]) 

				

		return mvir_12_4, mvir_13, mvir_14




	def prop_mass_rvir(self, haloID, propMerge_bulge, propMerge_disk,rvir_times=1,subvol_number=64, baryon_property=True):

		self.haloID 			= haloID
		self.propMerge_bulge 	= propMerge_bulge
		self.propMerge_disk 	= propMerge_disk

		prop_final_mass 		= []
		prop_central_mass 		= []
		prop_satellite_mass 	= []
		
		is_central_final 		= []
		final_vir 				= []

		if baryon_property == True:
			for k in range(0,subvol_number): 
				start = time.time()

				print(k)

				galaxy_prop 	= h5py.File(self.filepath + self.simulation + "/" + self.model + "/" + str(self.snapshot) + "/%s/"%k + "galaxies.hdf5", 'r')

				halo_ID 		= np.array(galaxy_prop['galaxies/%s' %self.haloID], dtype = dt)

				prop_merge		= (np.array(galaxy_prop['galaxies/%s' %self.propMerge_bulge], dtype= dt)/h + np.array(galaxy_prop['galaxies/%s' %self.propMerge_disk], dtype=dt)/h)/1.35 

				M_vir			= np.array(galaxy_prop['galaxies/mvir_hosthalo'], dtype=dt)/h

				is_central		= np.array(galaxy_prop['galaxies/type'], dtype=dt)

				position_x  	= np.array(galaxy_prop['galaxies/position_x'], dtype=dt)
				position_y  	= np.array(galaxy_prop['galaxies/position_y'], dtype=dt)				
				position_z  	= np.array(galaxy_prop['galaxies/position_z'], dtype=dt)				

				vvir_hosthalo 	= np.array(galaxy_prop['galaxies/vvir_hosthalo'], dtype=dt)

				galaxy_prop.close()

				##################################################################################################################################################
				### RVir - HI values
				###################################################################################################################################################
				if rvir_times == 0:
					maxR = 0.2
				else:
					maxR = rvir_times*(G*M_vir[M_vir > 10**12]/vvir_hosthalo[M_vir > 10**12]**2)

				(unique_elements, indices_elements, counts_elements) = np.unique(halo_ID[M_vir > 10**12], return_index=True,return_counts=True)

				idmax = max(unique_elements)

				Unique_HaloIDs 	= np.unique(halo_ID[M_vir > 10**12])
				print('number of haloes:', len(unique_elements))	

				is_central_checking = is_central[M_vir > 10**12]
				
				if rvir_times == 0:
					maxR = 0.2*np.ones(len(is_central_checking))
				else:
					maxR 				= maxR[is_central_checking == 0]

				for i in range(len(Unique_HaloIDs)):

					# if i%1000 == 0:
						# print(i)

					position_x_vir 		= position_x[np.where(halo_ID == Unique_HaloIDs[i])[0]]
					position_y_vir 		= position_y[np.where(halo_ID == Unique_HaloIDs[i])[0]]
					position_z_vir 		= position_z[np.where(halo_ID == Unique_HaloIDs[i])[0]]

					position_x_cen 		= np.ones((len(position_x_vir)))*position_x[np.where((halo_ID == Unique_HaloIDs[i])& (is_central == 0))[0]][0]  
					position_y_cen 		= np.ones((len(position_x_vir)))*position_y[np.where((halo_ID == Unique_HaloIDs[i])& (is_central == 0))[0]][0]
					position_z_cen 		= np.ones((len(position_x_vir)))*position_z[np.where((halo_ID == Unique_HaloIDs[i])& (is_central == 0))[0]][0]

					
					HI_unique 			= prop_merge[np.where(halo_ID == Unique_HaloIDs[i])[0]]
					
					# distance 			= np.zeros(len(position_x_vir))
					# merge_array 		= []

					distance 			= np.sqrt(((position_x_vir - position_x_cen))**2 + (position_y_vir - position_y_cen)**2)# + (position_z_vir - position_z_cen)**2)					
					merge_array 		= np.where(distance < maxR[i])[0]

					# for j in range(len(position_x_vir)):
					# 	distance[j]		= np.sqrt((position_x_vir[j] - position_x_cen)**2 + (position_y_vir[j] - position_y_cen)**2 + (position_z_vir[j] - position_z_cen)**2)

					# 	if distance[j] < maxR[i]:
					# 		merge_array = np.append(merge_array, j)

					a = [int(k) for k in merge_array]


					prop_final_mass 	= np.append(prop_final_mass, np.sum(HI_unique[a]))

					prop_central_mass 	= np.append(prop_central_mass, prop_merge[np.where((halo_ID == Unique_HaloIDs[i])& (is_central == 0))[0]])
					prop_satellite_mass	= np.append(prop_satellite_mass, np.sum(HI_unique[a]) - prop_merge[np.where((halo_ID == Unique_HaloIDs[i])& (is_central == 0))[0]])

					final_vir 			= np.append(final_vir, M_vir[np.where((halo_ID == Unique_HaloIDs[i])& (is_central == 0))[0]])

				print("Done reading in ",(time.time() - start))
			
				
		return prop_final_mass, prop_central_mass[final_vir > 10**12], prop_satellite_mass[final_vir > 10**12],final_vir[final_vir > 10**12]



############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
### Reading Data - Lightcone
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************



class LightconeReading:

	def __init__(self,filepath,subvolumes,filename):

		self.filepath = filepath
		self.subvolumes = subvolumes
		self.filename = filename




	def readIndividualFiles(self,fields):

		data = collections.OrderedDict()
		self.fields = fields 
		subvolumes_range = range(0,self.subvolumes)

		for idx,subv in enumerate(subvolumes_range):

			fname = os.path.join(self.filepath, self.filename +'_%.2d.hdf5'%subv)

			print('Reading data from %s'%fname)

			with h5py.File(fname,'r') as f:
		
				for gname, dsname in self.fields.items():
					print(gname)
					group = f[gname]

					for dsname in dsname:
						full_name = '%s/%s' % (gname, dsname)
						print(full_name)
						l = data.get(full_name, None)

						if l is None:
							l = group[dsname].value

						else:
							l = np.concatenate([l,group[dsname].value])

						data[full_name] = l

		return list(data.values())


	def readIndividualFiles_test(self,fields):

		data = collections.OrderedDict()
		self.fields = fields 
		subvolumes_range = self.subvolumes

		

		fname = os.path.join(self.filepath, self.filename +'_%.2d.hdf5'%subvolumes_range)

		print('Reading data from %s'%fname)

		with h5py.File(fname,'r') as f:
	
			for gname, dsname in self.fields.items():
				print(gname)
				group = f[gname]

				for dsname in dsname:
					full_name = '%s/%s' % (gname, dsname)
					print(full_name)
					l = data.get(full_name, None)

					if l is None:
						l = group[dsname].value

					else:
						l = np.concatenate([l,group[dsname].value])

					data[full_name] = l

		return list(data.values())




def abundanceMatchingMvir(mstar):

	return np.log10(10**12.35) + 0.43*np.log10((10**mstar)/(10**10.72)) + (((10**mstar)/(10**10.72)))**0.56/(1 + ((10**mstar)/(10**10.72))**(-1.54)) - 0.5


def abundanceMatchingMvir_luminosity(abs_rband_mag, abs_rband_mag_low=0):
	abs_rband_sun = 4.31
	luminosity = 10**(-0.4*(abs_rband_mag - abs_rband_sun))
	lgap 	   = luminosity/10**(-0.4*(abs_rband_mag_low - abs_rband_sun))
	alpha = -3.6*10**(-5) - 6*10**(-6)*np.log10(lgap)
	delta = 0.11 - 0.039*np.log10(lgap) + 0.0014*(np.log10(lgap))**2
	if abs_rband_mag_low != 0:
		return 10.595+4.370*(10**-4)*np.exp(np.log10(luminosity)/1.214) + alpha*(np.log10(luminosity) - 5.3)**(5.2) + delta
	else:
		return 10.595+4.370*(10**-4)*np.exp(np.log10(luminosity)/1.214) 

# def abundanceMatchingMvir_luminosity(abs_rband_mag):
# 	abs_rband_sun = 4.31
# 	luminosity = 10**(-0.4*(abs_rband_mag - abs_rband_sun))

# 	if np.log10(luminosity) >9:
# 		return 10.595+4.370*(10**-4)*np.exp(np.log10(luminosity)/1.214) + 0.5

# 	else:
# 		return 10.595+4.370*(10**-4)*np.exp(np.log10(luminosity)/1.214)



# def abundanceMatchingMvir_luminosity(abs_rband_mag, abs_rband_mag_low):
# 	abs_rband_sun = 4.31
# 	luminosity = 10**(-0.4*(abs_rband_mag - abs_rband_sun))
# 	lgap 	   = luminosity/10**(-0.4*(abs_rband_mag_low - abs_rband_sun))
# 	alpha = -3.6*10**5 - 6*10**6*np.log10(lgap)
# 	delta = 0.11 - 0.039*np.log10(lgap) + 0.0014*(np.log10(lgap))**2

# 	return 10.595+4.370*(10**-4)*np.exp(np.log10(luminosity)/1.214) + np.log10(alpha*(np.log10(luminosity) - 5.3)**(5.2) + delta)




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


def function_stacking(group_cat,lightcone_GAMA, lightcone_all, lightcone_all_GAMA, z_cut_halo=700, rvir_cut_halo=1, z_cut_central=700, rvir_cut_central=0.1, abundance_matching='luminosity'):
	
	Mvir_halo 		   = []
	Central_HI 		   = []
	HI_mass_halo	   = []

	matched_id_galaxy_sky = np.array(lightcone_GAMA['id_galaxy_sky'])

	ra_lightcone 	= np.array(lightcone_all['ra'])
	dec_lightcone 	= np.array(lightcone_all['dec'])

	vpec_z_lightcone = np.array(lightcone_all['vpec_z'])
	vpec_y_lightcone = np.array(lightcone_all['vpec_y'])
	vpec_x_lightcone = np.array(lightcone_all['vpec_x'])
	vpec_all_lightcone = np.sqrt(vpec_z_lightcone**2 + vpec_x_lightcone**2 + vpec_y_lightcone**2)
	vpec_r_lightcone = np.array(lightcone_all['vpec_r'])

	ra_selected     = np.array(lightcone_GAMA['ra'])
	dec_selected    = np.array(lightcone_GAMA['dec'])

	vpec_z_selected = np.array(lightcone_all_GAMA['vpec_z'])
	vpec_y_selected = np.array(lightcone_all_GAMA['vpec_y'])
	vpec_x_selected = np.array(lightcone_all_GAMA['vpec_x'])
	vpec_r_selected = np.array(lightcone_all_GAMA['vpec_r'])

	if group_cat == False:
		mvir_lightcone = np.array(lightcone_all_GAMA['mvir_hosthalo'])/h
		abs_rband_mag = np.array(lightcone_all_GAMA['r_ab'])
	else:
		mvir_lightcone = np.array(lightcone_all_GAMA['MassAfunc'])
		abs_rband_mag = np.array(lightcone_all_GAMA['r_ab'])

	zobs_lightcone = np.array(lightcone_all['zobs'])
	matom_lightcone = np.array(lightcone_all_GAMA['matom_all'])/h/1.35 
	mstar_lightcone = np.log10(np.array(lightcone_all_GAMA['mstar']))

	

	for i,j in zip(matched_id_galaxy_sky, range(len(matched_id_galaxy_sky))):
		
		print(j)
		
		trial = np.where(lightcone_all['id_galaxy_sky'] == i)[0]

		if abundance_matching == 'stellar':
			Mvir_value = abundanceMatchingMvir(mstar_lightcone[lightcone_all_GAMA['id_galaxy_sky']==i])
			Mvir_value = 10**Mvir_value
		elif abundance_matching == 'luminosity':
			Mvir_value = abundanceMatchingMvir_luminosity(abs_rband_mag[lightcone_all_GAMA['id_galaxy_sky']==i])
			Mvir_value = 10**Mvir_value
		else:
			Mvir_value = mvir_lightcone[lightcone_all_GAMA['id_galaxy_sky']==i]

		c1 = SkyCoord(ra=ra_selected[lightcone_GAMA['id_galaxy_sky']==i]*0.01745*u.rad, dec=dec_selected[lightcone_GAMA['id_galaxy_sky']==i]*0.01745*u.rad)
		c2 = SkyCoord(ra=ra_lightcone*0.01745*u.rad, dec=dec_lightcone*0.01745*u.rad)

		sep = c1.separation(c2)

		distance 			= (np.floor(sep.arcmin)+1)*np.pi/180/60
		distance 			= distance*Planck15.angular_diameter_distance(zobs_lightcone[trial])
		distance 			= distance.value

		host_virial_radius 	= np.ones(shape = len(distance))*Mvir2Rvir(mass=Mvir_value, z=zobs_lightcone[trial]) 

		
		######------------------------------------------
		###   zcut = 700
		######------------------------------------------

		distance_cut = np.where(distance < rvir_cut_halo*host_virial_radius)[0]

		vpec_ones = np.ones(len(lightcone_all))*vpec_r_selected[lightcone_GAMA['id_galaxy_sky']==i]
		
		merge_array = []

		for j in distance_cut: 
				
			if (abs(vpec_ones[j] - vpec_all_lightcone[j]) < z_cut_halo):

				merge_array = np.append(merge_array, j)

		a = [int(k) for k in merge_array]
		add_HI = lightcone_all['matom_all'][trial] + np.sum(lightcone_all['matom_all'][a])
		
		HI_mass_halo 			= np.append(HI_mass_halo, add_HI)


		######------------------------------------------
		###   zcut = 700 (central)
		######------------------------------------------

		distance_cut = np.where(distance < rvir_cut_central)[0]

		vpec_ones = np.ones(len(lightcone_all))*vpec_r_selected[lightcone_GAMA['id_galaxy_sky']==i]
		
		merge_array = []

		for j in distance_cut: 
				
			if (abs(vpec_ones[j] - vpec_all_lightcone[j]) < z_cut_central):

				merge_array = np.append(merge_array, j)

		a = [int(k) for k in merge_array]
		
		# print(len(a))

		add_HI = lightcone_all['matom_all'][trial] + np.sum(lightcone_all['matom_all'][a])

		Central_HI		= np.append(Central_HI, add_HI)


		Mvir_halo = np.append(Mvir_halo, Mvir_value)

	return Mvir_halo, HI_mass_halo, Central_HI



def function_stacking_vvir(group_cat,lightcone_GAMA, lightcone_all, lightcone_all_GAMA, vvir_cut_halo=1, rvir_cut_halo=1, vvir_cut_central=1, rvir_cut_central=0.1, abundance_matching=True):
	
	Mvir_halo 		   = []
	Central_HI 		   = []
	HI_mass_halo	   = []

	matched_id_galaxy_sky = np.array(lightcone_GAMA['id_galaxy_sky'])

	ra_lightcone 	= np.array(lightcone_all['ra'])
	dec_lightcone 	= np.array(lightcone_all['dec'])

	vpec_z_lightcone = np.array(lightcone_all['vpec_z'])
	vpec_y_lightcone = np.array(lightcone_all['vpec_y'])
	vpec_x_lightcone = np.array(lightcone_all['vpec_x'])
	vpec_all_lightcone = np.sqrt(vpec_z_lightcone**2 + vpec_x_lightcone**2 + vpec_y_lightcone**2)
	vpec_r_lightcone = np.array(lightcone_all['vpec_r'])

	ra_selected     = np.array(lightcone_GAMA['ra'])
	dec_selected    = np.array(lightcone_GAMA['dec'])

	vpec_z_selected = np.array(lightcone_all_GAMA['vpec_z'])
	vpec_y_selected = np.array(lightcone_all_GAMA['vpec_y'])
	vpec_x_selected = np.array(lightcone_all_GAMA['vpec_x'])
	vpec_r_selected = np.array(lightcone_all_GAMA['vpec_r'])

	
	if group_cat == False:
		mvir_lightcone = np.array(lightcone_all_GAMA['mvir_hosthalo'])/h
		abs_rband_mag = np.array(lightcone_all_GAMA['r_ab'])
	else:
		mvir_lightcone = np.array(lightcone_all_GAMA['MassAfunc'])
		abs_rband_mag = np.array(lightcone_all_GAMA['r_ab'])

	zobs_lightcone = np.array(lightcone_all['zobs'])
	matom_lightcone = np.array(lightcone_all_GAMA['matom_all'])/h/1.35 
	mstar_lightcone = np.log10(np.array(lightcone_all_GAMA['mstar']))

	virial_velocity = np.zeros(len(mvir_lightcone))


	if group_cat == False:

		for i in range(len(virial_velocity)):

			# virial_velocity[i] 	= (abundanceMatchingMvir_luminosity(mstar_lightcone[i])/2.33/1e5)**(1/3)
			virial_velocity[i] 	= 	300
		
	else:

		dispersion_all 	= np.array(lightcone_all_GAMA['VelDisp'])
	
		for i in range(len(virial_velocity)):

			# virial_velocity[i] 	= (abundanceMatchingMvir_luminosity(mstar_lightcone[i])/2.33/1e5)**(1/3)
			virial_velocity[i] 	= 	dispersion_all[i]


	for i,j in zip(matched_id_galaxy_sky, range(len(matched_id_galaxy_sky))):
		if j%1000 == 0:
			print(j)
	

		trial = np.where(lightcone_all['id_galaxy_sky'] == i)[0]

		if abundance_matching == 'stellar':
			Mvir_value = abundanceMatchingMvir(mstar_lightcone[lightcone_all_GAMA['id_galaxy_sky']==i])
			Mvir_value = 10**Mvir_value

		elif abundance_matching == 'luminosity':
			Mvir_value = abundanceMatchingMvir_luminosity(abs_rband_mag[lightcone_all_GAMA['id_galaxy_sky']==i])
			Mvir_value = 10**Mvir_value
		else:
			Mvir_value = mvir_lightcone[lightcone_all_GAMA['id_galaxy_sky']==i]


		virial_velocity 	= (Mvir_value/2.33/1e5)**(1/3)
		c1 = SkyCoord(ra=ra_selected[lightcone_GAMA['id_galaxy_sky']==i]*0.01745*u.rad, dec=dec_selected[lightcone_GAMA['id_galaxy_sky']==i]*0.01745*u.rad)
		c2 = SkyCoord(ra=ra_lightcone*0.01745*u.rad, dec=dec_lightcone*0.01745*u.rad)

		sep = c1.separation(c2)

		distance 			= (np.floor(sep.arcmin)+1)*np.pi/180/60
		distance 			= distance*Planck15.angular_diameter_distance(zobs_lightcone[trial])
		distance 			= distance.value

		host_virial_radius 	= np.ones(shape = len(distance))*Mvir2Rvir(mass=Mvir_value, z=zobs_lightcone[trial]) 

		
		######------------------------------------------
		###   zcut = 700
		######------------------------------------------

		distance_cut = np.where(distance < rvir_cut_halo*host_virial_radius)[0]

		vpec_ones = np.ones(len(lightcone_all))*vpec_r_selected[lightcone_GAMA['id_galaxy_sky']==i]
		
		merge_array = []

		for j in distance_cut: 
				
			if (abs(vpec_ones[j] - vpec_all_lightcone[j]) < vvir_cut_halo*virial_velocity):

				merge_array = np.append(merge_array, j)

		a = [int(k) for k in merge_array]
		add_HI = lightcone_all['matom_all'][trial] + np.sum(lightcone_all['matom_all'][a])
		
		HI_mass_halo 			= np.append(HI_mass_halo, add_HI)


		######------------------------------------------
		###   zcut = 700 (central)
		######------------------------------------------

		distance_cut = np.where(distance < rvir_cut_central)[0]

		vpec_ones = np.ones(len(lightcone_all))*vpec_r_selected[lightcone_GAMA['id_galaxy_sky']==i]
		
		merge_array = []

		for j in distance_cut: 
				
			if (abs(vpec_ones[j] - vpec_all_lightcone[j]) < vvir_cut_central*virial_velocity):

				merge_array = np.append(merge_array, j)

		a = [int(k) for k in merge_array]
		
		add_HI = lightcone_all['matom_all'][trial] + np.sum(lightcone_all['matom_all'][a])

		Central_HI		= np.append(Central_HI, add_HI)


		Mvir_halo = np.append(Mvir_halo, Mvir_value)

	return Mvir_halo, HI_mass_halo, Central_HI




def function_stacking_multiprocessing(galaxy_id_matched,group_cat,lightcone_GAMA, lightcone_all, lightcone_all_GAMA, z_cut_halo=700, rvir_cut_halo=1, z_cut_central=700, rvir_cut_central=0.1, rgas_value='False',abundance_matching='luminosity'):
	
	print(galaxy_id_matched)
	trial = np.where(lightcone_all['id_galaxy_sky'] == galaxy_id_matched)[0]

	if abundance_matching == 'stellar':
		Mvir_value = abundanceMatchingMvir(np.log10(np.array(lightcone_all_GAMA['mstar'][lightcone_all_GAMA['id_galaxy_sky']==galaxy_id_matched])))
		Mvir_value = 10**Mvir_value
	elif abundance_matching == 'luminosity':
		if group_cat=='False':
			Mvir_value = abundanceMatchingMvir_luminosity(np.array(lightcone_all_GAMA['r_ab'][lightcone_all_GAMA['id_galaxy_sky']==galaxy_id_matched]))
			group_id = -999
		else:
			# Mvir_value = abundanceMatchingMvir_luminosity(np.array(lightcone_all_GAMA['TotRmag'][lightcone_all_GAMA['id_galaxy_sky']==galaxy_id_matched]))
			Mvir_value = abundanceMatchingMvir_luminosity(np.array(lightcone_all_GAMA['r_ab'][lightcone_all_GAMA['id_galaxy_sky']==galaxy_id_matched]))
			group_id = lightcone_all_GAMA['GroupID'][lightcone_all_GAMA['id_galaxy_sky']==galaxy_id_matched]
		Mvir_value = 10**Mvir_value
	else:
		if group_cat=='False':
			Mvir_value = np.array(lightcone_all_GAMA['mvir_hosthalo'][lightcone_all_GAMA['id_galaxy_sky']==galaxy_id_matched])/h
			group_id = -999
		else:
			Mvir_value = np.array(lightcone_all_GAMA['MassAfunc'][lightcone_all_GAMA['id_galaxy_sky']==galaxy_id_matched])
			group_id = lightcone_all_GAMA['GroupID'][lightcone_all_GAMA['id_galaxy_sky']==galaxy_id_matched]
	
		

	c1 = SkyCoord(ra=np.array(lightcone_GAMA['ra'][lightcone_GAMA['id_galaxy_sky']==galaxy_id_matched])*0.01745*u.rad, dec=np.array(lightcone_GAMA['dec'][lightcone_GAMA['id_galaxy_sky']==galaxy_id_matched])*0.01745*u.rad)
	
	ra_selected_value 	= np.array(lightcone_GAMA['ra'][lightcone_GAMA['id_galaxy_sky']==galaxy_id_matched])

	dec_selected_value 	= np.array(lightcone_GAMA['dec'][lightcone_GAMA['id_galaxy_sky']==galaxy_id_matched])

	selected_lightcone  = lightcone_all.loc[(lightcone_all['ra']>= ra_selected_value[0] - 0.5) & (lightcone_all['ra']<= ra_selected_value[0] + 0.5) & (lightcone_all['dec']>= dec_selected_value[0] - 0.5) & (lightcone_all['dec']<= dec_selected_value[0] + 0.5)].copy()


	c2 = SkyCoord(ra=np.array(selected_lightcone['ra'])*0.01745*u.rad, dec=np.array(selected_lightcone['dec'])*0.01745*u.rad)

	sep = c1.separation(c2)

	distance 			= (np.floor(sep.arcmin)+1)*np.pi/180/60
	distance 			= distance*Planck15.angular_diameter_distance(np.array(lightcone_all['zobs'][trial]))
	distance 			= distance.value

	host_virial_radius 	= np.ones(shape = len(distance))*Mvir2Rvir(mass=Mvir_value, z=np.array(lightcone_all['zobs'][trial])) 
	rgas_intrinsic 		= 2*np.array(lightcone_all['rgas_disk_intrinsic'][trial])   # make it in kpc

	
	######------------------------------------------
	###   zcut = 700
	######------------------------------------------

	if rgas_value == 'True':
		distance_cut = np.where(distance < rvir_cut_central*rgas_intrinsic)[0]
	else:
		distance_cut = np.where(distance < rvir_cut_central)[0]


	vpec_ones = np.ones(len(selected_lightcone))*np.array(lightcone_all_GAMA['vpec_r'][lightcone_all_GAMA['id_galaxy_sky']==galaxy_id_matched])
	
	vpec_all_lightcone = np.sqrt(np.array(selected_lightcone['vpec_z'])**2 + np.array(selected_lightcone['vpec_y'])**2 + np.array(selected_lightcone['vpec_x'])**2)
	
	merge_array = []
	for j in distance_cut: 

		if (abs(vpec_ones[j] - vpec_all_lightcone[j]) < z_cut_central):

			merge_array = np.append(merge_array, j)

	a = [int(k) for k in merge_array]


	matom_sumation = np.array(selected_lightcone['matom_all'])
	add_HI = np.sum(matom_sumation[a]) + lightcone_all['matom_all'][trial] 
	HI_mass_add = np.array(add_HI)
	
	mstar_sumation = np.array(selected_lightcone['mstars_all'])
	add_star =  np.sum(mstar_sumation[a]) + lightcone_all['mstars_all'][trial]
	star_halo = np.array(add_star)

	z_galaxies_halo = []
	z_obs_add 		= np.array(selected_lightcone['zobs'])
	z_galaxies_halo = np.append(z_galaxies_halo, lightcone_all['zobs'][trial])
	z_galaxies_halo = np.append(z_galaxies_halo, z_obs_add[a])

	z_cos_halo = []
	z_cos_add 		= np.array(selected_lightcone['zcos'])
	z_cos_halo = np.append(z_cos_halo, lightcone_all['zcos'][trial])
	z_cos_halo = np.append(z_cos_halo, z_cos_add[a])

	type_galaxies_halo = []
	type_gal_add 		= np.array(selected_lightcone['type'])
	type_galaxies_halo = np.append(type_galaxies_halo, lightcone_all['type'][trial])
	type_galaxies_halo = np.append(type_galaxies_halo, type_gal_add[a])


	host_halo_id    = []
	host_halo_add	= np.array(selected_lightcone['id_halo_sam'])
	host_halo_id = np.append(host_halo_id, lightcone_all['id_halo_sam'][trial])
	host_halo_id = np.append(host_halo_id, host_halo_add[a])


	galaxy_halo_id = []
	galaxy_halo_add = np.array(selected_lightcone['id_galaxy_sky'])
	galaxy_halo_id = np.append(galaxy_halo_id, lightcone_all['id_galaxy_sky'][trial])
	galaxy_halo_id = np.append(galaxy_halo_id, galaxy_halo_add[a])


	######------------------------------------------
	###   zcut = 700 (central)
	######------------------------------------------

	distance_cut = np.where(distance < rvir_cut_central)[0]

	vpec_ones = np.ones(len(selected_lightcone))*np.array(lightcone_all_GAMA['vpec_r'][lightcone_all_GAMA['id_galaxy_sky']==galaxy_id_matched])
	
	vpec_all_lightcone = np.sqrt(np.array(selected_lightcone['vpec_z'])**2 + np.array(selected_lightcone['vpec_y'])**2 + np.array(selected_lightcone['vpec_x'])**2)
	
	merge_array = []

	for j in distance_cut: 
			
		if (abs(vpec_ones[j] - vpec_all_lightcone[j]) < z_cut_central):

			merge_array = np.append(merge_array, j)

	a = [int(k) for k in merge_array]
	

	matom_sumation = np.array(selected_lightcone['matom_all'])
	add_HI =  np.sum(matom_sumation[a]) + lightcone_all['matom_all'][trial]
	central_HI_add = np.array(add_HI)
	
	mstar_sumation = np.array(selected_lightcone['mstars_all'])
	add_star = np.sum(mstar_sumation[a]) + lightcone_all['mstars_all'][trial]
	star_central = np.array(add_star)

	z_galaxies_central = []
	z_obs_add 		= np.array(selected_lightcone['zobs'])
	z_galaxies_central = np.append(z_galaxies_central, lightcone_all['zobs'][trial])
	z_galaxies_central = np.append(z_galaxies_central, z_obs_add[a])

	z_cos_central = []
	z_cos_add 		= np.array(selected_lightcone['zcos'])
	z_cos_central = np.append(z_cos_central, lightcone_all['zcos'][trial])
	z_cos_central = np.append(z_cos_central, z_cos_add[a])



	type_galaxies_central = []
	type_gal_add 		= np.array(selected_lightcone['type'])
	type_galaxies_central = np.append(type_galaxies_central, lightcone_all['type'][trial])
	type_galaxies_central = np.append(type_galaxies_central, type_gal_add[a])


	host_central_id    = []
	host_halo_add	= np.array(selected_lightcone['id_halo_sam'])
	host_central_id = np.append(host_central_id, lightcone_all['id_halo_sam'][trial])
	host_central_id = np.append(host_central_id, host_halo_add[a])

	

	galaxy_central_id = []
	galaxy_central_add = np.array(selected_lightcone['id_galaxy_sky'])
	galaxy_central_id = np.append(galaxy_central_id, lightcone_all['id_galaxy_sky'][trial])
	galaxy_central_id = np.append(galaxy_central_id, galaxy_central_add[a])



	Mvir_add = Mvir_value

	return Mvir_add, HI_mass_add, central_HI_add, star_halo, star_central, z_galaxies_halo, z_galaxies_central, type_galaxies_halo, type_galaxies_central, host_halo_id, host_central_id, group_id, galaxy_halo_id, galaxy_central_id, z_cos_halo, z_cos_central





def function_stacking_vvir_multiprocessing(galaxy_id_matched,group_cat,lightcone_GAMA, lightcone_all, lightcone_all_GAMA, vvir_cut_halo=700, rvir_cut_halo=1, vvir_cut_central=700, rvir_cut_central=0.1,rgas_value='False', abundance_matching='luminosity', disperstion='True'):
	
	print(galaxy_id_matched)

	trial = np.where(lightcone_all['id_galaxy_sky'] == galaxy_id_matched)[0]

	if abundance_matching == 'stellar':
		Mvir_value = abundanceMatchingMvir(np.log10(np.array(lightcone_all_GAMA['mstar'][lightcone_all_GAMA['id_galaxy_sky']==galaxy_id_matched])))
		Mvir_value = 10**Mvir_value
	elif abundance_matching == 'luminosity':
		if group_cat=='False':
			Mvir_value = abundanceMatchingMvir_luminosity(np.array(lightcone_all_GAMA['r_ab'][lightcone_all_GAMA['id_galaxy_sky']==galaxy_id_matched]))
			group_id = -999
			Mvir_value = 10**Mvir_value
			if disperstion=='True':
				virial_velocity = 300
			else:
				virial_velocity = (Mvir_value/2.33/1e5)**(1/3)

		else:
			Mvir_value = abundanceMatchingMvir_luminosity(np.array(lightcone_all_GAMA['r_ab'][lightcone_all_GAMA['id_galaxy_sky']==galaxy_id_matched]))
			group_id = lightcone_all_GAMA['GroupID'][lightcone_all_GAMA['id_galaxy_sky']==galaxy_id_matched]
			Mvir_value = 10**Mvir_value
			if disperstion=='True':
				virial_velocity = np.array(lightcone_all_GAMA['VelDisp'][lightcone_all_GAMA['id_galaxy_sky']==galaxy_id_matched])
			else:
				virial_velocity = (Mvir_value/2.33/1e5)**(1/3)

	else:
		if group_cat=='False':
			Mvir_value = np.array(lightcone_all_GAMA['mvir_hosthalo'][lightcone_all_GAMA['id_galaxy_sky']==galaxy_id_matched])/h
			group_id = -999
		else:
			Mvir_value = np.array(lightcone_all_GAMA['MassAfunc'][lightcone_all_GAMA['id_galaxy_sky']==galaxy_id_matched])
			group_id = lightcone_all_GAMA['GroupID'][lightcone_all_GAMA['id_galaxy_sky']==galaxy_id_matched]
	
		

	c1 = SkyCoord(ra=np.array(lightcone_GAMA['ra'][lightcone_GAMA['id_galaxy_sky']==galaxy_id_matched])*0.01745*u.rad, dec=np.array(lightcone_GAMA['dec'][lightcone_GAMA['id_galaxy_sky']==galaxy_id_matched])*0.01745*u.rad)
	
	ra_selected_value 	= np.array(lightcone_GAMA['ra'][lightcone_GAMA['id_galaxy_sky']==galaxy_id_matched])

	dec_selected_value 	= np.array(lightcone_GAMA['dec'][lightcone_GAMA['id_galaxy_sky']==galaxy_id_matched])

	selected_lightcone  = lightcone_all.loc[(lightcone_all['ra']>= ra_selected_value[0] - 0.5) & (lightcone_all['ra']<= ra_selected_value[0] + 0.5) & (lightcone_all['dec']>= dec_selected_value[0] - 0.5) & (lightcone_all['dec']<= dec_selected_value[0] + 0.5)].copy()


	c2 = SkyCoord(ra=np.array(selected_lightcone['ra'])*0.01745*u.rad, dec=np.array(selected_lightcone['dec'])*0.01745*u.rad)

	sep = c1.separation(c2)

	distance 			= (np.floor(sep.arcmin)+1)*np.pi/180/60
	distance 			= distance*Planck15.angular_diameter_distance(np.array(lightcone_all['zobs'][trial]))
	distance 			= distance.value

	host_virial_radius 	= np.ones(shape = len(distance))*Mvir2Rvir(mass=Mvir_value, z=np.array(lightcone_all['zobs'][trial])) 


	# virial_velocity 	= (Mvir_value/2.33/1e5)**(1/3)
	rgas_intrinsic 		= 2*np.array(lightcone_all['rgas_disk_intrinsic'][trial])   # make it in kpc

	######------------------------------------------
	###   zcut = 700
	######------------------------------------------

	distance_cut = np.where(distance < rvir_cut_halo*host_virial_radius)[0]

	vpec_ones = np.ones(len(selected_lightcone))*np.array(lightcone_all_GAMA['vpec_r'][lightcone_all_GAMA['id_galaxy_sky']==galaxy_id_matched])
	
	vpec_all_lightcone = np.sqrt(np.array(selected_lightcone['vpec_z'])**2 + np.array(selected_lightcone['vpec_y'])**2 + np.array(selected_lightcone['vpec_x'])**2)
	
	merge_array = []
	for j in distance_cut: 

		if group_cat == 'False':
			if (abs(vpec_ones[j] - vpec_all_lightcone[j]) < virial_velocity):

				merge_array = np.append(merge_array, j)

		else:		
			if (abs(vpec_ones[j] - vpec_all_lightcone[j]) < vvir_cut_halo*virial_velocity):

				merge_array = np.append(merge_array, j)

	a = [int(k) for k in merge_array]


	matom_sumation = np.array(selected_lightcone['matom_all'])
	add_HI = np.sum(matom_sumation[a]) + lightcone_all['matom_all'][trial]
	HI_mass_add = np.array(add_HI)
	
	mstar_sumation = np.array(selected_lightcone['mstars_all'])
	add_star = np.sum(mstar_sumation[a]) + lightcone_all['mstars_all'][trial] 
	star_halo = np.array(add_star)

	z_galaxies_halo = []
	z_obs_add 		= np.array(selected_lightcone['zobs'])
	z_galaxies_halo = np.append(z_galaxies_halo, lightcone_all['zobs'][trial])
	z_galaxies_halo = np.append(z_galaxies_halo, z_obs_add[a])

	z_cos_halo = []
	z_cos_add 		= np.array(selected_lightcone['zcos'])
	z_cos_halo = np.append(z_cos_halo, lightcone_all['zcos'][trial])
	z_cos_halo = np.append(z_cos_halo, z_cos_add[a])


	type_galaxies_halo = []
	type_gal_add 		= np.array(selected_lightcone['type'])
	type_galaxies_halo = np.append(type_galaxies_halo, lightcone_all['type'][trial])
	type_galaxies_halo = np.append(type_galaxies_halo, type_gal_add[a])


	host_halo_id    = []
	host_halo_add	= np.array(selected_lightcone['id_halo_sam'])
	host_halo_id = np.append(host_halo_id, lightcone_all['id_halo_sam'][trial])
	host_halo_id = np.append(host_halo_id, host_halo_add[a])

	galaxy_halo_id = []
	galaxy_halo_add = np.array(selected_lightcone['id_galaxy_sky'])
	galaxy_halo_id = np.append(galaxy_halo_id, lightcone_all['id_galaxy_sky'][trial])
	galaxy_halo_id = np.append(galaxy_halo_id, galaxy_halo_add[a])


	######------------------------------------------
	###   zcut = 700 (central)
	######------------------------------------------

	if rgas_value == 'True':
		distance_cut = np.where(distance < rvir_cut_central*rgas_intrinsic)[0]
	else:
		distance_cut = np.where(distance < rvir_cut_central)[0]

	vpec_ones = np.ones(len(selected_lightcone))*np.array(lightcone_all_GAMA['vpec_r'][lightcone_all_GAMA['id_galaxy_sky']==galaxy_id_matched])
	
	vpec_all_lightcone = np.sqrt(np.array(selected_lightcone['vpec_z'])**2 + np.array(selected_lightcone['vpec_y'])**2 + np.array(selected_lightcone['vpec_x'])**2)
	
	merge_array = []

	for j in distance_cut: 
			
		if (abs(vpec_ones[j] - vpec_all_lightcone[j]) < 300):

			merge_array = np.append(merge_array, j)

	a = [int(k) for k in merge_array]
	

	matom_sumation = np.array(selected_lightcone['matom_all'])
	add_HI = np.sum(matom_sumation[a]) + lightcone_all['matom_all'][trial] 
	central_HI_add = np.array(add_HI)
	
	mstar_sumation = np.array(selected_lightcone['mstars_all'])
	add_star = np.sum(mstar_sumation[a]) + lightcone_all['mstars_all'][trial] 
	star_central = np.array(add_star)

	z_galaxies_central = []
	z_obs_add 		= np.array(selected_lightcone['zobs'])
	z_galaxies_central = np.append(z_galaxies_central, lightcone_all['zobs'][trial])
	z_galaxies_central = np.append(z_galaxies_central, z_obs_add[a])

	z_cos_central = []
	z_cos_add 		= np.array(selected_lightcone['zcos'])
	z_cos_central = np.append(z_cos_central, lightcone_all['zcos'][trial])
	z_cos_central = np.append(z_cos_central, z_cos_add[a])

	type_galaxies_central = []
	type_gal_add 		= np.array(selected_lightcone['type'])
	type_galaxies_central = np.append(type_galaxies_central, lightcone_all['type'][trial])
	type_galaxies_central = np.append(type_galaxies_central, type_gal_add[a])


	host_central_id    = []
	host_halo_add	= np.array(selected_lightcone['id_halo_sam'])
	host_central_id = np.append(host_central_id, lightcone_all['id_halo_sam'][trial])
	host_central_id = np.append(host_central_id, host_halo_add[a])
	

	
	galaxy_central_id = []
	galaxy_central_add = np.array(selected_lightcone['id_galaxy_sky'])
	galaxy_central_id = np.append(galaxy_central_id, lightcone_all['id_galaxy_sky'][trial])
	galaxy_central_id = np.append(galaxy_central_id, galaxy_central_add[a])

	Mvir_add = Mvir_value

	return Mvir_add, HI_mass_add, central_HI_add, star_halo, star_central, z_galaxies_halo, z_galaxies_central, type_galaxies_halo, type_galaxies_central, host_halo_id, host_central_id, group_id, galaxy_halo_id, galaxy_central_id, z_cos_halo, z_cos_central


