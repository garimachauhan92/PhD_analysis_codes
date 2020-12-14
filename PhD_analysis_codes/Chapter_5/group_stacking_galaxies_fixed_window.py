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
import statistics as statistics
import csv as csv
from Common_module import SharkDataReading
import Common_module
from Common_module import LightconeReading

import pandas as pd
# import splotch as spl

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck15
import sys

from multiprocessing import Pool
from argparse import ArgumentParser

############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
### Parser Arguments
######--------------------------------------------------------------------------------------------------------------------------------------
####****************************************************************************************************************************************

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
### Parser Arguments
######--------------------------------------------------------------------------------------------------------------------------------------
####****************************************************************************************************************************************

parser = ArgumentParser()
parser.add_argument('-zH',
					action='store',dest='zcut_halo',type=int,default=None,
					help='zcut for the halo')

parser.add_argument('-zC',
					action='store',dest='zcut_central',type=str,default=None,
					help='zcut for the central')


parser.add_argument('-rH',
					action='store',dest='rvir_cut_halo',type=str,default=None,
					help='halo aperture times')


parser.add_argument('-rC',
					action='store',dest='rvir_cut_central',type=str,default=None,
					help='central aperture value')



parser.add_argument('-grC',
					action='store',dest='group_cat',type=str,default=None,
					help='Group central or isolated central')



parser.add_argument('-rgasV',
					action='store',dest='rgas_value',type=str,default=None,
					help='whether you want an aperture equal to the rgas or not')



parser.add_argument('-name',
					action='store',dest='name_hdf5',type=str,default=None,
					help='file name for the hdf5')


parser.add_argument('-bunch',
					action='store',nargs='+',dest='bunch',type=int,default=None,
					help='which lot of file you want to process')


parser.add_argument('-n',
					action='store',dest='num',type=str,default=None,
					help='number of threads you need')


parser.add_argument('-restart',
					action='store',dest='restart',type=str,default=None,
					help='if we stopped the process or not')


args = parser.parse_args()

zcut_halo = int(args.zcut_halo)
zcut_central = int(args.zcut_central)

rvir_cut_halo = float(args.rvir_cut_halo)
rvir_cut_central = float(args.rvir_cut_central)

group_cat = args.group_cat
name_hdf5 = str(args.name_hdf5)
bunch = args.bunch
rgas_value = args.rgas_value

num=24
num=int(args.num)

restart = args.restart





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

# path_GAMA = "/group/pawsey0119/gchauhan/HI_stacking_paper/Matias_GAMA_new/"
path_GAMA = "/mnt/su3ctm/gchauhan/HI_stacking_paper/Matias_GAMA_new/"

if group_cat=='False':
	grand_isolated 				= pd.read_csv(path_GAMA + "grand_isolated.csv")
	grand_isolated_lightcone	= pd.read_csv(path_GAMA + "grand_isolated_lightcone.csv")
	lightcone_alfalfa 			= pd.read_csv(path_GAMA + "lightcone_alfalfa.csv")

else:
	grand_group 				= pd.read_csv(path_GAMA + "grand_group.csv")
	grand_group_lightcone 		= pd.read_csv(path_GAMA + "grand_group_lightcone.csv")
	lightcone_alfalfa 			= pd.read_csv(path_GAMA + "lightcone_alfalfa.csv")


############################################################################################################################################
######--------------------------------------------------------------------------------------------------------------------------------------
###                                                   Function development
######--------------------------------------------------------------------------------------------------------------------------------------
####*********************************************************


import time
start_time = time.time()

if __name__ == '__main__':

	num_threads = num

	with Pool(num_threads) as pool:

		if group_cat == 'False':
			matched_id_galaxy_sky 	= np.array(grand_isolated['id_galaxy_sky'])
			group_func 				= grand_isolated
			group_func_lightcone	= grand_isolated_lightcone
			

			if len(bunch) == 1:
				total_tasks = len(matched_id_galaxy_sky[bunch[0]:35300 + bunch[0]])
				tasks = matched_id_galaxy_sky[bunch[0]: bunch[0]+35300]
			
			elif len(bunch) == 2:
				total_tasks = len(matched_id_galaxy_sky[bunch[0]:len(matched_id_galaxy_sky)])
				tasks = matched_id_galaxy_sky[bunch[0]:len(matched_id_galaxy_sky)]
		
			else:
				total_tasks = len(matched_id_galaxy_sky)
				tasks = matched_id_galaxy_sky

			group_id_isol = tasks



		else:
			matched_id_galaxy_sky = np.array(grand_group['id_galaxy_sky'])
			group_func 				= grand_group
			group_func_lightcone	= grand_group_lightcone

			if len(bunch) == 1:
				total_tasks = len(matched_id_galaxy_sky[bunch[0]:35300 + bunch[0]])
				tasks = matched_id_galaxy_sky[bunch[0]: bunch[0]+35300]

		
			elif len(bunch) == 2:
				total_tasks = len(matched_id_galaxy_sky[bunch[0]:len(matched_id_galaxy_sky)])
				tasks = matched_id_galaxy_sky[bunch[0]:len(matched_id_galaxy_sky)]
		
			else:
				total_tasks = len(matched_id_galaxy_sky)
				tasks = matched_id_galaxy_sky


		if group_cat=='False':
			group_id = tasks



		for t in tasks:

			print(name_hdf5)
			print(restart)
			print(np.where(tasks == t)[0][0])
			
			temp = [pool.apply(Common_module.function_stacking_multiprocessing,(t,group_cat,group_func,lightcone_alfalfa,group_func_lightcone, zcut_halo, rvir_cut_halo, zcut_central, rvir_cut_central, rgas_value, 'luminosity'))]




			# pool.close()
			# pool.join()


			############################################################################################################################################
			######--------------------------------------------------------------------------------------------------------------------------------------
			###                                                    writing HDF5 file
			######--------------------------------------------------------------------------------------------------------------------------------------


			if restart == 'False':


				if t == tasks[0]:

					if group_cat == 'False':
						group_id = np.array(t)
						group_writing = int(t)
					else:
						group_id = np.array(temp[0][11])
						group_writing = int(temp[0][11])	

					mvir_temp = temp[0][0]
					HI_halo_temp = temp[0][1]
					HI_central_temp = temp[0][2]
					star_halo_temp=temp[0][3]
					star_central_temp=temp[0][4]
					zobs_halo={group_writing:temp[0][5]}
					zobs_central={group_writing:temp[0][6]}
					type_halo = {group_writing:temp[0][7]}
					type_central = {group_writing:temp[0][8]}
					host_halo = {group_writing:temp[0][9]}
					host_central = {group_writing:temp[0][10]}
					galaxy_halo_id={group_writing:temp[0][12]}
					galaxy_central_id={group_writing:temp[0][13]}
					zcos_halo={group_writing:temp[0][14]}
					zcos_central={group_writing:temp[0][15]}

					
					loop_length = len(temp)

					# hf = h5py.File('/group/pawsey0119/gchauhan/HI_stacking_paper/calculated_values/%s.hdf5'%name_hdf5, 'w')
					hf = h5py.File('/mnt/su3ctm/gchauhan/HI_stacking_paper/calculated_values/%s.hdf5'%name_hdf5, 'w')

					hf.create_dataset('mvir_abmatch', data = mvir_temp, chunks=(1,), maxshape=(None,))
					hf.create_dataset('mhi_halo', data = HI_halo_temp, chunks=(1,), maxshape=(None,))
					hf.create_dataset('mhi_central', data = HI_central_temp, chunks=(1,), maxshape=(None,))
					hf.create_dataset('mstar_halo', data = star_halo_temp, chunks=(1,), maxshape=(None,))
					hf.create_dataset('mstar_central', data = star_central_temp, chunks=(1,), maxshape=(None,))

					if group_cat=='False':
						hf.create_dataset('GAMA_GroupID', data = group_id_isol)
					else:
						hf.create_dataset('GAMA_GroupID', data = group_id, chunks=(1,), maxshape=(None,))

					zobs_halo_hdf5	=	hf.create_group('zobs_halo')
					zobs_central_hdf5	=	hf.create_group('zobs_central')
					zcos_halo_hdf5	=	hf.create_group('zcos_halo')
					zcos_central_hdf5	=	hf.create_group('zcos_central')


					id_halo_hdf5 = hf.create_group("hosthalo_id_halo")
					id_central_hdf5 = hf.create_group("hosthalo_id_central")
					gal_type_halo 	= hf.create_group("gal_type_halo")	
					gal_type_central = hf.create_group("gal_type_central")
					gal_id_halo = hf.create_group("id_galaxy_halo")
					gal_id_central = hf.create_group("id_galaxy_central")

					for k,v in zobs_halo.items():
						zobs_halo_hdf5.create_dataset('%s'%str(int(group_id)), data=v)


					for k,v in zobs_central.items():
						zobs_central_hdf5.create_dataset('%s'%str(int(group_id)), data=v)


					for k,v in zcos_halo.items():
						zcos_halo_hdf5.create_dataset('%s'%str(int(group_id)), data=v)


					for k,v in zcos_central.items():
						zcos_central_hdf5.create_dataset('%s'%str(int(group_id)), data=v)


					for k,v in host_halo.items():
						id_halo_hdf5.create_dataset('%s'%str(int(group_id)), data=v)


					for k,v in host_central.items():
						id_central_hdf5.create_dataset('%s'%str(int(group_id)), data=v)


					for k,v in type_halo.items():
						gal_type_halo.create_dataset('%s'%str(int(group_id)), data=v)


					for k,v in type_central.items():
						gal_type_central.create_dataset('%s'%str(int(group_id)), data=v)


					for k,v in galaxy_halo_id.items():
						gal_id_halo.create_dataset('%s'%str(int(group_id)), data=v)


					for k,v in galaxy_central_id.items():
						gal_id_central.create_dataset('%s'%str(int(group_id)), data=v)

					hf.close()


				else:

					# group_writing = int(temp[0][11])
					# group_id = np.array(temp[0][11])	
					if group_cat == 'False':
						group_id = np.array(t)
						group_writing = int(t)
					else:
						group_id = np.array(temp[0][11])
						group_writing = int(temp[0][11])	



					mvir_temp = temp[0][0]
					HI_halo_temp = temp[0][1]
					HI_central_temp = temp[0][2]
					star_halo_temp=temp[0][3]
					star_central_temp=temp[0][4]
					zobs_halo={group_writing:temp[0][5]}
					zobs_central={group_writing:temp[0][6]}
					type_halo = {group_writing:temp[0][7]}
					type_central = {group_writing:temp[0][8]}
					host_halo = {group_writing:temp[0][9]}
					host_central = {group_writing:temp[0][10]}
					galaxy_halo_id={group_writing:temp[0][12]}
					galaxy_central_id={group_writing:temp[0][13]}
					zcos_halo={group_writing:temp[0][14]}
					zcos_central={group_writing:temp[0][15]}



					loop_length = len(temp)

					# hf = h5py.File('/group/pawsey0119/gchauhan/HI_stacking_paper/calculated_values/%s.hdf5'%name_hdf5, 'a')
					hf = h5py.File('/mnt/su3ctm/gchauhan/HI_stacking_paper/calculated_values/%s.hdf5'%name_hdf5, 'a')

					hf['mvir_abmatch'].resize((hf['mvir_abmatch'].shape[0] + mvir_temp.shape[0]), axis = 0 )
					hf['mvir_abmatch'][-mvir_temp.shape[0]:] = mvir_temp

					hf['mhi_halo'].resize((hf['mhi_halo'].shape[0] + HI_halo_temp.shape[0]), axis = 0 )
					hf['mhi_halo'][-HI_halo_temp.shape[0]:] = HI_halo_temp

					hf['mhi_central'].resize((hf['mhi_central'].shape[0] + HI_central_temp.shape[0]), axis = 0 )
					hf['mhi_central'][-HI_central_temp.shape[0]:] = HI_central_temp

					hf['mstar_halo'].resize((hf['mstar_halo'].shape[0] + star_halo_temp.shape[0]), axis = 0 )
					hf['mstar_halo'][-star_halo_temp.shape[0]:] = star_halo_temp

					hf['mstar_central'].resize((hf['mstar_central'].shape[0] + star_central_temp.shape[0]), axis = 0 )
					hf['mstar_central'][-star_central_temp.shape[0]:] = star_central_temp

					if group_cat=='True':
						hf['GAMA_GroupID'].resize((hf['GAMA_GroupID'].shape[0] + group_id.shape[0]), axis = 0 )
						hf['GAMA_GroupID'][-group_id.shape[0]:] = group_id




					for k,v in zobs_halo.items():
						hf.create_dataset('/zobs_halo/%s'%str(int(group_id)), data=v, chunks=(1,), maxshape=(None,))


					for k,v in zobs_central.items():
						hf.create_dataset('/zobs_central/%s'%str(int(group_id)), data=v, chunks=(1,), maxshape=(None,))


					for k,v in zcos_halo.items():
						hf.create_dataset('/zcos_halo/%s'%str(int(group_id)), data=v, chunks=(1,), maxshape=(None,))


					for k,v in zcos_central.items():
						hf.create_dataset('/zcos_central/%s'%str(int(group_id)), data=v, chunks=(1,), maxshape=(None,))


					for k,v in host_halo.items():
						hf.create_dataset('/hosthalo_id_halo/%s'%str(int(group_id)), data=v, chunks=(1,), maxshape=(None,))


					for k,v in host_central.items():
						hf.create_dataset('/hosthalo_id_central/%s'%str(int(group_id)), data=v, chunks=(1,), maxshape=(None,))


					for k,v in type_halo.items():
						hf.create_dataset('/gal_type_halo/%s'%str(int(group_id)), data=v, chunks=(1,), maxshape=(None,))


					for k,v in type_central.items():
						hf.create_dataset('/gal_type_central/%s'%str(int(group_id)), data=v, chunks=(1,), maxshape=(None,))


					for k,v in galaxy_halo_id.items():
						hf.create_dataset('/id_galaxy_halo/%s'%str(int(group_id)), data=v, chunks=(1,), maxshape=(None,))


					for k,v in galaxy_central_id.items():
						hf.create_dataset('/id_galaxy_central/%s'%str(int(group_id)), data=v, chunks=(1,), maxshape=(None,))


					hf.close()


			else:

				if group_cat == 'False':
					group_id = np.array(t)
					group_writing = int(t)
				else:
					group_id = np.array(temp[0][11])
					group_writing = int(temp[0][11])	

				mvir_temp = temp[0][0]
				HI_halo_temp = temp[0][1]
				HI_central_temp = temp[0][2]
				star_halo_temp=temp[0][3]
				star_central_temp=temp[0][4]
				zobs_halo={group_writing:temp[0][5]}
				zobs_central={group_writing:temp[0][6]}
				type_halo = {group_writing:temp[0][7]}
				type_central = {group_writing:temp[0][8]}
				host_halo = {group_writing:temp[0][9]}
				host_central = {group_writing:temp[0][10]}
				galaxy_halo_id={group_writing:temp[0][12]}
				galaxy_central_id={group_writing:temp[0][13]}
				zcos_halo={group_writing:temp[0][14]}
				zcos_central={group_writing:temp[0][15]}



				loop_length = len(temp)

				# hf = h5py.File('/group/pawsey0119/gchauhan/HI_stacking_paper/calculated_values/%s.hdf5'%name_hdf5, 'a')
				hf = h5py.File('/mnt/su3ctm/gchauhan/HI_stacking_paper/calculated_values/%s.hdf5'%name_hdf5, 'a')
				
				hf['mvir_abmatch'].resize((hf['mvir_abmatch'].shape[0] + mvir_temp.shape[0]), axis = 0 )
				hf['mvir_abmatch'][-mvir_temp.shape[0]:] = mvir_temp

				hf['mhi_halo'].resize((hf['mhi_halo'].shape[0] + HI_halo_temp.shape[0]), axis = 0 )
				hf['mhi_halo'][-HI_halo_temp.shape[0]:] = HI_halo_temp

				hf['mhi_central'].resize((hf['mhi_central'].shape[0] + HI_central_temp.shape[0]), axis = 0 )
				hf['mhi_central'][-HI_central_temp.shape[0]:] = HI_central_temp

				hf['mstar_halo'].resize((hf['mstar_halo'].shape[0] + star_halo_temp.shape[0]), axis = 0 )
				hf['mstar_halo'][-star_halo_temp.shape[0]:] = star_halo_temp

				hf['mstar_central'].resize((hf['mstar_central'].shape[0] + star_central_temp.shape[0]), axis = 0 )
				hf['mstar_central'][-star_central_temp.shape[0]:] = star_central_temp

				if group_cat=='True':
					hf['GAMA_GroupID'].resize((hf['GAMA_GroupID'].shape[0] + group_id.shape[0]), axis = 0 )
					hf['GAMA_GroupID'][-group_id.shape[0]:] = group_id




				for k,v in zobs_halo.items():
					hf.create_dataset('/zobs_halo/%s'%str(int(group_id)), data=v, chunks=(1,), maxshape=(None,))


				for k,v in zobs_central.items():
					hf.create_dataset('/zobs_central/%s'%str(int(group_id)), data=v, chunks=(1,), maxshape=(None,))


				for k,v in zcos_halo.items():
					hf.create_dataset('/zcos_halo/%s'%str(int(group_id)), data=v, chunks=(1,), maxshape=(None,))


				for k,v in zcos_central.items():
					hf.create_dataset('/zcos_central/%s'%str(int(group_id)), data=v, chunks=(1,), maxshape=(None,))


				for k,v in host_halo.items():
					hf.create_dataset('/hosthalo_id_halo/%s'%str(int(group_id)), data=v, chunks=(1,), maxshape=(None,))


				for k,v in host_central.items():
					hf.create_dataset('/hosthalo_id_central/%s'%str(int(group_id)), data=v, chunks=(1,), maxshape=(None,))


				for k,v in type_halo.items():
					hf.create_dataset('/gal_type_halo/%s'%str(int(group_id)), data=v, chunks=(1,), maxshape=(None,))


				for k,v in type_central.items():
					hf.create_dataset('/gal_type_central/%s'%str(int(group_id)), data=v, chunks=(1,), maxshape=(None,))


				for k,v in galaxy_halo_id.items():
					hf.create_dataset('/id_galaxy_halo/%s'%str(int(group_id)), data=v, chunks=(1,), maxshape=(None,))


				for k,v in galaxy_central_id.items():
					hf.create_dataset('/id_galaxy_central/%s'%str(int(group_id)), data=v, chunks=(1,), maxshape=(None,))


				hf.close()

	# return Mvir_halo_return,HI_mass_halo_return,Central_HI_return
	# return a,b,c


print((time.time() - start_time)/60)



# 700 700 0.5 4 False True GAMA_isol_z700_rgT_rH0p5_rC4_3 10500
