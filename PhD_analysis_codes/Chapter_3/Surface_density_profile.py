import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as sci
import pandas as pd
import csv as csv
import mpl_toolkits.axisartist.floating_axes as floating_axes
import numpy as np
from matplotlib.projections import PolarAxes
import matplotlib.lines as mlines
import matplotlib.patches as mpatches


import matplotlib as mpl

figSize = (12,8)
labelFS = 22
tickFS = 20
titleFS = 24
textFS = 16
legendFS = 16
linewidth_plot = 3
markersize_lines_plot = 10

# Adjust axes
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['axes.axisbelow'] = 'line'

# Adjust Fonts
mpl.rcParams['font.family'] = "sans-serif"
mpl.rcParams['text.usetex'] = False
mpl.rcParams['text.latex.unicode'] = True
mpl.rcParams['mathtext.fontset'] = "dejavusans"


mpl.rcParams['axes.titlesize'] = titleFS
mpl.rcParams['axes.labelsize'] = labelFS
mpl.rcParams['xtick.labelsize'] = tickFS
mpl.rcParams['ytick.labelsize'] = tickFS
mpl.rcParams['legend.fontsize'] = legendFS

# Adjust line-widths
mpl.rcParams['lines.linewidth'] = linewidth_plot
mpl.rcParams['lines.markersize'] = markersize_lines_plot

#Adjust Legend
mpl.rcParams['legend.markerscale'] = 1

# Adjust ticks
for a in ['x','y']:
    mpl.rcParams['{0}tick.major.size'.format(a)] = 5.0
    mpl.rcParams['{0}tick.minor.size'.format(a)] = 2.5
    
    mpl.rcParams['{0}tick.major.width'.format(a)] = 1.0
    mpl.rcParams['{0}tick.minor.width'.format(a)] = 1.0
    
    mpl.rcParams['{0}tick.direction'.format(a)] = 'in'
    mpl.rcParams['{0}tick.minor.visible'.format(a)] = True

mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True

# Adjust figure and subplots
mpl.rcParams['figure.figsize'] = figSize
mpl.rcParams['figure.subplot.left'] = 0.1
mpl.rcParams['figure.subplot.right'] = 0.96
mpl.rcParams['figure.subplot.top'] = 0.96



###---------------------------------------------------------------------------------------------------------------------------
# Constants
###--------------------------------------------------------------------------------------------------------------------------

#G             = 4.301e-9     	# In Mpc
G             = 4.301e-6		# In Kpc
G_cgs         = 6.674e-8

M_solar_2_g   = 1.99e33
#Mpc_2_cm      = 3.086e24		# Mpc to cm
Mpc_2_cm      = 3.086e21		# Kpc to cm	
k_boltzmann   = 1.3807e-16

#cm_2_mpc      = 3.24e-25		# cm to Mpc
cm_2_mpc      = 3.24e-22		# cm to Kpc
g_2_M_solar   = 0.5e-33

h 			  = 0.6751

dt  = np.dtype(int)
df  = np.dtype(float)



# In[4]:


def movingaverage(interval, window_size):

	window = np.ones(int(window_size))/float(window_size)
	return np.convolve(interval, window, 'same')


# In[5]:


def surface_density(R_halo,R_disk,R_HI,R_bulge,M_cold_gas,M_stars_disk,M_halo,M_disk,M_bulge,c_halo,c_disk,c_bulge,c_HI, sini, M_gas_bulge, M_stars_bulge):
	v_g = 10

	## Making the Radius Channel - -------------------------------------------------------------------------

	
	xmax = 1
	dx = 0.004
	nx = int(xmax/dx)
	x = np.zeros(shape = nx)


	for i in range(1,nx + 1):
		x[i - 1] = (i - 0.5) * dx


	r_x = (R_halo/1.67)*x
	
	R = R_halo/1.67

	if R_HI == 0:
		R_HI = 3*R_disk/1.67	

		c_HI = R_halo/R_HI




	#if R_disk == 0:
	#	return  None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

	#if R_halo == 0:
	#	return  None, None, None, None, None, None, None, None, None, None, None, None, None, None, None




	## Making Surface Density Profile   DISK-------------------------------------------------------------------
    

	Gas_galaxy = np.zeros(shape = nx)
	P_star_galaxy = np.zeros(shape = nx)
	P_gas_galaxy = np.zeros(shape = nx)
	v_star_galaxy = np.zeros(shape=nx)
	hstar_galaxy = R_HI/7.3*Mpc_2_cm

	M_cold_gas_cgs = M_cold_gas*M_solar_2_g
	M_stars_disk_cgs = M_stars_disk*M_solar_2_g
	R_HI_cgs = R_HI*Mpc_2_cm

	#M_stars_bulge_cgs = M_stars_bulge*M_solar_2_g
	#R_bulge_cgs = R_bulge*Mpc_2_cm
	
	r_x_cgs = r_x*Mpc_2_cm



	A_cgs = -r_x_cgs/R_HI_cgs

	for i in range(0,nx):

		Gas_galaxy[i] = M_cold_gas_cgs/(2*np.pi*(R_HI_cgs)**2)*np.exp(A_cgs[i])

		
		P_star_galaxy[i] = ((M_stars_disk_cgs)/(2*np.pi*(R_HI_cgs)**2)*np.exp(A_cgs[i]))

		P_gas_galaxy[i] = (M_cold_gas_cgs/(2*np.pi*(R_HI_cgs)**2))*np.exp(A_cgs[i])

		v_star_galaxy[i] = np.sqrt(np.pi*G_cgs*hstar_galaxy*P_star_galaxy[i])


	P_ext_galaxy = np.pi/2*G_cgs*Gas_galaxy
	P_ext_galaxy = P_ext_galaxy*(P_gas_galaxy + (v_g*1e5/v_star_galaxy)*P_star_galaxy)
	P_ext_galaxy = P_ext_galaxy/k_boltzmann

	P_ext = max(P_ext_galaxy)
	
	#P_0 = 3.7*10e4
	P_0  = 34673

	#R_c_galaxy = (P_ext_galaxy/P_0)**0.8

	R_c_galaxy = (P_ext_galaxy/P_0)**0.92


	f_H1_galaxy = 1/(1 + R_c_galaxy)
	f_H2_galaxy = R_c_galaxy/(1 + R_c_galaxy)



	surface_H1_galaxy = f_H1_galaxy*Gas_galaxy
	H1 = np.trapz(r_x_cgs*surface_H1_galaxy, dx=dx*Mpc_2_cm)
	surface_H1_galaxy_disk = surface_H1_galaxy/H1/cm_2_mpc**2

	surface_H2_galaxy = f_H2_galaxy*Gas_galaxy
	H2 = np.trapz(r_x_cgs*surface_H2_galaxy,dx = dx*Mpc_2_cm) 
	surface_H2_galaxy_disk = surface_H2_galaxy/H2/cm_2_mpc**2
	#---------------------------------------------------------------------------------------------------------------------

	Gas_galaxy = np.zeros(shape = nx)
	P_star_galaxy = np.zeros(shape = nx)
	P_gas_galaxy = np.zeros(shape = nx)
	v_star_galaxy = np.zeros(shape=nx)
	hstar_galaxy = R_bulge/7.3*Mpc_2_cm

	M_cold_gas_cgs = M_gas_bulge*M_solar_2_g
	M_stars_disk_cgs = M_stars_bulge*M_solar_2_g
	R_HI_cgs = R_bulge*Mpc_2_cm

	#M_stars_bulge_cgs = M_stars_bulge*M_solar_2_g
	#R_bulge_cgs = R_bulge*Mpc_2_cm
	
	r_x_cgs = r_x*Mpc_2_cm



	A_cgs = -r_x_cgs/R_HI_cgs

	for i in range(0,nx):

		Gas_galaxy[i] = M_cold_gas_cgs/(2*np.pi*(R_HI_cgs)**2)*np.exp(A_cgs[i])

		
		P_star_galaxy[i] = ((M_stars_disk_cgs)/(2*np.pi*(R_HI_cgs)**2)*np.exp(A_cgs[i]))

		P_gas_galaxy[i] = (M_cold_gas_cgs/(2*np.pi*(R_HI_cgs)**2))*np.exp(A_cgs[i])

		v_star_galaxy[i] = np.sqrt(np.pi*G_cgs*hstar_galaxy*P_star_galaxy[i])


	P_ext_galaxy = np.pi/2*G_cgs*Gas_galaxy
	P_ext_galaxy = P_ext_galaxy*(P_gas_galaxy + (v_g*1e5/v_star_galaxy)*P_star_galaxy)
	P_ext_galaxy = P_ext_galaxy/k_boltzmann

	P_ext = max(P_ext_galaxy)
	
	#P_0 = 3.7*10e4
	P_0  = 34673

	#R_c_galaxy = (P_ext_galaxy/P_0)**0.8

	R_c_galaxy = (P_ext_galaxy/P_0)**0.92


	f_H1_galaxy = 1/(1 + R_c_galaxy)
	f_H2_galaxy = R_c_galaxy/(1 + R_c_galaxy)

	surface_H1_galaxy = f_H1_galaxy*Gas_galaxy
	H1 = np.trapz(r_x_cgs*surface_H1_galaxy, dx=dx*Mpc_2_cm)
	surface_H1_galaxy_bulge = surface_H1_galaxy/H1/cm_2_mpc**2

	surface_H2_galaxy = f_H2_galaxy*Gas_galaxy
	H2 = np.trapz(r_x_cgs*surface_H2_galaxy,dx = dx*Mpc_2_cm) 
	surface_H2_galaxy_bulge = surface_H2_galaxy/H2/cm_2_mpc**2
    
	return surface_H1_galaxy_disk, surface_H2_galaxy_disk, surface_H1_galaxy_bulge, surface_H2_galaxy_bulge, r_x    


# In[21]:


def velocity_profile(R_halo,R_disk,R_HI,R_bulge,M_cold_gas,M_stars_disk,M_halo,M_disk,M_bulge,c_halo,c_disk,c_bulge,c_HI, sini):     # K is the index of galaxy, sini - sine of angle of inclination [theta dataset]
	
	## adresses 
	# if M_halo > 10**10:
	# 	v_g = 10
	# else:	
	# 	v_g = 20

	v_g = 10

	## Making the Radius Channel - -------------------------------------------------------------------------

	
	xmax = 1
	dx = 0.004
	nx = int(xmax/dx)
	x = np.zeros(shape = nx)


	for i in range(1,nx + 1):
		x[i - 1] = (i - 0.5) * dx


	r_x = (R_halo/1.67)*x
	
	R = R_halo/1.67

	if R_HI == 0:
		R_HI = 3*R_disk/1.67	

		c_HI = R_halo/R_HI

        #---------------------------------------------------------------------------------------------------------

	## Making Velocity Profile------------------------------------------------------------------------


	A = r_x/R

	V_bulge_sqr = np.zeros(shape = nx)
	numerator_b = np.zeros(shape = nx)
	denominator_b = np.zeros(shape = nx)

	V_halo_sqr = np.zeros(shape = nx)
	numerator_h = np.zeros(shape = nx)


	V_disk_sqr = np.zeros(shape= nx)
	numerator_d = np.zeros(shape = nx)
	denominator_d = np.zeros(shape = nx)

	V_HI_sqr = np.zeros(shape= nx)
	numerator_g = np.zeros(shape = nx)
	denominator_g = np.zeros(shape = nx)


	
	for i in range(0,nx):

		numerator_b[i] = ((c_bulge*A[i])**2)*(c_bulge)

		denominator_b[i] = (1 + (c_bulge*A[i])**2)**1.5
		V_bulge_sqr[i] = (((G*M_bulge)/R)*(numerator_b[i]/denominator_b[i]))



	

	

		denominator_h = np.log(1 + c_halo) - ((c_halo/(1 + c_halo)))
		numerator_h[i] = np.log(1 + c_halo*A[i]) - ((c_halo*A[i])/(1 + c_halo*A[i]))
		V_halo_sqr[i] = (((G*M_halo)/R)*(numerator_h[i]/(A[i]*denominator_h)))

		



	
		numerator_d[i] = c_disk + 4.8*c_disk*(np.exp((-0.35*c_disk*A[i]) - (3.5/(c_disk*A[i]))))
		denominator_d[i] = (c_disk*A[i]) + (c_disk*A[i])**(-2) + 2*((c_disk*A[i]))**(-0.5)
		V_disk_sqr[i] = (((G*M_disk)/R)*(numerator_d[i]/denominator_d[i]))



	
		numerator_g[i] = c_HI + 4.8*c_HI*(np.exp((-0.35*c_HI*A[i]) - (3.5/(c_HI*A[i]))))
		denominator_g[i] = (c_HI*A[i]) + (c_HI*A[i])**(-2) + 2*((c_HI*A[i]))**(-0.5)
		V_HI_sqr[i] = (((G*M_cold_gas*0.73)/R)*(numerator_g[i]/denominator_g[i]))




	

	V_circ = np.sqrt(V_disk_sqr + V_halo_sqr + V_bulge_sqr + V_HI_sqr)

	

	max_Vcirc = max(V_circ)
	max_Vdisk = max(np.sqrt(V_disk_sqr))
	max_Vhalo = max(np.sqrt(V_halo_sqr))
	max_Vbulge = max(np.sqrt(V_bulge_sqr))
	max_VHI		= max(np.sqrt(V_HI_sqr))

    
	#plt.figure(figsize = (18,12))
	plt.plot(r_x[0:nx-1],np.sqrt(V_halo_sqr[0:nx-1]) , '--r', linewidth = 2, label = '$\\rm V^{Halo}_{Circ}$' )
	plt.plot(r_x[0:nx-1], np.sqrt(V_disk_sqr[0:nx-1]),'--b', linewidth = 2, label = '$\\rm V^{Disk}_{Circ}$' )
	plt.plot(r_x[0:nx-1], np.sqrt(V_bulge_sqr[0:nx-1]), '--g', linewidth = 2, label = '$\\rm V^{Bulge}_{Circ}$')
	plt.plot(r_x[0:nx-1], np.sqrt(V_HI_sqr[0:nx-1]), '--m', linewidth = 2, label =  '$\\rm V^{HI}_{Circ}$')
	plt.plot(r_x[0:nx-1], V_circ[0:nx-1], 'k', linewidth=2, label = '$\\rm V^{Total}_{Circ}$')
	plt.xlabel('$\\rm Radius\  [Kpc]$')
	plt.ylabel('$\\rm Circular\ Velocity\ [km/s]$')
	plt.legend(frameon=False)

	plt.xticks()
	plt.yticks()
	
	#plt.title('Velocity Profile', size=30)
	plt.show()

	





def plot_surface_density():
    
    plt.plot(r_x,np.log10(surface_H1_galaxy_disk),'-b', label='$\\rm \Sigma^{Disk}_{HI}$')
    plt.plot(r_x,np.log10(surface_H1_galaxy_bulge),'-r', label='$\\rm \Sigma^{Bulge}_{HI}$')
    
    plt.plot(r_x,np.log10(surface_H2_galaxy_disk),'--b', label='$\\rm \Sigma^{Disk}_{H_2}$')
    plt.plot(r_x,np.log10(surface_H2_galaxy_bulge),'--r', label='$\\rm \Sigma^{Bulge}_{H_2}$')
    
    Bulge_patch = mpatches.Patch(color='red', label='$\\rm Bulge$')
    Disk_patch = mpatches.Patch(color='blue', label='$\\rm Disk$')

    HI_gal = mlines.Line2D([],[],color='black', linestyle='-', label='$\\rm \Sigma_{HI}$')
    H2_gal = mlines.Line2D([],[],color='black', linestyle='--', label='$\\rm \Sigma_{H_{2}}$')

    plt.legend(handles=[Bulge_patch,Disk_patch,HI_gal,H2_gal], frameon=False)
    plt.xlabel('$\\rm Radius [kpc]$')
    plt.ylabel('$\\rm log_{10}(\Sigma)[(M_{\odot} Mpc^{-2}]$')
    plt.xticks()
    plt.yticks()
    plt.ylim(-5,2)
    plt.show()
    


# In[23]:


path_write 				= 	'/home/garima/Desktop/'

##------------------------------------------------------------------------------------

with np.errstate(divide = 'ignore', invalid = 'ignore'):


	#f = h5.File("/mnt/su3ctm/gchauhan/alfalfa_NH.hdf5",'r')
	f = h5.File("/home/garima/Desktop/mocksky.hdf5",'r')


	snapshot 		= np.array(f['galaxies/snapshot'], dtype = dt)
	subsnapshot		= np.array(f['galaxies/subvolume'], dtype = dt)
	id_galaxy_dan 	= np.array(f['galaxies/id_galaxy_sam'], dtype = dt)
	id_halo_dan 	= np.array(f['galaxies/id_halo_sam'], dtype = dt)
	dec 			= np.array(f['galaxies/dec'], dtype = df )
	ra 				= np.array(f['galaxies/ra'], dtype = df )
	inclination		= np.array(f['galaxies/inclination'], dtype = df )
	#mstars         = np.array(f['galaxies/mstars'], dtype = df)
	mvir_hosthalo	= np.array(f['galaxies/mvir_hosthalo'], dtype = df)
	mvir_subhalo	= np.array(f['galaxies/mvir_subhalo'], dtype = df)
	rgas_disk		= np.array(f['galaxies/rgas_disk_intrinsic'], dtype = df)*10**3
	rgas_bulge      = np.array(f['galaxies/rgas_bulge_intrinsic'], dtype=df)*10**3
	rstar_bulge 	= np.array(f['galaxies/rstar_bulge_intrinsic'], dtype = df)*10**3
	rstar_disk 		= np.array(f['galaxies/rstar_disk_intrinsic'], dtype = df)*10**3
	gal_type	 	= np.array(f['galaxies/type'], dtype = dt)
	zcos 			= np.array(f['galaxies/zcos'], dtype = df)
	zobs 			= np.array(f['galaxies/zobs'], dtype = df)
	distance 		= np.array(f['galaxies/dc'], dtype = df)

	c_halo	 		= np.array(f['galaxies/cnfw_subhalo'], dtype = df)
	matom_bulge 	= np.array(f['galaxies/matom_bulge'], dtype = df)
	matom_disk 		= np.array(f['galaxies/matom_disk'], dtype = df)

	mmol_disk 		= np.array(f['galaxies/mmol_disk'], dtype = df)
	mmol_bulge 		= np.array(f['galaxies/mmol_bulge'], dtype = df)

	mgas_disk 		= np.array(f['galaxies/mgas_disk'], dtype = df)
	mgas_bulge 		= np.array(f['galaxies/msgas_bulge'], dtype = df)

	mstars_bulge 	= np.array(f['galaxies/mstars_bulge'], dtype = df)	
	mstars_disk 	= np.array(f['galaxies/mstars_disk'], dtype = df)
	mstars          = mstars_bulge + mstars_disk
	vmax_halo 		= np.array(f['galaxies/vmax_subhalo'], dtype = df)
	vvir_hosthalo   = np.array(f['galaxies/vvir_hosthalo'],dtype = df)
	vvir_subhalo	= np.array(f['galaxies/vvir_subhalo'],dtype=df)


	print('Read the mock-sky File for ALFALFA')



# In[24]:


#R_S_halo = vmax_halo*10**3/10/67.51
R_S_halo                       =        G*mvir_hosthalo/vvir_hosthalo**2


# In[25]:


c_disk 	= np.zeros(len(rstar_disk))

c_bulge = np.zeros(len(rstar_bulge))
c_HI    = np.zeros(len(rgas_disk))


for i in range(len(rgas_disk)):
	if rgas_disk[i] == 0:
		c_HI[i] = 0
	else:
		c_HI[i] 	= 	R_S_halo[i]/(rgas_disk[i]/1.67)

	if rstar_disk[i] == 0:
		c_disk[i] = 0
	else:
		c_disk[i]	=	R_S_halo[i]/(rstar_disk[i]/1.67)

	if rstar_bulge[i] == 0:
		c_bulge[i] = 0
	else:
		c_bulge[i]	=	R_S_halo[i]/(1.7*rstar_bulge[i]/1.67)





M_bulge 			= 	mstars_bulge + mgas_bulge  					# Total mass of the bulge

M_disk 				= 	mstars_disk + mgas_disk						# Total mass of the disk	

mass_gas 			= 	mgas_disk/mstars

mass_galaxies 		= 	np.log10(mstars)

B_T 				=	mstars_bulge/mstars

theta 				= np.sin(inclination*np.pi/180)



r_bulge_not_zero = np.where(rgas_bulge != 0)[0]


for i in range(0,20):

	# if rgas_bulge[i] == 0:
	# 	continue
	# else:

	    # surface_H1_galaxy_disk, surface_H2_galaxy_disk, surface_H1_galaxy_bulge, surface_H2_galaxy_bulge, r_x  =surface_density(R_S_halo[i],rstar_disk[i],rgas_disk[i],rstar_bulge[i],mgas_disk[i],mstars_disk[i],mvir_subhalo[i],mstars_disk[i]+mgas_disk[i],mstars_bulge[i]+mgas_bulge[i],c_halo[i],c_disk[i],c_bulge[i],c_HI[i],theta[i],mgas_bulge[i],mstars_bulge[i])
	    
	    # plot_surface_density()
	    
	velocity_profile(R_S_halo[i],rstar_disk[i],rgas_disk[i],rstar_bulge[i],mgas_disk[i],mstars_disk[i],mvir_subhalo[i],mstars_disk[i]+mgas_disk[i],mstars_bulge[i]+mgas_bulge[i],c_halo[i],c_disk[i],c_bulge[i],c_HI[i],theta[i])
		 # i = 12
    
# surface_H1_galaxy_disk, surface_H2_galaxy_disk, surface_H1_galaxy_bulge, surface_H2_galaxy_bulge, r_x  =surface_density(R_S_halo[i],rstar_disk[i],rgas_disk[i],rstar_bulge[i],mgas_disk[i],mstars_disk[i],mvir_subhalo[i],mstars_disk[i]+mgas_disk[i],mstars_bulge[i]+mgas_bulge[i],c_halo[i],c_disk[i],c_bulge[i],c_HI[i],theta[i],mgas_bulge[i],mstars_bulge[i])

# plot_surface_density()
    


# In[88]:





# In[28]:


def skymap():
	theta = np.radians(ra[(matom_disk != 0) & (matom_disk > 10**6) ])
	radius = zcos[(matom_disk != 0) & (matom_disk > 10**6) ]

	# theta = np.random.uniform(low=0.0,high=60.0,size=(500,))
	# radius = np.random.uniform(low=0.0,high=1.0,size=(500,))

	thetaLims = (0.0,np.radians(60.0))

	zLims = (0.0,0.07)

	fig = plt.figure(figsize=(24,20))

	tr = PolarAxes.PolarTransform() 
	grid_helper = floating_axes.GridHelperCurveLinear(tr, extremes=(*thetaLims, *zLims))

	ax = floating_axes.FloatingSubplot(fig, 111, grid_helper=grid_helper)
	fig.add_subplot(ax)

	# adjust axis
	ax.axis["left"].set_axis_direction("bottom")
	ax.axis["right"].set_axis_direction("top")
	ax.axis["bottom"].set_visible(False)
	ax.axis["top"].set_axis_direction("bottom")
	ax.axis["top"].toggle(ticklabels=True, label=True)
	ax.axis["top"].major_ticklabels.set_axis_direction("top")
	ax.axis["top"].label.set_axis_direction("top")
	ax.axis["left"].label.set_text(r"z")
	ax.axis["top"].label.set_text(r"RA")

	# create a parasite axes whose transData in RA, cz

	aux_ax = ax.get_aux_axes(tr)
	aux_ax.patch = ax.patch  
	ax.patch.zorder = 0.9  

	scatter_yo = aux_ax.scatter(theta, radius, s= 0.5, c = np.log10(matom_disk[(matom_disk != 0) & (matom_disk > 10**6) ]), cmap='plasma')
	aux_ax.grid(True)
	aux_ax.set_axisbelow(True)
	cbar = plt.colorbar(scatter_yo, orientation='horizontal', shrink=0.7)
	plt.title('Lightcone - cosmological z - All ')
	cbar.ax.set_title('$log_{10}[M_{HI}$ / ($M_{\odot}$)]', size=14)
	plt.show()


# In[105]:


#skymap()


# In[103]:


max(matom_disk)

