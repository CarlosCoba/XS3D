import numpy as np
import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
from matplotlib import gridspec
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
from itertools import product


from matplotlib.offsetbox import AnchoredText
from src.axes_params import axes_ambient as axs 
from src.cbar import colorbar as cb
from src.colormaps_CLC import vel_map


#params =   {'text.usetex' : True }
#plt.rcParams.update(params)

def vmin_vmax(data,pmin=5,pmax=98,base=None):
	vmin,vmax=np.nanpercentile(np.unique(data),pmin),np.nanpercentile(np.unique(data),pmax)
	if base is not None:
		vmin,vmax=(vmin//base+1)*base,(vmax//base+1)*base
	return vmin,vmax

def zero2nan(data):	
	data[(data==0) & (~np.isfinite(data))]=np.nan
	return data

cmap = vel_map()

def plot_rmommaps(galaxy,momms_mdls,momms_obs,vsys,ext,vmode,hdr,out):
	mom0_mdl,mom1_mdl,mom2_mdl_kms,mom2_mdl_A,cube_mdl,velmap_intr,sigmap_intr,twoDmodels= momms_mdls
	mom0,mom1,mom2=momms_obs
	mom0,mom1,mom2=zero2nan(mom0),zero2nan(mom1),zero2nan(mom2)
	mom0_mdl,mom1_mdl,mom2_mdl=zero2nan(mom0_mdl),zero2nan(mom1_mdl),zero2nan(mom2_mdl_kms)

	
	mom1_mdl=mom1_mdl-vsys
	mom1=mom1-vsys


	width, height = 12, 5.5 # width [cm]
	#width, height = 3, 15 # width [cm]
	cm_to_inch = 0.393701 # [inch/cm]
	figWidth = width * cm_to_inch # width [inch]
	figHeight = height * cm_to_inch # width [inch]
  
	fig = plt.figure(figsize=(figWidth, figHeight), dpi = 300)
	#nrows x ncols
	widths = [1,1,1]
	heights = [1]
	gs2 = gridspec.GridSpec(1, 3)
	gs2.update(left=0.07, right=0.99,top=0.8,bottom=0.13, hspace = 0.03, wspace = 0)


	ax0=plt.subplot(gs2[0,0])
	ax1=plt.subplot(gs2[0,1])
	ax2=plt.subplot(gs2[0,2])		
	axes=[ax0,ax1,ax2]


	# moment zero maps:
	vmin,vmax=vmin_vmax(mom0)
	im0=ax0.imshow(mom0,origin='lower',cmap=cmap,extent=ext,vmin=vmin,vmax=vmax,aspect='auto')


	#moment 1 maps	
	vmin = abs(np.nanmin(mom1))
	vmax = abs(np.nanmax(mom1))
	vmin,vmax=vmin_vmax(mom1)

	max_vel = np.nanmax([vmin,vmax])	
	vmin = -(max_vel//50 + 1)*50
	vmax = (max_vel//50 + 1)*50
	im1=ax1.imshow(mom1,origin='lower',cmap=cmap,extent=ext,vmin=-200,vmax=200,aspect='auto')

	#moment 2 maps

	vmin,vmax=vmin_vmax(mom2,1,99,base=10)
	im2=ax2.imshow(mom2,origin='lower',cmap=cmap,extent=ext,vmin=vmin,vmax=vmax,aspect='auto')


	# If you dont want to add these lines ontop of the 2D plots, then comment the following.
	# it seems that  calling astropy.convolution causes that some package is downloaded each time this is run, why ?
	"""
	from astropy.convolution import Gaussian2DKernel
	from astropy.convolution import convolve
	import matplotlib
	matplotlib.rcParams['contour.negative_linestyle']= 'solid'

	kernel = Gaussian2DKernel(x_stddev=4)
	vloss = convolve(vel_ha, kernel)
	z = np.ma.masked_array(vloss.astype(int),mask= np.isnan(vel_ha) )
	N = vmax // 50
	ax.contour(z, levels = [i*50 for i in np.arange(-N,N+1)], colors = "k", alpha = 1, zorder = 1e3, extent = ext, linewidths = 0.6)
	ax1.contour(z, levels = [i*50 for i in np.arange(-N,N+1)], colors = "k", alpha = 1, zorder = 1e3, extent = ext, linewidths = 0.6)
	#ax2.contour(z, levels = [i*50 for i in np.arange(-N,N+1)], colors = "k", alpha = 1, zorder = 1e3, extent = ext, linewidths = 0.6)
	#if np.nanmean(Vtan) < 0 :
	#	Vtan = -Vtan
	#	Vrad = -Vrad
	"""

	#axs(ax,tickscolor = "k")
	
	for k in axes: axs(k)
	for k in range(3): axes[k].set_xlabel('$\mathrm{ \Delta RA~(arcsec)}$',fontsize=10,labelpad=0)
	ax0.set_ylabel('$\mathrm{ \Delta Dec~(arcsec)}$',fontsize=10,labelpad=0)

	axs(ax1,remove_yticks= True)
	axs(ax2,remove_yticks= True)
		
	txt = AnchoredText('$\mathrm{mom0_{res}}$', loc="upper left", pad=0.1, borderpad=0, prop={"fontsize":11},zorder=1e4);ax0.add_artist(txt)
	txt = AnchoredText('$\mathrm{mom1_{res}}$', loc="upper left", pad=0.1, borderpad=0, prop={"fontsize":11},zorder=1e4);ax1.add_artist(txt)
	txt = AnchoredText('$\mathrm{mom2_{res}}$', loc="upper left", pad=0.1, borderpad=0, prop={"fontsize":11},zorder=1e4);ax2.add_artist(txt)	
				

	#try:
	#	spec_axis_units=hdr['CUNIT3']
	#	spec_u=spec_axis_units.lower()
	#except(KeyError):
	#	spec_u='lambda'
	
	spec_u = 'km/s'			
	cb(im0,ax0,orientation = "horizontal", colormap = cmap, bbox= (0,1.1,0.5,1),width = "100%", height = "5%",label_pad = -24, label = "$\mathrm{flux*%s}$"%(spec_u),labelsize=10, ticksfontsize=8)

	cb(im1,ax1,orientation = "horizontal", colormap = cmap, bbox= (0,1.1,0.5,1),width = "100%", height = "5%",label_pad = -24, label = "$\mathrm{km/s}$",labelsize=10, ticksfontsize=8)
	cb(im2,ax2,orientation = "horizontal", colormap = cmap, bbox= (0,1.1,0.5,1),width = "100%", height = "5%",label_pad = -24, label = "$\mathrm{km/s}$",labelsize=10, ticksfontsize = 8)


	plt.savefig("%sfigures/res_mommaps_%s_model_%s.png"%(out,vmode,galaxy))
	plt.clf()




