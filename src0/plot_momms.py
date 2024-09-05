import numpy as np
import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
from matplotlib import gridspec
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from itertools import product


from matplotlib.offsetbox import AnchoredText
from src0.axes_params import axes_ambient as axs 
from src0.cbar import colorbar as cb
from src0.colormaps_CLC import vel_map
from src0.barscale import bscale
from src0.ellipse import drawellipse

#params =   {'text.usetex' : True }
#plt.rcParams.update(params)

def vmin_vmax(data,pmin=5,pmax=99,base=None):
	vmin,vmax=np.nanpercentile(np.unique(data),pmin),np.nanpercentile(np.unique(data),pmax)
	if base is not None:
		vmin,vmax=(vmin//base+1)*base,(vmax//base+1)*base
	return vmin,vmax

def zero2nan(data):
	#data[data==0]=np.nan
	return data

cmap = vel_map()
cmap_mom0 = vel_map('mom0')

def plot_mommaps(galaxy,momms_mdls,momms_obs,vsys,ext,vmode,hdr,config,pixel,out):
	mom0_mdl,mom1_mdl,mom2_mdl_kms,mom2_mdl_A,cube_mdl,velmap_intr,sigmap_intr= momms_mdls
	mom0,mom1,mom2=momms_obs
	mom0,mom1,mom2=zero2nan(mom0),zero2nan(mom1),zero2nan(mom2)
	mom0_mdl,mom1_mdl,mom2_mdl=zero2nan(mom0_mdl),zero2nan(mom1_mdl),zero2nan(mom2_mdl_kms)

	
	mom1_mdl=mom1_mdl-vsys
	mom1=mom1-vsys
	
	rnorm=1
	if np.max(ext)>80:
		rnorm=60
		rlabel='$\'$'
	else:
		rlabel='$\'\'$'	
	ext = ext/rnorm
				
	width, height = 10, 13 # width [cm]
	#width, height = 3, 15 # width [cm]
	cm_to_inch = 0.393701 # [inch/cm]
	figWidth = width * cm_to_inch # width [inch]
	figHeight = height * cm_to_inch # width [inch]
  
	fig = plt.figure(figsize=(figWidth, figHeight), dpi = 300)
	#nrows x ncols
	widths = [1,1,1]
	heights = [1]
	gs2 = gridspec.GridSpec(3, 3)
	gs2.update(left=0.08, right=0.97,top=0.93,bottom=0.06, hspace = 0.3, wspace = 0)



	axes=[plt.subplot(gs2[j,i]) for j,i in product(np.arange(3),np.arange(3))]
	[ax0,ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8]=axes


	# moment zero maps:
	res_mom0=mom0-mom0_mdl
	vmin,vmax=vmin_vmax(mom0_mdl)
	ax0.imshow(mom0,origin='lower',cmap=cmap_mom0,extent=ext,vmin=vmin,vmax=vmax,aspect='auto')
	im1=ax1.imshow(mom0_mdl,origin='lower',cmap=cmap_mom0,extent=ext,vmin=vmin,vmax=vmax,aspect='auto')	
	vmin,vmax=vmin_vmax(res_mom0,2,98)
	mean=(abs(vmax)+abs(vmin))/2
	im2=ax2.imshow(res_mom0,origin='lower',cmap=cmap_mom0,extent=ext,vmin=-mean,vmax=mean,aspect='auto')

	#moment 1 maps	
	res_mom1=mom1-mom1_mdl
	vmin = abs(np.nanmin(mom1_mdl))
	vmax = abs(np.nanmax(mom1_mdl))
	vmin,vmax=vmin_vmax(mom1_mdl)
	max_vel = np.nanmax([vmin,vmax])	
	vmin = -(max_vel//50 + 1)*50
	vmax = (max_vel//50 + 1)*50

	ax3.imshow(mom1,origin='lower',cmap=cmap,extent=ext,vmin=vmin,vmax=vmax,aspect='auto')
	im4=ax4.imshow(mom1_mdl,origin='lower',cmap=cmap,extent=ext,vmin=vmin,vmax=vmax,aspect='auto')	
	vmin,vmax=vmin_vmax(res_mom1)
	im5=ax5.imshow(res_mom1,origin='lower',cmap=cmap,extent=ext,vmin=-50,vmax=50,aspect='auto')

	#moment 2 maps
	res_mom2=mom2-mom2_mdl

	vmin,vmax=vmin_vmax(mom2_mdl,1,99,base=10)
	ax6.imshow(mom2,origin='lower',cmap=cmap,extent=ext,vmin=vmin,vmax=vmax,aspect='auto')
	im7=ax7.imshow(mom2_mdl,origin='lower',cmap=cmap,extent=ext,vmin=vmin,vmax=vmax,aspect='auto')
	#vmin,vmax=vmin_vmax(res_mom2, base=10)
	vmin,vmax=vmin_vmax(res_mom2,2,98)
	mean=(abs(vmax)+abs(vmin))//2		
	im8=ax8.imshow(res_mom2,origin='lower',cmap=cmap,extent=ext,vmin=-mean,vmax=mean,aspect='auto')



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
	for k in range(9):
		if k not in [0,3,6]:
			axs(axes[k],remove_yticks= True)
		if k not in [6,7,8]:
				axes[k].xaxis.set_major_formatter(plt.NullFormatter())
				
		#	axs(axes[k],remove_xticks= True)	


	for k in range(-3,0,1): axes[k].set_xlabel('$\mathrm{ \Delta RA }$ (%s)'%rlabel,fontsize=10,labelpad=1)
	for k in range(1,10,3): axes[k-1].set_ylabel('$\mathrm{ \Delta Dec}$ (%s)'%rlabel,fontsize=10,labelpad=1)


	txt0=['mom0','mom1', 'mom2']
	txt1=['obs','mdl']

		

	k=1
	for j,i in product(np.arange(3),np.arange(2)):
		txt = AnchoredText('$\mathrm{%s_{%s}}$'%(txt0[j],txt1[i]), loc="upper left", pad=0.1, borderpad=0, prop={"fontsize":10},zorder=1e4);axes[k-1].add_artist(txt)
		k+=1
		if not k % 3:
			txt = AnchoredText('$\mathrm{residual}$', loc="upper left", pad=0.1, borderpad=0, prop={"fontsize":10},zorder=1e4);axes[k-1].add_artist(txt)		
			k+=1



	#try:
	#	spec_axis_units=hdr['CUNIT3']
	#	spec_u=spec_axis_units.lower()
	#except(KeyError):
	#	spec_u='lambda'
	
	spec_u = 'km/s'			
	cb(im1,ax0,orientation = "horizontal", colormap = cmap, bbox= (0.5,1.12,1,1),width = "100%", height = "5%",label_pad = -21, label = "$\mathrm{flux*%s}$"%(spec_u),labelsize=10, ticksfontsize=8)
	cb2=cb(im2,ax2,orientation = "horizontal", colormap = cmap, bbox= (0.1,1.12,0.8,1),width = "100%", height = "5%",label_pad = -21, label = "$\mathrm{flux*%s}$"%(spec_u),labelsize=10, ticksfontsize = 8,power=True)


	cb(im4,ax3,orientation = "horizontal", colormap = cmap, bbox= (0.5,1.12,1,1),width = "100%", height = "5%",label_pad = -21, label = "$\mathrm{km/s}$",labelsize=10, ticksfontsize=8)
	cb(im5,ax5,orientation = "horizontal", colormap = cmap, bbox= (0.1,1.12,0.8,1),width = "100%", height = "5%",label_pad = -21, label = "$\mathrm{km/s}$",labelsize=10, ticksfontsize = 8)

	cb(im7,ax6,orientation = "horizontal", colormap = cmap, bbox= (0.5,1.12,1,1),width = "100%", height = "5%",label_pad = -21, label = "$\mathrm{km/s}$",labelsize=10, ticksfontsize=8)
	cb(im8,ax8,orientation = "horizontal", colormap = cmap, bbox= (0.1,1.12,0.8,1),width = "100%", height = "5%",label_pad = -21, label = "$\mathrm{km/s}$",labelsize=10, ticksfontsize = 8)
	
	
	[ny,nx]=mom0.shape
	bar_scale_arc,bar_scale_u,unit=bscale(vsys,nx,pixel)
	bar_scale_arc_norm=bar_scale_arc/rnorm
	
	ax0.text(ext[0]*(4/5.+1/10),ext[2]*(5/6)*0.95, '%s${\'\'}$:%s%s'%(bar_scale_arc,bar_scale_u,unit),fontsize=8)	
	ax0.plot([ext[0]*(4/5.),ext[0]*(4/5.)+bar_scale_arc_norm],[ext[2]*(5/6),ext[2]*(5/6)],'k-')

	ax3.text(ext[0]*(4/5.+1/10),ext[2]*(5/6)*0.95, '%s${\'\'}$:%s%s'%(bar_scale_arc,bar_scale_u,unit),fontsize=8)	
	ax3.plot([ext[0]*(4/5.),ext[0]*(4/5.)+bar_scale_arc_norm],[ext[2]*(5/6),ext[2]*(5/6)],'k-')

	ax6.text(ext[0]*(4/5.+1/10),ext[2]*(5/6)*0.95, '%s${\'\'}$:%s%s'%(bar_scale_arc,bar_scale_u,unit),fontsize=8)	
	ax6.plot([ext[0]*(4/5.),ext[0]*(4/5.)+bar_scale_arc_norm],[ext[2]*(5/6),ext[2]*(5/6)],'k-')


	# plot PSF ellipse ?
	config_general = config['general']	
	psf_arc=config_general.getfloat('psf_fwhm',None)
	bmaj_arc=config_general.getfloat('bmaj',None)
	bmin_arc=config_general.getfloat('bmin',None)
	bpa=config_general.getfloat('bpa',0)
				
	psf=None
	bmaj,bmin=None,None
	if psf_arc is not None:
		psf=psf_arc/rnorm
	if bmaj_arc is not None:		
		bmaj=bmaj_arc/rnorm
	if bmin_arc is not None:		
		bmin=bmin_arc/rnorm
			
	if psf is not None:
		x0,y0=ext[1]*(5/6.),ext[2]*(5/6)	
		x,y=drawellipse(x0,y0,psf/2.,0,bminor=psf/2.)
		ax0.plot(x,y,'g-',lw=0.5)
		ax3.plot(x,y,'g-',lw=0.5)
		ax6.plot(x,y,'g-',lw=0.5)
	if bmaj_arc is not None and bmin_arc is not None:
		x0,y0=ext[1]*(5/6.),ext[2]*(5/6)	
		x,y=drawellipse(x0,y0,bmaj,bpa,bminor=bmin)	
		ax0.plot(x,y,'g-',lw=0.5)
		ax3.plot(x,y,'g-',lw=0.5)
		ax6.plot(x,y,'g-',lw=0.5)						
			
	plt.savefig("%sfigures/mommaps_%s_model_%s.png"%(out,vmode,galaxy))
	#plt.clf()
	plt.close()




