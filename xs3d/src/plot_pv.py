import numpy as np
import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
from matplotlib import gridspec
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
from mpl_toolkits.axes_grid1.anchored_artists import (AnchoredEllipse,AnchoredSizeBar)
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
from itertools import product
import matplotlib.colors as colors
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from matplotlib.transforms import Affine2D
from matplotlib.offsetbox import AnchoredText

from .axes_params import axes_ambient as axs
from .cbar import colorbar as cb
from .colormaps_CLC import vel_map
from .barscale import bscale
from .constants import __c__
from .ellipse import drawellipse
from .conv import conv2d,gkernel,gkernel1d
from .conv_fftw import fftconv,data_2N,fftconv_numpy
from .psf_lsf import PsF_LsF
from .utils import vmin_vmax
from .pixel_params import eps_2_inc,e_eps2e_inc,inc_2_eps
cmap=vel_map()


#params =   {'text.usetex' : True }
#plt.rcParams.update(params)



def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

arr = np.linspace(0, 50, 100).reshape((10, 10))

cmap = plt.get_cmap('magma_r')
new_cmap = truncate_colormap(cmap, 0, 0.6)
cmap = vel_map()
cmap_pvd = vel_map('pvd')

def plot_pvd(galaxy,out_pvd,best,const,vmode,rms,moms_mod,moms_obs,datacube,hdr_info,psf_lsf,config,out):

	R=best['radius']
	nrings=len(R)	
	[v_sys,inc,pa,x_center,y_center,phi_bar,rmax]=const['v_sys'],const['inc'],const['pa'],const['x_center'],const['y_center'],const['phi_bar'],const['rmax']
	eps = inc_2_eps(inc)
	scalar_fields = ["v_rot"]
	vels = {k:best[k] for k in scalar_fields}
	vrot = vels['v_rot']		
	

	mom0_obs,mom1_obs,mom2_obs=moms_obs
	mom0_mod,mom1_mod,mom2_mod=moms_mod	

	pixel 	= hdr_info.scale	
	pvds,ext =out_pvd

	[ext0,ext1,_] = ext


	ext0[2]=ext0[2]-v_sys
	ext1[2]=ext1[2]-v_sys
	ext0[3]=ext0[3]-v_sys
	ext1[3]=ext1[3]-v_sys

	pvd_maj,pvd_min,pvd_maj_mod,pvd_min_mod=pvds[0],pvds[1],pvds[2],pvds[3]

	# lets remove negative fluxes
	pvd_maj[pvd_maj<0]=0
	pvd_min[pvd_min<0]=0

	msk=np.isfinite(mom0_obs*mom0_mod/mom0_obs)

	pa_maj = pa % 360
	pa_min = (pa+90) % 360
	pa_maj = int(round(pa_maj))
	pa_min = int(round(pa_min))
	[ny,nx]=mom0_obs.shape
	extimg=np.dot([-x_center,nx-x_center,-y_center,ny-y_center],pixel); xc =0; yc =0

	vrot=vrot*np.sin(inc*np.pi/180)
	max_vrot=np.nanmax(vrot)

	loc_txt_pv='upper left'
	rnorm=1
	if rmax > 80 and np.all(abs(extimg)>80):
		rnorm=60
		rlabel='$\'$'
	else:
		rlabel='$\'\'$'
	R=R/rnorm
	extimg=extimg/rnorm
	ext0[:2]=ext0[:2]/rnorm
	ext1[:2]=ext1[:2]/rnorm

	slit_w 		= psf_lsf.slit_w
	slit_w		= slit_w/rnorm
	
	
	mom1_mod-=v_sys
	mom1_obs-=v_sys
	vmin = abs(np.nanmin(mom1_mod[msk]))
	vmax = abs(np.nanmax(mom1_mod[msk]))
	max_vel = np.nanmax([vmin,vmax])
	vminv = -(max_vel//10 + 1)*10
	vmaxv = (max_vel//10 + 1)*10


	width, height = 18, 7 # width [cm]
	cm_to_inch = 0.393701 # [inch/cm]
	figWidth = width * cm_to_inch # width [inch]
	figHeight = height * cm_to_inch # width [inch]

	levels=2**np.arange(0,7,1,dtype=float)
	levelso=levels
	#"""
	pixelconv=1
	bmajconv=bminconv=1
	psf2d=gkernel(pvd_maj_mod.shape,fwhm=None,bmaj=bmajconv,bmin=bminconv,pixel_scale=pixelconv)
	padded_pvd_maj, cube_slices = data_2N(pvd_maj_mod, axes=[0, 1])
	padded_pvd_min, _ = data_2N(pvd_min_mod, axes=[0, 1])
	padded_psf, _ = data_2N(psf2d, axes=[0, 1])

	dft=fftconv_numpy(padded_pvd_maj,padded_psf,threads=2,axes = [0,1])
	pvd_maj_mod=dft.conv_DFT(cube_slices)
	dft=fftconv_numpy(padded_pvd_min,padded_psf,threads=2,axes = [0,1])
	pvd_min_mod=dft.conv_DFT(cube_slices)

	padded_pvd_maj, cube_slices = data_2N(pvd_maj, axes=[0, 1])
	padded_pvd_min, _ = data_2N(pvd_min, axes=[0, 1])
	dft=fftconv_numpy(padded_pvd_maj,padded_psf,threads=2,axes = [0,1])
	pvd_maj=dft.conv_DFT(cube_slices)
	dft=fftconv_numpy(padded_pvd_min,padded_psf,threads=2,axes = [0,1])
	pvd_min=dft.conv_DFT(cube_slices)


	# normalize by the rms
	pvd_min_mod/=rms
	pvd_maj_mod/=rms
	pvd_min/=rms
	pvd_maj/=rms

	#"""

	# Crop the FOV in case the object is too small
	xlength=nx # in pixels
	rmax_norm=rmax/rnorm
	xmin,xmax=extimg[0],extimg[1]
	ymin,ymax=extimg[2],extimg[3]
	if np.all(abs(extimg[:2])>rmax_norm):
		xmin,xmax=-rmax_norm*(4/3.),rmax_norm*(4/3.)
		xlength=2*xmax*rnorm/pixel
	if np.all(abs(extimg[-2:])>rmax_norm):
		ymin,ymax=-rmax_norm*(4/3.),rmax_norm*(4/3.)

	# bar scale
	highz=config['high_z']
	redshift=highz.getfloat('redshift',0)
	v_sysz=v_sys + redshift*__c__
	bar_scale_arc,bar_scale_u,unit=bscale(v_sysz,xlength,pixel,config)
	bar_scale_arc_norm=bar_scale_arc/rnorm


	# plot 2 pv versions:
	# one for publication
	# other for visualization
	for nplots in range(2):
		if nplots==0:
			figWidth0 = 19 * cm_to_inch # width [inch]
			figHeight0 = 19*(4/10.) * cm_to_inch # width [inch]
			fig = plt.figure(figsize=(figWidth0, figHeight0), dpi = 300)
			widths = [0.3,0.3,0.7,1,1]
			heights = [1,1,0.4,1,1]
			gs2 = fig.add_gridspec(nrows=1, ncols=2, left=0.08, right=0.99, wspace=0.25, bottom=0.13, top = 0.94)
		else:
			fig = plt.figure(figsize=(figWidth, figHeight), dpi = 300)
			widths = [0.3,0.3,0.7,1,1]
			heights = [1,1,0.4,1,1]
			gs1 = fig.add_gridspec(nrows=2, ncols=1, left=0.08, right=0.2, top=0.9, bottom=0.1, wspace=0.2)
			gs2 = fig.add_gridspec(nrows=1, ncols=2, left=0.3, right=0.99, wspace=0.2, bottom=0.1)

		#
		#define color of lines
		clines = '#279dc5'
		dashline = (5, (10, 3))

		# PVD major
		vmin,vmax=vmin_vmax(pvd_maj,pmax=99.8)
		if vmin<0: vmin = 0
		#norm = colors.LogNorm(vmin=vmin, vmax=vmax)
		# gamma=0.5 is equivalent to a square root normalization
		norm = colors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
		ax0=plt.subplot(gs2[0,0])
		axs(ax0,rotation='horizontal',fontsize_ticklabels=10)
		txt = AnchoredText('$\mathrm{PV_{MAJ}}$', loc=loc_txt_pv, pad=0.1, borderpad=0, prop={"fontsize":10},zorder=1e4);txt.patch.set_alpha(1);ax0.add_artist(txt)
		txt = AnchoredText(f'PA {pa_maj}$^\circ$', loc="lower right", pad=0.1, borderpad=0, prop={"fontsize":10},zorder=1e4,bbox_to_anchor=(1., 1.), bbox_transform=ax0.transAxes);txt.patch.set_alpha(0);ax0.add_artist(txt)
		ax0.imshow(pvd_maj,norm=norm,cmap=cmap_pvd,origin = "lower",extent=ext0,aspect='auto',alpha=0.7)#,vmin=vmin,vmax=vmax)
		cnt=ax0.contour(pvd_maj,levels=levelso,colors='k', linestyles='solid',zorder=10,extent=ext0,linewidths=1,alpha=1)
		cnt=ax0.contour(pvd_maj_mod,levels=levels,colors=clines, linestyles='solid',zorder=10,extent=ext0,linewidths=1,alpha=1)

		ax0.scatter(R,vrot,s=25,marker='X',c='#ffb703',edgecolor='k',lw=0.3,zorder=20)
		ax0.scatter(-R,-vrot,s=25,marker='X',c='#ffb703',edgecolor='k',lw=0.3,zorder=20)
		ax0.plot((ext0[0],ext0[1]),(0,0),color='black',linestyle=dashline,lw=0.5,zorder=10)
		ax0.plot((0,0),(ext0[2],ext0[3]),color='black',linestyle=dashline,lw=0.5,zorder=10)
		ax0.set_ylabel('$\mathrm{V_{LOS}~(km\,s^{-1})}$',fontsize=10,labelpad=0)
		ax0.set_xlabel(f'Offset ({rlabel})',fontsize=10,labelpad=1)

		# PVD minor
		ax1=plt.subplot(gs2[0,1])
		axs(ax1,rotation='horizontal',fontsize_ticklabels=10)
		txt = AnchoredText('$\mathrm{PV_{MIN}}$', loc=loc_txt_pv, pad=0.1, borderpad=0, prop={"fontsize":10},zorder=1e4);txt.patch.set_alpha(1);ax1.add_artist(txt)
		txt = AnchoredText(f'PA {pa_min}$^\circ$', loc="lower right", pad=0.1, borderpad=0, prop={"fontsize":10},zorder=1e4,bbox_to_anchor=(1., 1.), bbox_transform=ax1.transAxes);txt.patch.set_alpha(0);ax1.add_artist(txt)
		ax1.imshow(pvd_min,norm=norm,cmap=cmap_pvd,origin='lower',extent=ext1,aspect='auto',alpha=0.7)#,vmin=vmin,vmax=vmax)
		ax1.contour(pvd_min,levels=levelso,colors='k', linestyles='solid',zorder=10,extent=ext1,linewidths=1,alpha=1)
		ax1.contour(pvd_min_mod,levels=levels,colors=clines, linestyles='solid',zorder=10,extent=ext1,linewidths=1,alpha=1)
		ax1.plot((ext1[0],ext1[1]),(0,0),color='black',linestyle=dashline,lw=0.5,zorder=10)
		ax1.plot((0,0),(ext1[2],ext1[3]),color='black',linestyle=dashline,lw=0.5,zorder=10)
		ax1.set_xlabel(f'Offset ({rlabel})',fontsize=10,labelpad=1)

		if nplots==0:
			ax1.set_ylabel('$\mathrm{V_{LOS}~(km\,s^{-1})}$',fontsize=10,labelpad=0)
		else:
			lines = [Line2D([0], [0], color='k',lw=0.8), Line2D([0], [0], color='#279dc5',lw=0.8)];labels=['data','model']
			ax1.legend(lines,labels,loc='upper left',ncol=2,borderaxespad=0,handlelength=0.6,handletextpad=0.5,frameon=False, columnspacing=0.5,fontsize=10,bbox_to_anchor=(
0, 1.11), bbox_transform=ax1.transAxes)
			ax0.legend(lines,labels,loc='upper left',ncol=2,borderaxespad=0,handlelength=0.6,handletextpad=0.5,frameon=False, columnspacing=0.5,fontsize=10,bbox_to_anchor=(
0, 1.11), bbox_transform=ax0.transAxes)

		# plot PSF ellipse ?
		config_general = config['general']
		eline=config_general.getfloat('eline')
		specres=config_general.getfloat('fwhm_inst',None)

		bmaj_arc=psf_lsf.bmaj
		bmin_arc=psf_lsf.bmin
		bpa= psf_lsf.bpa
		fwhm_kms=psf_lsf.fwhm_inst_kms
		if fwhm_kms is None:
			fwhm_kms=hdr_info.cdelt3_kms

		bmin=bmin_arc/rnorm
		bmaj=bmaj_arc/rnorm	

		for Axes in [ax0, ax1]:
			beam=AnchoredEllipse(Axes.transData, width=bmaj, height=fwhm_kms, angle=0, loc='lower left', pad=0.2, borderpad=0, frameon=False, zorder=30)
			beam.ellipse.set(edgecolor='blue', facecolor='none', hatch=5*'.')
			Axes.add_artist(beam)				

		#for Axes in [ax0, ax1]:
		#	if np.any( abs(np.array([ext0[2],ext0[3]]))  > max_vrot*(4/3.) ):
		#		vmin,vmax= -(max_vrot+2*fwhm_kms), (max_vrot+2*fwhm_kms)
		#		Axes.set_ylim(vmin,vmax)


		if nplots==0:
			fig.tight_layout()
			plt.savefig("%sfigures/pvds_%s_model_%s.png"%(out,vmode,galaxy))
			plt.close()







	ax2=plt.subplot(gs1[0,0])
	broadband=np.nansum(datacube,axis=0)
	broadband[broadband<=0]=np.nan
	vmin,vmax=vmin_vmax(broadband)
	norm = colors.LogNorm(vmin=vmin, vmax=vmax)	 if (vmin>0) & (np.log10(vmax/vmin)>1)  else colors.Normalize(vmin=vmin, vmax=vmax)
	im2=ax2.imshow(broadband,norm=norm,cmap=cmap_pvd,aspect='auto',origin='lower',extent=extimg)
	axs(ax2,rotation='horizontal',remove_xyticks=True)
	clb=cb(im2, ax2, labelsize=10, colormap = cmap_pvd, bbox=(-0.25, 0.2, 0.05, 0.7), ticksfontsize=0, ticks = [], label = "flux", label_pad = -20, colors  = "k",orientation='vertical')
	
	v_min=round(vmin,1)
	v_max=round(vmax,1)
	clb.ax.text(0.5, -0.01, f'{v_min}', transform=clb.ax.transAxes, va='top', ha='center', fontsize=10)
	clb.ax.text(0.5, 1.0, f'{v_max}', transform=clb.ax.transAxes, va='bottom', ha='center', fontsize=10)
	
	ax2.text(xmin*(4/5.+1/10),ymax*(7/6)*0.95, '%s${\'\'}$:%s%s'%(bar_scale_arc,bar_scale_u,unit),fontsize=8)


	ax3=plt.subplot(gs1[1,0])	
	#moment 1
	vmin = abs(np.nanmin(mom1_obs))
	vmax = abs(np.nanmax(mom1_obs))
	base=10
	if vmax < 10 or vmin < 10:
		base = 0.1
	vmin,vmax=vmin_vmax(mom1_obs,pmin=2,pmax=98,base=base,symmetric=True)
		
	im3=ax3.imshow(mom1_obs,cmap=cmap,aspect='auto',vmin=vminv,vmax=vmaxv,origin='lower',extent=extimg)
	axs(ax3,rotation='horizontal',remove_yticks=True,fontsize_ticklabels=10)
	clb=cb(im3, ax3, labelsize=10, colormap = cmap, bbox=(-0.25, 0.2, 0.05, 0.7), ticksfontsize=0, ticks = [], label = "$\mathrm{V_{LOS}}$/ km\,s$^{-1}$", label_pad = -20, colors  = "k",orientation='vertical')
	
	v_min=int(round(vminv,1))
	v_max=int(round(vmaxv,1))
	clb.ax.text(0.5, -0.01, f'{v_min}', transform=clb.ax.transAxes, va='top', ha='center', fontsize=10)
	clb.ax.text(0.5, 1.0, f'{v_max}', transform=clb.ax.transAxes, va='bottom', ha='center', fontsize=10)

	ax3.set_xlabel('$\mathrm{\Delta RA}$ (%s)'%rlabel,fontsize=10,labelpad=1)


	for Axes in [ax2, ax3]:
		Axes.set_xlim(xmin,xmax)
		Axes.set_ylim(ymin,ymax)


	def slits(Axes, pa_rad, rmax):
		hw   = slit_w/2
		verts= np.array([[-rmax, -hw], [ rmax, -hw],
								  [ rmax,  hw], [-rmax,  hw]])

		t	= Affine2D().rotate(pa_rad+np.pi/2).translate(0, 0) + Axes.transData
		rect = Polygon(verts, closed=True, fill=False, edgecolor='k', lw=0.6, ls='-', transform=t)
		Axes.add_patch(rect)
			

	pa_maj_rad = np.radians(pa)
	pa_mnr_rad = np.radians(pa+90)
	pas = {'major': pa_maj_rad, 'minor': pa_mnr_rad}
	
	rmax = rmax_norm
	slits(ax2,pa_maj_rad, rmax)
	slits(ax3,pa_maj_rad, rmax)
	
	rmax = rmax_norm*(1-eps)
	slits(ax2,pa_mnr_rad, rmax)	
	slits(ax3,pa_mnr_rad, rmax)
				
	fig.tight_layout()
	plt.savefig("%sfigures/pvd_%s_model_%s.png"%(out,vmode,galaxy), bbox_inches='tight')
	plt.close()
