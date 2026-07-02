import numpy as np
import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
from matplotlib import gridspec
import matplotlib.colors as colors
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
from mpl_toolkits.axes_grid1.anchored_artists import (AnchoredEllipse,AnchoredSizeBar)
from itertools import product
from matplotlib.patches import Polygon
from matplotlib.transforms import Affine2D	
from matplotlib.offsetbox import AnchoredText

from .axes_params import axes_ambient as axs
from .cbar import colorbar as cb
from .colormaps_CLC import vel_map
from .barscale import bscale
from .ellipse import drawellipse, drawrectangle
from .psf_lsf import PsF_LsF
from .constants import __c__
#params =   {'text.usetex' : True }
#plt.rcParams.update(params)


from .pixel_params import inc_2_eps, eps_2_inc,e_eps2e_inc

def vmin_vmax(data,pmin=2,pmax=99.5,base=None,symmetric =False,mask=None):
	if mask is not None:
		data = data[mask]
	vmin,vmax=np.nanpercentile(np.unique(data),pmin),np.nanpercentile(np.unique(data),pmax)
	vsym = (vmax+abs(vmin))*0.5
	if symmetric: vmin,vmax=-1*vsym,vsym
	if base is not None:
		vmin,vmax=(vmin//base+1)*base,(vmax//base+1)*base
		if symmetric: vmin,vmax=-1*(vsym//base+1)*base,(vsym//base+1)*base
	return vmin,vmax

def zero2nan(data):
	#data[data==0]=np.nan
	return data

cmap = vel_map()
cmap_mom0 = vel_map('mom0')

def plot_mommaps(galaxy,mom_mod,momms_obs,const,vmode,psf_lsf,hdr_info,config,out):

	pixel = hdr_info.pix_arcs
	nx = hdr_info.nx
	ny = hdr_info.ny
	nz = hdr_info.nz			

	mom0_mod,mom1_mod,mom2_mod_kms= mom_mod
	mom0,mom1,mom2=momms_obs
	mom0,mom1,mom2=zero2nan(mom0),zero2nan(mom1),zero2nan(mom2)
	mom0_mod,mom1_mod,mom2_mod=zero2nan(mom0_mod),zero2nan(mom1_mod),zero2nan(mom2_mod_kms)
	mask=np.isfinite(mom0_mod)*np.isfinite(mom0)
	
	scalar_fields = ["v_sys", "inc", "pa","x_center", "y_center","phi_bar", "rmax"]
	[vsys,inc,pa,xc,yc,phi_bar,rmax] = [const[k] for k in scalar_fields]		
	eps=inc_2_eps(inc)
	
	# shift the extent to the kinematic centre
	ext=np.dot([-xc,nx-xc,-yc,ny-yc],pixel); xc =0; yc =0

	mom1_mod=mom1_mod-vsys
	mom1=mom1-vsys

	rnorm=1
	if np.max(rmax)>80 and np.all(abs(ext)>80):
		rnorm=60
		rlabel='$\'$'
	else:
		rlabel='$\'\'$'
	ext = ext/rnorm

	# Crop the FOV in case the object is too small
	xlength=nx # in pixels
	rmax_norm=rmax/rnorm
	xmin,xmax=ext[0],ext[1]
	ymin,ymax=ext[2],ext[3]
	if np.all(abs(ext[:2])>rmax_norm):
		xmin,xmax=-rmax_norm*(4/3.),rmax_norm*(4/3.)
		xlength=2*xmax*rnorm/pixel
	if np.all(abs(ext[-2:])>rmax_norm):
		ymin,ymax=-rmax_norm*(4/3.),rmax_norm*(4/3.)

	width, height = 10, 13 # width [cm]
	width, height = 14, 18 # width [cm]
	#width, height = 3, 15 # width [cm]
	cm_to_inch = 0.393701 # [inch/cm]
	figWidth = width * cm_to_inch # width [inch]
	figHeight = height * cm_to_inch # width [inch]

	fig = plt.figure(figsize=(figWidth, figHeight), dpi = 300)
	#nrows x ncols
	widths = [1,1,1]
	heights = [1]
	gs2 = gridspec.GridSpec(3, 3)
	#gs2.update(left=0.1, right=0.98,top=0.93,bottom=0.06, hspace = 0.3, wspace = 0)
	gs2.update(hspace = 0.3, wspace = 0, right=0.98, bottom=0.06, top=0.93 )



	axes=[plt.subplot(gs2[j,i]) for j,i in product(np.arange(3),np.arange(3))]
	[ax0,ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8]=axes

	units_res_mom1='km/s'
	units_res_mom2='km/s'

	# moment zero maps:
	mom0_mod=abs(mom0_mod)
	res_mom0=mom0-mom0_mod
	vmin,vmax=vmin_vmax(mom0_mod)
	norm = colors.LogNorm(vmin=vmin, vmax=vmax) if (vmin>0) & (np.log10(vmax/vmin)>1) else colors.Normalize(vmin=vmin, vmax=vmax)
	ax0.imshow(mom0,norm=norm,origin='lower',cmap=cmap_mom0,extent=ext,aspect='auto')
	im1=ax1.imshow(mom0_mod,norm=norm,origin='lower',cmap=cmap_mom0,extent=ext,aspect='auto')
	vmin,vmax=vmin_vmax(res_mom0,2,98,symmetric=True)
	lvmin, lvmax=int(np.log10(abs(vmin))),int(np.log10(abs(vmax)))
	# to place the exponent above the color bar
	exp = None
	if lvmin == lvmax:
		exp = lvmin - 1
		res_mom0 = res_mom0*10**abs(exp)
		vmin,vmax = vmin*10**abs(exp), vmax*10**abs(exp),
	im2=ax2.imshow(res_mom0,origin='lower',cmap=cmap_mom0,extent=ext,vmin=vmin,vmax=vmax,aspect='auto')

	#moment 1 maps
	res_mom1=mom1-mom1_mod
	vmin = abs(np.nanmin(mom1_mod))
	vmax = abs(np.nanmax(mom1_mod))
	base=10
	if vmax < 10 or vmin < 10:
		base = 0.1
	vmin,vmax=vmin_vmax(mom1_mod,base=base,symmetric=True,mask=mask)


	ax3.imshow(mom1,origin='lower',cmap=cmap,extent=ext,vmin=vmin,vmax=vmax,aspect='auto')
	im4=ax4.imshow(mom1_mod,origin='lower',cmap=cmap,extent=ext,vmin=vmin,vmax=vmax,aspect='auto')
	vmin,vmax=vmin_vmax(res_mom1,base=base,symmetric=True,mask=mask)
	# if residuals are lower than 1km/s then express them in m/s
	if vmax<1:
		res_mom1*=1000
		vmin,vmax=vmin*1000,vmax*1000
		units_res_mom1='m/s'
	im5=ax5.imshow(res_mom1,origin='lower',cmap=cmap,extent=ext,vmin=vmin,vmax=vmax,aspect='auto')

	#moment 2 maps
	res_mom2=mom2-mom2_mod
	vmin,vmax=vmin_vmax(mom2_mod,2,95,base=base,mask=mask)
	ax6.imshow(mom2,origin='lower',cmap=cmap,extent=ext,vmin=vmin,vmax=vmax,aspect='auto')
	im7=ax7.imshow(mom2_mod,origin='lower',cmap=cmap,extent=ext,vmin=vmin,vmax=vmax,aspect='auto')
	vmin,vmax=vmin_vmax(res_mom2,2,98,symmetric=True,mask=mask)
	# if residuals are lower than 1km/s then express them in m/s
	if vmax<1:
		res_mom2*=1000
		vmin,vmax=vmin*1000,vmax*1000
		units_res_mom2='m/s'
	im8=ax8.imshow(res_mom2,origin='lower',cmap=cmap,extent=ext,vmin=vmin,vmax=vmax,aspect='auto')


	for k in axes: axs(k, rotation='horizontal')
	for k in range(9):
		if k not in [0,3,6]:
			axs(axes[k],remove_yticks= True)
		if k not in [6,7,8]:
				axes[k].xaxis.set_major_formatter(plt.NullFormatter())

		#	axs(axes[k],remove_xticks= True)


	for k in range(-3,0,1): axes[k].set_xlabel('$\mathrm{ \Delta RA }$ (%s)'%rlabel,fontsize=12,labelpad=0)
	for k in range(1,10,3): axes[k-1].set_ylabel('$\mathrm{ \Delta Dec}$ (%s)'%rlabel,fontsize=12,labelpad=0)


	txt0=['Mom0','Mom1', 'Mom2']
	txt1=['Obs','Mod']

	k=1
	for j,i in product(np.arange(3),np.arange(2)):
		txt = AnchoredText('$\mathrm{%s}$'%(txt1[i]), loc="upper left", pad=0.1, borderpad=0, prop={"fontsize":12},zorder=1e4);axes[k-1].add_artist(txt)
		k+=1
		if not k % 3:
			txt = AnchoredText('$\mathrm{Residual}$', loc="upper left", pad=0.1, borderpad=0, prop={"fontsize":12},zorder=1e4);axes[k-1].add_artist(txt)
			k+=1

	spec_u = 'km\,s$^{-1}$'
	cb(im1,ax0,orientation = "horizontal", colormap = cmap, bbox= (0.5,1.12,1,1),width = "100%", height = "3%",label_pad = -27, label = "Mom\,0 ($\mathrm{flux\,km\,s^{-1}}$)",labelsize=11, ticksfontsize=11)
	label_res = '$\mathrm{flux\,km\,s^{-1}}$' if exp is None else '$\mathrm{flux*km/s~(x10^{%s})}$'%(exp)
	cb2=cb(im2,ax2,orientation = "horizontal", colormap = cmap, bbox= (0.1,1.12,0.8,1),width = "100%", height = "3%",label_pad = -28, label = label_res,labelsize=11, ticksfontsize=11,power=True)

	cb(im4,ax3,orientation = "horizontal", colormap = cmap, bbox= (0.5,1.12,1,1),width = "100%", height = "3%",label_pad = -27, label = "Mom\,1 ($\mathrm{km\,s^{-1}}$)",labelsize=11, ticksfontsize=11)
	cb(im5,ax5,orientation = "horizontal", colormap = cmap, bbox= (0.1,1.12,0.8,1),width = "100%", height = "3%",label_pad = -27, label = "$\mathrm{\Delta}$Mom\,1 ($\mathrm{km\,s^{-1}}$)",labelsize=11, ticksfontsize=11)

	cb(im7,ax6,orientation = "horizontal", colormap = cmap, bbox= (0.5,1.12,1,1),width = "100%", height = "3%",label_pad = -27, label = "Mom\,2 ($\mathrm{km\,s^{-1}}$)",labelsize=11, ticksfontsize=11)
	cb(im8,ax8,orientation = "horizontal", colormap = cmap, bbox= (0.1,1.12,0.8,1),width = "100%", height = "3%",label_pad = -27, label = "$\mathrm{\Delta}$Mom\,2 ($\mathrm{km\,s^{-1}}$)",labelsize=11, ticksfontsize=11)


	highz=config['high_z']
	redshift=highz.getfloat('redshift',0)
	vsysz=vsys + redshift*__c__
	bar_scale_arc,bar_scale_u,unit=bscale(vsysz,xlength,pixel,config)
	bar_scale_arc_norm=bar_scale_arc/rnorm

	ax0.text(xmin*(4/5.+1/10),ymin*(5/6)*0.95, '%s${\'\'}$:%s%s'%(bar_scale_arc,bar_scale_u,unit),fontsize=10)
	ax0.plot([xmin*(4/5.),xmin*(4/5.)+bar_scale_arc_norm],[ymin*(5/6),ymin*(5/6)],'k-')

	ax3.text(xmin*(4/5.+1/10),ymin*(5/6)*0.95, '%s${\'\'}$:%s%s'%(bar_scale_arc,bar_scale_u,unit),fontsize=10)
	ax3.plot([xmin*(4/5.),xmin*(4/5.)+bar_scale_arc_norm],[ymin*(5/6),ymin*(5/6)],'k-')

	ax6.text(xmin*(4/5.+1/10),ymin*(5/6)*0.95, '%s${\'\'}$:%s%s'%(bar_scale_arc,bar_scale_u,unit),fontsize=10)
	ax6.plot([xmin*(4/5.),xmin*(4/5.)+bar_scale_arc_norm],[ymin*(5/6),ymin*(5/6)],'k-')


	# plot PSF ellipse ?
	bmaj_arc=psf_lsf.bmaj
	bmin_arc=psf_lsf.bmin
	bpa= psf_lsf.bpa
	psf_arc=psf_lsf.fwhm_psf_arc
	psf_pix=psf_lsf.fwhm_psf_pix

	psf=None
	bmaj,bmin=None,None
	if psf_arc is not None:
		psf=psf_arc/rnorm
	if bmaj_arc is not None:
		bmaj=bmaj_arc/rnorm
	if bmin_arc is not None:
		bmin=bmin_arc/rnorm

	if psf is not None:
		for Axes in [ax0,ax1,ax6]:
			beam=AnchoredEllipse(Axes.transData, width=bmin,height=bmaj, angle=bpa, loc='lower right',pad=0.2, borderpad=0,frameon=True, )
			Axes.add_artist(beam)

	if bmaj_arc is not None and bmin_arc is not None:
		for Axes in [ax0,ax1,ax3,ax4,ax6,ax7]:
			beam=AnchoredEllipse(Axes.transData, width=bmin,height=bmaj, angle=bpa, loc='lower right',pad=0.2, borderpad=0,frameon=True)
			beam.ellipse.set(color='gray')
			Axes.add_artist(beam)


	for Axes in [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:

		elipse_mjr=drawellipse(xc,yc,bmajor=0.1*rmax_norm,pa_deg=pa+45,eps=1)
		x,y=elipse_mjr[0],elipse_mjr[1]
		Axes.plot(x, y, linestyle='--', color = '#393d42',  lw=1.5)

		elipse_mnr=drawellipse(xc,yc,bmajor=0.1*rmax_norm,pa_deg=pa+90+45,eps=1)
		x,y=elipse_mnr[0],elipse_mnr[1]
		Axes.plot(x, y, linestyle='--', color = '#393d42',  lw=1.5)


		pa_rad = np.radians(pa)
		if inc >= 85:
			hw   = max(rmax_norm * np.cos(np.radians(inc)), psf_arc / 2)
			verts= np.array([[-rmax_norm, -hw], [ rmax_norm, -hw],
							  [ rmax_norm,  hw], [-rmax_norm,  hw]])

			t	= Affine2D().rotate(pa_rad+np.pi/2).translate(0, 0) + Axes.transData
			rect = Polygon(verts, closed=True, fill=False,
						   edgecolor='k', lw=1.2, ls='-', transform=t)
			Axes.add_patch(rect)
		else:		
			elipse=drawellipse(xc,yc,bmajor=rmax_norm,pa_deg=pa,eps=eps)
			x,y=elipse[0],elipse[1]
			Axes.plot(x, y, '-', color = '#393d42',  lw=1.5)
		
		
	for Axes in axes:
		Axes.set_xlim(xmin,xmax)
		Axes.set_ylim(ymin,ymax)

	fig.tight_layout()
	plt.savefig("%sfigures/mommaps_%s_model_%s.png"%(out,vmode,galaxy),bbox_extra_artists=cb2)
	#plt.clf()
	plt.close()
