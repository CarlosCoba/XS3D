import numpy as np
import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
from matplotlib import gridspec
import matplotlib.colors as colors
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
from matplotlib.patches import Ellipse
from itertools import product
from matplotlib.patches import Polygon
from matplotlib.transforms import Affine2D
from matplotlib.offsetbox import AnchoredText
import matplotlib.ticker as ticker
from matplotlib.transforms import blended_transform_factory

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
from .utils import vmin_vmax

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

	bmaj_arc=psf_lsf.bmaj
	bmin_arc=psf_lsf.bmin
	bpa= psf_lsf.bpa
	psf_arc=psf_lsf.fwhm_psf_arc
	psf_pix=psf_lsf.fwhm_psf_pix
	config_clouds=config['clouds']
	zscale=config_clouds.getfloat('z_scale', 0)


	mom0_mod,mom1_mod,mom2_mod= mom_mod
	mom0,mom1,mom2=momms_obs
	mask=np.isfinite(mom0_mod)*np.isfinite(mom0)

	# extract scalar values
	scalar_fields = ["v_sys", "inc", "pa","x_center", "y_center","phi_bar", "rmax"]
	[vsys,inc,pa,xc,yc,phi_bar,rmax] = [const[k] for k in scalar_fields]
	eps=inc_2_eps(inc)

	# shift the extent to the kinematic centre
	ext=np.dot([-xc,nx-xc,-yc,ny-yc],pixel); xc =0; yc =0

	# remove systemic velocity
	mom1_mod=mom1_mod-vsys
	mom1=mom1-vsys

	# normalize
	rnorm=1
	if np.max(rmax)>80 and np.all(abs(ext)>80):
		rnorm=60
		rlabel='$\'$'
	else:
		rlabel='$\'\'$'

	ext = ext/rnorm
	bmin=bmin_arc/rnorm
	bmaj=bmaj_arc/rnorm
	z_scale=zscale/rnorm

	# Crop the FOV in case the object is too small
	xlength=nx # in pixels
	rmax_norm=rmax/rnorm
	xmin,xmax=ext[0],ext[1]
	ymin,ymax=ext[2],ext[3]
	# inclide the beam size in case a coarse beam is in the FOV
	rmax_tmp = rmax_norm+bmaj
	if np.all(abs(ext[:2])>rmax_norm):
		xmin,xmax=-rmax_tmp*(4/3.),rmax_tmp*(4/3.)
		xlength=2*xmax*rnorm/pixel
	if np.all(abs(ext[-2:])>rmax_norm):
		ymin,ymax=-rmax_tmp*(4/3.),rmax_tmp*(4/3.)

	width, height = 18, 21 # width [cm]
	cm_to_inch = 0.393701 # [inch/cm]
	figWidth = width * cm_to_inch # width [inch]
	figHeight = height * cm_to_inch # width [inch]

	fig = plt.figure(figsize=(figWidth, figHeight), dpi = 300)
	#nrows x ncols
	gs2 = gridspec.GridSpec(3, 3)
	gs2.update(hspace = 0.25, wspace = 0.1, right=0.98, bottom=0.05, top=0.93, left = 0.1 )

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
	lvmin, lvmax=np.log10(abs(vmin)),np.log10(abs(vmax))
	lvmin_int,lvmax_int=int(lvmin), int(lvmax)

	# to place the exponent above the color bar
	exp = None
	if lvmin == lvmax:
		if lvmax>0: # means the flux is larger than 1
			exp = lvmin_int
			res_mom0 = res_mom0*10**(-exp)
			vmin,vmax = vmin*10**(-exp), vmax*10**(-exp)
		if lvmax<0:  # means the flux is lower than 1
			exp = lvmin_int-1
			res_mom0 = res_mom0*10**abs(exp)
			vmin,vmax = vmin*10**abs(exp), vmax*10**abs(exp)

	im2=ax2.imshow(res_mom0,origin='lower',cmap=cmap_mom0,extent=ext,vmin=vmin,vmax=vmax,aspect='auto')

	#moment 1 maps
	res_mom1=mom1-mom1_mod
	vmin = abs(np.nanmin(mom1_mod))
	vmax = abs(np.nanmax(mom1_mod))
	base=10
	if vmax < 10 or vmin < 10:
		base = 0.1
	vmin,vmax=vmin_vmax(mom1_mod,pmin=2,pmax=98,base=base,symmetric=True,mask=mask)


	ax3.imshow(mom1,origin='lower',cmap=cmap,extent=ext,vmin=vmin,vmax=vmax,aspect='auto')
	im4=ax4.imshow(mom1_mod,origin='lower',cmap=cmap,extent=ext,vmin=vmin,vmax=vmax,aspect='auto')
	vmin,vmax=vmin_vmax(res_mom1,pmin=2,pmax=98,base=base,symmetric=True,mask=mask)
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

	cb1=cb(im1,ax0,orientation = "horizontal", colormap = cmap, bbox= (0.55,1,1,1),width = "100%", height = '5%',label_pad = -31, label = "$\mathrm{Moment\,0}$ ($\mathrm{flux\,km\,s^{-1}}$)",labelsize=11, ticksfontsize=11)
	cb1.ax.xaxis.set_ticks_position('top')

	label_res = '$\mathrm{\Delta Mom\,0}$  ($\mathrm{flux\,km\,s^{-1}}$)' if exp is None else '$\mathrm{\Delta Mom\,0}$ ($\mathrm{flux\,km\,s^{-1}\,(x10^{%s})}$)'%(exp)
	cb2=cb(im2,ax2,orientation = "horizontal", colormap = cmap, bbox= (0.1,1,0.8,1),width = "100%", height = '5%',label_pad = -31, label = label_res,labelsize=11, ticksfontsize=11,power=True)
	cb2.ax.xaxis.set_ticks_position('top')

	cb3=cb(im4,ax3,orientation = "horizontal", colormap = cmap, bbox= (0.55,1,1,1),width = "100%", height = '5%',label_pad = -31, label = "$\mathrm{Moment\,1}$ ($\mathrm{km\,s^{-1}}$)",labelsize=11, ticksfontsize=11)
	cb3.ax.xaxis.set_ticks_position('top')

	cb4=cb(im5,ax5,orientation = "horizontal", colormap = cmap, bbox= (0.1,1,0.8,1),width = "100%", height = '5%',label_pad = -31, label = "$\mathrm{\Delta Mom\,1}$  ($\mathrm{km\,s^{-1}}$)",labelsize=11, ticksfontsize=11)
	cb4.ax.xaxis.set_ticks_position('top')


	cb5=cb(im7,ax6,orientation = "horizontal", colormap = cmap, bbox= (0.55,1,1,1),width = "100%", height = '5%',label_pad = -31, label = "$\mathrm{Moment\,2}$ ($\mathrm{km\,s^{-1}}$)",labelsize=11, ticksfontsize=11)
	cb5.ax.xaxis.set_ticks_position('top')


	cb6=cb(im8,ax8,orientation = "horizontal", colormap = cmap, bbox= (0.1,1,0.8,1),width = "100%", height = '5%',label_pad = -31, label = "$\mathrm{\Delta Mom\,2}$  ($\mathrm{km\,s^{-1}}$)",labelsize=11, ticksfontsize=11)
	cb6.ax.xaxis.set_ticks_position('top')

	highz=config['high_z']
	redshift=highz.getfloat('redshift',0)
	vsysz=vsys + redshift*__c__
	bar_scale_arc,bar_scale_u,unit=bscale(vsysz,xlength,pixel,config,hdr_info)
	bar_scale_arc_norm=bar_scale_arc/rnorm

	'''
	# This works perfectly
	ax0.text(xmin*(4/5.+1/10),ymin*(5/6)*0.95, '%s${\'\'}$:%s%s'%(bar_scale_arc,bar_scale_u,unit),fontsize=10)
	ax0.plot([xmin*(4/5.),xmin*(4/5.)+bar_scale_arc_norm],[ymin*(5/6),ymin*(5/6)],'k-', lw =3)

	ax3.text(xmin*(4/5.+1/10),ymin*(5/6)*0.95, '%s${\'\'}$:%s%s'%(bar_scale_arc,bar_scale_u,unit),fontsize=10)
	ax3.plot([xmin*(4/5.),xmin*(4/5.)+bar_scale_arc_norm],[ymin*(5/6),ymin*(5/6)],'k-', lw =3)

	ax6.text(xmin*(4/5.+1/10),ymin*(5/6)*0.95, '%s${\'\'}$:%s%s'%(bar_scale_arc,bar_scale_u,unit),fontsize=10)
	ax6.plot([xmin*(4/5.),xmin*(4/5.)+bar_scale_arc_norm],[ymin*(5/6),ymin*(5/6)],'k-', lw =3)
	'''

	for Axes in [ax0,ax1,ax3,ax4,ax6,ax7]:
		x0 = 0.95*xmax - bmin/2
		y0 = 0.95*ymin + bmaj/2
		ellipse = Ellipse((x0, y0),width=bmin,height=bmaj,angle=bpa,fill=0,edgecolor='blue', facecolor='none', hatch=5*'.')
		Axes.add_patch(ellipse)


	for Axes in [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
		color = '#393d42'
		Axes.plot(xc, yc, marker='+', color = 'black', markeredgewidth=1, zorder=100)
		pa_rad = np.radians(pa)

		if inc >= 89:
			n_sigma_z=2
			zscale_obs=np.sqrt(z_scale**2 + (bmaj/2)**2 )
			z_hw=n_sigma_z * zscale_obs * np.sin(inc)
			hw   = max(rmax_norm * np.cos(np.radians(inc)), z_hw)
			verts= np.array([[-rmax_norm, -hw], [ rmax_norm, -hw],
							  [ rmax_norm,  hw], [-rmax_norm,  hw]])

			t	= Affine2D().rotate(pa_rad+np.pi/2).translate(0, 0) + Axes.transData
			rect = Polygon(verts, closed=True, fill=False,
						   edgecolor=color, lw=1, ls='-', transform=t)
			Axes.add_patch(rect)
		else:
			elipse=drawellipse(xc,yc,bmajor=rmax_norm,pa_deg=pa,eps=eps)
			x,y=elipse[0],elipse[1]
			line = (0, (2, 6))
			Axes.plot(x, y, linestyle=line, color = color,  lw=1)


	for Axes in axes:
		Axes.set_xlim(xmin,xmax)
		Axes.set_ylim(ymin,ymax)
		# Limit the display to a maximum of 5 ticks per axis
		Axes.xaxis.set_major_locator(ticker.MaxNLocator(5))
		Axes.yaxis.set_major_locator(ticker.MaxNLocator(5))



	for Axs in [ax0, ax3, ax6]:
		blended = blended_transform_factory(
			Axs.transData,   # x axis: data coordinates (pixels/arcsec/kpc)
			Axs.transAxes    # y axis: axes fraction    (0=bottom, 1=top, 1.03=just above)
		)

		scalebar_data   = bar_scale_arc_norm   # same units as the axes
		x_left          = xmin*0.94            # near the left edge of the image
		x_right         = x_left + scalebar_data

		y_bar  = 1.03    # bar sits 3% above the top edge of the axes
		y_text = 1.05   # text floats a bit higher

		Axs.annotate('',
			xy=(x_right, y_bar), xytext=(x_left, y_bar),
			xycoords=blended, textcoords=blended,
			arrowprops=dict(arrowstyle='|-|', color='k', lw=2, mutation_scale=1),
			annotation_clip=False)

		x_mid = (x_left + x_right)/2
		text = '%s${\'\'}$:%s%s'%(bar_scale_arc,bar_scale_u,unit)
		Axs.text(x_mid, y_text, text,
			transform=blended, ha='center', va='bottom',
			color='k', fontsize=9, clip_on=False)


	#fig.tight_layout()
	plt.savefig("%sfigures/mommaps_%s_model_%s.png"%(out,vmode,galaxy),bbox_extra_artists=cb2)
	#plt.clf()
	plt.close()
