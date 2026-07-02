import numpy as np
import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
from matplotlib import gridspec
import matplotlib.colors as colors
from matplotlib.lines import Line2D
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
from mpl_toolkits.axes_grid1.anchored_artists import (AnchoredEllipse,AnchoredSizeBar)
from itertools import product

from matplotlib.offsetbox import AnchoredText
from .axes_params import axes_ambient as axs
from .cbar import colorbar as cb
from .colormaps_CLC import vel_map
from .barscale import bscale
from .ellipse import drawellipse, drawrectangle
from .psf_lsf import PsF_LsF
from .constants import __c__
from .conv import conv2d,gkernel,gkernel1d
from .conv_fftw import fftconv,data_2N,fftconv_numpy
from .pixel_params import eps_2_inc,e_eps2e_inc,inc_2_eps

#params =   {'text.usetex' : True }
#plt.rcParams.update(params)


cmap = vel_map()
cmap_mom0 = vel_map('mom0')
cmap_mom0 = vel_map('pvd')

def plot_channels(galaxy,datacube,cube_mod,const,vmode,hdr_info,psf_lsf,config,rms,out):


	[v_sys,inc,pa,x_center,y_center,phi_bar,rmax]=const['v_sys'],const['inc'],const['pa'],const['x_center'],const['y_center'],const['phi_bar'],const['rmax']
	eps = inc_2_eps(inc)
	wave_kms=hdr_info.wave_kms
	[nz,ny,nx]=datacube.shape
	pixel=hdr_info.scale
	ext=np.dot([-x_center,nx-x_center,-y_center,ny-y_center],pixel); xc =0; yc =0	


	pixelconv=pixel
	bmajconv=bminconv=2*pixel
	tmp=np.isfinite(cube_mod)
	tmp_mdl=np.copy(cube_mod)
	tmp_mdl[~tmp]=0
	psf2d=gkernel([ny,nx],fwhm=None,bmaj=bmajconv,bmin=bminconv,pixel_scale=pixelconv)
	padded_mdl, cube_slices = data_2N(tmp_mdl, axes=[1, 2])
	psf3d=np.ones_like(cube_mod)*psf2d
	padded_psf, _ = data_2N(psf3d, axes=[1, 2])
	dft=fftconv(padded_mdl,padded_psf,threads=2,axes = [1,2])
	cube_mod_conv=dft.conv_DFT(cube_slices)
	del tmp_mdl, tmp

	# plot PSF ellipse ?
	bmaj_arc=psf_lsf.bmaj
	bmin_arc=psf_lsf.bmin
	bpa= psf_lsf.bpa
	psf_arc=psf_lsf.fwhm_psf_arc
	
	rnorm=1
	if rmax>80 and np.all(abs(ext)>80):
		rnorm=60
		rlabel='$\'$'
	else:
		rlabel='$\'\'$'
		
	ext = ext/rnorm
	bmin=bmin_arc/rnorm
	bmaj=bmaj_arc/rnorm	

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


	width, height = 17, 17*(5/6) # width [cm]
	width, height = 18*(5/6), 17 # width [cm]	
	cm_to_inch = 0.393701 # [inch/cm]
	figWidth = width * cm_to_inch # width [inch]
	figHeight = height * cm_to_inch # width [inch]
	fig = plt.figure(figsize=(figWidth, figHeight), dpi = 300)

	meanflux_chan = np.array([np.mean(datacube[k], where=( (datacube[k]!=0) & (np.isfinite(datacube[k]))) )/rms for k in range(nz)])
	chan_sig=(meanflux_chan>0.5) & np.isfinite(meanflux_chan)
	ngood=np.sum(chan_sig)

	l0=7
	l=np.sqrt(ngood)
	l=np.ceil(l)
	if l < l0:
		l0=l

	l0=int(l0)
	gs2 = gridspec.GridSpec(l0, l0)
	gs2.update(left=0.12, right=0.86,top=0.93,bottom=0.1, hspace = 0, wspace = 0)
	axes=[plt.subplot(gs2[j,i]) for j,i in product(np.arange(l0),np.arange(l0))]


	channels=np.arange(nz)[chan_sig] # index of channels with signal
	chanplot=np.linspace(0, ngood-1, l0**2) # we cannot plot all of them, just l0**2.
	chanplot=np.round(chanplot)
	u, c = np.unique(chanplot, return_counts=True)
	dup = u[c > 1]
	if len(dup)>0:
		chanplot=np.arange(ngood)
		print('Too many channels to show: Channels wont be displayed in multiples of the channel width')
	chanplot=chanplot.astype(int)

	vmin=rms*(2**-1)
	vmax=np.percentile(cube_mod[cube_mod!=0],99.5)
	norm = colors.LogNorm(vmin=vmin, vmax=vmax) if vmax > 1 else colors.Normalize(vmin=vmin, vmax=vmax)
	clines = '#279dc5'
	
	dv=psf_lsf.cdelt3_kms
	for j,k in enumerate(chanplot):
		if j<=ngood:
			kk=channels[k]
			chanmap=datacube[kk]
			chanmap_mdl=(cube_mod_conv[kk])/rms
			chanmap[chanmap==0]=np.nan
			axes[j].imshow(chanmap,norm=norm,origin='lower',cmap=cmap_mom0,extent=ext,aspect='auto',alpha=1)
			levels=2**np.arange(-1,7,1,dtype=float)
			axes[j].contour(chanmap_mdl,levels=levels,colors=clines, linestyles='solid',zorder=1,extent=ext,linewidths=0.4,alpha=1)

			v_chan=round(wave_kms[kk],2)
			vchan=int(v_chan) if dv>10 else v_chan
			txt = AnchoredText(f'{vchan}', loc="upper left", pad=0.1, borderpad=0.1, prop={"fontsize":10},zorder=1e4);txt.patch.set_alpha(0);axes[j].add_artist(txt)

			#axs(axes[k],rotation='horizontal', remove_ticks_all=True)

	for j,Axes in enumerate(axes):
		if j==(l0**2-l0):
			axs(Axes,rotation='horizontal', direction='in', fontsize_ticklabels=10, tick_minor=2, tick_major=3)
		else:
			axs(Axes,rotation='horizontal', direction='in', remove_xyticks=True, fontsize_ticklabels=12, tick_minor=2, tick_major=3)

	lines = [Line2D([0], [0], color=clines,lw=0.8)];labels=['model']
	ax_legend=axes[l0-1]
	ax_legend.legend(lines,labels,loc='lower left',borderaxespad=0,handlelength=0.6,handletextpad=0.5,frameon=False, fontsize=12, bbox_to_anchor=(0, 1), bbox_transform=ax_legend.transAxes)


	rms_int=int(np.log10(rms))
	rms_round= round(rms/10**rms_int,5)
	txt = AnchoredText(f'rms={rms_round}e{rms_int} [flux units]', loc="lower left", frameon=False, prop={"fontsize":12}, bbox_to_anchor=(0, 1), bbox_transform=axes[0].transAxes);axes[0].add_artist(txt)
	from matplotlib.cm import ScalarMappable
	cmappable = ScalarMappable(colors.Normalize(vmin/rms,vmax/rms), cmap=cmap_mom0)
	w=int(100*l0)
	cb(cmappable,axes[-1],orientation = "vertical", colormap = cmap, bbox= (1.1,0,1,1), height = f"{w}%", width = "10%",label_pad = 0, label = "flux/rms",labelsize=12, ticksfontsize=9)

	for Axes in axes:
		if vmode == 'edgeon':
			rec=drawrectangle(xc,yc,bmajor=rmax_norm,pa_deg=pa,eps=eps)
			x,y = rec[0], rec[1]
			Axes.plot(x, y, '-', color = '#393d42',  lw=0.5)			
		else:	
			elipse=drawellipse(xc,yc,bmajor=rmax_norm,pa_deg=pa,eps=eps)
			x,y=elipse[0],elipse[1]#pixel*(elipse[0]-nx/2)/rnorm,pixel*(elipse[1]-ny/2)/rnorm
			Axes.plot(x, y, '-', color = '#393d42',  lw=0.5)

		Axes.plot(xc, yc, marker='+', color = 'black', markeredgewidth=1, zorder=100)	



	for Axes in axes:
		Axes.set_xlim(xmin,xmax)
		Axes.set_ylim(ymin,ymax)
	
	for Axes in axes:
		beam=AnchoredEllipse(Axes.transData, width=bmin, height=bmaj, angle=bpa, loc='lower right', pad=0.2, borderpad=0, frameon=False)
		beam.ellipse.set(edgecolor='blue', facecolor='none', hatch=5*'.')
		Axes.add_artist(beam)
						

	indx=(l0**2-l0)
	axes[indx].set_xlabel('$\mathrm{ \Delta RA }$ (%s)'%rlabel,fontsize=10,labelpad=0)
	axes[indx].set_ylabel('$\mathrm{ \Delta Dec}$ (%s)'%rlabel,fontsize=10,labelpad=0)

	txt = AnchoredText(f'Channel units: km/s', loc="lower left", pad=0.1, borderpad=0, prop={"fontsize":12},zorder=1e4, bbox_to_anchor=(0, -0.3), bbox_transform=axes[-1].transAxes);txt.patch.set_alpha(0);axes[-2].add_artist(txt)

	fig.tight_layout()
	fig.savefig("%sfigures/channels_cube_%s_model_%s.png"%(out,vmode,galaxy), bbox_inches='tight')
	#plt.clf()
	plt.close()

	return None
