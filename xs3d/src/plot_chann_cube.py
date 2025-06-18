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
from .ellipse import drawellipse
from .psf_lsf import PsF_LsF
from .constants import __c__
from .conv import conv2d,gkernel,gkernel1d
from .conv_fftw import fftconv,data_2N,fftconv_numpy
#params =   {'text.usetex' : True }
#plt.rcParams.update(params)

def vmin_vmax(data,pmin=2,pmax=99.5,base=None,symmetric =False):
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


def plot_channels(galaxy,datacube,momms_mdls,const,ext,vmode,hdr_cube,hdr_info,config,rms,pixel,out):

	mom0_mdl,mom1_mdl,mom2_mdl_kms,mom2_mdl_A,cube_mdl,velmap_intr,sigmap_intr,twoDmodels= momms_mdls
	[pa,eps,inc,xc,yc,vsys,phi_bar,rmax]=const
	wave_kms=hdr_info.wave_kms
	[nz,ny,nx]=cube_mdl.shape
	ext=np.dot([-xc,nx-xc,-yc,ny-yc],pixel); xc =0; yc =0


	pixelconv=pixel
	bmajconv=bminconv=2*pixel
	tmp=np.isfinite(cube_mdl)
	tmp_mdl=np.copy(cube_mdl)
	tmp_mdl[~tmp]=0
	psf2d=gkernel([ny,nx],fwhm=None,bmaj=bmajconv,bmin=bminconv,pixel_scale=pixelconv)
	padded_mdl, cube_slices = data_2N(tmp_mdl, axes=[1, 2])
	psf3d=np.ones_like(cube_mdl)*psf2d
	padded_psf, _ = data_2N(psf3d, axes=[1, 2])
	dft=fftconv(padded_mdl,padded_psf,threads=2,axes = [1,2])
	cube_mdl_conv=dft.conv_DFT(cube_slices)
	del tmp_mdl, tmp


	rnorm=1
	if rmax>80 and np.all(abs(ext)>80):
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



	width, height = 17, 17*(5/6) # width [cm]
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

	vmin=0
	vmax=np.nanpercentile(cube_mdl[cube_mdl!=0],99.5)
	norm = colors.LogNorm(vmin=vmin, vmax=vmax) if vmax > 1 else colors.Normalize(vmin=vmin, vmax=vmax)

	for j,k in enumerate(chanplot):
		if j<=ngood:
			kk=channels[k]
			chanmap=datacube[kk]
			chanmap_mdl=(cube_mdl_conv[kk])/rms
			axes[j].imshow(chanmap,norm=norm,origin='lower',cmap=cmap_mom0,extent=ext,aspect='auto',alpha=1)
			levels=2**np.arange(-1,7,1,dtype=float)
			axes[j].contour(chanmap_mdl,levels=levels,colors='navy', linestyles='solid',zorder=1,extent=ext,linewidths=0.4,alpha=1)

			vchan=round(wave_kms[kk],2)
			txt = AnchoredText(f'{vchan}', loc="upper left", pad=0.1, borderpad=0, prop={"fontsize":10},zorder=1e4);txt.patch.set_alpha(0);axes[j].add_artist(txt)

			#axs(axes[k],rotation='horizontal', remove_ticks_all=True)

	for j,Axes in enumerate(axes):
		if j==(l0**2-l0):
			axs(Axes,rotation='horizontal', direction='out', fontsize_ticklabels=12)
		else:
			axs(Axes,rotation='horizontal', remove_xyticks=True, direction='out', fontsize_ticklabels=12)


	lines = [Line2D([0], [0], color='navy',lw=0.8)];labels=['model']
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
		elipse=drawellipse(xc,yc,bmajor=rmax_norm,pa_deg=pa,eps=eps)
		x,y=elipse[0],elipse[1]#pixel*(elipse[0]-nx/2)/rnorm,pixel*(elipse[1]-ny/2)/rnorm
		Axes.plot(x, y, '-', color = '#393d42',  lw=0.5)

		elipse_mjr=drawellipse(xc,yc,bmajor=0.5*rmax_norm,pa_deg=pa,eps=1)
		x,y=elipse_mjr[0],elipse_mjr[1]#pixel*(elipse_mjr[0]-nx/2)/rnorm,pixel*(elipse_mjr[1]-ny/2)/rnorm
		Axes.plot(x, y, linestyle='--', color = '#393d42',  lw=0.5)

		elipse_mnr=drawellipse(xc,yc,bmajor=0.5*(1-eps)*rmax_norm,pa_deg=pa+90,eps=1)
		x,y=elipse_mnr[0],elipse_mnr[1]#pixel*(elipse_mnr[0]-nx/2)/rnorm,pixel*(elipse_mnr[1]-ny/2)/rnorm
		Axes.plot(x, y, linestyle='--', color = '#393d42',  lw=0.5)



	for Axes in axes:
			Axes.set_xlim(xmin,xmax)
			Axes.set_ylim(ymin,ymax)


	# plot PSF ellipse ?
	psf_lsf=PsF_LsF(hdr_cube,config)
	bmaj_arc=psf_lsf.bmaj
	bmin_arc=psf_lsf.bmin
	bpa= psf_lsf.bpa
	psf_arc=psf_lsf.fwhm_psf_arc

	psf=None
	bmaj,bmin=None,None
	if psf_arc is not None:
		psf=psf_arc/rnorm
	if bmaj_arc is not None:
		bmaj=bmaj_arc/rnorm
	if bmin_arc is not None:
		bmin=bmin_arc/rnorm

	if psf is not None:
		for Axes in axes:
			beam=AnchoredEllipse(Axes.transData, width=bmin,height=bmaj, angle=bpa, loc='lower right',pad=0.2, borderpad=0,frameon=True)
			beam.ellipse.set(color='gray')
			Axes.add_artist(beam)

	if bmaj_arc is not None and bmin_arc is not None:
		for Axes in axes:
			beam=AnchoredEllipse(Axes.transData, width=bmin,height=bmaj, angle=bpa, loc='lower right',pad=0.2, borderpad=0,frameon=True)
			beam.ellipse.set(color='gray')
			Axes.add_artist(beam)

	indx=(l0**2-l0)
	axes[indx].set_xlabel('$\mathrm{ \Delta RA }$ (%s)'%rlabel,fontsize=12,labelpad=1)
	axes[indx].set_ylabel('$\mathrm{ \Delta Dec}$ (%s)'%rlabel,fontsize=12,labelpad=1)

	txt = AnchoredText(f'Channel units: km/s', loc="lower left", pad=0.1, borderpad=0, prop={"fontsize":12},zorder=1e4, bbox_to_anchor=(0, -0.3), bbox_transform=axes[-1].transAxes);txt.patch.set_alpha(0);axes[-1].add_artist(txt)

	plt.savefig("%sfigures/channels_cube_%s_model_%s.png"%(out,vmode,galaxy))
	#plt.clf()
	plt.close()

	return None
