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
from matplotlib.lines import Line2D
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
cmap=vel_map()


#params =   {'text.usetex' : True }
#plt.rcParams.update(params)

def vmin_vmax(data,pmin=2,pmax=99.5,base=None,symmetric =False):
	vmin,vmax=np.nanpercentile(np.unique(data),pmin),np.nanpercentile(np.unique(data),pmax)
	vsym = (vmax+abs(vmin))*0.5
	if symmetric: vmin,vmax=-1*vsym,vsym
	if base is not None:
		vmin,vmax=(vmin//base)*base,(vmax//base)*base
		if symmetric: vmin,vmax=-1*(vsym//base)*base,(vsym//base)*base
	return vmin,vmax

def zero2nan(data):
	data[data==0]=np.nan
	return data

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

arr = np.linspace(0, 50, 100).reshape((10, 10))
fig, ax = plt.subplots(ncols=2)

cmap = plt.get_cmap('magma_r')
new_cmap = truncate_colormap(cmap, 0, 0.6)
cmap = vel_map()
cmap_mom0 = vel_map('mom0')

def plot_pvd(galaxy,out_pvd,vt,R,const,vmode,rms,momms_mdls,momaps,datacube,pixel,hdr_cube,hdr_info,config,out):
	pvds,slits,ext=out_pvd
	slit_major,slit_minor = slits
	[ext0,ext1,_]=ext
	[pa,eps,inc,xc,yc,vsys,phi_bar,rmax]=const

	ext0[2]=ext0[2]-vsys
	ext1[2]=ext1[2]-vsys
	ext0[3]=ext0[3]-vsys
	ext1[3]=ext1[3]-vsys

	mom0,mom1,mom2=momaps
	mom0_mdl,mom1_mdl,mom2_mdl_kms,mom2_mdl_A,cube_mdl,velmap_intr,sigmap_intr,twoDmodels = momms_mdls
	pvd_maj,pvd_min,pvd_maj_mdl,pvd_min_mdl=pvds[0],pvds[1],pvds[2],pvds[3]
	msk=np.isfinite(mom0*mom0_mdl/mom0)
	slit_major*=msk
	slit_minor*=msk

	pa_maj = pa % 360
	pa_min = (pa+90) % 360
	pa_maj = int(round(pa_maj))
	pa_min = int(round(pa_min))
	[ny,nx]=mom0.shape
	extimg=np.dot([-nx/2.,nx/2.,-ny/2.,ny/2.],pixel)
	extimg=np.dot([-xc,nx-xc,-yc,ny-yc],pixel); xc =0; yc =0

	vt=vt*np.sin(inc*np.pi/180)
	max_vt=np.nanmax(vt)

	# Upper quadrant if:
	if np.cos(pa_maj*np.pi/180)>0:
		s=+1
	else:
		s=-1
	R*=s
	rnorm=1
	if rmax > 80:
		rnorm=60
		rlabel='$\'$'
	else:
		rlabel='$\'\'$'
	R=R/rnorm
	extimg=extimg/rnorm

	mom1_mdl-=vsys
	mom1-=vsys
	vmin = abs(np.nanmin(mom1_mdl[msk]))
	vmax = abs(np.nanmax(mom1_mdl[msk]))
	max_vel = np.nanmax([vmin,vmax])
	vminv = -(max_vel//10 + 1)*10
	vmaxv = (max_vel//10 + 1)*10


	width, height = 16, 16*(4./10) # width [cm]
	#width, height = 3, 15 # width [cm]
	cm_to_inch = 0.393701 # [inch/cm]
	figWidth = width * cm_to_inch # width [inch]
	figHeight = height * cm_to_inch # width [inch]

	fig = plt.figure(figsize=(figWidth, figHeight), dpi = 300)
	#nrows x ncols
	widths = [0.3,0.3,0.7,1,1]
	heights = [1,1,0.4,1,1]


	gs1 = fig.add_gridspec(nrows=2, ncols=1, left=0.08, right=0.2, top=0.9, bottom=0.2, wspace=0.2)
	gs2 = fig.add_gridspec(nrows=1, ncols=2, left=0.3, right=0.99, wspace=0.2, bottom=0.2)


	#gs2 = gridspec.GridSpec(3, 7, width_ratios=widths)#, height_ratios=heights)
	#gs2.update(left=0.05, right=0.97,top=0.9,bottom=0.15, hspace = 0.0, wspace = 0.0)
	levels=2**np.arange(1,7,1,dtype=float)


	#"""
	pixelconv=1
	bmajconv=bminconv=1
	psf2d=gkernel(pvd_maj_mdl.shape,fwhm=None,bmaj=bmajconv,bmin=bminconv,pixel_scale=pixelconv)
	padded_pvd_maj, cube_slices = data_2N(pvd_maj_mdl, axes=[0, 1])
	padded_pvd_min, _ = data_2N(pvd_min_mdl, axes=[0, 1])
	padded_psf, _ = data_2N(psf2d, axes=[0, 1])

	dft=fftconv_numpy(padded_pvd_maj,padded_psf,threads=2,axes = [0,1])
	pvd_maj_mdl=dft.conv_DFT(cube_slices)
	dft=fftconv_numpy(padded_pvd_min,padded_psf,threads=2,axes = [0,1])
	pvd_min_mdl=dft.conv_DFT(cube_slices)

	padded_pvd_maj, cube_slices = data_2N(pvd_maj, axes=[0, 1])
	padded_pvd_min, _ = data_2N(pvd_min, axes=[0, 1])
	dft=fftconv_numpy(padded_pvd_maj,padded_psf,threads=2,axes = [0,1])
	pvd_maj=dft.conv_DFT(cube_slices)
	dft=fftconv_numpy(padded_pvd_min,padded_psf,threads=2,axes = [0,1])
	pvd_min=dft.conv_DFT(cube_slices)


	# normalize by the rms
	pvd_min_mdl/=rms
	pvd_maj_mdl/=rms
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
	vsysz=vsys + redshift*__c__
	bar_scale_arc,bar_scale_u,unit=bscale(vsysz,xlength,pixel,config)
	bar_scale_arc_norm=bar_scale_arc/rnorm

	ax2=plt.subplot(gs1[0,0])
	broadband=np.nansum(datacube,axis=0)
	broadband[broadband==0]=np.nan
	vmin,vmax=vmin_vmax(broadband)
	norm = colors.LogNorm(vmin=vmin, vmax=vmax)	 if (vmin>0) & (np.log10(vmax/vmin)>1)  else colors.Normalize(vmin=vmin, vmax=vmax)
	im2=ax2.imshow(broadband,norm=norm,cmap=cmap_mom0,aspect='auto',origin='lower',extent=extimg)
	ax2.contour(slit_major, levels =[0.95], colors = "k", alpha = 1, linewidths = 0.5,zorder=10,extent=extimg)
	ax2.contour(slit_minor, levels =[0.95], colors = "k", alpha = 1, linewidths = 0.5,zorder=10,extent=extimg)
	axs(ax2,rotation='horizontal',remove_xyticks=True)
	clb=cb(im2, ax2, labelsize=10, colormap = cmap_mom0, bbox=(-0.25, 0.2, 0.05, 0.7), ticksfontsize=0, ticks = [vmin, vmax], label = "flux", label_pad = -20, colors  = "k",orientation='vertical')
	clb.text(-1,-0.2,round(vmin,1),transform=clb.transAxes,fontsize=8)
	clb.text(-1,1.03,round(vmax,1),transform=clb.transAxes,fontsize=8)

	ax2.text(xmin*(4/5.+1/10),ymax*(7/6)*0.95, '%s${\'\'}$:%s%s'%(bar_scale_arc,bar_scale_u,unit),fontsize=8)
	#ax2.plot([extimg[0]*(4/5.),extimg[0]*(5/6)+bar_scale_arc_norm],[extimg[3]*(7/6),extimg[3]*(7/6)],'k-')


	ax3=plt.subplot(gs1[1,0])
	vmin,vmax=vmin_vmax(mom1,base=25)
	if abs(vmin) < 10 or abs(vmax) < 10:
		vminv,vmaxv=vmin_vmax(mom1,symmetric=True)
	im3=ax3.imshow(mom1,cmap=cmap,aspect='auto',vmin=vminv,vmax=vmaxv,origin='lower',extent=extimg)
	ax3.contour(slit_major, levels =[0.95], colors = "k", alpha = 1, linewidths = 0.5,zorder=10,extent=extimg)
	ax3.contour(slit_minor, levels =[0.95], colors = "k", alpha = 1, linewidths = 0.5,zorder=10,extent=extimg)
	axs(ax3,rotation='horizontal',remove_yticks=True,fontsize_ticklabels=10)
	clb=cb(im3, ax3, labelsize=10, colormap = cmap, bbox=(-0.25, 0.2, 0.05, 0.7), ticksfontsize=0, ticks = [vminv, vmaxv], label = "$\mathrm{V_{LOS}}$/km/s", label_pad = -20, colors  = "k",orientation='vertical')
	clb.text(-2,-0.2,int(vminv),transform=clb.transAxes,fontsize=8)
	clb.text(-2,1.03,int(vmaxv),transform=clb.transAxes,fontsize=8)
	ax3.set_xlabel('$\mathrm{\Delta RA}$ (%s)'%rlabel,fontsize=13,labelpad=1)
	#ax3.text(extimg[0]*(4/5.+1/10),extimg[2]*(5/6)*0.95, '%s${\'\'}$:%s%s'%(bar_scale_arc,bar_scale_u,unit),fontsize=8)
	#ax3.plot([extimg[0]*(4/5.),extimg[0]*(5/6)+bar_scale_arc_norm],[extimg[2]*(5/6),extimg[2]*(5/6)],'k-')



	# PVD major
	vmin,vmax=vmin_vmax(pvd_maj,pmax=99.5)
	if vmin<=0: vmin = 1
	norm = colors.LogNorm(vmin=vmin, vmax=vmax)
	ax0=plt.subplot(gs2[0,0])
	axs(ax0,rotation='horizontal',fontsize_ticklabels=10)
	txt = AnchoredText('$\mathrm{PV_{major}}$', loc="upper left", pad=0.1, borderpad=0, prop={"fontsize":12},zorder=1e4);txt.patch.set_alpha(0.5);ax0.add_artist(txt)
	txt = AnchoredText(f'PA:{pa_maj}$^\circ$', loc="lower right", pad=0.1, borderpad=0, prop={"fontsize":12},zorder=1e4,bbox_to_anchor=(1., 1.), bbox_transform=ax0.transAxes);txt.patch.set_alpha(0);ax0.add_artist(txt)
	ax0.imshow(pvd_maj,norm=norm,cmap=cmap_mom0,origin = "lower",extent=ext0,aspect='auto',alpha=0.7)#,vmin=vmin,vmax=vmax)
	#levels=np.linspace(np.nanmin(pvd_maj_mdl),np.nanmax(pvd_maj_mdl),10)
	cnt=ax0.contour(pvd_maj_mdl,levels=levels,colors='navy', linestyles='solid',zorder=10,extent=ext0,linewidths=1,alpha=1)

	lines = [Line2D([0], [0], color='navy',lw=0.8)];labels=['model']
	ax0.legend(lines,labels,loc='upper left',borderaxespad=0,handlelength=0.6,handletextpad=0.5,frameon=False,fontsize=10,bbox_to_anchor=(0, 1.11), bbox_transform=ax0.transAxes)
	ax0.scatter(R,vt,s=20,marker='s',c='#5ea1ba',edgecolor='k',lw=0.3,zorder=20)
	ax0.scatter(-R,-vt,s=20,marker='s',c='#5ea1ba',edgecolor='k',lw=0.3,zorder=20)
	ax0.plot((ext0[0],ext0[1]),(0,0),"k-",lw=0.5)
	ax0.plot((0,0),(ext0[2],ext0[3]),"k-",lw=0.5)
	ax0.set_ylabel('$\mathrm{V_{LOS}~(km/s)}$',fontsize=13,labelpad=1)
	ax0.set_xlabel(f'r ({rlabel})',fontsize=13,labelpad=1)
	#Nmultiple=50*( (abs(ext1[2])//2) // 50 )


	# PVD minor
	ax1=plt.subplot(gs2[0,1])
	axs(ax1,rotation='horizontal',fontsize_ticklabels=10)
	txt = AnchoredText('$\mathrm{PV_{minor}}$', loc="upper left", pad=0.1, borderpad=0, prop={"fontsize":12},zorder=1e4);txt.patch.set_alpha(0.5);ax1.add_artist(txt)
	txt = AnchoredText(f'PA:{pa_min}$^\circ$', loc="lower right", pad=0.1, borderpad=0, prop={"fontsize":12},zorder=1e4,bbox_to_anchor=(1., 1.), bbox_transform=ax1.transAxes);txt.patch.set_alpha(0);ax1.add_artist(txt)
	ax1.imshow(pvd_min,norm=norm,cmap=cmap_mom0,origin='lower',extent=ext1,aspect='auto',alpha=0.7)#,vmin=vmin,vmax=vmax)
	#levels=np.linspace(np.nanmin(pvd_min_mdl),np.nanmax(pvd_min_mdl),10)
	ax1.contour(pvd_min_mdl,levels=levels,colors='navy', linestyles='solid',zorder=10,extent=ext1,linewidths=1,alpha=1)
	ax1.plot((ext1[0],ext1[1]),(0,0),"k-",lw=0.5)
	ax1.plot((0,0),(ext1[2],ext1[3]),"k-",lw=0.5)
	ax1.set_xlabel(f'r ({rlabel})',fontsize=13,labelpad=1)
	#ax1.set_ylabel('$V\mathrm{_{LOS}~(km/s)}$',fontsize=13,labelpad=1)
	ax1.legend(lines,labels,loc='upper left',borderaxespad=0,handlelength=0.6,handletextpad=0.5,frameon=False, fontsize=10,bbox_to_anchor=(0, 1.11), bbox_transform=ax1.transAxes)


	# plot PSF ellipse ?
	"""
	psf_arc=config_general.getfloat('psf_fwhm',None)
	bmaj_arc=config_general.getfloat('bmaj',None)
	"""
	config_general = config['general']
	eline=config_general.getfloat('eline')
	specres=config_general.getfloat('fwhm_inst',None)


	psf_lsf=PsF_LsF(hdr_cube,config)
	bmaj_arc=psf_lsf.bmaj
	bmin_arc=psf_lsf.bmin
	bpa= psf_lsf.bpa
	psf_arc=psf_lsf.fwhm_psf_arc
	fwhm_kms=psf_lsf.fwhm_inst_kms
	if fwhm_kms is None:
		fwhm_kms=hdr_info.cdelt3_kms


	psf=None
	if psf_arc is not None:
		psf=psf_arc/rnorm
	if bmaj_arc is not None:
		psf=bmaj_arc/rnorm

	if psf is not None and 	fwhm_kms is not None:
		for Axes in [ax0, ax1]:
			#x0,y0=ext0[0]*(5/6.),ext0[2]*(5/6)
			#x,y=drawellipse(x0,y0,fwhm_kms,0,bminor=psf/2.)
			#Axes.plot(x,y,'k-',lw=0.5)

			beam=AnchoredEllipse(Axes.transData, width=psf,height=fwhm_kms, angle=0, loc='lower left',pad=0.2, borderpad=0,frameon=True, zorder=30)
			beam.ellipse.set(color='gray')
			Axes.add_artist(beam)


	for Axes in [ax2, ax3]:
		elipse=drawellipse(xc,yc,bmajor=rmax_norm,pa_deg=pa,eps=eps)
		x,y=elipse[0],elipse[1]#pixel*(elipse[0]-nx/2)/rnorm,pixel*(elipse[1]-ny/2)/rnorm
		Axes.plot(x, y, '-', color = '#393d42',  lw=0.5)

		elipse_mjr=drawellipse(xc,yc,bmajor=0.5*rmax_norm,pa_deg=pa,eps=1)
		x,y=elipse_mjr[0],elipse_mjr[1]#pixel*(elipse_mjr[0]-nx/2)/rnorm,pixel*(elipse_mjr[1]-ny/2)/rnorm
		Axes.plot(x, y, linestyle='--', color = '#393d42',  lw=0.5)

		elipse_mnr=drawellipse(xc,yc,bmajor=0.5*(1-eps)*rmax_norm,pa_deg=pa+90,eps=1)
		x,y=elipse_mnr[0],elipse_mnr[1]#pixel*(elipse_mnr[0]-nx/2)/rnorm,pixel*(elipse_mnr[1]-ny/2)/rnorm
		Axes.plot(x, y, linestyle='--', color = '#393d42',  lw=0.5)


	#ax0.set_xlim(ext0[0],ext0[1])
	#ax1.set_xlim(ext1[0],ext1[1])


	for Axes in [ax2, ax3]:
		Axes.set_xlim(xmin,xmax)
		Axes.set_ylim(ymin,ymax)

	for Axes in [ax0, ax1]:
		if np.any( abs(np.array([ext0[2],ext0[3]]))  > max_vt*(4/3.) ):
			vmin,vmax= -(max_vt+2*fwhm_kms), (max_vt+2*fwhm_kms)
			Axes.set_ylim(vmin,vmax)


	#if Nmultiple>0: ax0.yaxis.set_major_locator(MultipleLocator(Nmultiple))
	#if Nmultiple>0: ax1.yaxis.set_major_locator(MultipleLocator(Nmultiple))


	fig.tight_layout()
	plt.savefig("%sfigures/pvd_%s_model_%s.png"%(out,vmode,galaxy))
	#plt.clf()
	plt.close()
