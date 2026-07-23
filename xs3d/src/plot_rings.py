import numpy as np
import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
from matplotlib import gridspec
import matplotlib.colors as colors
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
from itertools import product
from matplotlib.patches import Polygon
from matplotlib.transforms import Affine2D	
from matplotlib.offsetbox import AnchoredText
import matplotlib.ticker as ticker
from matplotlib.transforms import blended_transform_factory

from .conv import conv2d,gkernel,gkernel1d
from .conv_fftw import fftconv,data_2N,fftconv_numpy
from .axes_params import axes_ambient as axs
from .cbar import colorbar as cb
from .colormaps_CLC import vel_map
from .barscale import bscale
from .ellipse import drawellipse, drawrectangle
from .psf_lsf import PsF_LsF
from .constants import __c__
from .rings_to_sky  import project_ring, project_ring_edges, overlay_rings

#params =   {'text.usetex' : True }
#plt.rcParams.update(params)

from .pixel_params import inc_2_eps, eps_2_inc,e_eps2e_inc

def vmin_vmax(data,pmin=2,pmax=99,base=None,symmetric =False,mask=None):
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
	data[data==0]=np.nan
	return data

cmap = vel_map()
cmap_mom0=vel_map('mom0')

def plot_rings_sky(galaxy,momms_obs,best_rings,const,vmode,psf_lsf,hdr_info,config,rms,out):

	pixel = hdr_info.pix_arcs
	nx = hdr_info.nx
	ny = hdr_info.ny
	nz = hdr_info.nz
	dv = hdr_info.dv
	dx_arcsec = pixel
				

	mom0,mom1,mom2=momms_obs
	mom0[~np.isfinite(mom0)]=0
	mask=np.isfinite(mom0)
	
	pixel_conv=1
	bmaj = 2
	psf2d=gkernel(mom0.shape,fwhm=bmaj,pixel_scale=pixel_conv)
	padded_mom0, mom0_slices = data_2N(mom0, axes=[0, 1])
	padded_psf, _ = data_2N(psf2d, axes=[0, 1])

	dft=fftconv_numpy(padded_mom0,padded_psf,threads=2,axes = [0,1])
	mom0=dft.conv_DFT(mom0_slices)
		
	mom0_norm=mom0/(rms*nz*dv)
	
	scalar_fields = ["v_sys", "inc", "pa","x_center", "y_center","phi_bar", "rmax"]
	[vsys,inc,pa,xc,yc,phi_bar,rmax] = [const[k] for k in scalar_fields]		
	eps=inc_2_eps(inc)
	
	# shift the extent to the kinematic centre
	ext=np.dot([-xc,nx-xc,-yc,ny-yc],pixel)

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

	width, height = 10, 10 # width [cm]
	cm_to_inch = 0.393701 # [inch/cm]
	figWidth = width * cm_to_inch # width [inch]
	figHeight = height * cm_to_inch # width [inch]

	fig = plt.figure(figsize=(figWidth, figHeight), dpi = 300)
	#nrows x ncols
	gs2 = gridspec.GridSpec(1, 1)
	gs2.update(hspace = 0, wspace = 0, right=0.99, bottom=0.01, top=0.99, left = 0.01 )

	ax0 = fig.add_subplot(gs2[0, 0])


	ax0.plot(xc, yc, marker='+', color = 'black', markeredgewidth=1, zorder=100)
	max_f = np.max(mom0_norm)
	min_f = np.min(mom0_norm[mom0_norm>0])
	
	lmax = np.log10(max_f)
	lmin = np.log10(0.01)	
	levels = np.logspace(lmin,lmax,num=8,endpoint=True)

	#levels=2**np.arange(-2,7,1,dtype=float)
	indices = np.digitize(mom0_norm.ravel(), levels)
	indices_ = np.unique(indices)[::-1]
	
	# Fetch the colormap (e.g., 'plasma', 'viridis', 'jet')
	cmap = plt.colormaps['copper_r']
	cmap = vel_map('pvd')

	# Generate colors at regular intervals spanning 0 to 1
	colors = cmap(np.linspace(0, 1, len(indices_)))
	cnt=ax0.contourf(mom0_norm, levels=levels,  linestyles='solid', zorder=-10, linewidths=2,alpha=0.6, colors = colors)

	ax0.set_xticks([])
	ax0.set_yticks([])	
	

	# Bare locus lines
	#overlay_rings(ax0, best_rings, dx_arcsec, mode='locus',receding_color='tomato', approaching_color='lime',spine_lw=2.0)

	# Filled band showing the ring width
	overlay_rings(ax0, best_rings, dx_arcsec, mode='both',col_fill='#314F40', fill_alpha=0.2, show_major_axis = True)
	plt.legend()                            

	ax0.set_facecolor('white')

	#fig.tight_layout()
	plt.savefig("%sfigures/rings_%s_model_%s.png"%(out,vmode,galaxy))
	#plt.clf()
	plt.close()
	
