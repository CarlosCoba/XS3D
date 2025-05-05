import numpy as np
import matplotlib.pylab as plt
from itertools import product
import logging as logger
#logger = logging.getLogger(__name__)

from .constants import __c__
from .conv import conv2d,gkernel,gkernel1d
from .conv_spec1d import gaussian_filter1d,convolve_sigma
from .momtools import mask_wave
from .conv_fftw import fftconv,data_2N
from .start_messenge import Print
from .tools_fits import get_fits_data
from .utils import vparabola3D
from .nan_percentile import nan_percentile # this is much faster than nunmpy

def rmse(data_array):
	data=data_array[np.isfinite(data_array) & (data_array!=0)]
	mean=np.nanmean(data)
	N=len(data)
	y=(data-mean)**2
	root_ms = np.sqrt(np.nansum(y)/N)
	root_ms=0 if ~np.isfinite(root_ms) else root_ms
	return root_ms


def mask_cube(data,config,hdr,f=5,clip=None,msk_user=None):
	Print().status("Estimating RMS")
	config_general = config['general']
	config_others = config['others']
	if clip is None:
		clip=config_others.getfloat('clip',6)

	nthreads=config_general.getint('nthreads',1)
	dv=config_others.getint('dv',4)
	ds=config_others.getint('ds',2)

	# Following the procedure of Dame 2011
	# https://arxiv.org/pdf/1101.1499

	#(1) rms noise.
	cube = np.copy(data)
	isnan=np.isfinite(cube)
	cube[~isnan]=0
	iszero=cube!=0
	noise_msk=cube<0
	[nz,ny,nx]=cube.shape

	# select spectra with low signal
	avg2d=np.nansum(cube, axis=0)

	# mask spectra that have low signal (on average)
	msk_signal = (avg2d>0)
	if msk_user!=None:
		msk_usr=get_fits_data(msk_user).astype(bool)
	else:
		msk_usr=np.ones((ny,nx)).astype(bool)


	noise=cube*noise_msk.astype(float)
	noisep=abs(noise)
	noise_flat=noisep[noise_msk]
	noisep[noisep==0]=np.nan


	#median absolute deviation from the mean on the noise cube
	#mad_map=abs(noisep-np.nanpercentile(noisep,50,axis=0))
	mad_map=abs(noisep-nan_percentile(noisep,50))
	mad_map*=(1/noise_msk)
	#noise per spectrum
	#sigma_map=np.nanpercentile(mad_map,50,axis=0)*1.4826
	sigma_map=nan_percentile(mad_map,50)*1.4826


	#mad_global=abs(noise_flat-np.nanpercentile(noise_flat,50,axis=0))
	mad_global=abs(noise_flat-nan_percentile(noise_flat,50))
	#rms_global=np.nanpercentile(mad_global,50,axis=0)*1.4826
	rms_global=nan_percentile(mad_global,50)*1.4826

	# calculate the rms on each channel of the original cube
	rms_channels=np.array([rmse(cube[k]) for k in range(nz) ])

	rms_mean=np.median(rms_channels)
	rms_cube = rms_global if rms_global !=0 else rms_mean
	Print().out("Original cube RMS",round(rms_cube,10))

	#(2) calculate smooted cube
	# smooth the cube spectrally and spatially by dv and ds pixels
	sigma_inst_pix_spec=dv
	sigma_inst_pix_spat=ds

	if dv!=0 or ds!=0:
		psd2d=np.ones((ny,nx))
		lsf1d=np.ones(nz)
		if dv!=0:
			lsf1d=gkernel1d(nz,sigma_pix=sigma_inst_pix_spec)
		if ds!=0:
			psf2d=gkernel((ny,nx),sigma_inst_pix_spat,pixel_scale=1)

		psf3d_1 = psf2d * lsf1d[:, None, None]
		if dv!=0 and ds !=0:
			axes=[0,1,2]
		if dv!=0 and ds==0:
			axes=[0]
		if dv==0 and ds!=0:
			axes=[1,2]

		padded_cube, cube_slices = data_2N(cube, axes=axes)
		padded_psf, psf_slices = data_2N(psf3d_1, axes=axes)

		dft=fftconv(padded_cube,padded_psf,threads=nthreads)
		cube_smooth=dft.conv_DFT(cube_slices)
		#Do not forget to recover the zeros
		cube_smooth*=iszero


		msk_neg=cube_smooth<0
		#rms per channel on the smoothed cube
		cube_smooth_neg=cube_smooth*msk_neg
		rms_channels=np.array([rmse(cube_smooth_neg[k]) for k in range(nz) ])
		#avg rms
		rms_channel=np.median(rms_channels)



		noisep=abs(cube_smooth_neg)
		noisep=noisep[noisep!=0]

		#median absolute deviation from the mean on the smooth cube
		#mad_sm=abs(noisep-np.nanpercentile(noisep,50,axis=0))
		mad_sm=abs(noisep-nan_percentile(noisep,50))
		#sigma_sm=np.nanpercentile(mad_sm,50,axis=0)*1.4826
		sigma_sm=nan_percentile(mad_sm,50)*1.4826
		Print().out("Smoothed cube RMS",round(sigma_sm,10))

		if ~np.isfinite(sigma_sm) or sigma_sm==0:
			Print().status("The input cube does not contain a proper noise ?")
			sigma_sm=rms_cube
			Print().status("Changing RMS to original cube value")

		#the rms on the smoothed cube:
		global_rmse=sigma_sm
		#the rms that will be passed
		rms_cube = global_rmse*clip

		msk_rms=(cube_smooth) > global_rmse*clip

		msk_cube=np.copy(msk_rms)
		if ds!=0 and dv!=0:
			for i,j,k in product(np.arange(nx),np.arange(ny),np.arange(nz)):
				if msk_rms[k,j,i] :
					dS=sigma_inst_pix_spat//2
					dV=sigma_inst_pix_spec//2
					msk_cube[k-dV:k+dV+1,j-dS:j+dS+1,i-dS:i+dS+1]=True

	else:
		msk_cube=cube > rms_cube*clip

	# apply the user mask and the SN msk
	msk_cube*=(msk_usr)

	plot=0
	if plot:
		fig,ax=plt.subplots(1,1)
		xc,yc=26,52
		ori=cube[:,yc,xc]
		sm=cube_smooth[:,yc,xc]
		ori_msk=(cube*(msk_cube))[:,yc,xc]
		ax.plot(np.arange(nz), np.ones(nz)*(global_rmse*clip) , 'g--', lw =2, label = 'rms*clip')
		ax.plot(np.arange(nz), np.ones(nz)*sigma_map[yc][xc] , 'k--', lw =3, label='local noise')
		ax.plot(np.arange(nz), sm , 'y-', lw =5, label = 'smoothed')
		ax.plot(np.arange(nz), sm*(sm<0) , 'b-', lw =5, label = 'smooth -')
		ax.plot(np.arange(nz), ori_msk , 'r-', lw =5, label='observed-msked')
		ax.plot(np.arange(nz), ori , 'k-', lw = 1, label = 'observed');ax.legend();plt.show()


	# calculate the peak velocity using the smoothed cube
	vpeak=config_others.getboolean('vpeak',False)
	if vpeak:
		wave_cover_kms=hdr.wave_kms
		vpeak2D=vparabola3D(cube*msk_cube,wave_cover_kms)
	else:
		vpeak2D=None
	return msk_cube,rms_cube,vpeak2D








def ecube(cube,rms,box=5):
	[nz,ny,nx]=cube.shape
	# estimate the error in every channel based on the
	# negative values on the spectra along the spectral axis.

	noise_cube=cube<0
	noise=cube*noise_cube
	# calculate the rms on each channel of the original cube
	rms_channels=np.array([rmse(cube[k]) for k in range(nz) ])
	# calculate the rms on each channel of the noise cube
	rms_channels_neg=np.array([rmse(noise[k]) for k in range(nz) ])
	msk=rms_channels_neg==0

	rms_vz = rms_channels_neg
	rms_vz[msk]=rms_channels[msk]

	errorcube=np.ones_like(cube)*(rms_vz[:,None,None])
	#errorcube=np.sqrt(rms**2 + error2D**2) + errorcube

	return 	errorcube




def cstats(cube,f=10):
	[nz,ny,nx]=cube.shape

	cube_mod_conv=np.copy(cube)
	cube_mod_conv[cube_mod_conv==0]=np.nan
	# select spectra with low signal

	avg2d=np.nanmean(cube_mod_conv, axis=0)
	p5=np.nanpercentile(avg2d,f)
	#plt.imshow(avg2d);plt.show()

	# mask spectra that have low SN (on average)
	msk = avg2d<p5
	c=cube_mod_conv*msk*np.ones(nz)[:,None,None]
	msk=c!=0
	# calculate the rms of those line-less spectra
	rms=rmse(c[msk])
	m1=avg2d>2*rms
	return m1*np.ones(nz)[:,None,None],rms



from scipy.sparse.linalg import spsolve
from scipy.linalg import cholesky
# from scikits.sparse.cholmod import cholesky
from scipy import sparse
from scipy.stats import norm



def als(y, lam=1e6, p=0.1, itermax=10):
	r"""
	Implements an Asymmetric Least Squares Smoothing
	baseline correction algorithm (P. Eilers, H. Boelens 2005)

	Baseline Correction with Asymmetric Least Squares Smoothing
	based on https://github.com/vicngtor/BaySpecPlots

	Baseline Correction with Asymmetric Least Squares Smoothing
	Paul H. C. Eilers and Hans F.M. Boelens
	October 21, 2005

	Description from the original documentation:

	Most baseline problems in instrumental methods are characterized by a smooth
	baseline and a superimposed signal that carries the analytical information: a series
	of peaks that are either all positive or all negative. We combine a smoother
	with asymmetric weighting of deviations from the (smooth) trend get an effective
	baseline estimator. It is easy to use, fast and keeps the analytical peak signal intact.
	No prior information about peak shapes or baseline (polynomial) is needed
	by the method. The performance is illustrated by simulation and applications to
	real data.


	Inputs:
		y:
			input data (i.e. spectrum)
		lam:
			parameter that can be adjusted by user. The larger lambda is,
			the smoother the resulting background, z
		p:
			wheighting deviations. 0.5 = symmetric, <0.5: negative
			deviations are stronger suppressed
		itermax:
			number of iterations to perform
	Output:
		the fitted background vector

	"""
	L = len(y)
	#D = sparse.csc_matrix(np.diff(np.eye(L), 2))
	D = sparse.eye(L, format='csc')
	D = D[1:] - D[:-1]  # numpy.diff( ,2) does not work with sparse matrix. This is a workaround.
	D = D[1:] - D[:-1]
	D = D.T
	w = np.ones(L)
	for i in range(itermax):
		W = sparse.diags(w, 0, shape=(L, L))
		Z = W + lam * D.dot(D.T)
		z = spsolve(Z, w * y)
		w = p * (y > z) + (1 - p) * (y < z)
	return z



def baselinecor(cube,config):
	config_general = config['general']
	baseline_cor=config_general.getboolean('baseline', False)
	if baseline_cor:
		basecube=np.zeros_like(cube)
		[nz,ny,nx]=cube.shape
		for i,j in product(np.arange(nx),np.arange(ny)):
			flux=cube[:,j,i]
			if np.sum(flux)!=0:
				basecube[:,j,i]=als(flux)
		return cube-basecube,basecube
	else:
		return cube,None
