import numpy as np
import matplotlib.pylab as plt
from itertools import product
import logging as logger
#logger = logging.getLogger(__name__)

from src0.constants import __c__
from src0.conv import conv2d,gkernel,gkernel1d
from src0.conv_spec1d import gaussian_filter1d,convolve_sigma
from src0.momtools import mask_wave
from src0.conv_fftw import fftconv,data_2N
from src0.start_messenge import Print

def rmse(data_array):
	data=data_array[np.isfinite(data_array) & (data_array!=0)] 
	mean=np.nanmean(data)
	N=len(data)
	y=(data-mean)**2
	root_ms = np.sqrt(np.nansum(y)/N)
	return root_ms



def mask_cube(data,config,f=5,clip=None):
	Print().status("Estimating RMS")
	config_general = config['general']
	if clip is None:
		clip=config_general.getfloat('clip',6)

	nthreads=config_general.getint('nthreads',1)			
		
	# Following the procedure of Dame 2011
	# https://arxiv.org/pdf/1101.1499

	#(1) rms noise. Use regions without emission.
	cube = np.copy(data)
	#cube[cube==0]=np.nan
	cube[~np.isfinite(cube)]=0	
	[nz,ny,nx]=cube.shape
		
	# select spectra with low signal
	avg2d=np.nanmean(cube, axis=0)
	p5=np.nanpercentile(np.unique(avg2d),f)
	# just for comparason
	p95=np.nanpercentile(np.unique(avg2d),95)
	#print('p5/p95',p5/p95)

	# mask spectra that have low signal (on average)
	msk = (avg2d<p5) & (avg2d!=0)
	c=cube*msk*np.ones(nz)[:,None,None]
	msk=c!=0
	# calculate the rms on the original cube
	rms_ori=rmse(c[msk])

	#(2) calculate smooted cube
	# smooth the cube spectrally and spatially by a factor of 2
	sigma_inst_pix=2
	lsf1d=gkernel1d(nz,sigma_pix=sigma_inst_pix)
	psf2d=gkernel((ny,nx),2,pixel_scale=1)
	psf3d_1 = psf2d * lsf1d[:, None, None]

	padded_cube, cube_slices = data_2N(cube, axes=[0, 1, 2])
	padded_psf, psf_slices = data_2N(psf3d_1, axes=[0, 1, 2])

	a=fftconv(padded_cube,padded_psf,threads=nthreads)
	cube_smooth=a.convolve_3d_same(cube_slices)
	
	#cube_smooth=convolve_3d_same(cube, psf3d_1)[0]
	c=cube_smooth*msk*np.ones(nz)[:,None,None]
	# calculate the rms on the smooth cube
	rms_sm=rmse(c[msk])
	clip_level=rms_sm*clip

	Print().out("Cube RMS",rms_sm)				
	# if clip is too high, try to reduce it.
	#if clip_level/p95>0.2:
	#	logger.warning(f'Clip level of {clip} seems to be high. I will try to reduce it.')
	#	clip_level=p95*1e-2
			
	msk_cube=np.zeros_like(cube,dtype=bool)
	msk_cube[cube_smooth>clip_level]=True
	msk_cube2=np.copy(msk_cube)
	
	for i,j,k in product(np.arange(nx),np.arange(ny),np.arange(nz)):
		if msk_cube[k,j,i]:
			msk_cube2[k-1:k+2,j-1:j+2,i-1:i+2]=True


	#calculate rms of the clean cube
	rms_clean=rmse(cube[msk])	
	return msk_cube2,rms_clean


	
	
def ecube(cube,box=5):
	# estimate the error in every pixel based on the
	# negative values on the spectra along the spectral axis.
	# Each spectrum has a unique error.
	neg_msk=cube<0
	neg_cube=abs(cube*neg_msk)
	neg_cube[neg_cube==0]=np.nan
	avg_z=np.nanmedian(neg_cube, axis=0)
	
	if not np.nansum(avg_z):
		avg_z=np.ones_like(avg_z)

	return 	avg_z	
	
	
	

def cstats(cube,f=10):
	[nz,ny,nx]=cube.shape
	"""
	# smooth the cube spectrally and spatially
	sigma_inst_pix=2
	
	lsf3d=np.ones((ny,nx))*gkernel1d(nz,sigma_pix=sigma_inst_pix)[:,None,None]
	cube_mod_lsf=convolve_1d(cube, lsf3d)[0]
	cube_mod_conv=cube_mod_lsf
	
	lsf1d=gkernel1d(nz,sigma_pix=sigma_inst_pix)
	psf2d=gkernel((ny,nx),2,pixel_scale=1)	
	psf3d_1 = psf2d * lsf1d[:, None, None]
	cube_mod_conv3d=convolve_3d_same(cube, psf3d_1)[0]
	cube_mod_conv=cube_mod_conv3d
	"""
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
	
	
	



