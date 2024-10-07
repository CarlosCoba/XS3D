import numpy as np
from astropy.io import fits
from scipy.signal import convolve2d 
from astropy.convolution import convolve,convolve_fft
import matplotlib.pylab as plt
from src0.constants import __sigma_2_FWHM__,__FWHM_2_sigma__

def gkernel(shape, fwhm, bmaj=None,bmin=None, pa0 = 0, pixel_scale = 1.):
	# INPUTS
	# shape - [ny,nx] shape of the array
	# bmaj  - major axis of psf in arcsec
	# bmin  - minor axis of psf in arcsec
	# fwhm  - fwhm of the psf in arcsec
	# pa	- position angle of the beam in deg.
	# pixel size in arcsec/pix
	
	pa=pa0*np.pi/180
	if bmaj==None: bmaj=fwhm
	if bmin==None: bmin=fwhm
	

	[ny,nx]=shape
	x0,y0 = nx/2., ny/2.
	y, x = np.indices(shape)

	#r2 = np.square(x-x0) + np.square(y-y0)
	#kernel = np.exp(-0.5 * (r2) / sig_pix**2)
	q = bmin/bmaj
	eps = (1-q)
		
	X = (- (x-x0)*np.sin(pa) + (y-y0)*np.cos(pa))
	Y = (- (x-x0)*np.cos(pa) - (y-y0)*np.sin(pa))
	r2= (X**2+(Y/(1-eps))**2).astype('float64')

	if fwhm is not None:	
		sigma_arc = fwhm*__FWHM_2_sigma__		
		sig_pix = sigma_arc/ pixel_scale
		sig_pix2=(sig_pix)**2
		r2=r2/sig_pix2

	'''
	if fwhm is None:
		bmin_pix=bmin/pixel_scale
		rx2= X**2/bmin_pix**2
		bmaj_pix=bmaj/pixel_scale
		ry2=(Y/(1-eps))**2/bmaj_pix**2
		r2=rx2+ry2
	'''
	r2=r2.astype(np.float64)
	kernel = np.exp(-0.5*r2)
	kernel = kernel/np.sum(kernel)
	
	return kernel
	



def conv2d(image, fwhm, pixel_scale):
	""" INPUT
	2D image
	fwhm resolution in arcsec
	kernel size in arcsec """

	[ny,nx] = image.shape
	image_copy = np.copy(image)


	extend = np.zeros((3*ny,3*nx))
	gauss_kern = gkernel(extend.shape,fwhm,pixel_scale=pixel_scale)
	extend[ny:2*ny,nx:2*nx] = image_copy
	
	img_conv = convolve_fft(extend, gauss_kern, mask = extend == 0 )
	model_conv = img_conv[ny:2*ny,nx:2*nx]
	model_conv[image == 0]  = 0
	


	extend = np.zeros((ny,nx))
	gauss_kern = gkernel(extend.shape,fwhm,pixel_scale=pixel_scale)	
	model_conv=convolve_2d(image,gauss_kern)[0]
	return model_conv
	
	
	
def gaussian(x, mu, sigma):
		"""
		Non-normalized gaussian function.

		x : float|numpy.ndarray
			Input value(s)
		mu : float
			Position of the peak on the x-axis
		sigma : float
			Standard deviation

		:rtype: Float value(s) after transformation, of the same shape as input x.
		"""
		return np.exp((x - mu) ** 2 / (-2. * sigma ** 2))

	
	
def gkernel1d(nz,sigma_pix=None,fwhm_pix=None):
	# Std deviation from FWHM

	if sigma_pix is not None:
		sigma=sigma_pix
	if fwhm_pix is not None:
		sigma = fwhm_pix * __FWHM_2_sigma__
	# Resulting vector shape
	depth = nz
	# Assymmetric range around 0
	zo = (depth - 1) / 2 - (depth % 2 - 1) / 2
	z_range = np.arange(depth) - zo
	# Compute gaussian (we assume peak is at 0, ie. µ=0)
	lsf_1d = gaussian(z_range, 0, sigma)
	# Normalize and serve
	return lsf_1d / lsf_1d.sum()
	
	

