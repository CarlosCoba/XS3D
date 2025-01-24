import numpy as np
from astropy.io import fits
from scipy.signal import convolve2d 
from astropy.convolution import convolve,convolve_fft
import matplotlib.pylab as plt
from .constants import __sigma_2_FWHM__,__FWHM_2_sigma__
from .ellipse import drawellipse

def gkernel(shape,fwhm,bmaj=None,bmin=None,bpa = 0,pixel_scale = 1.):
	# INPUTS
	# shape - [ny,nx] shape of the array
	# bmaj  - major axis of psf in arcsec
	# bmin  - minor axis of psf in arcsec
	# fwhm  - fwhm of the psf in arcsec
	# bpa	- position angle of the beam in deg.
	# pixel size in arcsec/pix


	Bpa=bpa*np.pi/180
	Bmaj = bmaj*__FWHM_2_sigma__ if bmaj is not None else fwhm
	Bmin = bmin*__FWHM_2_sigma__ if bmin is not None else fwhm
		
	#if bmaj==None: bmaj=fwhm*__FWHM_2_sigma__
	#if bmin==None: bmin=fwhm*__FWHM_2_sigma__

	[ny,nx]=shape
	x0,y0 = nx/2., ny/2.
	y, x = np.indices(shape)

	q = Bmin/Bmaj
	eps = (1-q)
		
	X = (- (x-x0)*np.sin(Bpa) + (y-y0)*np.cos(Bpa))
	Y = (- (x-x0)*np.cos(Bpa) - (y-y0)*np.sin(Bpa))
	r2= (X**2+(Y/(1-eps))**2).astype('float64')

	# if only fwhm is passed.
	if fwhm is not None:	
		sigma_arc=fwhm*__FWHM_2_sigma__		
		sig_pix=sigma_arc/pixel_scale
		sig_pix2=(sig_pix)**2
		r2=(r2/sig_pix2)
	# if BMAJ and BMIN passed.
	if fwhm is None:
		sig_pix2=(Bmaj/pixel_scale)**2		
		r2=(r2/sig_pix2)

	r2=r2.astype(np.float64)
	kernel = np.exp(-0.5*r2)
	kernel = kernel/np.sum(kernel)
	
	plot=0
	if plot:
		bmjr,bmnr=Bmaj/pixel_scale,Bmin/pixel_scale
		x,y=drawellipse(x0,y0,bmjr*4,Bpa,bmnr*4)
		plt.plot(x,y,'k-')
		x,y=drawellipse(x0,y0,bmjr,Bpa,bmnr)
		plt.plot(x,y,'r-')			
		plt.imshow((kernel), origin='lower');plt.show()
	
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
	
	

