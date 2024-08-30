import numpy as np
from astropy.convolution import convolve,convolve_fft


def gkernel(l=5, sig=1.,pixel_scale = 1.):

	"""
	creates gaussian kernel with side length l and a sigma of sig
	"""
	l = l / pixel_scale
	l = int(l)
	sig = sig / pixel_scale

	ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
	xx, yy = np.meshgrid(ax, ax)

	kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / sig**2)

	return kernel / np.nansum(kernel)



def deconv_2D(image, psf, kernel_size, pixel_scale):
	""" INPUT
	2D image
	psf resolution in arcsec
	kernel size in arcsec """


	[ny,nx] = image.shape
	image_copy = np.copy(image)
	gauss_kern = gkernel(l=kernel_size,sig=psf,pixel_scale=pixel_scale)

	extend = np.zeros((3*ny,3*nx))
	extend[ny:2*ny,nx:2*nx] = image_copy
	img_conv = convolve_fft(extend, gauss_kern, mask = extend == 0 )
	model_conv = img_conv[ny:2*ny,nx:2*nx]
	model_conv[image == 0]  = 0

	return model_conv

