import numpy as np


def gkernel(l=5, sig=1.,pixel_scale = 1.):

	"""\
	creates gaussian kernel with side length l and a sigma of sig
	"""
	l = l / pixel_scale
	l = int(l)
	sig = sig / pixel_scale

	ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
	xx, yy = np.meshgrid(ax, ax)

	kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / sig**2)

	return kernel / np.nansum(kernel)

