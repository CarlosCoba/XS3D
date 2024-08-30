import numpy as np
from scipy.ndimage import convolve1d

#Taken from pypipe3d
def convolve_sigma(flux, sigma, side_box=None):
	"""
	Convolves `flux` using a Gaussian-kernel with standard deviation `sigma`.
	The kernel have dimension 2*`side_box` + 1.

	Parameters
	----------
	flux : array like
		Spectrum to be convolved.
	sigma : float
		Sigma of the Gaussian-kernel.
	N_side: float
		Will define the range size of the Gaussian-kernel.

	Returns
	-------
	array like
		Convolved `flux` by the weights defined by the Gaussian-kernel.
	"""
	kernel_function = lambda x: np.exp(-0.5*(((x - side_box)/sigma)**2))
	N = 2*side_box + 1
	kernel = np.array(list(map(kernel_function, np.arange(N))))
	norm = kernel.sum()
	kernel = kernel/norm
	return convolve1d(flux, kernel, mode='nearest')




#
# Taken from ppxf/ppxf_util.py
#

def gaussian_filter1d(spec, sig):
	"""
	Convolve a spectrum by a Gaussian with different sigma for every
	pixel, given by the vector "sigma" with the same size as "spec".
	If all sigma are the same this routine produces the same output as
	scipy.ndimage.gaussian_filter1d, except for the border treatment.
	Here the first/last p pixels are filled with zeros.
	When creating  template library for SDSS data, this implementation
	is 60x faster than the naive loop over pixels.
	"""

	sig = sig.clip(0.01)  # forces zero sigmas to have 0.01 pixels
	p = int(np.ceil(np.max(3*sig)))
	m = 2*p + 1  # kernel size
	x2 = np.linspace(-p, p, m)**2

	n = spec.size
	a = np.zeros((m, n))
	for j in range(m):   # Loop over the small size of the kernel
		a[j, p:-p] = spec[j:n-m+j+1]

	gau = np.exp(-x2[:, None]/(2*sig**2))
	gau /= np.sum(gau, 0)[None, :]  # Normalize kernel
	conv_spectrum = np.sum(a*gau, 0)

	return conv_spectrum

