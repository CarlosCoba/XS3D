import numpy as np
import numpy.fft as fft
import pyfftw

pyfftw.config.PLANNER_EFFORT = 'FFTW_ESTIMATE'
from pyfftw.interfaces.numpy_fft import rfftn, irfftn, fftshift
pyfftw.interfaces.cache.enable()

def convolve_3d_same(cube, psf, compute_fourier=True):
	"""
	Convolve a 3D cube with PSF & LSF.
	PSF can be the PSF data or its Fourier transform.
	if compute_fourier then compute the fft transform of the PSF.
	if False then assumes that the fft is given..

	This convolution has edge effects (and is slower when using numpy than pyfftw).

	cube: The cube we want to convolve
	psf: The Point Spread Function or its Fast Fourier Transform
	"""

	# Pad to power of 2
	padded_cube, cube_slices = padding(cube, axes=[0, 1, 2])

	size = np.array(np.shape(padded_cube)[slice(0, 3)])

	if compute_fourier:
		padded_psf, psf_slices = padding(psf, axes=[0, 1, 2])
		fft_psf = np.fft.rfftn(padded_psf, s=size, axes=[0, 1, 2])
	else:
		fft_psf = psf

	fft_img = np.fft.rfftn(padded_cube, s=size, axes=[0, 1, 2])

	# Convolution
	fft_cube = np.real(np.fft.fftshift(np.fft.irfftn(fft_img * fft_psf, s=size, axes=[0, 1, 2]), axes=[0, 1, 2]))

	# Remove padding
	cube_conv = fft_cube[cube_slices]

	return cube_conv#, fft_psf


def convolve_3d_xy(cube, psf, compute_fourier=True):
	"""
	Convolve 3D cube along spatial directions only,
	using provided Point Spread Function.
	"""

	# Compute needed padding
	cubep, boxcube = padding(cube, axes=[1, 2])

	size = np.array(np.shape(cubep)[slice(1, 3)])

	if compute_fourier:
		psfp, boxpsf = padding(psf, axes=[1, 2])
		fftpsf = np.fft.rfftn(psfp, s=size, axes=[1, 2])

	else:
		fftpsf = psf

	fftimg = np.fft.rfftn(cubep, s=size, axes=[1, 2])

	#Convolution
	fft_cube = np.fft.fftshift(np.fft.irfftn(fftimg * fftpsf, s=size, axes=[1, 2]), axes=[1, 2]).real

	# Remove padding
	cube_conv = fft_cube[boxcube]

	return cube_conv#, fftpsf


def convolve_1d(data, psf, compute_fourier=True, axis=0):
	"""
	Convolve data with PSF only along one dimension specified by axis (default: 0)
	PSF can be the PSF data or its Fourier transform
	if compute_fourier then compute the fft transform of the PSF.
	if False then assumes that the fft is given.
	"""

	axis = np.array([axis])

	# Compute needed padding
	cubep, boxcube = padding(data, axes=axis)
	# Get the size of the axis
	size = np.array(np.shape(cubep)[slice(axis[0], axis[0] + 1)])

	if compute_fourier:
		psfp, boxpsf = padding(psf, axes=axis)
		fftpsf = np.fft.rfftn(psfp, s=size, axes=axis)
	else:
		fftpsf = psf

	fftimg = np.fft.rfftn(cubep, s=size, axes=axis)

	# Convolution
	fft_cube = np.fft.fftshift(np.fft.irfftn(fftimg * fftpsf, s=size, axes=axis), axes=axis).real

	# Remove padding
	cube_conv = fft_cube[boxcube]

	return cube_conv#, fftpsf


def padding(cube, axes=None):
		"""
		Computes padding needed for a cube to make sure it has
		a power of 2 shape along dimensions of passed axes (default [0,1])
		Returns padded cube and cube slices,
		which are the indices of the actual data in the padded cube.
		"""

		if axes is None:
			axes = [0, 1]

		# Compute padding size for each axis
		old_shape = np.shape(cube)
		new_shape = np.array(old_shape)

		for axis in axes:
			zdim = cube.shape[axis]
			s = np.binary_repr(zdim - 1)
			s = s[:-1] + '0'
			new_shape[axis] = 2 ** len(s)

		cube_padded = np.zeros(new_shape)
		#cube_slices = np.empty(len(old_shape), slice).tolist()
		cube_slices = [None for i in range(len(old_shape))]

		for i, v in enumerate(old_shape):
			cube_slices[i] = slice(0, old_shape[i])

		for axis in axes:
			diff = new_shape[axis] - old_shape[axis]
			if (diff & 1):
				half = diff // 2 + 1
			else:
				half = diff // 2
			cube_slices[axis] = slice(half, old_shape[axis] + half)

		cube_slices = tuple(cube_slices)
		# Copy cube contents into padded cube
		cube_padded[cube_slices] = cube.copy()
		#cube_padded=cube_padded.astype(np.float64)
		return cube_padded, cube_slices
