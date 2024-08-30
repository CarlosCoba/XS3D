import numpy as np
import numpy.fft as fft
import pyfftw

pyfftw.config.PLANNER_EFFORT = 'FFTW_ESTIMATE'
from pyfftw.interfaces.numpy_fft import rfftn, irfftn, fftshift
pyfftw.interfaces.cache.enable()

class fftconv:
	def __init__(self,cube,psf,threads,axes=[0,1,2],slice_cube=slice(0, 3)):
		self.cube=cube
		self.psf=psf
		self.axes=axes
		self.slice=slice_cube
		shape = (np.shape(cube)[self.slice])
		# the complex
		if np.iscomplexobj(cube) and np.iscomplexobj(psf):
			self.fft_cube = pyfftw.builders.fftn(cube, s=shape, threads=threads,axes=axes,auto_align_input=False, auto_contiguous=False,avoid_copy=True)
			self.fft_psf = pyfftw.builders.fftn(psf, s=shape, threads=threads,axes=axes,auto_align_input=False, auto_contiguous=False,avoid_copy=True)
			self.ifft_obj = pyfftw.builders.ifftn(self.fft_cube.get_output_array(), s=shape,threads=threads,axes=axes,auto_align_input=False, auto_contiguous=False,avoid_copy=True)                
        # real
		else:
			self.fft_cube = pyfftw.builders.rfftn(cube, s=shape, threads=threads,axes=axes,auto_align_input=False, auto_contiguous=False,avoid_copy=True)
			self.fft_psf = pyfftw.builders.rfftn(psf, s=shape, threads=threads,axes=axes,auto_align_input=False, auto_contiguous=False,avoid_copy=True)
			self.ifft_obj = pyfftw.builders.irfftn(self.fft_cube.get_output_array(), s=shape,threads=threads,axes=axes,auto_align_input=False, auto_contiguous=False,avoid_copy=True)        
                
                



	def convolve_3d_same(self,cube_slices):
		# Pad to power of 2
		#padded_cube, cube_slices = padding(cube, axes=[0, 1, 2])
		#padded_psf, psf_slices = padding(psf, axes=[0, 1, 2])

		fft_padded_cube0 = self.fft_cube(self.cube)
		fft_padded_psf0 = self.fft_psf(self.psf)
		ret= self.ifft_obj(fft_padded_cube0 * fft_padded_psf0)
		fft_cube = np.real(np.fft.fftshift(ret,axes=self.axes))
		
		# Remove padding
		cube_conv = fft_cube[cube_slices]
		return cube_conv
				
	def convolve_3d_xy(self,cube_slices):
		return self.convolve_3d_same(cube_slices)	
	

	def convolve_1d(self,cube_slices):
		return self.convolve_3d_same(cube_slices)	



		
#####################################################################33
"""
To optimize the FFT we need to make sure that each dimension of the datacube
has a lenght of 2**N. This will necesesaerly increase the actual dimmensions
of the datacube but will make the FFT much faster. 
"""

#nx=2**N --> log2(nx)/log(2)=N
def data_2N(data,axes=None):
	dims=np.shape(data)
	naxes=len(dims)
	if axes is None:
		axes=list(np.arange(naxes))
		
	nwshape=np.array(dims)
	for axis in axes:
		nx=dims[axis]
		check=np.log2(nx)
		if check % 1 == 0:
			nxnew=nx
		else:
			f=np.log2(nx)/np.log2(2)
			N=int(f)+1
			nxnew=2**N
		nwshape[axis]=nxnew
			
	newcube=np.zeros(nwshape)
	nwshape=newcube.shape
	# free array space
	free=nwshape-np.array(dims)
	check_pair=free % 2 # -> zeros if all pairs
	free+=check_pair
	halfside=(free/2).astype(int)		
	
	
	# set default slices
	slices=[slice(0,k) for k in nwshape]
		
	for k in axes:
		low=int(halfside[k]) 
		up=int(nwshape[k]+check_pair[k]-halfside[k])
		slices[k]=slice(low,up)

	slices=tuple(slices)
	newcube[slices]=data.copy()
	return newcube,slices
	
	

