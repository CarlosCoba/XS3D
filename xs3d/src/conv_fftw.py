import numpy as np
import numpy.fft as fft
import pyfftw
import math

pyfftw.config.PLANNER_EFFORT = 'FFTW_ESTIMATE'
from pyfftw.interfaces.numpy_fft import rfftn, irfftn, fftshift
pyfftw.interfaces.cache.enable()


"""
	Following we apply the convolution theorem, https://en.wikipedia.org/wiki/Convolution_theorem.
	`The Fourier transform of a convolution of two functions is the
	 product of their Fourier transforms.`
	 
	 Given two functions a(array) & k(kernel) with Fourier (F) transform A & K, respectively :
	 	A = F(a)
	 	K = F(k)
	 The convolution * of a & k is defined as:
	 	r = a * k
	 The convolution theorem states that:
	 	F(r) = A.K
	 Applying the iverse F-1 -->
	 	r = a * k = F-1(A.K)	 	
"""
class fftconv:
	def __init__(self,cube,psf,threads,axes=[0,1,2]):
		self.cube=cube
		self.psf=psf
		self.axes=axes
		self.slice=slice(axes[0],axes[-1]+1)
		shape = (np.shape(cube)[self.slice])
		size = np.array(cube.shape)[self.slice]

		# axes: Vectors along which the transform is computed.
		# size: Shape of the result.
		
		# Accoding to pyfftw documentation the dtype of the input array must match the transform.
								
		# if the input is complex
		if np.iscomplexobj(cube) and np.iscomplexobj(psf):
			self.fft_cube = pyfftw.builders.fftn(cube, s=size, threads=threads,axes=axes,auto_align_input=False, auto_contiguous=False,avoid_copy=True)
			self.fft_psf = pyfftw.builders.fftn(psf, s=size, threads=threads,axes=axes,auto_align_input=False, auto_contiguous=False,avoid_copy=True)
			self.ifft_obj = pyfftw.builders.ifftn(self.fft_cube.get_output_array(), s=size,threads=threads,axes=axes,auto_align_input=False, auto_contiguous=False,avoid_copy=True)                
		# if the input is real
		else:
			self.fft_cube = pyfftw.builders.rfftn(cube, s=size, threads=threads,axes=axes,auto_align_input=False, auto_contiguous=False,avoid_copy=True)
			self.fft_psf = pyfftw.builders.rfftn(psf, s=size, threads=threads,axes=axes,auto_align_input=False, auto_contiguous=False,avoid_copy=True)
			self.ifft_obj = pyfftw.builders.irfftn(self.fft_cube.get_output_array(), s=size,threads=threads,axes=axes,auto_align_input=False, auto_contiguous=False,avoid_copy=True)        
                

	def conv_DFT(self,cube_slices):	

		fft_cube = self.fft_cube(self.cube)
		fft_psf = self.fft_psf(self.psf)
		# Here apply the convolution theorem		
		conv= self.ifft_obj(fft_cube * fft_psf)
		conv_real = np.real(np.fft.fftshift(conv,axes=self.axes))
		
		# Remove padding
		return conv_real[cube_slices]



"""
To optimize the FFT we need to make sure that each dimension of the datacube
has a lenght of 2**N. This will necesesaerly increase the actual dimmensions
of the datacube but will make the FFT much faster. 

nx=2**N --> log2(nx)/log(2)=N

"""
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
			#N=int(f)+1
			N=math.ceil(f)
			nxnew=2**N
		nwshape[axis]=nxnew
			
	newcube=np.zeros(nwshape)
	nwshape=newcube.shape
	# free array space
	free=nwshape-np.array(dims)
	
	check_pair=free % 2 # -> zeros if all pairs
	halfside_l=np.ceil(free/2).astype(int)
	halfside_u=np.ceil(free/2).astype(int) - check_pair 
	
	# set default slices
	slices=[slice(0,k) for k in nwshape]
		
	for k in axes:
		#low=int(halfside[k]) 
		#up=int(nwshape[k]+check_pair[k]-halfside[k])
		low=halfside_l[k] 
		up=int(nwshape[k]-halfside_u[k])		
		slices[k]=slice(low,up)

	slices=tuple(slices)
	newcube[slices]=data.copy()
	return newcube,slices



"""
	Following we apply the convolution theorem, https://en.wikipedia.org/wiki/Convolution_theorem.
	`The Fourier transform of a convolution of two functions is the
	 product of their Fourier transforms.`
	 
	 Given two functions a(array) & k(kernel) with Fourier (F) transform A & K, respectively :
	 	A = F(a)
	 	K = F(k)
	 The convolution * of a & k is defined as:
	 	r = a * k
	 The convolution theorem states that:
	 	F(r) = A.K
	 Applying the iverse F-1 -->
	 	r = a * k = F-1(A.K)	 	
"""
class fftconv_numpy:
	def __init__(self,cube,psf,threads,axes=[0,1,2]):
		self.cube=cube
		self.psf=psf
		self.axes=axes
		self.slice=slice(axes[0],axes[-1]+1)
		shape = (np.shape(cube)[self.slice])
		self.size = np.array(cube.shape)[self.slice]				
		self.conv=self.conv_3D(self.cube,self.psf)

	# axes: Vectors along which the transform is computed.
	# size: Shape of the result.

	def fft(self,array):
		fft = np.fft.fftn(array,axes=self.axes,s=self.size)
		return fft

	def ifft(self,array):
		ifft = np.fft.ifftn(array,axes=self.axes,s=self.size).real
		return ifft

	def conv_3D(self,array, kernel):
		# Here apply the convolution theorem
		conv = self.ifft(self.fft(array)*self.fft(kernel))
		conv_shift=np.fft.fftshift(conv,axes=self.axes)
		return conv_shift

	def conv_DFT(self,cube_slices):
		fft_cube = self.conv	
		# Remove padding
		return fft_cube[cube_slices]
		
		
		
