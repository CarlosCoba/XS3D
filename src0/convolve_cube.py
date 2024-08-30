import numpy as np
import matplotlib.pylab as plt
from itertools import product
import time
from scipy import interpolate
from multiprocessing import Pool
from src0.read_hdr import Header_info
from src0.momtools import GaussProf,trapecium,trapecium3d


from src0.constants import __c__,__sigma_2_FWHM__,__FWHM_2_sigma__
from src0.conv import conv2d,gkernel,gkernel1d
from src0.conv_spec1d import gaussian_filter1d,convolve_sigma
from src0.conv_fftw import fftconv,padding,data_2N
from src0.momtools import mask_wave


class Cube_creation:

	def __init__(self,datacube,header,mommaps,config):
	


		self.nz,self.ny,self.nx = datacube.shape
		self.h=header
		self.mommaps_obs=mommaps
		[self.mom0,self.mom1,self.mom2]=mommaps
		self.vel_map = self.mommaps_obs[1]
		self.datacube=datacube
		self.config = config
		self.crval3,self.cdelt3,self.pixel_scale=Header_info(self.h,config).read_header()
		
		
		config_const = config['constant_params']
		config_general = config['general']
		self.vary_disp=config_general.getboolean('fit_dispersion',False)
		
		self.wmin,self.wmax=config_general.getfloat('wmin',None),config_general.getfloat('wmax',None)												
		self.eline_A=config_general.getfloat('eline',None)
		self.fwhm_inst_A=config_general.getfloat('fwhm_inst',None)
		self.sigma_inst_A=self.fwhm_inst_A*__FWHM_2_sigma__ if self.fwhm_inst_A is not None else None
		self.sigma_inst_kms=(self.sigma_inst_A/self.eline_A)*__c__ if self.fwhm_inst_A is not None else None
		self.sigma_inst_pix=(self.sigma_inst_A/self.cdelt3)*np.ones(self.nz) if self.fwhm_inst_A is not None else None

		try:
			self.bmaj=self.h['BMAJ']*3600
			self.bmin=self.h['BMIN']*3600
			self.bpa=self.h['BPA']
		except(KeyError):		
			self.fwhm_psf_arc=config_general.getfloat('psf_fwhm',None)
			self.bpa=config_general.getfloat('bpa',0)
			self.bmaj=config_general.getfloat('bmaj',self.fwhm_psf_arc)
			self.bmin=config_general.getfloat('bmin',self.fwhm_psf_arc)
			self.fit_psf=False
			if self.fwhm_psf_arc is not None or self.bmaj is not None:
				self.fit_psf=True			

		self.hdr=Header_info(header,config)
		self.wave_cover_kms=self.hdr.wave_kms
		self.cdelt3_kms=self.hdr.cdelt3_kms
		self.ones2d=np.ones((self.ny,self.nx))
		self.ones3d=np.ones((self.nz,self.ny,self.nx))		
		self.psf2d=gkernel(self.ones2d.shape,self.fwhm_psf_arc,bmaj=self.bmaj,bmin=self.bmin,pixel_scale=self.pixel_scale) if self.fit_psf else None
		self.mom0,self.mom1,self.mom2=self.obs_mommaps()
		#self.mask_cube=np.isfinite(self.mom0)			
		self.nthreads=config_general.getint('nthreads',2)			
		self.eflux2d=0
		
		self.x_=self.ones2d*self.wave_cover_kms[:,None,None]
				
		
	def gaussian_cube(self,vxy,sigmaxy,f0=1):
		#x_ = self.ones2d*self.wave_cover_kms[:,None,None] # shape: (nz,ny,nx)
		vxy_ = vxy*np.ones(self.nz)[:,None,None] # shape: (nz,ny,nx)
		delta_v2=np.square(self.x_-vxy_)
		sig2= np.square(sigmaxy)*np.ones(self.nz)[:,None,None] # shape: (nz,ny,nx)
		cube_mod=f0*np.exp(-0.5*delta_v2/sig2 )
		cube_mod[~np.isfinite(cube_mod)]=0 
		return cube_mod
		
	def obs_mommaps(self):
		mom0= trapecium3d(self.datacube,self.cdelt3_kms)
		Fdv=trapecium3d(self.datacube*self.ones2d*self.wave_cover_kms[:,None,None],self.cdelt3_kms)
		mom1=np.divide(Fdv,mom0,where=mom0!=0,out=np.zeros_like(mom0))
		
		dv2= self.datacube*np.square(self.ones3d*self.wave_cover_kms[:,None,None]-mom1*self.ones3d)
		#mom2=np.sqrt( abs(trapecium3d(dv2,self.cdelt3_kms)/mom0) )
		mom2=np.sqrt( abs(np.divide(trapecium3d(dv2,self.cdelt3_kms),mom0,where=mom0!=0,out=np.zeros_like(mom0))) )
				
		return [mom0,mom1,mom2]

	def obs_emommaps(self,individual_run=0):
	
		sigma_spectral = self.sigma_inst_kms/1. if self.sigma_inst_kms is not None else self.cdelt3_kms/1.
		sigma_spectral*=0.5
		runs = [individual_run]
				
		for k in runs:
			# randomly draw a sample of the observed spectrum
			newfluxcube=self.datacube+np.random.randn(self.nz,self.ny,self.nx)*self.eflux2d
			newspectral=self.wave_cover_kms[:,None,None]+np.random.randn(self.nz,self.ny,self.nx)*sigma_spectral
			#newcube0=np.copy(self.datacube)
			for i,j in product(np.arange(self.nx),np.arange(self.ny)):
				fi = interpolate.interp1d(newspectral[:,j,i],newfluxcube[:,j,i],fill_value='extrapolate')	
				newfluxcube[:,j,i]=fi(self.wave_cover_kms)
				            				
			#newcube=newcube0	            				
			mom0= trapecium3d(newfluxcube,self.cdelt3_kms)
			Fdv=trapecium3d(newfluxcube*self.ones2d*self.wave_cover_kms[:,None,None],self.cdelt3_kms)
			mom1=np.divide(Fdv,mom0,where=mom0!=0,out=np.ones_like(mom0))
			
			dv2= newfluxcube*np.square(self.ones3d*self.wave_cover_kms[:,None,None]-mom1*self.ones3d)
			#mom2=np.sqrt( abs(trapecium3d(dv2,self.cdelt3_kms)/mom0) )
			mom2=np.sqrt( abs(np.divide(trapecium3d(dv2,self.cdelt3_kms),mom0,where=mom0!=0,out=np.ones_like(mom0))) )
		
		del newfluxcube	
		return mom0,mom1,mom2


	def obs_emommaps_boots(self,niter):
		mom0_cube=np.ones((niter,self.ny,self.nx))
		mom1_cube=np.ones((niter,self.ny,self.nx))
		mom2_cube=np.ones((niter,self.ny,self.nx))
		with Pool(self.nthreads) as pool:
			result=pool.map(self.obs_emommaps,np.arange(niter))
		
		for k in range(niter):
			mom0_cube[k] = result[k][0]
			mom1_cube[k] = result[k][1]
			mom2_cube[k] = result[k][2]		
		
		emom0_2d=np.nanstd(mom0_cube,axis=0)
		emom1_2d=np.nanstd(mom1_cube,axis=0)
		emom2_2d=np.nanstd(mom2_cube,axis=0)
							
		return [emom0_2d,emom1_2d,emom2_2d],[mom0_cube,mom1_cube,mom2_cube]

				
	def cube_convolved(self,cube,norm=False):
		mom0= trapecium3d(cube,self.cdelt3_kms)
		cube_mod_psf_norm=cube*np.divide(self.mom0,mom0,where=mom0!=0,out=np.zeros_like(mom0)) #if norm else cube_mod
		mom0_norm=trapecium3d(cube_mod_psf_norm,self.cdelt3_kms)
		
		Fdv=trapecium3d(cube_mod_psf_norm*self.ones2d*self.wave_cover_kms[:,None,None],self.cdelt3_kms)
		#mom1=Fdv/mom0_norm
		mom1=np.divide(Fdv,mom0_norm,where=mom0_norm!=0,out=np.zeros_like(mom0_norm))
				
		dv2= cube_mod_psf_norm*np.square(self.ones3d*self.wave_cover_kms[:,None,None]-mom1*self.ones3d)
		#mom2=np.sqrt( trapecium3d(dv2,self.cdelt3_kms)/mom0_norm)
		mom2=np.sqrt( abs(np.divide(trapecium3d(dv2,self.cdelt3_kms),mom0_norm,where=mom0_norm!=0,out=np.zeros_like(mom0_norm))) )
		return mom0_norm,mom1,mom2,cube_mod_psf_norm
		
	def create_cube(self,velmap,sigmap,padded_cube=None,padded_psf=None,cube_slices=None, pass_cube=True, fit_cube=False):
		cube_mod=self.gaussian_cube(velmap,sigmap,f0=1)

		#(2) lsf convolution only.
		if self.fwhm_inst_A is not None and not self.fit_psf:			
			lsf3d=np.ones_like(velmap)*gkernel1d(self.nz,sigma_pix=self.sigma_inst_pix[0])[:,None,None]
			#cube_mod_conv=convolve_1d(cube_mod,lsf3d)
			
			padded_cube, cube_slices = data_2N(cube_mod, axes=[0])
			padded_lsf, psf_slices = data_2N(lsf3d, axes=[0])

			a=fftconv(padded_cube,padded_lsf,self.nthreads,slice_cube=slice(0,1),axes=[0])
			cube_mod_conv=a.convolve_1d(cube_slices)
		
		# fit PSF and fixed broadening
		if 	self.fit_psf and self.fwhm_inst_A is None or self.vary_disp==False:
			# spatial convolution py the PSF in each channel
			psf3d=self.psf2d*np.ones(self.nz)[:,None,None]
			#cube_mod_conv=convolve_3d_xy(cube_mod,psf3d)
			
			padded_cube, cube_slices = data_2N(cube_mod, axes=[1,2])
			padded_psf, psf_slices = data_2N(psf3d, axes=[1,2])

			#start_time = time.time()
			a=fftconv(padded_cube,padded_psf,self.nthreads,slice_cube=slice(1,3),axes=[1,2])
			cube_mod_conv=a.convolve_3d_xy(cube_slices)

		#(3) fit PSF and LSF											
		if 	self.fit_psf and self.vary_disp and self.fwhm_inst_A is not None :	
			lsf1d=gkernel1d(self.nz,sigma_pix=self.sigma_inst_pix[0])
			psf3d_1 = self.psf2d * lsf1d[:, None, None]
			
			if padded_cube is not None:
				padded_cube[cube_slices]=cube_mod
				padded_psf[cube_slices]=psf3d_1

			else:	
				padded_cube, cube_slices = data_2N(cube_mod, axes=[0, 1, 2])
				padded_psf, psf_slices = data_2N(psf3d_1, axes=[0, 1, 2])

			#start_time = time.time()
			a=fftconv(padded_cube,padded_psf,self.nthreads)
			cube_mod_conv=a.convolve_3d_same(cube_slices)
			#print("--- %s seconds ---" % (time.time() - start_time))


		mom0,mom1_kms,mom2_kms,cube_mod_psf_norm=self.cube_convolved(cube_mod_conv, norm=True)
		msk_mdl = (velmap!=0) & (self.mom0!=0)
		msk_mom0=(self.mom0!=0)	
		#msk_mdl=1
		mom0*=msk_mdl
		mom1_kms*=msk_mdl
		mom2_kms*=msk_mdl
		cube_mod_psf_norm*=msk_mdl
		mom2=mom2_kms
		
		# For saving memory purposes
		if pass_cube:
			return mom0,mom1_kms,mom2_kms,mom2,cube_mod_psf_norm
		else:	
			return mom0,mom1_kms,mom2_kms,mom2,0		
			
	
class Zeropadding:
	def __init__(self,cube,h,config):

		self.datacube=cube
		config_general = config['general']
		self.vary_disp=config_general.getboolean('fit_dispersion',False)
		self.fwhm_inst_A=config_general.getfloat('fwhm_inst',None)

		try:
			self.bmaj=h['BMAJ']*3600
			self.bmin=h['BMIN']*3600
		except(KeyError):		
			self.fwhm_psf_arc=config_general.getfloat('psf_fwhm',None)
			self.bpa=config_general.getfloat('bpa',0)
			self.bmaj=config_general.getfloat('bmaj',self.fwhm_psf_arc)
			self.fit_psf=False
			if self.fwhm_psf_arc is not None or self.bmaj is not None:
				self.fit_psf=True			



	def create_cube_pad(self):
		if self.fwhm_inst_A is not None and not self.fit_psf:
			padded_cube, cube_slices = data_2N(self.datacube, axes=[0])

		if 	self.fit_psf and self.fwhm_inst_A is None or self.vary_disp==False:			
			padded_cube, cube_slices = data_2N(self.datacube, axes=[1,2])
			
		if 	self.fit_psf and self.vary_disp and self.fwhm_inst_A is not None :	
			padded_cube, cube_slices = data_2N(self.datacube, axes=[0, 1, 2])

		padded_cube*=0
		return padded_cube, cube_slices
					
	def __call__(self):
		return self.create_cube_pad()#padded_cube, cube_slices
					

