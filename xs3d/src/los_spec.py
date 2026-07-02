import numpy as np
import matplotlib.pylab as plt
from itertools import product
import time
from scipy import interpolate
from multiprocessing import Pool
from scipy.interpolate import interp1d

from .read_hdr import Header_info
from .momtools import GaussProf,trapecium,trapecium3d


from .constants import __c__,__sigma_2_FWHM__,__FWHM_2_sigma__
from .conv import conv2d,gkernel,gkernel1d
from .conv_spec1d import gaussian_filter1d,convolve_sigma
from .conv_fftw import fftconv,data_2N
from .momtools import mask_wave
from .utils import parabola
from .psf_lsf import PsF_LsF


from .pixel_params import R_x, R_z, eps_2_inc

class LOS_spectrum:
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
		config_others = config['others']
		self.vary_disp=config_general.getboolean('fit_dispersion',False)

		self.wmin,self.wmax=config_general.getfloat('wmin',None),config_general.getfloat('wmax',None)

		psf_lsf= PsF_LsF(self.h, config)
		self.fit_psf=psf_lsf.fit_psf
		self.bmaj=psf_lsf.bmaj
		self.bmin=psf_lsf.bmin
		self.bpa= psf_lsf.bpa
		self.fwhm_psf_arc=psf_lsf.fwhm_psf_arc
		self.sigma_inst_pix=psf_lsf.sigma_inst_pix


		self.hdr=Header_info(header,config)
		self.wave_cover_kms=self.hdr.wave_kms
		self.cdelt3_kms=self.hdr.cdelt3_kms
		self.dV=abs(self.cdelt3_kms)
		self.ones2d=np.ones((self.ny,self.nx))
		self.ones3d=np.ones((self.nz,self.ny,self.nx))
		self.psf2d=gkernel(self.ones2d.shape,self.fwhm_psf_arc,bmaj=self.bmaj,bmin=self.bmin, bpa=self.bpa,pixel_scale=self.pixel_scale, norm=True) if self.fit_psf else None
		self.vpeak=config_others.getboolean('vpeak',False)
		#self.mask_cube=np.isfinite(self.mom0)
		self.nthreads=config_general.getint('nthreads',2)
		self.eflux3d=0

		# ------ Vertical height scale	------	
		self.hz_arcs = config_others.getfloat('hz_arcs', 1)
		self.z_profile = config_others.get('z_profile', 'exp')		
		self.gamma_arcs = config_others.getfloat('gamma_arcs', 0)				
		self.gamma_kms = config_others.getfloat('gamma_kms', 0)						
		self.x_=self.ones2d*self.wave_cover_kms[:,None,None]

		if self.gamma_kms != 0:
			self.gamma_arcs = self.gamma_kms
		
		self.Z_disk_arcs = 1

	def j_z(self, z_arr):
		z_disk = z_arr / self.hz_arcs
		# z_disk = z_disk_arcs / hz_arcs
		z_profile = self.z_profile
		if z_profile == 'exp':
			fz = np.exp(-np.abs(z_disk))
			# Normalize profile
			fz /= 2*self.hz_arcs
		elif z_profile == 'exp2':
			fz = np.exp(-1*(0.5*z_disk )**2)
			fz /= np.sqrt(2*np.pi*self.hz_arcs**2)												
		elif z_profile == 'sech2':
			z0 = z_disk/2
			fz = np.cosh(z0)**2
			fz /= 2*self.hz_arcs
		else:
			print('Unknown vertical profile')	

		return fz	
	

	def j_R(self, R_disk):
		# Relation between hR and hz from Ranaivoharimina +2024
		# https://ui.adsabs.harvard.edu/abs/2024ApJ...977...66R/abstract
		lh_R = ( np.log10(self.hz_arcs) + 0.81 ) / 0.90
		h_R = 10**lh_R
		#h_R = 7*self.hz_arcs
		r = R_disk / h_R
		fr = np.exp(-r)
		# This does not requires Normalize profile
		return fr	

	def create_cube_vyx_test2(self,cube, xy_mesh, v3d, sig3d, z_arr, fzR, ds, msk, f0=1):
		Xs,Ys = xy_mesh
		
		
		# flatten coordinates and valid pixel mask (early radial test)
		Xf = Xs.ravel(); Yf = Ys.ravel()
    		
		valid0 =  msk
		idx_valid = np.where(valid0.ravel())[0]

		Xv = Xf[idx_valid]; Yv = Yf[idx_valid]
		npix_valid = Xv.size

		s_block = 70
		block_pix = max(1, min(npix_valid,  int( max(1, npix_valid // 2000) ))) 
		block_pix = 10000

		[nchan, ny, nx] = cube.shape
		x_lambda = self.wave_cover_kms # (nchan,) for broadcasting		
		nchan = len(self.wave_cover_kms)
		
		# For memory, process pixels in blocks
		for p0 in range(0, npix_valid, block_pix):
			p1 = min(p0 + block_pix, npix_valid)
			Xb = Xv[p0:p1]; Yb = Yv[p0:p1]
			nb = Xb.size
			# accumulate spectral arrays for block
			block_spec = np.zeros((nb, nchan), dtype=np.float64)

			for iz, z_d in enumerate(z_arr):
				vlos_z = (v3d[iz])[msk]
				disp_z = (sig3d[iz])[msk]
				emissivity_z = (fzR[iz])[msk]

				vlos = vlos_z[p0:p1]
				disp = disp_z[p0:p1]
				emissivity = emissivity_z[p0:p1]

				# radial mask
				mask_valid = vlos != 0
				if not np.any(mask_valid):
					continue
				
				vlos_valid = vlos[mask_valid]
				disp_valid = disp[mask_valid]
				emm_valid = (emissivity[mask_valid])*ds


				# spectral contribution: Gaussian per voxel. We treat sigma_vals as velocity dispersion.
				# Precompute gaussian prefactors for masked entries
				for ii, pix_idx in enumerate(np.where(mask_valid)[0]):
				
					pref = 1.0 / (np.sqrt(2*np.pi) * disp_valid[ii])
					spec = pref * np.exp(-0.5 * ((x_lambda - vlos_valid[ii])/disp_valid[ii])**2)
  
					# multiply by weight (contrib per pixel per layer)
					block_spec[pix_idx, :] += emm_valid[ii] * spec



			# scatter block_spec into cube
			# map block pixels back to global (jy,ix)
			global_indices = idx_valid[p0:p1]
			jy = (global_indices // nx).astype(int)
			ix = (global_indices % nx).astype(int)
			for k in range(nb):
				cube[:, jy[k], ix[k]] += block_spec[k, :]


	def create_cube_vyx(self,cube, xy_mesh, v3d, sig3d, z_arr, fzR, ds, msk, f0=1):
		[nchan,ny,nx] = cube.shape
		nlos = len(z_arr)
		# Set the maximum RAM memory for this loop.
		

		threshold_mem = 2 # GB
		# each float64 = 8bytes		
		nGB = (nchan*ny*nx*nlos*8) / 1e9

		print('nGB = ', nGB)
		# If the hypercube is larger than 2GB then go for the loop, else vectorize
		if nGB > threshold_mem:
			#  Slow becauce of te loop			
			vlos = (v3d*msk)
			disp = (sig3d*msk)
			emm = (fzR*msk)
			x_lambda = self.wave_cover_kms
					
			x_lambda = self.wave_cover_kms # (nchan,) for broadcasting		
			for iv, vel in enumerate(x_lambda):
				pref = 1.0 / (np.sqrt(2*np.pi) * disp)
				diff = np.add(vel, -vlos, where = vlos != 0)
				
				spec = pref * np.exp(-0.5 * ((diff)/disp)**2)
				Isvxy = spec*emm*ds
				Isvxy[~np.isfinite(Isvxy)]=0
				cube[iv] = np.sum(Isvxy, axis=0) 		
		else:
			# Faster 
			vlos = (v3d*msk)[:,None,:,:]
			disp = (sig3d*msk)[:,None,:,:]
			emm = (fzR*msk)[:,None,:,:]
			x_lambda = self.wave_cover_kms[None,:,None,None]
			diff = vlos - x_lambda	
			pref = 1.0 / (np.sqrt(2*np.pi) * disp)
			spec = pref * np.exp(-0.5 * ((diff)/disp)**2, where = disp != 0)
			Isvxy = spec*emm*ds
			Isvxy[~np.isfinite(Isvxy)]=0
			cube += np.sum(Isvxy, axis = 0)		
			
		return None

	# -------------------------
	# Helper: ring interpolants
	# -------------------------
	def make_interp(self, r_ring, v_ring, kind='linear', fill_value=0.0):
		r = np.asarray(r_ring)
		v = np.asarray(v_ring)
		return interp1d(r, v, kind=kind, bounds_error=False, fill_value=fill_value, assume_sorted=True)


	def vectorize_voxel(self, xy_mesh, x0, y0, pa, eps, vsys, pixel_scale, s_arr, rk, vrot_ring, sigma_ring, cube, mask_, gamma_kms_arcs ):
		(x,y) = xy_mesh
		Sx = (x-x0)*pixel_scale
		Sy = (y-y0)*pixel_scale	
		inc = eps_2_inc(eps)
		R_max = np.max(rk)
		ds = s_arr[1] - s_arr[0]

		Sx = Sx[None, :, :]		# (n_samp, ny, nx)
		Sy = Sy[None, :, :]		# (n_samp, ny, nx)
		Sz = s_arr[:, None, None] 
	
		[nv, ny, nx] = cube.shape
		Rz = R_z(pa)
		Rx = R_x(inc)
		# -------------------------
		# Build rotation matrix
		# -------------------------	
		R = Rz @ Rx # disk -> sky
		Rt = R.T # sky->disk

		# Pre-extract Rt elements for speed
		Rt00, Rt01, Rt02 = Rt[0,0], Rt[0,1], Rt[0,2]
		Rt10, Rt11, Rt12 = Rt[1,0], Rt[1,1], Rt[1,2]
		Rt20, Rt21, Rt22 = Rt[2,0], Rt[2,1], Rt[2,2]
		
		R20, R21 = R[2,0], R[2,1]  # used to project velocity components into sky z   

		jr = self.j_R(rk)
		# prepare callable radial profiles
		vt_func = self.make_interp(rk, vrot_ring, kind='linear')
		sigma_func = self.make_interp(rk, sigma_ring, kind='linear')
		jr_func = self.make_interp(rk, jr, kind='linear', fill_value=0.0)
				
		# Disk coordinates
		Dx = Rt[0,0]*Sx + Rt[0,1]*Sy + Rt[0,2]*Sz
		Dy = Rt[1,0]*Sx + Rt[1,1]*Sy + Rt[1,2]*Sz
		Dz = Rt[2,0]*Sx + Rt[2,1]*Sy + Rt[2,2]*Sz

		# Disk radius and azimuth (in disk frame)
		R_disk = np.hypot(Dx, Dy)
		# theta_disk = arctan2(y_disk, x_disk) -> angle measured from x_disk toward y_disk
		theta = np.arctan2(Dy, Dx)			  # (n_samp, ny, nx)
		cos_theta = Dx/R_disk


		# Interpolated values
		vt_intp = vt_func(R_disk)
		disp_intp = sigma_func(R_disk)
		jr_intp = self.j_R(R_disk)
		jz_intp = self.j_z(Dz)
		# Emissivity
		emm_intp = jr_intp*jz_intp
		
		mask_R = R_disk	< R_max
		msk = mask_R & mask_


		gamma = gamma_kms_arcs
		
		gamma1 = gamma_kms_arcs
		gamma2 = 1.5*gamma_kms_arcs
		
		# Azimuthal lagg in the vertical direction
		vlagg = np.zeros_like(Dz)
		vlagg[Dz>0] = gamma1*abs(Dz[Dz>0])
		vlagg[Dz<0] = gamma2*abs(Dz[Dz<0])
						
		vt = vt_intp - vlagg
		Vrot = np.maximum(vt, 0 )
		v3d = Vrot*cos_theta*R21
		disp3d = disp_intp*msk
		emm3d = emm_intp*msk
		
		
		vlos3d = v3d + msk*vsys
		vlos = vlos3d[:,None,:,:]
		disp = disp3d[:,None,:,:]
		emm = emm3d[:,None,:,:]	
		x_lambda = self.wave_cover_kms[None,:,None,None]


		diff = vlos - x_lambda	
		pref = 1.0 / (np.sqrt(2*np.pi) * disp)
		spec = pref * np.exp(-0.5 * ((diff)/disp)**2, where = disp != 0)
		Isvxy = spec*emm*ds
		Isvxy[~np.isfinite(Isvxy)]=0
		cube += np.sum(Isvxy, axis = 0)	

		'''
		vlos = (v3d*msk)[:,None,:,:]
		disp = (sig3d*msk)[:,None,:,:]
		emm = (fzR*msk)[:,None,:,:]
		x_lambda = self.wave_cover_kms[None,:,None,None]
		diff = vlos - x_lambda	
		pref = 1.0 / (np.sqrt(2*np.pi) * disp)
		spec = pref * np.exp(-0.5 * ((diff)/disp)**2, where = disp != 0)
		Isvxy = spec*emm*ds
		Isvxy[~np.isfinite(Isvxy)]=0
		cube += np.sum(Isvxy, axis = 0)	
		'''
		return None
		





	
	

	def create_cube_vyx_edgeon(self,v3d, sig3d, jR_per_voxel, z_disk, n_depth, f0=1):

		x_lambda = self.wave_cover_kms[:, None, None]  # (nchan, 1, 1) for broadcasting		
		fz = self.j_z(z_disk)
		
		[_, ny, nx] = v3d.shape
		[nchan, _, _] = x_lambda.shape

		# emissivity per voxel for this z-layer (n_depth, ny, nx)
		emissivity_layer = jR_per_voxel * fz
		
				
		# accumulate contributions from each depth sample (loop over depth)
		# inner loop is n_depth (e.g. 201). Each iteration computes gauss (nchan, ny, nx),
		# multiplies by a 2D emissivity map and accumulates into cube.
		cube = np.zeros((nchan, ny, nx), dtype=float)		
		for idepth in range(n_depth):
			emiss2d = emissivity_layer[idepth]   # (ny, nx)
			if not emiss2d.any():
				continue

			Vlos2d = v3d[idepth]		  # (ny, nx)
			Sig2d = sig3d[idepth]
			# spectral kernel: gauss( v - (Vsys + Vlos) ), shape (nchan, ny, nx)
			delta_v = x_lambda - ( Vlos2d)[None, :, :]
			gauss = np.exp(-0.5 * (delta_v / Sig2d) ** 2)
			gauss /= (np.sqrt(2.0 * np.pi) * Sig2d)
			gauss[~np.isfinite(gauss)]=0				
			cube += gauss 

		return cube












