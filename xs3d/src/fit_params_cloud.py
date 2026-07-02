import numpy as np
import matplotlib.pylab as plt
import scipy
import sys
from lmfit import Model, Parameters, fit_report, minimize, Minimizer
from matplotlib.gridspec import GridSpec
import configparser
import random
from itertools import product,chain
import time

from .kin_components import CIRC_MODEL
from .kin_components import HARMONIC_MODEL
from .kin_components import SIGMA_MODEL
from .kin_components import EDGEON_MODEL
from .kin_components import AZIMUTHAL_ANGLE,SIN_COS
from .pixel_params import pixels,v_interp,eps_2_inc, Rings, Cilinder, Zdist, R_disk, R_disk_edgeon
from .weights_interp import weigths_w
from .create_2D_kin_models import bidi_models
from .create_3D_cube_model import best_3d_model
from .create_dataset import dataset_to_2D
from .read_hdr import Header_info
from .momtools import GaussProf,trapecium
from .convolve_cube import Cube_creation,Zeropadding
from .los_spec import LOS_spectrum
from .psf_lsf import PsF_LsF
from .lum_dist import Angdist

from .constants import __c__
from .conv import conv2d,gkernel,gkernel1d
from .conv_spec1d import gaussian_filter1d,convolve_sigma



class Least_square_fit:
	#def __init__(self,vel_map, emoms, guess, vary, vmode, config, rings_pos, ring_space, fit_method, e_ISM, pixel_scale, frac_pixel, v_center, m_hrm = 1, N_it = 1):
	def __init__(self,datacube, edatacube, header, mommaps, emoms, guess, vary, vmode, config, rings_pos, ring_space, frac_pixel, v_center, m_hrm = 1, N_it = 1):

		"""
		vary = [Vrot,Vrad,Vtan,PA,INC,XC,YC,VSYS,theta]
		"""

		self.N_it = N_it
		if self.N_it == 0:
			vary,self.vary_kin = vary*0,0
			vary,self.vary_kin = vary,1
		else:
			self.vary_kin = 1
		if "hrm" in vmode:
			self.sig0, self.c_k0, self.s_k0,self.pa0,self.eps0,self.xc0,self.yc0,self.vsys0,self.phi_bar = guess
			guess_kin=[self.sig0, self.c_k0, self.s_k0]
			guess=list(chain(*guess_kin))+[self.pa0,self.eps0,self.xc0,self.yc0,self.vsys0,self.phi_bar]
			constant_params = [self.pa0,self.eps0,self.xc0,self.yc0,self.vsys0]
			self.vary_pa,self.vary_eps,self.vary_xc,self.vary_yc,self.vary_vsys,self.vary_phib = vary
			self.vary_sk, self.vary_ck = True, True
			self.sig0=np.asarray(self.sig0)

		else:
			self.sig0, self.vrot0,self.vrad0,self.vtan0,self.pa0,self.eps0,self.xc0,self.yc0,self.vsys0,self.phi_bar = guess
			n_circ,n_noncirc = len(self.vrot0),len(self.vrad0[self.vrad0!=0])
			#if n_noncirc !=n_circ:
			#	self.vrad0[n_noncirc],self.vtan0[n_noncirc]=1e-3,1e-3
			constant_params = [self.pa0,self.eps0,self.xc0,self.yc0,self.vsys0,self.phi_bar]
			self.vary_pa,self.vary_eps,self.vary_xc,self.vary_yc,self.vary_vsys,self.vary_phib = vary
			self.vary_vrot, self.vary_vrad, self.vary_vtan = 1*self.vary_kin, 1*self.vary_kin, 1*self.vary_kin
			if vmode == 'edgeon':
				self.vary_eps = 0
				self.eps0 = 1 

		#flat parameters:
		self.params=np.hstack(guess)
		self.nparams=len(self.params[self.params!=0])
		self.m_hrm = m_hrm
		self.nz,self.ny,self.nx = datacube.shape
		self.h=header
		self.rings_pos = rings_pos['R_pos']
		self.rings_nc =  rings_pos['R_NC']
		self.r1st=self.rings_pos[0]
		self.rmax=self.rings_pos[-1]
		self.nrings = len(self.rings_pos)
		self.n_annulus = self.nrings - 1
		self.mommaps_obs=mommaps
		[self.mom0,self.mom1,self.mom2]=mommaps
		self.vel_map = self.mommaps_obs[1]
		self.datacube=datacube
		self.emoms = emoms
		[self.emom0,self.emom1,self.emom2]=emoms
		self.vmode = vmode
		self.ring_space = ring_space
		self.fit_method = 'nelder'#'least_squares'
		self.config = config
		self.constant_params = constant_params
		self.osi = ["-", ".", ",", "#","%", "&", ""]

		# Assume 4% outliers
		self.mom1[abs(self.mom1-self.vsys0)>1e3]=0
		p1=np.nanpercentile(np.unique(self.mom1),2)
		p99=np.nanpercentile(np.unique(self.mom1),98)
		msk_outliers=(self.mom1>p1)*(self.mom1<p99)
		self.mom0=self.mom0*(msk_outliers)
		self.mom1=self.mom1*(msk_outliers)
		self.mom2=self.mom2*(msk_outliers)

		# args to pass to minimize function
		self.kws={'maxiter':500,'fatol':1e-4}
		self.kwargs = {}
		if self.N_it == 0:
			self.kwargs = {"ftol":1e8}

		X = np.arange(0, self.nx, 1) + 0.5
		Y = np.arange(0, self.ny, 1) + 0.5
		self.XY_mesh = np.meshgrid(X,Y)

		if self.xc0 % int(self.xc0) == 0 or self.yc0 % int(self.yc0) == 0 :
			self.xc0, self.yc0 =  self.xc0 + 1e-5, self.yc0 + 1e-5
		self.crval3,self.cdelt3,self.pixel_scale=Header_info(self.h,self.config).read_header()
		self.wave_kms=Header_info(self.h,self.config).wave_kms
		self.r_n = Rings(self.XY_mesh,self.pa0*np.pi/180,self.eps0,self.xc0,self.yc0,self.pixel_scale, self.vmode)	
		self.r_n = np.asarray(self.r_n, dtype = np.longdouble)
		self.frac_pixel = frac_pixel
		self.v_center = v_center
		
		# Compute lum distance
		dist = Angdist(self.vsys0/__c__)
		dL, scale_pc_arc = dist.comv_distance()
		
		config_others = config['others']
		config_const = config['constant_params']
		config_general = config['general']		

		# ----- Intensity profile ------		
		self.normalize = config_others.getboolean('norm', 1)
				
		# ------ Vertical height scale	------	
		self.hz_arcs = config_others.getfloat('hz_arcs', 1)

		
		# 	Create vertical (disk internal) samples for LOS integration
		# 	Most emission comes from roughly +-(2-4)hz.
		# 	Past 4*hz contribution is negligible for exponential profile.  		
		n_hz = 5
		samples_per_hz = 6
		
		# cover +/- n_hz * hz in zdisk
		z_ext_arcs = self.hz_arcs * n_hz
		# 	Number of vertical layers
		#	nzlayer should be impar to ensure z = 0
		nzlayer = 2*n_hz*samples_per_hz + 1
		
		#	Intrinsic disk vertical coordinate.
		#	Extent for vertical integration		
		self.z_arr_arcs = np.linspace(-z_ext_arcs, z_ext_arcs, nzlayer)
		# dz for integration
		self.dz_arcs = self.z_arr_arcs[1] - self.z_arr_arcs[0]
		
		# pack z scale properties				
		self.z_scale_p = (self.z_arr_arcs, self.hz_arcs, self.dz_arcs)

		# pixel area in arcsec square
		self.pixel_arcs2 = self.pixel_scale*self.pixel_scale


		# ----- Edge on case -----
		s_min, s_max = None, None
		if s_min is None: s_min = -self.rmax
		if s_max is None: s_max =  self.rmax

		self.n_depth = 101
		self.s_arr = np.linspace(s_min, s_max, self.n_depth)
		self.ds = self.s_arr[1] - self.s_arr[0]

		# ------ vertical lagging ------
		gamma_kpc = config_others.getfloat('gamma', 15)	 # km/s/kpc
		self.gamma_pc = gamma_kpc/1000
		self.gamma = 0

		interp_model = np.zeros((self.ny,self.nx))
		self.index_v0 = 123456
		if vmode in ["circular",'edgeon']: self.Vk = 1+1
		if vmode == "radial" or vmode == 'vertical' or self.vmode == 'ff': self.Vk = 2+1
		if vmode == "bisymmetric": self.Vk = 3+1


		self.Vrot, self.Vrad, self.Vtan = 0,0,0
		if "hrm" not in self.vmode:
			self.V_k = [self.vrot0*0,self.vrot0,self.vrad0,self.vtan0]
			self.V_k_std = [0,0,0,0]
		else:
			self.V_k = [0]*(2*self.m_hrm)+[0]
			self.V_k_std = [0]*(2*self.m_hrm)+[0]
			self.Vk = 2*self.m_hrm+1

		##############################################
		# ----- Constrains of Fitted parameters ----
		##############################################
		self.Vmin, self.Vmax = -550, 550
		eps_min, eps_max = 1-np.cos(5*np.pi/180),1-np.cos(90*np.pi/180)
		self.PAmin,self.PAmax,self.vary_pa = config_const.getfloat('MIN_PA', -360*2), config_const.getfloat('MAX_PA', 360*2),config_const.getboolean('FIT_PA', self.vary_pa)
		self.INCmin,self.INCmax,self.vary_eps = config_const.getfloat('MIN_INC', eps_min), config_const.getfloat('MAX_INC', eps_max),config_const.getboolean('FIT_INC', self.vary_eps)
		# To change input INCmin (in deg) and INCmax (in deg) values from the config file to eps
		if self.INCmin >1:  self.INCmin = 1-np.cos(self.INCmin*np.pi/180)
		if self.INCmax >1:  self.INCmax = 1-np.cos(self.INCmax*np.pi/180)
		self.X0min,self.X0max,self.vary_xc = config_const.getfloat('MIN_X0', 0), config_const.getfloat('MAX_X0', self.nx),config_const.getboolean('FIT_X0', self.vary_xc)
		self.Y0min,self.Y0max,self.vary_yc = config_const.getfloat('MIN_Y0', 0), config_const.getfloat('MAX_Y0', self.ny),config_const.getboolean('FIT_Y0', self.vary_yc)
		self.VSYSmin,self.VSYSmax,self.vary_vsys = config_const.getfloat('MIN_VSYS', 0), config_const.getfloat('MAX_VSYS', 10*__c__),config_const.getboolean('FIT_VSYS', self.vary_vsys)
		self.PAbarmin,self.PAbarmax,self.vary_phib = config_const.getfloat('MIN_PHI_BAR', -2*np.pi),config_const.getfloat('MAX_PHI_BAR', 2*np.pi), config_const.getboolean('FIT_PHI_BAR', self.vary_phib)
		
		#	Rename Geometry to capital letters
		self.PA = self.pa0
		self.EPS =  self.eps0
		self.X0 = self.xc0
		self.Y0 = self.yc0
		self.VSYS = self.vsys0
		self.PHI_BAR = self.phi_bar
		
		#	Control weight and convergency		
		self.WEIGHT=config_const.getint('WEIGHT',0)
		self.XTOL=config_const.getfloat('XTOL',1e-5)
		self.MAXF=config_const.getint('MAXF',15)
		outliers = config_general.getboolean('outliers', False)
		if outliers: self.kwargs["loss"]="soft_l1"


		self.vary_disp=config_general.getboolean('fit_dispersion',False)
		psf_lsf= PsF_LsF(self.h, config)
		self.sigma_inst_kms=psf_lsf.sigma_inst_kms
		self.sigma_inst_pix=psf_lsf.sigma_inst_pix


		# ----------- LSF ---------------
		# if there are mom2[y][x] < sigma_inst, then assign the instrumental
		if self.sigma_inst_kms is not None:
			mom2_msk=((self.mom2<self.sigma_inst_kms) & (self.mom0!=0))
			self.mom2[mom2_msk]=self.sigma_inst_kms

		self.min_sig=0.1
		if self.sigma_inst_pix is not None:
			#remove the instrumental dispersion
			sig0 = np.sqrt(self.sig0**2-self.sigma_inst_kms**2)
			#check if there are nans
			nan_sigmas=~np.all(np.isfinite(sig0))
			if nan_sigmas:
				# if only some values are nan
				if np.nanmean(sig0)!=0:
					msk_inst=~np.isfinite(sig0)
					sig0[msk_inst]=np.nanmean(sig0)
				# if all values are nan
				else:
					sig0=np.ones_like(sig0)
			self.sig0=sig0


		if not self.vary_disp and self.sigma_inst_pix is not None:
			self.sig0=np.ones_like(self.sig0)*self.sigma_inst_kms



		self.cube_modl = Cube_creation(datacube,header,mommaps,config)
		self.spec3d = LOS_spectrum(datacube,header,mommaps,config)
		# update these variables from the above classes
		self.spec3d.z_arr_arcs = self.z_arr_arcs
		self.cube_modl.normalize = self.normalize
		
		self.ecube=edatacube
		a=Zeropadding(datacube,header,config)
		pad=a()
		self.padded_cube, self.cube_slices=pad[0],pad[1]
		self.padded_psf=np.copy(self.padded_cube)
		self.mask_cube=np.ones_like(self.datacube,dtype=bool)*(self.mom0!=0)
		self.fit_from_cube=config_general.getboolean('fit_from_cube',False)


		# precompute masks
		self.mom0_msk = self.mom0 != 0
		
		
		self.peakI0=np.nanmax(self.datacube,axis=0)
	def iter_cb(self,params, iter, resid, *args, **kws):
		sumres=0.5*np.sum(resid**2)
		print(iter,sumres)
		pass


class Config_params(Least_square_fit):

		def assign_constpars(self,pars):

			#if self.config in self.osi:
			pars.add('Vsys', value=self.VSYS, vary = self.vary_vsys, min = self.VSYSmin, max = self.VSYSmax)
			pars.add('pa', value=self.PA, vary = self.vary_pa, min = self.PAmin, max = self.PAmax)
			pars.add('eps', value=self.EPS, vary = self.vary_eps, min = self.INCmin, max = self.INCmax)
			pars.add('x0', value=self.X0, vary = self.vary_xc,  min = self.X0min, max = self.X0max)
			pars.add('y0', value=self.Y0, vary = self.vary_yc, min = self.Y0min, max = self.Y0max)
			if self.vmode == "bisymmetric":
				pars.add('phi_b', value=self.PHI_BAR, vary = self.vary_phib, min = self.PAbarmin , max = self.PAbarmax)
			if self.vmode == "ff":
				pars.add('alpha', value=0, vary = True, min = 0 , max = 1)


		def tune_velocities(self,pars,iy):
				if "hrm" not in self.vmode:
					if self.vmode == "radial" or self.vmode == 'vertical' or self.vmode == 'ff':
						if self.vrad0[iy] == 0:
							self.vary_vrad = False
						else:
							self.vary_vrad = True*self.vary_kin
					if self.vmode == "bisymmetric":
						if self.vrad0[iy] == 0  and self.vtan0[iy] ==0:
							self.vary_vrad = False
							self.vary_vtan = False
						else:
							self.vary_vrad = True*self.vary_kin
							self.vary_vtan = True*self.vary_kin
				else:
					if self.s_k0[0][iy] == 0:
						self.vary_sk = False
						self.vary_ck = False
					else:
						self.vary_sk = True
						self.vary_ck = True



		def assign_vels(self,pars):
			for iy in range(self.nrings):
				pars.add('Sig_%i' % (iy),value=self.sig0[iy], vary = self.vary_disp, min = self.min_sig, max = 1000)
				if "hrm" not in self.vmode:
					pars.add('Vrot_%i' % (iy),value=self.vrot0[iy], vary = self.vary_vrot, min = 0, max = self.Vmax)

					#if self.vrad0[iy] == 0 and self.vtan0[iy] ==0:
					#	self.vary_vrad = False
					#	self.vary_vtan = False

					if self.vmode == "radial" or self.vmode == 'vertical':
						self.tune_velocities(pars,iy)
						pars.add('Vrad_%i' % (iy), value=self.vrad0[iy], vary = self.vary_vrad, min = self.Vmin, max = self.Vmax)

					if self.vmode == "bisymmetric":
						self.tune_velocities(pars,iy)
						pars.add('Vrad_%i' % (iy), value=self.vrad0[iy], vary = self.vary_vrad, min = self.Vmin, max = self.Vmax)
						pars.add('Vtan_%i' % (iy), value=self.vtan0[iy], vary = self.vary_vtan, min = self.Vmin, max = self.Vmax)

				if "hrm" in self.vmode:
					pars.add('C1_%i' % (iy),value=self.c_k0[0][iy], vary = True, min = self.Vmin, max = self.Vmax)
					self.tune_velocities(pars,iy)
					k = 1
					for j in range(1,self.m_hrm+1):
						if k != self.m_hrm and self.m_hrm != 1:
							pars.add('C%s_%i' % (k+1,iy), value=self.c_k0[k][iy], vary = self.vary_ck, min = self.Vmin, max = self.Vmax)
						pars.add('S%s_%i' % (j,iy), value=self.s_k0[j-1][iy], vary = self.vary_sk, min = self.Vmin, max = self.Vmax)
						k = k + 1


class Models(Config_params):

			def kinmdl_dataset(self, pars, i, xy_mesh, r_space, r_n, r_2 = None, disp = False):

				pars = pars.valuesdict()
				pa = pars['pa'] % 360
				eps = pars['eps']
				x0,y0 = pars['x0'],pars['y0']

				# For inner interpolation
				r1, r2 = self.rings_pos[0], self.rings_pos[1]
				if "hrm" not in self.vmode and self.v_center != 0:
					if self.v_center == "extrapolate":
						v1, v2 = pars["Vrot_0"], pars["Vrot_1"]
						v_int =  v_interp(0, r2, r1, v2, v1 )
						pars["Vrot_%i" % (self.index_v0)] = v_int

						if self.vmode == "radial" or self.vmode == "bisymmetric":
							v1, v2 = pars["Vrad_0"], pars["Vrad_1"]
							v_int =  v_interp(0, r2, r1, v2, v1 )
							pars["Vrad_%i" % (self.index_v0)] = v_int
						if self.vmode == "bisymmetric":
							v1, v2 = pars["Vtan_0"], pars["Vtan_1"]
							v_int =  v_interp(0, r2, r1, v2, v1 )
							pars["Vtan_%i" % (self.index_v0)] = v_int
					else:
						# This only applies to Vt component in circ
						# I made up this index.
						if self.vmode in ["circular", 'edgeon']:
							pars["Vrot_%i" % (self.index_v0)] = self.v_center


				'''
				Dispersion and Vz are always extrapolated to the origin.
				But only if the first ring fitted is different from zero.
				'''
				if self.r1st !=0:
					s1, s2 = pars["Sig_0"], pars["Sig_1"]
					s_int =  v_interp(0, r2, r1, s2, s1 )
					pars["Sig_%i" % (self.index_v0)] = s_int
					if self.vmode == 'vertical':
						vz1, vz2 = pars["Vrad_0"], pars["Vrad_1"]
						vz_int =  v_interp(0, r2, r1, vz2, vz1 )
						pars["Vrad_%i" % (self.index_v0)] = vz_int

					if  "hrm" in self.vmode and self.v_center == "extrapolate":
						for k in range(1,self.m_hrm+1) :
							v1, v2 = pars['C%s_%i'% (k,0)], pars['C%s_%i'% (k,1)]
							v_int =  v_interp(0, r2, r1, v2, v1 )
							pars["C%s_%i" % (k, self.index_v0)] = v_int

							v1, v2 = pars['S%s_%i'% (k,0)], pars['S%s_%i'% (k,1)]
							v_int =  v_interp(0, r2, r1, v2, v1 )
							pars["S%s_%i" % (k, self.index_v0)] = v_int

				# Weights are computed on the midplane z = 0, so this is ok.
				Weights_xy = weigths_w(xy_mesh,pa,eps,x0,y0,r_2,r_space,r_n,pixel_scale=self.pixel_scale,vmode=self.vmode)
				Sig = pars['Sig_%i'% i]
				modl0 = (SIGMA_MODEL(xy_mesh,Sig,pa,eps,x0,y0))*Weights_xy
				if disp:
					return modl0


				if "hrm" not in self.vmode:
					Vrot = pars['Vrot_%i'% i]

				if self.vmode in ["circular", 'edgeon']:
					modl1 = (SIGMA_MODEL(xy_mesh,Vrot,pa,eps,x0,y0))*Weights_xy
					return modl0,modl1
				if self.vmode == "radial" or self.vmode == 'vertical':
					Vrad = pars['Vrad_%i'% i]
					v1 = (SIGMA_MODEL(xy_mesh,Vrot,pa,eps,x0,y0))*Weights_xy
					v2 = (SIGMA_MODEL(xy_mesh,Vrad,pa,eps,x0,y0))*Weights_xy
					return modl0,v1,v2
				if self.vmode == 'ff':
					toggle=self.vrad0[i] != 0 if i in range(self.nrings) else 1
					Vrad=toggle*Vrot
					v1 = (SIGMA_MODEL(xy_mesh,Vrot,pa,eps,x0,y0))*Weights_xy
					v2 = (SIGMA_MODEL(xy_mesh,Vrad,pa,eps,x0,y0))*Weights_xy
					return modl0,v1,v2
				if self.vmode == "bisymmetric":
					Vrad = pars['Vrad_%i'% i]
					Vtan = pars['Vtan_%i'% i]
					if Vrad != 0 and Vtan != 0:
						phi_b = pars['phi_b'] % (2*np.pi)
						v2 = (SIGMA_MODEL(xy_mesh,Vrad,pa,eps,x0,y0))*Weights_xy
						v3 = (SIGMA_MODEL(xy_mesh,Vtan,pa,eps,x0,y0))*Weights_xy
						v1 = (SIGMA_MODEL(xy_mesh,Vrot,pa,eps,x0,y0))*Weights_xy
					else:
						v1 = (SIGMA_MODEL(xy_mesh,Vrot,pa,eps,x0,y0))*Weights_xy
						v2=v1*0
						v3=v1*0
					return modl0,v1,v2,v3
				if "hrm" in self.vmode:
					C_k, S_k  = [pars['C%s_%i'% (k,i)] for k in range(1,self.m_hrm+1)], [pars['S%s_%i'% (k,i)] for k in range(1,self.m_hrm+1)]
					Ck = [(SIGMA_MODEL(xy_mesh,ck,pa,eps,x0,y0))*Weights_xy for ck in C_k]
					Sk = [(SIGMA_MODEL(xy_mesh,sk,pa,eps,x0,y0))*Weights_xy for sk in S_k]
					vels=[Ck,Sk]
					flatCS=list(chain(*vels))
					return [modl0]+flatCS




class Fit_kin_mdls(Models):

		def residual(self, pars):
			Vsys = pars['Vsys']
			pa = pars['pa'] % 360
			eps = pars['eps']
			x0, y0 = pars['x0'],pars['y0']
			inc = eps_2_inc(eps)
			print(pars)
			
			# Weights (ny, nx)
			theta, cos_theta0 = AZIMUTHAL_ANGLE([self.ny,self.nx],pa,eps,x0,y0)
			if self.WEIGHT == 0:
				cos_theta = 1
			elif self.WEIGHT == 1:
				cos_theta = abs(cos_theta0)
			elif self.WEIGHT == -1:
				cos_theta = abs(cos_theta0)
				cos_theta = np.exp(-cos_theta)
			else :
				cos_theta = 1

			# Add Lagging in the vertical direction 
			dist_tmp = Angdist(Vsys/__c__)
			_, scale_pc_arc_tmp = dist_tmp.comv_distance()
			if self.gamma_pc !=0:
				# gamma units km/s/arcs
				self.gamma = self.gamma_pc*scale_pc_arc_tmp # (km/s/pc * pc/arcs)
			
			# Compute R, and the azimuthal angle in the disk plane (n_depth, ny, nx)
			#if self.vmode == 'edgeon':
			if True:			
				rk, vk,sigmak = np.empty_like(self.rings_pos, dtype=float),np.empty_like(self.rings_pos, dtype=float),np.empty_like(self.rings_pos, dtype=float) 
				for k in range(self.nrings):
					try:
						rk[k] = self.rings_pos[k]
						vk[k] = pars[f'Vrot_{k}']
						sigmak[k] = pars[f'Sig_{k}']						
					except(IndexError): pass
				
				mask = self.mom0 != 0
				cube = np.zeros_like(self.datacube)
		
			if self.vmode == 'edgeon':				
				_ = self.spec3d.vectorize_voxel(self.XY_mesh,x0,y0,pa*np.pi/180,eps,Vsys,self.pixel_scale,self.s_arr,rk,vk,sigmak,cube,mask,self.gamma)
			else:
				# Compute R, theta for every zdisk-vertical distance (n_hz, ny, nx)
				r_n_3d, theta_3d, Sz, Dz =  R_disk(self.XY_mesh,self.z_arr_arcs,pa*np.pi/180,eps,x0,y0,self.pixel_scale,self.vmode)
				theta = theta_3d


				# Azimuthal lagg in the vertical direction
				gamma1 = self.gamma
				gamma2 = self.gamma

				vlagg = np.zeros_like(Dz)
				vlagg[Dz>0] = gamma1*abs(Dz[Dz>0])
				vlagg[Dz<0] = gamma2*abs(Dz[Dz<0])

															
				# prepare cube
				cube = np.zeros_like(self.datacube)
				sigma_3d = np.zeros_like(theta_3d)
				velmap_3d = np.zeros_like(theta_3d)
						
			#########################################################
			# ----- Construct vertical disk structure ----
			#########################################################			
			for k, z_sample in enumerate(self.z_arr_arcs):
				if self.vmode == 'edgeon': continue
				
				if self.vmode != 'edgeon':
					r_n = r_n_3d[k] 
					cos = np.cos(theta_3d[k]) 
					sin = np.sin(theta_3d[k])
					
				twoDmdls = dataset_to_2D(
					[self.ny,self.nx], self.n_annulus, self.rings_pos, r_n, self.XY_mesh, self.kinmdl_dataset,
					self.vmode, self.v_center, pars, self.index_v0, nmodls=self.Vk
					)
				sigmap = twoDmdls[0]
				interp_model = twoDmdls[1:]
				
				"""
				---------*       Interpolation task       *------------------------
				Analysis of the inner radius.
				The analysis is performed only if the first ring starts at r[0] !=0.
				This does not make sence for edgeon model so lets save some time here

				"""
				if self.r1st !=0 and self.vmode != 'edgeon':
					mask_inner = np.where( (r_n < self.rings_pos[0] ) )
					x_r0,y_r0 = self.XY_mesh[0][mask_inner], self.XY_mesh[1][mask_inner]
					r_space_0 = self.rings_pos[0]
					R_n = r_n[mask_inner]					

					#(a) velocity rises linearly from zero: r1 = 0, v1 = 0
					if self.v_center == 0 or (self.v_center != "extrapolate" and self.vmode != "circular"):
						#Velocity and Sigma
						VS_xy_mdl = self.kinmdl_dataset(pars, 0, (x_r0,y_r0), r_n = R_n, r_2 = 0, r_space = r_space_0)
						S_xy_mdl=VS_xy_mdl[0]
						V_xy_mdl=VS_xy_mdl[1:]
						for k,mdl2d in enumerate(interp_model):
							v_new_2 = V_xy_mdl[k][1]
							mdl2d[mask_inner] = v_new_2
					else:
						r2 = self.rings_pos[0] 		# ring posintion
						v1_index = self.index_v0	# index of velocity
						#Velocity and Sigma
						VS_xy_mdl0 = self.kinmdl_dataset(pars, v1_index, (x_r0,y_r0), r_n = R_n, r_2 = r2, r_space = r_space_0)
						S_xy_mdl0=VS_xy_mdl0[0]
						V_xy_mdl0=VS_xy_mdl0[1:]

						r1 = 0 					# ring posintion
						v2_index = 0			# index of velocity
						#Velocity and Sigma
						VS_xy_mdl1 = self.kinmdl_dataset(pars, v2_index, (x_r0,y_r0), r_n = R_n, r_2 = r1, r_space = r_space_0)
						S_xy_mdl1=VS_xy_mdl1[0]
						V_xy_mdl1=VS_xy_mdl1[1:]

						for k in range(len(interp_model)):
							v_new_1=V_xy_mdl0[k][0]
							v_new_2=V_xy_mdl1[k][1]
							v_new = v_new_1 + v_new_2
							(interp_model[k])[mask_inner] = v_new

					# Dispersion is always extrpolated:
					r2 = self.rings_pos[0] 		# ring posintion
					v1_index = self.index_v0	# index of velocity
					S_xy_mdl = self.kinmdl_dataset(pars, v1_index, (x_r0,y_r0), r_n = R_n, r_2 = r2, r_space = r_space_0, disp= True )
					v_new_1 = S_xy_mdl[0]

					r1 = 0 					# ring posintion
					v2_index = 0			# index of velocity
					S_xy_mdl = self.kinmdl_dataset(pars, v2_index, (x_r0,y_r0), r_n = R_n, r_2 = r1, r_space = r_space_0, disp = True)
					v_new_2 = S_xy_mdl[1]

					v_new = v_new_1 + v_new_2
					sigmap[mask_inner] = v_new
					
				if self.vmode=='circular':
					vt=interp_model[0]
					vt*=cos*np.sin(inc)
					msk=vt!=0
					velmap=vt+msk*Vsys
				if self.vmode=='radial':
					[vt,vr]=interp_model
					vt*=np.sin(inc)*cos
					vr*=np.sin(inc)*sin
					velsum=vt+vr
					msk=velsum!=0
					velmap=velsum+msk*Vsys
				if self.vmode=='vertical':
					[vt,vz]=interp_model
					vt*=np.sin(inc)*cos
					vz*=np.cos(inc)
					velsum=vt+vz
					msk=velsum!=0
					velmap=velsum+msk*Vsys
				if self.vmode=='ff':
					alpha = pars['alpha']
					[vt,vr]=interp_model
					p=np.sqrt(2*(1-alpha**2))
					vt*=np.sin(inc)*cos
					vr*=-p*np.sin(inc)*sin
					velsum=vt+vr
					msk=velsum!=0
					velmap=velsum+msk*Vsys
				if self.vmode=='bisymmetric':
					phi_b = pars['phi_b'] % (2*np.pi)
					[vt,v2r,v2t]=interp_model
					vt*=np.sin(inc)*cos
					theta_b=theta-phi_b
					v2r*=-1*np.sin(inc)*sin*np.sin(2*theta_b)
					v2t*=-1*np.sin(inc)*cos*np.cos(2*theta_b)
					velsum=vt+v2r+v2t
					msk=velsum!=0
					velmap=velsum+msk*Vsys
				if 'hrm' in self.vmode:
					velsum=0
					for k in range(self.m_hrm):
						CkSk=interp_model[k]*np.cos((k+1)*theta)*np.sin(inc)+interp_model[k+self.m_hrm]*np.sin((k+1)*theta)*np.sin(inc)
						velsum+=CkSk
					msk=velsum!=0
					velmap=velsum+msk*Vsys


				velmap_3d[k] = velmap
				sigma_3d[k] = sigmap
			
				#plt.imshow(velmap*(velmap/velmap), origin = 'lower');plt.show()
				#plt.imshow(velmap*(sigmap/sigmap), origin = 'lower');plt.show()
				
			#########################################################
			# ----- Integration along the LOS for every s_sample ----
			#########################################################			
				
			if self.vmode != 'edgeon':
				# --- Luminosity density --
				j_z = self.spec3d.j_z(self.z_arr_arcs)[:,None,None]
				j_R = np.ones_like(r_n_3d) if self.normalize else self.spec3d.j_R(r_n_3d)
				# luminosity density  
				j_Rz = j_R*j_z
				ds = abs(self.dz_arcs/(1-eps))
				_ = self.spec3d.create_cube_vyx(cube,self.XY_mesh,velmap_3d,sigma_3d,self.z_arr_arcs,j_Rz, ds, self.mom0_msk)						
				mom0_tmp = np.sum(sigma_3d, axis = 0)
			else:
				cube = cube
				mom0_tmp = np.sum(cube, axis = 0)				

			# Create model cube affected by beamsmearing		
			mom0_mdl,mom1_mdl,mom2_mdl_kms,mom2_mdl_A,cube_mdl=self.cube_modl.create_cube(
			self.mom1,mom0_tmp,cube,self.padded_cube,self.padded_psf,self.cube_slices,pass_cube=self.fit_from_cube
			)
			ntotal_d=(self.nx*self.ny)
			neff_d=np.sum( (np.isfinite(self.mom1)) & (self.mom0!=0) )
			neff_m=np.sum( (np.isfinite(mom1_mdl)) & (mom1_mdl!=0) & (self.mom0!=0))
			msk = (mom0_tmp!=0) & (self.mom0>0)
			if neff_m == 0 : neff_m = 1

			#plt.imshow((mom1_mdl-self.mom1)*(mom1_mdl/mom1_mdl), vmin=-200, vmax=200, origin = 'lower');plt.show()
			#plt.imshow((mom1_mdl-Vsys)*(mom1_mdl/mom1_mdl), vmin=-200, vmax=200, origin = 'lower');plt.show()
			#plt.imshow(mom2_mdl_kms*(mom1_mdl/mom1_mdl), origin = 'lower');plt.show()
			#xc, yc = 23, 28
			#xc, yc = self.nx//2, self.ny//2
			#plt.plot(np.arange(self.nz), cube_mdl[:, yc, xc], 'r-' ); plt.plot(np.arange(self.nz), self.datacube[:, yc, xc], 'k-' );
			#plt.plot(np.arange(self.nz), cube[:, yc, xc], 'g-' ); plt.show()#;quit()						
			if self.fit_from_cube:
				residual_xy=np.sum((self.datacube-cube_mdl)**2,axis=0)
				residual_xy*=msk
				del cube_mdl
				residual = msk*( (self.mom2-mom2_mdl_kms)**2) + msk*((self.mom1-mom1_mdl)**2 )*cos_theta
			else:
				if self.vary_disp:
					residual = msk*( (self.mom2-mom2_mdl_kms)**2) + msk*((self.mom1-mom1_mdl)**2)*cos_theta + msk*((self.peakI0-cube_mdl)**2 )
				else:
					residual = msk*( (self.mom2-mom2_mdl_kms)**2) + msk*((self.mom1-mom1_mdl)**2)*cos_theta + msk*((self.peakI0-cube_mdl)**2 )
					#n=len(residual)

			# weight effective number of data
			w_neff = (ntotal_d**2)/(neff_d*neff_m)
			if self.fit_from_cube:
				a = np.sqrt( (residual_xy + residual)*w_neff).ravel()
				return a
			else:
				return np.sqrt( residual*w_neff ).ravel()


		def reduce_func(res,x):
			N  = len(res)
			out = np.nansum(res*res)/N
			return out


		def run_mdl(self):
			pars = Parameters()
			self.assign_vels(pars)
			self.assign_constpars(pars)
			res = self.residual(pars)

			out1 = Minimizer(self.residual, pars)#, iter_cb=self.iter_cb)
			options={'verbose':2,'max_nfev':self.MAXF*(self.nparams+1),'xtol':self.XTOL,'gtol':self.XTOL,'ftol':self.XTOL}
			#out=out1.minimize(method='least_squares',**options)
			out=out1.minimize(method='nelder')			

			return out

		def results(self):
			out = self.run_mdl()
			best = out.params
			N_free = out.nfree
			N_nvarys = out.nvarys
			N_data = out.ndata
			bic, aic = out.bic, out.aic
			red_chi = out.redchi

			phi_b=best["phi_b"].value if self.vmode=='bisymmetric' else 0
			std_phi_b=best["phi_b"].stderr if self.vmode=='bisymmetric' else 0
			alpha=best["alpha"].value if self.vmode=='ff' else 0
			std_alpha=best["alpha"].stderr if self.vmode=='ff' else 0
			if self.vmode=='ff': phi_b,std_phi_b=alpha,std_alpha

			constant_parms = np.asarray( [best["pa"].value, best["eps"].value, best["x0"].value,best["y0"].value, best["Vsys"].value, phi_b] )
			e_constant_parms =  [ best["pa"].stderr, best["eps"].stderr, best["x0"].stderr, best["y0"].stderr, best["Vsys"].stderr, std_phi_b]
			constant_parms[0]=constant_parms[0]%360
			constant_parms[-1]=constant_parms[-1]%(2*np.pi)

			pa, eps, x0, y0, Vsys, phi_b = constant_parms
			std_pa, std_eps, std_x0, std_y0, std_Vsys, std_phi_b = list(e_constant_parms)

			if "hrm" not in self.vmode:
				v_kin = ["Sig", "Vrot","Vrad", "Vtan"]
				nvkin=self.Vk if self.vmode != 'ff' else 2
				for i in range(nvkin):
					self.V_k[i] = [ best["%s_%s"%(v_kin[i],iy)].value for iy in range(self.nrings) ]
					self.V_k_std[i] = [ best["%s_%s"%(v_kin[i],iy)].stderr for iy in range(self.nrings) ]
					# In case something goes wrong with errors:
					if None in self.V_k_std[i] :  self.V_k_std[i] = len(self.V_k[i])*[1e-3]
			else:
				v_kin = ["C","S"]
				k = 0
				for j in range(len(v_kin)):
					for i in range(self.m_hrm):
						self.V_k[k] = [ best["%s%s_%s"%(v_kin[j],i+1,iy)].value for iy in range(self.nrings) ]
						self.V_k_std[k] = [ best["%s%s_%s"%(v_kin[j],i+1,iy)].stderr for iy in range(self.nrings) ]
						# In case something goes wrong with errors:
						if None in self.V_k_std[k] :  self.V_k_std[k] = len(self.V_k[k])*[1e-3]
						k = k + 1
				# Add dispersion at last position
				self.V_k[-1] = [ best["%s_%s"%('Sig',iy)].value for iy in range(self.nrings) ]
				self.V_k_std[-1] = [ best["%s_%s"%('Sig',iy)].stderr for iy in range(self.nrings) ]
				if None in self.V_k_std[-1] :  self.V_k_std[-1] = len(self.V_k[-1])*[1e-3]


			if None in e_constant_parms:  e_constant_parms = [1e-3]*len(constant_parms)
			if np.nan in e_constant_parms:  e_constant_parms = [1e-3]*len(constant_parms)

			create_3D =	best_3d_model(
			self.mommaps_obs,self.datacube,self.h,self.config,self.vmode,self.V_k,pa,eps,x0,y0,
			Vsys,self.rings_pos,self.ring_space,self.pixel_scale,self.v_center,self.m_hrm,phi_b,
			self.Vk,self.z_scale_p,self.s_arr
			)
			# Update these variables
			create_3D.n_depth = self.n_depth 
			create_3D.y_depth = self.s_arr
			create_3D.gamma = self.gamma
			mdls_3D = create_3D.model3D()
			mom01d,mom0axi,mom0_mdl,mom1_mdl,mom2_mdl_kms,mom2_mdl_A,cube_mdl,velmap_intr,sigmap_intr,twoDmodels = mdls_3D

			# We need to re-compute chisquare with the best model !
			# Compute the residuals
			res = ( self.datacube - cube_mdl) #/ self.ecube
			msk2D = np.isfinite(mom1_mdl) & (self.mom0!=0)
			msk3D = (self.datacube!=0)*(msk2D*np.ones(self.nz)[:,None,None]).astype(bool)
			cost = res[msk3D]

			N_data=len(cost)
			N_free = N_data - N_nvarys

			# Residual sum of squares
			rss2 = (cost)**2
			rss=np.nansum(rss2)
			# Compute reduced chisquare
			chisq =rss
			red_chi = chisq/ (N_free)

			chisq = chisq if np.isfinite(chisq) else 1e4
			red_chi = red_chi if np.isfinite(red_chi) else 1e4
			rss = rss if np.isfinite(rss) else 1e4

			# Akaike Information Criterion
			aic = N_data*np.log(rss/N_data) + 2*N_nvarys
			#Bayesian Information Criterion
			bic = N_data*np.log(rss/N_data) + np.log(N_data)*N_nvarys

			for k in range(self.Vk):
				self.V_k[k] = np.asarray(self.V_k[k])
				self.V_k_std[k] = np.asarray(self.V_k_std[k])
			errors = [[],[],[]]
			if "hrm" not in self.vmode:
				errors[0],errors[1] = self.V_k_std,e_constant_parms
			else:
				errors[0],errors[1] = [self.V_k_std[0:self.m_hrm],self.V_k_std[self.m_hrm:-1],self.V_k_std[-1]],e_constant_parms


			if len(self.V_k) != len(self.V_k_std)  : self.V_k_std = [1e-3]*len(self.V_k)

			out_data = [N_free, N_nvarys, N_data, bic, aic, red_chi]
			return mdls_3D, self.V_k, pa, eps , x0, y0, Vsys, phi_b, out_data, errors, self.rings_pos
