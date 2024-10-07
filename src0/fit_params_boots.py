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

def Rings(xy_mesh,pa,eps,x0,y0,pixel_scale):
	(x,y) = xy_mesh

	X = (- (x-x0)*np.sin(pa) + (y-y0)*np.cos(pa))
	Y = (- (x-x0)*np.cos(pa) - (y-y0)*np.sin(pa))

	R= np.sqrt(X**2+(Y/(1-eps))**2)

	return R*pixel_scale


from src0.kin_components import CIRC_MODEL
from src0.kin_components import HARMONIC_MODEL
from src0.kin_components import SIGMA_MODEL
from src0.kin_components import AZIMUTHAL_ANGLE,SIN_COS
from src0.pixel_params import pixels,v_interp,eps_2_inc
from src0.weights_interp import weigths_w
from src0.create_2D_kin_models import bidi_models
from src0.create_3D_cube_model import best_3d_model
from src0.create_dataset import dataset_to_2D
from src0.read_hdr import Header_info
from src0.momtools import GaussProf,trapecium
from src0.convolve_cube import Cube_creation,Zeropadding

from src0.constants import __c__
from src0.conv import conv2d,gkernel,gkernel1d
from src0.conv_spec1d import gaussian_filter1d,convolve_sigma



class Least_square_fit:
	#def __init__(self,vel_map, e_vel_map, guess, vary, vmode, config, rings_pos, ring_space, fit_method, e_ISM, pixel_scale, frac_pixel, v_center, m_hrm = 1, N_it = 1):
	def __init__(self,datacube, header, mommaps, e_vel_map, guess, vary, vmode, config, rings_pos, ring_space, frac_pixel, v_center, m_hrm = 1, N_it = 1):
	
		"""
		vary = [Vrot,Vrad,Vtan,PA,INC,XC,YC,VSYS,theta]
		"""

		self.N_it = N_it
		if self.N_it == 0:
			vary,self.vary_kin = vary*0,0
			vary,self.vary_kin = vary,1
		else:
			self.vary_kin = 0
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
			
		
		#flat parameters:
		self.params=np.hstack(guess)
		self.nparams=len(self.params[self.params!=0])
		self.m_hrm = m_hrm
		self.h=header
		self.rings_pos = rings_pos
		self.nrings = len(self.rings_pos)
		self.n_annulus = self.nrings - 1
		self.mommaps_obs=mommaps
		[self.mom0,self.mom1,self.mom2]=mommaps
		self.vel_map = self.mommaps_obs[1]
		self.ny,self.nx = self.vel_map.shape		
		self.e_vel_map = e_vel_map
		[self.emom0,self.emom1,self.emom2]=e_vel_map		
		self.vmode = vmode
		self.ring_space = ring_space
		self.fit_method = 'nelder'#'least_squares'
		self.config = config
		self.constant_params = constant_params
		self.osi = ["-", ".", ",", "#","%", "&", ""]

		# args to pass to minimize function
		self.kws={'maxiter':500,'fatol':1e-4,'adaptivessss':True}
		self.kwargs = {}
		#self.kwargs = {'ftol':1e-4,'gtol':1e-6,'xtol':1e-6,'max_nfev':max_nfev,'verbose':2}
		if self.N_it == 0:
			self.kwargs = {"ftol":1e8}
			
		X = np.arange(0, self.nx, 1)
		Y = np.arange(0, self.ny, 1)
		self.XY_mesh = np.meshgrid(X,Y)

		if self.xc0 % int(self.xc0) == 0 or self.yc0 % int(self.yc0) == 0 : 
			self.xc0, self.yc0 =  self.xc0 + 1e-5, self.yc0 + 1e-5
		self.crval3,self.cdelt3,self.pixel_scale=Header_info(self.h,self.config).read_header()
		self.r_n = Rings(self.XY_mesh,self.pa0*np.pi/180,self.eps0,self.xc0,self.yc0,self.pixel_scale)
		self.r_n = np.asarray(self.r_n, dtype = np.longdouble)
		self.frac_pixel = frac_pixel
		self.v_center = v_center

					
		self.e_ISM = 0
		interp_model = np.zeros((self.ny,self.nx))
		self.index_v0 = 123456
		if vmode == "circular": self.Vk = 1+1
		if vmode == "radial" or vmode == 'vertical': self.Vk = 2+1
		if vmode == "bisymmetric": self.Vk = 3+1
		

		self.Vrot, self.Vrad, self.Vtan = 0,0,0
		if "hrm" not in self.vmode:
			self.V_k = [self.vrot0*0,self.vrot0*0, self.vrot0*0, self.vrot0*0] 
			self.V_k_std = [0,0,0,0]
		else: 
			self.V_k = [0]*(2*self.m_hrm)+[0] 
			self.V_k_std = [0]*(2*self.m_hrm)+[0]
			self.Vk = 2*self.m_hrm+1

		config_const = config['constant_params']
		self.Vmin, self.Vmax = -450, 450
		eps_min, eps_max = 1-np.cos(10*np.pi/180),1-np.cos(80*np.pi/180)
		self.PAmin,self.PAmax,self.vary_pa = config_const.getfloat('MIN_PA', -360*2), config_const.getfloat('MAX_PA', 360*2),config_const.getboolean('FIT_PA', self.vary_pa)
		self.INCmin,self.INCmax,self.vary_eps = config_const.getfloat('MIN_INC', eps_min), config_const.getfloat('MAX_INC', eps_max),config_const.getboolean('FIT_INC', self.vary_eps)
		# To change input INCmin (in deg) and INCmax (in deg) values from the config file to eps 
		if self.INCmin >1:  self.INCmin = 1-np.cos(self.INCmin*np.pi/180)
		if self.INCmax >1:  self.INCmax = 1-np.cos(self.INCmax*np.pi/180)
		self.X0min,self.X0max,self.vary_xc = config_const.getfloat('MIN_X0', 0), config_const.getfloat('MAX_X0', self.nx),config_const.getboolean('FIT_X0', self.vary_xc)
		self.Y0min,self.Y0max,self.vary_yc = config_const.getfloat('MIN_Y0', 0), config_const.getfloat('MAX_Y0', self.ny),config_const.getboolean('FIT_Y0', self.vary_yc)
		self.VSYSmin,self.VSYSmax,self.vary_vsys = config_const.getfloat('MIN_VSYS', 0), config_const.getfloat('MAX_VSYS', 10*__c__),config_const.getboolean('FIT_VSYS', self.vary_vsys)
		self.PAbarmin,self.PAbarmax,self.vary_phib = config_const.getfloat('MIN_PHI_BAR', -np.pi), config_const.getfloat('MAX_PHI_BAR', np.pi),config_const.getboolean('FIT_PHI_BAR', self.vary_phib)
		self.WEIGHT=config_const.getint('WEIGHT',0)
		self.XTOL=config_const.getfloat('XTOL',1e-5)
		self.MAXF=config_const.getint('MAXF',15)

		config_general = config['general']
		outliers = config_general.getboolean('outliers', False)
		if outliers: self.kwargs["loss"]="soft_l1"
		#self.kwargs["loss"]="soft_l1"

		
		self.eline_A=config_general.getfloat('eline',None)
		self.vary_disp=config_general.getboolean('fit_dispersion',False)
		self.fwhm_inst_A=config_general.getfloat('fwhm_inst',None)
		self.sigma_inst_A=self.fwhm_inst_A/(np.sqrt(8*np.log(2))) if self.fwhm_inst_A is not None else None
		self.sigma_inst_kms=(self.sigma_inst_A/self.eline_A)*__c__ if self.fwhm_inst_A is not None else None


		self.min_sig=0
		if self.sigma_inst_A is not None:
			sig0 = np.sqrt(self.sig0**2-self.sigma_inst_kms**2)
			#if donomitated by instrumental the sig0 is NaN:
			msk_inst=~np.isfinite(sig0)
			sig0[msk_inst]=self.sig0[msk_inst]
			self.sig0=sig0
			self.min_sig=0#self.sigma_inst_kms*1
		
		if not self.vary_disp and self.fwhm_inst_A is not None:
			self.sig0=np.ones_like(self.sig0)*self.sigma_inst_kms					
		

		# Rename to capital letters
		self.PA = self.pa0
		self.EPS =  self.eps0 
		self.X0 = self.xc0
		self.Y0 = self.yc0
		self.VSYS = self.vsys0
		self.PHI_BAR = self.phi_bar
		self.vary_disp=0

    
class Config_params(Least_square_fit):

		def assign_constpars(self,pars):

			#if self.config in self.osi:
			pars.add('Vsys', value=self.VSYS, vary = self.vary_vsys, min = self.VSYSmin, max = self.VSYSmax)
			pars.add('pa', value=self.PA, vary = self.vary_pa, min = self.PAmin, max = self.PAmax)
			pars.add('eps', value=self.EPS, vary = self.vary_eps, min = self.INCmin, max = self.INCmax)
			pars.add('x0', value=self.X0, vary = self.vary_xc,  min = self.X0min, max = self.X0max)
			pars.add('y0', value=self.Y0, vary = self.vary_yc, min = self.Y0min, max = self.Y0max)
			if self.vmode == "bisymmetric":
				pars.add('phi_b', value=self.PHI_BAR, vary = self.vary_phib)#, min = 0, self.PAbarmin , max = self.PAbarmax)


		def tune_velocities(self,pars,iy):
				if "hrm" not in self.vmode:
					if self.vmode == "radial" or self.vmode == "vertical":
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
					pars.add('Vrot_%i' % (iy),value=self.vrot0[iy], vary = self.vary_vrot, min = self.Vmin, max = self.Vmax)
						
					#if self.vrad0[iy] == 0 and self.vtan0[iy] ==0:
					#	self.vary_vrad = False
					#	self.vary_vtan = False
										
					if self.vmode == "radial" or self.vmode == "vertical":
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

			def kinmdl_dataset(self, pars, i, xy_mesh, r_space, r_2 = None, disp = False):

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
						if self.vmode in ["circular"]:
							pars["Vrot_%i" % (self.index_v0)] = self.v_center
						
				
				# Dispersion and Vz are  always extrapolated
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


				Sig = pars['Sig_%i'% i]
				modl0 = (SIGMA_MODEL(xy_mesh,Sig,pa,eps,x0,y0))*weigths_w(xy_mesh,pa,eps,x0,y0,r_2,r_space,pixel_scale=self.pixel_scale)
				if disp:
					return modl0


				if "hrm" not in self.vmode:
					Vrot = pars['Vrot_%i'% i]
					
				if self.vmode == "circular":
					modl1 = (SIGMA_MODEL(xy_mesh,Vrot,pa,eps,x0,y0))*weigths_w(xy_mesh,pa,eps,x0,y0,r_2,r_space,pixel_scale=self.pixel_scale)
					return modl0,modl1
																								
				if self.vmode == "radial" or self.vmode == 'vertical':
					Vrad = pars['Vrad_%i'% i]
					v1 = (SIGMA_MODEL(xy_mesh,Vrot,pa,eps,x0,y0))*weigths_w(xy_mesh,pa,eps,x0,y0,r_2,r_space,pixel_scale=self.pixel_scale)																								
					v2 = (SIGMA_MODEL(xy_mesh,Vrad,pa,eps,x0,y0))*weigths_w(xy_mesh,pa,eps,x0,y0,r_2,r_space,pixel_scale=self.pixel_scale)					
					return modl0,v1,v2
				if self.vmode == "bisymmetric":
					Vrad = pars['Vrad_%i'% i]
					Vtan = pars['Vtan_%i'% i]
					if Vrad != 0 and Vtan != 0:
						phi_b = pars['phi_b'] % (2*np.pi)
						v2 = (SIGMA_MODEL(xy_mesh,Vrad,pa,eps,x0,y0))*weigths_w(xy_mesh,pa,eps,x0,y0,r_2,r_space,pixel_scale=self.pixel_scale)						
						v3 = (SIGMA_MODEL(xy_mesh,Vtan,pa,eps,x0,y0))*weigths_w(xy_mesh,pa,eps,x0,y0,r_2,r_space,pixel_scale=self.pixel_scale)
						v1 = (SIGMA_MODEL(xy_mesh,Vrot,pa,eps,x0,y0))*weigths_w(xy_mesh,pa,eps,x0,y0,r_2,r_space,pixel_scale=self.pixel_scale)																								
					else:
						v1 = (SIGMA_MODEL(xy_mesh,Vrot,pa,eps,x0,y0))*weigths_w(xy_mesh,pa,eps,x0,y0,r_2,r_space,pixel_scale=self.pixel_scale)
						v2=v1*0
						v3=v1*0
					return modl0,v1,v2,v3
				if "hrm" in self.vmode:
					C_k, S_k  = [pars['C%s_%i'% (k,i)] for k in range(1,self.m_hrm+1)], [pars['S%s_%i'% (k,i)] for k in range(1,self.m_hrm+1)]
					Ck = [(SIGMA_MODEL(xy_mesh,ck,pa,eps,x0,y0))*weigths_w(xy_mesh,pa,eps,x0,y0,r_2,r_space,pixel_scale=self.pixel_scale) for ck in C_k]
					Sk = [(SIGMA_MODEL(xy_mesh,sk,pa,eps,x0,y0))*weigths_w(xy_mesh,pa,eps,x0,y0,r_2,r_space,pixel_scale=self.pixel_scale) for sk in S_k]																
					vels=[Ck,Sk]
					flatCS=list(chain(*vels))
					return [modl0]+flatCS




class Fit_kin_mdls(Models):

		def Vk2D(self,pars):
			pa = pars['pa'] % 360
			eps = pars['eps']
			x0,y0 = pars['x0'],pars['y0']
			
			self.r_n = Rings(self.XY_mesh,pa*np.pi/180,eps,x0,y0,self.pixel_scale)									
			twoDmdls= dataset_to_2D([self.ny,self.nx], self.n_annulus, self.rings_pos, self.r_n, self.XY_mesh, self.kinmdl_dataset, self.vmode, self.v_center, pars, self.index_v0, nmodls=self.Vk)
			interp_sig_model=twoDmdls[0]
			interp_model=twoDmdls[1:]
			
			mask_inner = np.where( (self.r_n < self.rings_pos[0] ) )
			x_r0,y_r0 = self.XY_mesh[0][mask_inner], self.XY_mesh[1][mask_inner] 
			r_space_0 = self.rings_pos[0]


			for mdl2d in interp_model:
				#(a) velocity rises linearly from zero: r1 = 0, v1 = 0
				if self.v_center == 0 or (self.v_center != "extrapolate" and self.vmode != "circular"):
					V_xy_mdl = self.kinmdl_dataset(pars, 0, (x_r0,y_r0), r_2 = 0, r_space = r_space_0)[1]
					v_new_2 = V_xy_mdl[1]
					mdl2d[mask_inner] = v_new_2
				else:
					r2 = self.rings_pos[0] 		# ring posintion
					v1_index = self.index_v0	# index of velocity
					V_xy_mdl = self.kinmdl_dataset(pars, v1_index, (x_r0,y_r0), r_2 = r2, r_space = r_space_0 )[1]
					v_new_1 = V_xy_mdl[0]

					r1 = 0 					# ring posintion
					v2_index = 0			# index of velocity
					V_xy_mdl = self.kinmdl_dataset(pars, v2_index, (x_r0,y_r0), r_2 = r1, r_space = r_space_0 )[1]
					v_new_2 = V_xy_mdl[1]

					v_new = v_new_1 + v_new_2
					mdl2d[mask_inner] = v_new
								
			#"""
			# Dispersion is always extrpolated to r=0:			
			r2 = self.rings_pos[0] 		# ring posintion
			v1_index = self.index_v0	# index of velocity
			V_xy_mdl = self.kinmdl_dataset(pars, v1_index, (x_r0,y_r0), r_2 = r2, r_space = r_space_0, disp= True )
			v_new_1 = V_xy_mdl[0]

			r1 = 0 					# ring posintion
			v2_index = 0			# index of velocity
			V_xy_mdl = self.kinmdl_dataset(pars, v2_index, (x_r0,y_r0), r_2 = r1, r_space = r_space_0, disp = True)
			v_new_2 = V_xy_mdl[1]

			v_new = v_new_1 + v_new_2
			interp_sig_model[mask_inner] = v_new
			#"""
			return interp_sig_model,interp_model

		

		def residual(self, pars):
			#print(pars)
			Vsys = pars['Vsys']
			pa = pars['pa'] % 360
			eps = pars['eps']
			x0,y0 = pars['x0'],pars['y0']
			inc=eps_2_inc(eps)
			theta,cos_theta0=AZIMUTHAL_ANGLE([self.ny,self.nx],pa,eps,x0,y0)
			sin,cos=SIN_COS(self.XY_mesh,pa,eps,x0,y0)
			
			if self.WEIGHT==0:
				cos_theta=1
			else:
				cos_theta=abs(cos_theta0) if self.WEIGHT==1 else (abs(cos_theta0))**self.WEIGHT
				
			#interp_sig_model,interp_model=self.Vk2D(pars)
			######

			self.r_n = Rings(self.XY_mesh,pa*np.pi/180,eps,x0,y0,self.pixel_scale)									
			twoDmdls= dataset_to_2D([self.ny,self.nx], self.n_annulus, self.rings_pos, self.r_n, self.XY_mesh, self.kinmdl_dataset, self.vmode, self.v_center, pars, self.index_v0, nmodls=self.Vk)
			sigmap=twoDmdls[0]
			interp_model=twoDmdls[1:]
			
			"""
			Analysis of the inner radius

			"""			
			mask_inner = np.where( (self.r_n < self.rings_pos[0] ) )
			x_r0,y_r0 = self.XY_mesh[0][mask_inner], self.XY_mesh[1][mask_inner] 
			r_space_0 = self.rings_pos[0]

			#(a) velocity rises linearly from zero: r1 = 0, v1 = 0
			if self.v_center == 0 or (self.v_center != "extrapolate" and self.vmode != "circular"):
				#Velocity and Sigma
				VS_xy_mdl = self.kinmdl_dataset(pars, 0, (x_r0,y_r0), r_2 = 0, r_space = r_space_0)
				S_xy_mdl=VS_xy_mdl[0]
				V_xy_mdl=VS_xy_mdl[1:]
				for k,mdl2d in enumerate(interp_model):
					v_new_2 = V_xy_mdl[k][1]
					mdl2d[mask_inner] = v_new_2
			else:
				r2 = self.rings_pos[0] 		# ring posintion
				v1_index = self.index_v0	# index of velocity
				#Velocity and Sigma			
				VS_xy_mdl0 = self.kinmdl_dataset(pars, v1_index, (x_r0,y_r0), r_2 = r2, r_space = r_space_0)			
				S_xy_mdl0=VS_xy_mdl0[0]
				V_xy_mdl0=VS_xy_mdl0[1:]
				
				r1 = 0 					# ring posintion
				v2_index = 0			# index of velocity
				#Velocity and Sigma
				VS_xy_mdl1 = self.kinmdl_dataset(pars, v2_index, (x_r0,y_r0), r_2 = r1, r_space = r_space_0)			
				S_xy_mdl1=VS_xy_mdl1[0]
				V_xy_mdl1=VS_xy_mdl1[1:]
				

				for k in range(len(interp_model)):								
					v_new_1=V_xy_mdl0[k][0]				
					v_new_2=V_xy_mdl1[k][1]
					v_new = v_new_1 + v_new_2
					(interp_model[k])[mask_inner] = v_new
								
			#"""
			# Dispersion is always extrpolated:			
			r2 = self.rings_pos[0] 		# ring posintion
			v1_index = self.index_v0	# index of velocity
			S_xy_mdl = self.kinmdl_dataset(pars, v1_index, (x_r0,y_r0), r_2 = r2, r_space = r_space_0, disp= True )
			v_new_1 = S_xy_mdl[0]

			r1 = 0 					# ring posintion
			v2_index = 0			# index of velocity
			S_xy_mdl = self.kinmdl_dataset(pars, v2_index, (x_r0,y_r0), r_2 = r1, r_space = r_space_0, disp = True)
			v_new_2 = S_xy_mdl[1]

			v_new = v_new_1 + v_new_2
			sigmap[mask_inner] = v_new	

			######
			
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


			msk=(velmap!=0) & (self.mom0!=0)
			mom1_mdl,mom2_mdl_kms=velmap,sigmap

			if self.vary_disp:
				residual =    msk*((self.mom1-mom1_mdl)**2)*cos_theta				
			else:									
				residual =   msk*((self.mom1-mom1_mdl)**2)*cos_theta				

			residual[~np.isfinite(residual)]=0
			n=len(residual)
			residual=np.sqrt(residual/n)			
			return residual


		def reduce_func(res,x):
			N  = len(res)
			out = np.nansum(res*res)/N
			return out	
			

		def run_mdl(self):
			pars = Parameters()
			self.assign_vels(pars)
			self.assign_constpars(pars)			
			res = self.residual(pars)
						
			out1 = Minimizer(self.residual, pars)
			options={'verbose':0,'max_nfev':self.MAXF*(self.nparams+1),'xtol':self.XTOL,'gtol':self.XTOL,'ftol':self.XTOL}		
			out=out1.minimize(method='least_squares',**options)		

			#fit_kws={'tol':1} 
			#out = out1.scalar_minimize(**fit_kws)
		
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
			
			constant_parms = np.asarray( [best["pa"].value, best["eps"].value, best["x0"].value,best["y0"].value, best["Vsys"].value, phi_b] )
			e_constant_parms =  [ best["pa"].stderr, best["eps"].stderr, best["x0"].stderr, best["y0"].stderr, best["Vsys"].stderr, std_phi_b] 
			constant_parms[0]=constant_parms[0]%360
			constant_parms[-1]=constant_parms[-1]%(2*np.pi)			
			
			pa, eps, x0, y0, Vsys, phi_b = constant_parms
			std_pa, std_eps, std_x0, std_y0, std_Vsys, std_phi_b = list(e_constant_parms)
			

			if "hrm" not in self.vmode:
				v_kin = ["Sig", "Vrot","Vrad", "Vtan"]
				for i in range(self.Vk):
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
				# Add dispersion at las position
				self.V_k[-1] = [ best["%s_%s"%('Sig',iy)].value for iy in range(self.nrings) ]
				self.V_k_std[-1] = [ best["%s_%s"%('Sig',iy)].stderr for iy in range(self.nrings) ]
				if None in self.V_k_std[-1] :  self.V_k_std[-1] = len(self.V_k[-1])*[1e-3]				
														
			for k in range(self.Vk):
				self.V_k[k] = np.asarray(self.V_k[k])
				self.V_k_std[k] = np.asarray(self.V_k_std[k])
			errors = [[],[]]
			if "hrm" not in self.vmode:
				errors[0],errors[1] = self.V_k_std,e_constant_parms
			else:
				errors[0],errors[1] = [self.V_k_std[0:self.m_hrm],self.V_k_std[self.m_hrm:-1],self.V_k_std[-1]],e_constant_parms[:-1]



			if len(self.V_k) != len(self.V_k_std)  : self.V_k_std = [1e-3]*len(self.V_k)
			
			return self.V_k, pa, eps , x0, y0, Vsys, phi_b


