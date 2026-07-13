import numpy as np
import time
import os
import matplotlib.pylab as plt
from scipy.stats import circstd,circmean
from multiprocessing import Pool, cpu_count

from .eval_tab_model import tab_mod_vels
from .phi_bar_sky import pa_bar_sky
from .fit_params import Fit_kin_mdls as fit
from .fit_params_boots import Fit_kin_mdls as fit_boots
from .tools_fits import array_2_fits
from .create_2D_vlos_model import best_2d_model
from .read_hdr import Header_info
from .pixel_params import inc_2_eps, eps_2_inc,e_eps2e_inc, Rings, make_outer_mask_sky
from .convolve_cube import Cube_creation,Zeropadding,Cube_operations
from .utils import circmean

from .cloud_tilted_rings import TiltedRingModel, CubeConfig, Ring
from .cloud_fit_engine import (
    build_params, fit_rings, make_weight_map,
    residual_cube, rotation_curve,
    set_bounds, _print_params_summary,
)
from .params import Set_params
from .extract_prms import extractp,extract_harmonics

class Harmonic_model:
	def __init__(self, vmode, galaxy, obs_cube, header, mommaps, emoms, guess0, vary, n_it, rstart, rfinal, ring_space, frac_pixel, inner_interp, delta, bar_min_max, config, outdir,cube_class,psf_lsf,m_hrm):


		self.galaxy=galaxy
		self.obs_cube=obs_cube
		self.hdr=header
		self.mommaps=mommaps
		self.vel_copy=np.copy(self.mommaps[1])
		self.vel=self.mommaps[1]
		self.emoms=emoms
		self.guess0=guess0
		self.vary=vary
		self.n_it,self.n_it0=n_it,n_it
		self.rstart=rstart
		self.rfinal=rfinal
		self.ring_space=ring_space
		self.frac_pixel=frac_pixel
		self.inner_interp=inner_interp
		self.rwidth=delta
		self.bar_min_max=bar_min_max
		self.config=config
		self.m_hrm = m_hrm
		self.pixel_scale=header.scale
		self.psf_lsf=psf_lsf

		if self.n_it == 0: self.n_it = 1
		rend = self.rfinal
		if (self.rfinal-self.rstart) % self.ring_space == 0 :
			# To include the last ring :
			rend = self.rfinal + self.ring_space


		self.rings = np.arange(self.rstart, rend, self.ring_space)
		self.nrings = len(self.rings)

		self.r_bar_min, self.r_bar_max = self.bar_min_max

		self.pa0,self.eps0,self.x0,self.y0,self.vsys0,self.theta_b = self.guess0
		self.inc0=eps_2_inc(self.eps0)
		self.vmode = "hrm"
		[nz,ny,nx] = obs_cube.shape
		self.shape = [ny,nx]
		self.nx,self.ny = nx, ny



		#outs
		self.PA,self.EPS,self.XC,self.YC,self.VSYS,self.THETA = 0,0,0,0,0,0
		self.GUESS = []
		self.C_k, self.S_k = [],[]
		self.chisq_global = np.inf
		self.aic_bic = 0
		self.best_vlos_2D_model = 0
		self.best_kin_2D_models = 0
		self.Rings = 0
		self.std_errors = 0
		self.GUESS = 0
		self.n_circ = 0
		self.n_noncirc = 0


		config_boots	= config['bootstrap']
		config_general	= config['general']
		config_others	= config['others']
		config_clouds	= config['clouds']
		config_lsq 		= config['fitting']
		self.n_boot		= config_boots.getint('Nboots', 0)


		self.bootstrap_contstant_prms = np.zeros((self.n_boot, 6))
		self.bootstrap_kin_c, self.bootstrap_kin_s = 0, 0
		self.bootstrap_mom1d = 0

		self.cube_class	= cube_class
		self.outdir 	= outdir
		self.momscube	= 0
		self.emomscube	= 0
		self.nthreads	= config_general.getint('nthreads',1)
		self.nclouds 	= config_clouds.getint('nclouds', 1)
		self.nsubclouds = config_clouds.getint('nsubclouds', 50)
		self.z_scale 	= config_clouds.getfloat('z_scale', 0.1)
		self.z_profile 	= config_clouds.get('z_profile', 'sech2')

		self.disp_kms	= 	psf_lsf.sigma_inst_kms
		self.vary_disp	= 	psf_lsf.vary_disp

		# fitting
		self.rweight		= config_lsq.getint('rweight', 0)
		self.zweight		= config_lsq.getboolean('zweight', 0)
		self.weights		= (self.rweight,self.zweight)
		self.fitmethod 		= config_lsq.get('optimethod', 'nelder')
		self.seed			= 40
		self.vary_nc		= config_lsq.getfloat('vary_nc', 2)



		"""

		 					Harmonic model


		"""


	def lsq(self, fit_routine=fit):

		c1_tab_it, c3_tab_it, s1_tab_it, s3_tab_it = np.zeros(self.nrings,), np.zeros(self.nrings,), np.zeros(self.nrings,),np.zeros(self.nrings,)
		for it in np.arange(self.n_it):
			# Here we create the tabulated model
			_, disp_tab, c_tab, s_tab, R_pos = tab_mod_vels(self.rings,self.mommaps, self.emoms, self.pa0,self.eps0,self.x0,self.y0,self.vsys0,self.theta_b,self.rwidth,self.pixel_scale,self.vmode,self.shape,self.frac_pixel,self.r_bar_min, self.r_bar_max, self.m_hrm)
			c1_tab = c_tab[0]
			c1_tab[abs(c1_tab) > 400] = np.nanmedian(c1_tab)
			# Try to correct the PA if velocities are negative
			if np.nanmean(c1_tab) < 0 :
				self.pa0 = self.pa0 + 180
				c_tab[0]=-1*c_tab[0]
			# convert arrays to list
			c_tab=[list(c_tab[k]) for k in range(self.m_hrm)]
			s_tab=[list(s_tab[k]) for k in range(self.m_hrm)]
			#disp_tab=list(disp_tab)
			disp_tab = np.clip(disp_tab, self.disp_kms, None)
			disp_tab = np.sqrt(disp_tab**2 - self.disp_kms**2)
			if not self.vary_disp:
				disp_tab = np.ones_like(disp_tab)*self.disp_kms

			guess = [disp_tab,c_tab,s_tab,self.pa0,self.eps0,self.x0,self.y0,self.vsys0,self.theta_b]

			R_nc = (R_pos >= self.r_bar_min) & (R_pos <= self.r_bar_max)
			r_nc_vary = (R_nc * self.vary_nc).astype(float)
			R={'R_pos':R_pos, 'R_nc': r_nc_vary}

			vels = [disp_tab,c_tab,s_tab]
			rmax=np.max(R_pos)
			rmax_px=rmax/self.pixel_scale
			guess_common = dict(
				v_sys          = self.vsys0,
				inc            = self.inc0,
				pa             = self.pa0 % 360,
				x_center       = self.x0,
				y_center       = self.y0,
				z_scale        = self.z_scale,
				z_profile      = self.z_profile,
				n_clouds       = self.nclouds,
				n_subclouds    = self.nsubclouds,
				velocity_model = self.vmode,
				phi_bar		   = 45
			)

			cnf_prms=Set_params(self.vmode, self.psf_lsf, R, self.ring_space, self.vary, self.hdr,guess_common,self.m_hrm)
			guess_rings = cnf_prms.harm(vels)
			spec = cnf_prms.prms(self.vmode)
			lmfit_prm=cnf_prms

			#assume 2% outliers
			mom0_obs=self.mommaps[0]
			mom1_obs=self.mommaps[1]*(self.mommaps[1]/self.mommaps[1])
			mom2_obs=self.mommaps[2]
			msk1 = 	(abs(mom1_obs-self.vsys0)>1e3)
			msk2 = 	(mom2_obs>1000)

			mom1_obs[msk1*msk2]=0
			p1=np.nanpercentile(np.unique(mom1_obs),1)
			p99=np.nanpercentile(np.unique(mom1_obs),99)
			msk_outliers=(mom1_obs>p1)*(mom1_obs<p99)
			mom0_obs=mom0_obs*(msk_outliers)

			# ============================================================
			# 1.  Cube configuration
			# ============================================================

			cube_oper=Cube_operations(self.hdr, self.config, self.psf_lsf)

			# ============================================================
			# 5.  Fit using Nelder-Mead
			# ============================================================
			minmethod='Nelder-Mead' if self.fitmethod=='nelder'
            minmethod='Levenberg-Marquardt' if self.fitmethod=='leastsq'
            minmethod='Powell' if self.fitmethod=='powell'

			method = self.fitmethod
			if method    == 'nelder':
				options = {'xatol'  : 1e-3,'fatol'  : 1e-3,'maxiter': 3000, 'adaptive': True}
				fit_kws = {'options': options,}
			if method    == 'leastsq':
				fit_kws = {}
            if method   == 'powell'
                options = {'xtol': 1e-2, 'ftol': 1e-2}
                fit_kws =  {'options': options}

			best_rings, result = fit_rings(
				self.obs_cube*msk_outliers,
				self.mommaps,
				guess_rings,
				spec, cnf_prms,
				self.hdr,
				self.psf_lsf,
				cube_oper,
				weight_alpha = self.weights,
				method       = method,
				seed         = self.seed,
				verbose      = True,
				fit_kws      = fit_kws,
			)

			obs_cube = self.obs_cube
			# ============================================================
			# 7.  Build best-fit model cube for diagnostics
			# ============================================================
			best_model	= TiltedRingModel(self.hdr, self.psf_lsf, seed=self.seed)
			mod_cube 	= best_model.build(best_rings, verbose=False)
			res_cube 	= residual_cube(obs_cube, mod_cube)
			best_vals	= extractp(best_rings)

			mom0_obs,_,_=cube_oper.obs_mommaps(obs_cube)
			mom_mod=cube_oper.obs_mommaps(mod_cube)
			[mom0_mod,mom1_mod,mom0_mod2] = mom_mod

			mod_cube=mod_cube*np.divide(mom0_obs,mom0_mod,where=mom0_mod!=0,out=np.zeros_like(mom0_mod))

			best_vals['pa'] = best_vals['pa'] % 360
			best_vals['phi_bar'] = best_vals['phi_bar'] % 360

			best_const = extractp(best_rings,self.vmode)
			best_vels=extract_harmonics(best_rings)
			best_vals_all = [best_const,best_vels]

			scalar_fields	= ["v_sys","inc","pa","x_center","y_center","phi_bar"]
			operator 		= [np.mean,circmean,circmean,np.mean,np.mean,circmean]
			const = {p : opr(best_vals[p]) for p,opr in zip(scalar_fields, operator)}

			# get the final mask
			W_cur= make_weight_map(mom0_obs,self.psf_lsf, best_rings,alpha=self.weights, r_max_px=rmax_px, n_sigma_z=4)
			msk = (W_cur !=0).astype(float)
			mod_cube*=msk


			return mod_cube,best_rings,best_vals_all,result


			'''
			# Minimization
			fitting = fit_routine(self.obs_cube, self.eobs_cube, self.h, self.mommaps, self.emoms, guess, self.vary, self.vmode, self.config, R, self.ring_space, self.frac_pixel, self.inner_interp, self.m_hrm, N_it=self.n_it0)
			kin_3D_modls, Vk , self.pa0, self.eps0, self.x0, self.y0, self.vsys0, self.theta_b, out_data, Errors, true_rings = fitting.results()
			xi_sq = out_data[-1]

			disp, c_k, s_k = Vk[-1], Vk[0:self.m_hrm],Vk[self.m_hrm:]
			self.c_k, self.s_k = c_k, s_k
			# The first circular and first radial components
			c1 = c_k[0]
			s1 = s_k[0]

			# Keep the best fit
			#if xi_sq < self.chisq_global:
			if True:
				self.PA,self.EPS,self.XC,self.YC,self.VSYS = self.pa0, self.eps0, self.x0, self.y0, self.vsys0
				self.C_k, self.S_k = c_k, s_k
				self.Disp = np.asarray(disp)
				self.chisq_global = xi_sq
				self.aic_bic = out_data
				self.best_kin_3D_models = kin_3D_modls
				self.Rings = true_rings
				self.std_errors = Errors
				self.GUESS = [self.Disp, self.C_k, self.S_k, self.PA, self.EPS, self.XC, self.YC, self.VSYS,self.theta_b]
				self.n_circ = len(self.C_k[0])
				self.n_noncirc = len((self.S_k[0])[self.S_k[0]!=0])
				self.bootstrap_kin = np.zeros((self.n_boot, (2*self.m_hrm+1)*self.n_circ))
				self.bootstrap_mom1d = np.zeros((self.n_boot, self.n_circ))

		'''
	""" Following, the error computation.
	"""



	def boots(self,individual_run=0):
		self.frac_pixel = 0
		self.n_it,self.n_it0 = 1, 1
		runs = [individual_run]
		[mom0_cube,mom1_cube,mom2_cube]=self.momscube
		[emom0,emom1,emom2]=self.emomscube

		for k in runs:
			mommaps=[mom0_cube[k],mom1_cube[k],mom2_cube[k]]
			emommaps=[emom0,emom1,emom2]

			self.pa0,self.eps0,self.x0,self.y0,self.vsys0,self.theta_b = self.GUESS[-6:]
			np.random.seed()
			pa = self.pa0 + 5*np.random.normal()
			inc= eps_2_inc(self.eps0) + (5*np.pi/180)*np.random.normal() # rad
			eps=inc_2_eps(inc*180/np.pi)
			# setting chisq to -inf will preserve the leastsquare results
			self.chisq_global = -np.inf
			if (k+1) % 5 == 0 : print("%s/%s \t bootstraps" %((k+1),self.n_boot))

			intens_tab,disp_tab, c_tab, s_tab, R_pos = tab_mod_vels(self.Rings,mommaps, emommaps,pa,eps,self.x0,self.y0,self.vsys0,self.theta_b,self.rwidth,self.pixel_scale,self.vmode,self.shape,self.frac_pixel,self.r_bar_min, self.r_bar_max, self.m_hrm)
			c_tab=[list(c_tab[j]) for j in range(self.m_hrm)]
			s_tab=[list(s_tab[j]) for j in range(self.m_hrm)]
			kin=[c_tab, s_tab, disp_tab]
			vels=list(chain(*kin))
			#self.bootstrap_kin[k,:] = np.hstack(vels)

			guess = [disp_tab,c_tab,s_tab,self.pa0,self.eps0,self.x0,self.y0,self.vsys0,self.theta_b]
			# Minimization
			R={'R_pos':R_pos, 'R_NC': R_pos>self.r_bar_min }
			fitting = fit_boots(None, self.h, mommaps, emommaps, guess, self.vary, self.vmode, self.config, R, self.ring_space, self.frac_pixel, self.inner_interp,N_it=1)
			# outs
			_ , pa0, eps0, x0, y0, vsys0, theta_b = fitting.results()
			# convert PA to rad:
			pa0=pa0*np.pi/180
			#self.bootstrap_contstant_prms[k,:] = np.array ([ pa0, eps0, x0, y0, vsys0, theta_b ] )

			return([[ pa0, eps0, x0, y0, vsys0, theta_b ], np.hstack(vels), intens_tab])

	def run_boost_para(self):
		ncpu = self.nthreads
		with Pool(ncpu) as pool:
			result=pool.map(self.boots,np.arange(self.n_boot),chunksize=1)
		for k in range(self.n_boot):
			self.bootstrap_contstant_prms[k,:] = result[k][0]
			self.bootstrap_kin[k,:] = result[k][1]
			self.bootstrap_mom1d[k,:] = result[k][2]

		p=np.nanpercentile(self.bootstrap_mom1d,[15.865, 50, 84.135],axis=0).reshape((3,len(self.bootstrap_mom1d[0])))
		d=np.diff(p,axis=0)
		std_mom1d= 0.5 * np.sum(d,axis=0)

		p = np.nanpercentile(self.bootstrap_kin,[15.865, 50, 84.135],axis=0).reshape((3,len(self.bootstrap_kin[0])))
		d=np.diff(p,axis=0)
		std_kin=0.5 * np.sum(d,axis=0)
		eCSSig=np.array_split(std_kin,2*self.m_hrm+1)
		eCSS= [eCSSig[0:self.m_hrm],eCSSig[self.m_hrm:-1],eCSSig[-1]]

		p=np.nanpercentile(self.bootstrap_contstant_prms,[15.865, 50, 84.135],axis=0).reshape((3,6))
		d=np.diff(p,axis=0)
		std_const= 0.5 * np.sum(d,axis=0)# abs(sigma1u + sigma1l)
		std_pa=abs(circstd(self.bootstrap_contstant_prms[:,0]))*180/np.pi # rad ---> deg
		std_phi_bar=abs(circstd(self.bootstrap_contstant_prms[:,-1])) # rad
		std_const[0],std_const[-1]=std_pa,std_phi_bar

		self.std_errors = [eCSS,std_const,std_mom1d]

	def output(self):
		#least
		out = self.lsq()
		return out

	def __call__(self):
		out = self.output()
		return out
