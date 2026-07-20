import numpy as np
import time
import os
import matplotlib.pylab as plt
from scipy.stats import circstd,circmean
from multiprocessing import Pool, cpu_count

from .start_messenge import Print
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
	def __init__(self, vmode, galaxy, obs_cube, eobs_cube, header, mommaps, emoms, guess0, vary, n_it, rstart, rfinal, ring_space, frac_pixel, inner_interp, delta, bar_min_max, config, outdir,cube_class,psf_lsf,m_hrm):


		self.galaxy       = galaxy
		self.obs_cube     = obs_cube
		self.eflux2d      = eobs_cube
		self.hdr          = header
		self.mommaps      = mommaps
		self.vel_copy     = np.copy(self.mommaps[1])
		self.vel          = self.mommaps[1]
		self.emoms        = emoms
		self.guess0       = guess0
		self.vary         = vary
		self.n_it         = (n_it+1)
		self.rstart       = rstart
		self.rfinal       = rfinal
		self.ring_space   = ring_space
		self.frac_pixel   = frac_pixel
		self.inner_interp = inner_interp
		self.rwidth       = delta
		self.bar_min_max  = bar_min_max
		self.config       = config
		self.m_hrm        = m_hrm
		self.pixel_scale  = header.scale
		self.psf_lsf      = psf_lsf
        self.vary_params  = {}
		self.rms          = header.rms        

        # print function
        P=Print()
        self.P=P

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


	def lsq(self,obs_cube=None, verbose=True, bootstrap=False):
		if obs_cube is None: obs_cube = self.obs_cube

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

			# ============================================================
			# 1.  Cube configuration
			# ============================================================

			cube_oper=Cube_operations(self.hdr, self.config, self.psf_lsf)

			# ============================================================
			# 5.  Fit using Nelder-Mead
			# ============================================================
			if self.fitmethod=='nelder': minmethod='Nelder-Mead'
			if self.fitmethod=='leastsq': minmethod='Levenberg-Marquardt'
			if self.fitmethod=='powell': minmethod='Powell'

			method = self.fitmethod
			if method    == 'nelder':
				options = {'xatol'  : 1e-3,'fatol'  : 1e-3,'maxiter': 3000, 'adaptive': True}
				fit_kws = {'options': options,}
			if method    == 'leastsq':
				fit_kws = {}
			if method   == 'powell':
				options = {'xtol': 1e-2, 'ftol': 1e-2}
				fit_kws =  {'options': options}

			best_rings, result = fit_rings(
				obs_cube,
                self.eflux2d,
				self.mommaps,
				guess_rings,
				spec, cnf_prms,
				self.hdr,
				self.psf_lsf,
				cube_oper,
				weight_alpha = self.weights,
				method       = method,
				seed         = self.seed,
				verbose      = verbose,
				fit_kws      = fit_kws,
			)

            self.P.status("Best model found !")
			if bootstrap: return  best_rings, result
			self.vary_params = spec

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

            # update inital values
            [self.pa0,self.inc0,self.x0,self.y0,self.vsys0,self.theta_b]=[const['pa'],const['inc'],const['x_center'],const['y_center'],const['v_sys'],const['phi_bar'] ]

			# get the final mask
			W_cur= make_weight_map(mom0_obs,self.psf_lsf, best_rings,alpha=self.weights, r_max_px=rmax_px, n_sigma_z=4)
			msk = (W_cur !=0).astype(float)
			mod_cube*=msk

		return mod_cube,best_rings,best_vals_all,result

	def run_boost(self,output=None):

        self.P.status('Computing Errors on parameters')
        self.P.status('N bootstraps    %s'%self.n_boot )
        print(self.P.deli)

		[obs_cube,best_rings,best_vals,result]=output
		n_boot    = self.n_boot
		msk       = obs_cube == 0
		params_ref  = build_params(best_rings, self.vary_params)
		free_names  = [n for n, p in params_ref.items() if p.vary]
		samples		= {n: [] for n in free_names}

		for k in range(n_boot):
			if (k+1) % 5 == 0 : print("%s/%s \t bootstraps" %((k+1),self.n_boot))
			rng 	= np.random.default_rng()
			noise	= rng.standard_normal(obs_cube.shape)
			obs_cube_tmp= obs_cube +  self.rms*noise
			obs_cube_tmp[msk] = self.obs_cube[msk]

			best_rings_k, result_k = self.lsq(obs_cube_tmp, verbose=0, bootstrap=True)

			for name in free_names:
                try:
                    samples[name].append(result_k.params[name].value)
                else KeyError:
                    samples[name].append(np.nan)

		stderr = {}; median = {}; ci_68 = {}; ci_95 = {}
		for name in free_names:
			arr = np.array(samples[name])
			if len(arr) < 2:
				stderr[name] = np.nan
				median[name] = np.nan
				ci_68[name]  = (np.nan, np.nan)
				ci_95[name]  = (np.nan, np.nan)
			else:
				stderr[name] = float(np.nanstd(arr))
				median[name] = float(np.nanmedian(arr))
				ci_68[name]  = (float(np.nanpercentile(arr, 16)),float(np.nanpercentile(arr, 84)))
				ci_95[name]  = (float(np.nanpercentile(arr,  2.5)),float(np.nanpercentile(arr, 97.5)))

			# Include the standard deviation only in the results file
			par = result.params[name]
			std = par.stderr # this is None by default
			(output[-1].params[name]).stderr  = stderr[name]
		return None

	def output(self):
		out = self.lsq()
		if self.n_boot !=0: self.run_boost(out)
		return out

	def __call__(self):
		out = self.output()
		return out
