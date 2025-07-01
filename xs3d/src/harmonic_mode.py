import numpy as np
import time
import itertools
from itertools import chain
from scipy.stats import circstd,circmean
from .eval_tab_model import tab_mod_vels
from .fit_params import Fit_kin_mdls as fit
from .fit_params_boots import Fit_kin_mdls as fit_boots
from .tools_fits import array_2_fits
from .create_2D_vlos_model import best_2d_model
from .read_hdr import Header_info
import os
from multiprocessing import Pool, cpu_count
from .pixel_params import inc_2_eps, eps_2_inc,e_eps2e_inc

class Harmonic_model:
	def __init__(self, galaxy, datacube, edatacube, header, mommaps, emoms, guess0, vary, n_it, rstart, rfinal, ring_space, frac_pixel, inner_interp, delta, bar_min_max, config, m_hrm, outdir,cube_class):


		self.galaxy=galaxy
		self.datacube=datacube
		self.edatacube=edatacube
		self.h=header
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
		self.delta=delta
		self.bar_min_max=bar_min_max
		self.config=config
		self.pixel_scale=Header_info(self.h,config).scale
		self.m_hrm = m_hrm

		if self.n_it == 0: self.n_it = 1
		rend = self.rfinal
		if (self.rfinal-self.rstart) % self.ring_space == 0 :
			# To include the last ring :
			rend = self.rfinal + self.ring_space


		self.rings = np.arange(self.rstart, rend, self.ring_space)
		self.nrings = len(self.rings)

		self.r_bar_min, self.r_bar_max = self.bar_min_max

		self.pa0,self.eps0,self.x0,self.y0,self.vsys0,self.theta_b = self.guess0
		self.vmode = "hrm"
		[ny,nx] = (self.vel).shape
		self.shape = [ny,nx]



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


		config_boots = config['bootstrap']
		config_general = config['general']
		self.n_boot = config_boots.getint('Nboots', 0)


		self.bootstrap_contstant_prms = np.zeros((self.n_boot, 6))
		self.bootstrap_kin_c, self.bootstrap_kin_s = 0, 0
		self.bootstrap_mom1d = 0

		self.cube_class=cube_class
		self.outdir = outdir
		self.momscube=0
		self.emomscube=0
		self.nthreads=config_general.getint('nthreads',1)



		"""

		 					Harmonic model


		"""


	def lsq(self, fit_routine=fit):

		c1_tab_it, c3_tab_it, s1_tab_it, s3_tab_it = np.zeros(self.nrings,), np.zeros(self.nrings,), np.zeros(self.nrings,),np.zeros(self.nrings,)
		for it in np.arange(self.n_it):
			# Here we create the tabulated model
			_, disp_tab, c_tab, s_tab, R_pos = tab_mod_vels(self.rings,self.mommaps, self.emoms, self.pa0,self.eps0,self.x0,self.y0,self.vsys0,self.theta_b,self.delta,self.pixel_scale,self.vmode,self.shape,self.frac_pixel,self.r_bar_min, self.r_bar_max, self.m_hrm)
			c1_tab = c_tab[0]
			c1_tab[abs(c1_tab) > 400] = np.nanmedian(c1_tab)
			# Try to correct the PA if velocities are negative
			if np.nanmean(c1_tab) < 0 :
				self.pa0 = self.pa0 + 180
				c_tab[0]=-1*c_tab[0]
			# convert arrays to list
			c_tab=[list(c_tab[k]) for k in range(self.m_hrm)]
			s_tab=[list(s_tab[k]) for k in range(self.m_hrm)]
			disp_tab=list(disp_tab)
			guess = [disp_tab,c_tab,s_tab,self.pa0,self.eps0,self.x0,self.y0,self.vsys0,self.theta_b]
			R={'R_pos':R_pos, 'R_NC': R_pos>self.r_bar_min }
			# Minimization
			fitting = fit_routine(self.datacube, self.edatacube, self.h, self.mommaps, self.emoms, guess, self.vary, self.vmode, self.config, R, self.ring_space, self.frac_pixel, self.inner_interp, self.m_hrm, N_it=self.n_it0)
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

			intens_tab,disp_tab, c_tab, s_tab, R_pos = tab_mod_vels(self.Rings,mommaps, emommaps,pa,eps,self.x0,self.y0,self.vsys0,self.theta_b,self.delta,self.pixel_scale,self.vmode,self.shape,self.frac_pixel,self.r_bar_min, self.r_bar_max, self.m_hrm)
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
		ecovar = self.lsq()
		#bootstrap
		if self.n_boot>0:
			print("------------------------------------")
			print("starting bootstrap analysis ..")
			print("------------------------------------")
			self.emomscube,self.momscube=(self.cube_class).obs_emommaps_boots(self.n_boot)
			eboots=self.run_boost_para()



	def __call__(self):
		out = self.output()
		return self.PA,self.EPS,self.XC,self.YC,self.VSYS,self.Rings,self.Disp,self.C_k,self.S_k,self.best_kin_3D_models,self.aic_bic,self.std_errors
