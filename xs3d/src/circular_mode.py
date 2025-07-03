import numpy as np
import time
import os
from scipy.stats import circstd,circmean
from .eval_tab_model import tab_mod_vels
from .phi_bar_sky import pa_bar_sky
from .fit_params import Fit_kin_mdls as fit
from .fit_params_boots import Fit_kin_mdls as fit_boots
from .tools_fits import array_2_fits
from .create_2D_vlos_model import best_2d_model
from .read_hdr import Header_info
from multiprocessing import Pool, cpu_count
from .pixel_params import inc_2_eps, eps_2_inc,e_eps2e_inc
import matplotlib.pylab as plt
#first_guess_it = []

class Circular_model:
	def __init__(self, vmode, galaxy, datacube, edatacube, header, mommaps, emoms, guess0, vary, n_it, rstart, rfinal, ring_space, frac_pixel, inner_interp, delta, bar_min_max, config, outdir,cube_class):

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

		if self.n_it == 0: self.n_it =1


		rend = self.rfinal
		if (self.rfinal-self.rstart) % self.ring_space == 0 :
			# To include the last ring :
			rend = self.rfinal + self.ring_space


		self.rings = np.arange(self.rstart, rend, self.ring_space)
		self.nrings = len(self.rings)

		self.r_bar_min, self.r_bar_max = self.bar_min_max
		self.pa0,self.eps0,self.x0,self.y0,self.vsys0,self.theta_b = self.guess0
		self.vmode = vmode
		[ny,nx] = (self.vel).shape
		self.shape = [ny,nx]


		#outs
		self.PA,self.EPS,self.XC,self.YC,self.VSYS,self.THETA = 0,0,0,0,0,0
		self.GUESS = []
		self.Disp,self.Vrot,self.Vrad,self.Vtan = [],[],[],[]
		self.chisq_global = np.inf
		self.aic_bic = 0
		self.best_vlos_2D_model = 0
		self.best_kin_3D_models = 0
		self.Rings = 0
		self.std_errors = 0
		self.nvels=0


		config_boots = config['bootstrap']
		config_general = config['general']
		self.n_boot = config_boots.getint('Nboots', 0)

		self.bootstrap_contstant_prms = np.zeros((self.n_boot, 6))
		self.bootstrap_kin = 0
		self.bootstrap_mom1d = 0


		self.cube_class=cube_class
		self.outdir = outdir
		self.momscube=0
		self.emomscube=0
		self.nthreads=config_general.getint('nthreads',1)
		"""

		 					CIRCULAR MODEL

		"""

	def lsq(self,fit_routine=fit):

		vrad_it, vtan_it = np.zeros(100,), np.zeros(100,)
		vrot_tab_it, vrad_tab_it, vtan_tab_it = np.zeros(self.nrings,), np.zeros(self.nrings,), np.zeros(self.nrings,)

		for it in np.arange(self.n_it):

			# Here we create the tabulated model
			_,disp_tab, vrot_tab, vrad_tab, vtan_tab, R_pos = tab_mod_vels(self.rings,self.mommaps, self.emoms, self.pa0,self.eps0,self.x0,self.y0,self.vsys0,self.theta_b,self.delta,self.pixel_scale,self.vmode,self.shape,self.frac_pixel,self.r_bar_min, self.r_bar_max)
			vrot_tab[abs(vrot_tab) > 400] = np.nanmedian(vrot_tab)

			# Try to correct the PA if velocities are negative
			if np.nanmean(vrot_tab) < 0 :
				self.pa0 = self.pa0 + 180
				vrot_tab*=-1

			guess = [disp_tab,vrot_tab,vrad_tab,vtan_tab,self.pa0,self.eps0,self.x0,self.y0,self.vsys0,self.theta_b]
			if it == 0: first_guess_it = guess

			R={'R_pos':R_pos, 'R_NC': R_pos>self.r_bar_min}
			# Minimization
			fitting = fit_routine(self.datacube, self.edatacube, self.h, self.mommaps, self.emoms, guess, self.vary, self.vmode, self.config, R, self.ring_space, self.frac_pixel, self.inner_interp,N_it=self.n_it0)
			# outs
			kin_3D_modls, Vk , self.pa0, self.eps0, self.x0, self.y0, self.vsys0, self.theta_b, out_data, Errors, true_rings = fitting.results()
			xi_sq = out_data[-1]
			#Unpack velocities
			disp, vrot, vrad, vtan = Vk
			self.disp, self.vrot, self.vrad, self.vtan=disp, vrot, vrad, vtan

			# Keep the best fit
			#if xi_sq < self.chisq_global:
			if True:
				self.PA,self.EPS,self.XC,self.YC,self.VSYS,self.THETA = self.pa0, self.eps0, self.x0, self.y0,self.vsys0,self.theta_b
				self.Vrot = np.asarray(vrot);self.n_circ = len(vrot)
				self.Vrad = np.asarray(vrad)
				self.Vtan = np.asarray(vtan)
				self.Disp = np.asarray(disp)
				self.chisq_global = xi_sq
				self.aic_bic = out_data
				self.best_kin_3D_models = kin_3D_modls
				self.Rings = true_rings
				self.std_errors = Errors
				self.GUESS = [self.Disp, self.Vrot, self.Vrad, self.Vtan, self.PA, self.EPS, self.XC, self.YC, self.VSYS, self.THETA]
				self.bootstrap_kin = np.zeros((self.n_boot, 4*self.n_circ))
				self.bootstrap_mom1d = np.zeros((self.n_boot, self.n_circ))
	""" Following, the error computation.
	"""



	def boots(self,individual_run=0):
		self.frac_pixel = 0
		self.n_it,self.n_it0 = 1, 1
		runs = [individual_run]
		[mom0_cube,mom1_cube,mom2_cube] = self.momscube
		[emom0,emom1,emom2] = self.emomscube

		for k in runs:
			mommaps = [mom0_cube[k],mom1_cube[k],mom2_cube[k]]
			emommaps = [emom0,emom1,emom2]

			self.pa0,self.eps0,self.x0,self.y0,self.vsys0,self.theta_b = self.GUESS[-6:]
			np.random.seed()
			pa = self.pa0 + 5*np.random.normal()
			inc = eps_2_inc(self.eps0) + (5*np.pi/180)*np.random.normal() # rad
			eps = inc_2_eps(inc*180/np.pi)
			# setting chisq to -inf will preserve the leastsquare results
			self.chisq_global = -np.inf
			if (k+1) % 5 == 0 : print("%s/%s \t bootstraps" %((k+1),self.n_boot))
			intens_tab,disp_tab, vrot_tab, vrad_tab, vtan_tab, R_pos = tab_mod_vels(self.Rings,mommaps,emommaps,pa,eps,self.x0,self.y0,self.vsys0,self.theta_b,self.delta,self.pixel_scale,self.vmode,self.shape,self.frac_pixel,self.r_bar_min, self.r_bar_max)
			vels = list(disp_tab)+list(vrot_tab)+list(vrad_tab)+list(vtan_tab)

			guess = [disp_tab,vrot_tab,vrad_tab,vtan_tab,self.pa0,self.eps0,self.x0,self.y0,self.vsys0,self.theta_b]
			R = {'R_pos':R_pos, 'R_NC': R_pos>self.r_bar_min}
			# Minimization
			fitting = fit_boots(None, self.h, mommaps, emommaps, guess, self.vary, self.vmode, self.config, R, self.ring_space, self.frac_pixel, self.inner_interp,N_it=1)
			# outs
			_ , pa0, eps0, x0, y0, vsys0, theta_b = fitting.results()
			# convert PA to rad:
			pa0 = pa0*np.pi/180

			return([[ pa0, eps0, x0, y0, vsys0, theta_b ], np.concatenate([disp_tab, vrot_tab, vrad_tab, vtan_tab]), intens_tab])

	def run_boost_para(self):
		ncpu = self.nthreads
		with Pool(ncpu) as pool:
			result=pool.map(self.boots,np.arange(self.n_boot),chunksize=1)
		for k in range(self.n_boot):
			self.bootstrap_contstant_prms[k,:] = result[k][0]
			self.bootstrap_kin[k,:] = result[k][1]
			self.bootstrap_mom1d[k,:] = result[k][2]

		p = np.nanpercentile(self.bootstrap_mom1d,[15.865, 50, 84.135],axis=0).reshape((3,len(self.bootstrap_mom1d[0])))
		d=np.diff(p,axis=0)
		std_mom01d=0.5 * np.sum(d,axis=0)

		p = np.nanpercentile(self.bootstrap_kin,[15.865, 50, 84.135],axis=0).reshape((3,len(self.bootstrap_kin[0])))
		d=np.diff(p,axis=0)
		std_kin=0.5 * np.sum(d,axis=0)

		p=np.nanpercentile(self.bootstrap_contstant_prms,[15.865, 50, 84.135],axis=0).reshape((3,6))
		d=np.diff(p,axis=0)
		std_const= 0.5 * np.sum(d,axis=0)
		std_pa=abs(circstd(self.bootstrap_contstant_prms[:,0]))*180/np.pi # rad ---> deg
		std_phi_bar=abs(circstd(self.bootstrap_contstant_prms[:,-1])) # rad
		std_const[0],std_const[-1]=std_pa,std_phi_bar
		self.std_errors = [np.array_split(std_kin,4),std_const,std_mom01d]

	def output(self):
		#least
		ecovar = self.lsq()
		#bootstrap
		if self.n_boot>0:
			print("------------------------------------")
			print("starting bootstrap analysis ..")
			print("------------------------------------")
			self.emomscube,self.momscube=(self.cube_class).obs_emommaps_boots(self.n_boot)
			eboots= self.run_boost_para()


	def __call__(self):
		out = self.output()
		# Get sky bar PA
		PA_bar_major = pa_bar_sky(self.PA,self.EPS,self.THETA)
		PA_bar_minor = pa_bar_sky(self.PA,self.EPS,self.THETA-np.pi/2)
		return self.PA,self.EPS,self.XC,self.YC,self.VSYS,self.THETA,self.Rings,self.Disp,self.Vrot,self.Vrad,self.Vtan,self.best_kin_3D_models,PA_bar_major,PA_bar_minor,self.aic_bic,self.std_errors
