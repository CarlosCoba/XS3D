import numpy as np
import matplotlib.pylab as plt
from astropy.io import fits
import warnings
warnings.filterwarnings('ignore')
import os.path
from os import path
import time
from time import gmtime,strftime

from .kinematic_centre_vsys import kincenter as  KC
from .geometric_moments import geom_moms
from .cbar import colorbar as cb
from .write_table import write
from .circular_mode import Circular_model
from .harmonic_mode import Harmonic_model
from .start_messenge import Print
from .pixel_params import inc_2_eps, eps_2_inc,e_eps2e_inc
from .momtools import mommaps
from .save_2D_kin_models import  save_2d_models
from .save_fits_1D_model_harmonic import save_model_h
from .plot_models_harmonic import plot_kin_models_h
from .plot_models import plot_kin_models
from .save_fits_1D_model import save_model
from .save_fits_table import save_table
from .save_fits_mommaps import save_momments,save_rmomments
from .save_fits_pvd import save_pvds
from .plot_momms import plot_mommaps
from .plot_resmoms import plot_rmommaps
from .plot_chann_cube import plot_channels
from .plot_chann_rcube import plot_rchannels
from .create_directories import direc_out
from .read_hdr import Header_info
from .convolve_cube import Cube_creation,Cube_operations
from .momtools import mask_wave
from .pv import pv_array2
from .plot_pv import plot_pvd
from .cube_stats import cstats,baselinecor,ecube,mask_cube
valid_strings_for_optional_inputs=["", "-", ".", ",", "#","%", "&","None"]

from .psf_lsf import PsF_LsF
from .utils import nan2zero,zero2nan,circmean
from .plot_rings import plot_rings_sky
from .save_vrot_z import save_vrot_z_fits


def guess_vals(pa_g,inc_g,xc_g,yc_g,vsys_g,PHI_B ):
	# List of guess values
	guess=[pa_g,inc_g,xc_g,yc_g,vsys_g,PHI_B]
	return guess

class Run_models:

	def __init__(self, galaxy, datacube, msk_cube, vsys_g, pa_g, inc_g, xc_g, yc_g, PHI_B, n_it, vary_pa_g, vary_inc_g, vary_XC, vary_YC, vary_vsys_g, vary_PHI, delta, rstart, rfinal, ring_space, frac_pixel, inner_interp, bar_min_max, vmode, survey, config, prefix, osi):
		
		#set time
		self.start_time=time.time()

		# print start messenge
		P=Print(); P()

		self.P=P
		
		self.outdir=direc_out(config)
		
		self.vmode=vmode
		
		self.galaxy=galaxy

		self.datacube, self.hdr_ori=fits.getdata(datacube,header=True)
		
		# Read header information
		self.hdr_info=Header_info(self.hdr_ori, config)
		
		self.P.cubehdr(self.hdr_info)
		
		# Print
		self.P.out('V Doppler',self.hdr_info.vdoppler)
		self.P.configprint(self.hdr_info,config)
		
		# remove NaN values
		self.datacube[~np.isfinite(self.datacube)]=0

		#if cut wavelenghts apply here
		msk_w,hdr_tmp,cut_spec=mask_wave(self.hdr_ori,config)
		
		if cut_spec:
			self.datacube= self.datacube[:,None][msk_w[:,None]]
			# Change header if axes are cut
			self.hdr_ori= hdr_tmp
			# update header
			self.hdr_info=Header_info(self.hdr_ori, config)

		#baseline correction
		self.datacube,self.baselcube=baselinecor(self.datacube,config)

		# apply rms-based-mask to the cube
		if msk_cube in osi: msk_cube=None
		self.msk2d_cube=msk_cube
		rms3d,self.rms_cube,vpeak2D=mask_cube(self.datacube,config,self.hdr_info,msk_user=self.msk2d_cube)
		self.datacube=self.datacube*rms3d
		self.hdr_ori['RMS']=self.rms_cube
		self.hdr_info.rms = self.rms_cube

		# psf class
		self.psf_lsf=PsF_LsF(self.hdr_info, config)
			
		# cube class
		self.cube_class=Cube_operations(self.hdr_info,config,self.psf_lsf)		

		# create error cube
		self.errcube=ecube(self.datacube,self.rms_cube)
		self.cube_class.eflux3d=self.errcube

		#create observed momemnt maps
		self.mom_obs=self.cube_class.obs_mommaps(self.datacube)
		# create random moments instead ?
		#self.mom_obs=self.cube_class.obs_mommaps_rnd(individual_run=1)

		[self.mom0,self.mom1,self.mom2]=self.mom_obs
		#if vpeak2D is not None: self.mom1=vpeak2D # vpeak from smoothed cube
		# create temporary error moment maps
		self.emoms=[np.ones_like(self.mom0),np.ones_like(self.mom1),np.ones_like(self.mom2)]

		pixel_scale=self.hdr_info.scale

		self.PA_bar_mjr,self.PA_bar_mnr,self.PHI_BAR=0,0,0
		self.survey=survey
		self.m_hrm=3
		self.config=config 
        	

		geom=[pa_g,inc_g,xc_g,yc_g]
		
		#check whether the disk geometry will be computed or not
		compute_geom=np.any([True if k in geom else False for k in osi])
		if compute_geom:
			# estimate geometric moments with mom0 map.
			geom_start=geom_moms(self.mom0,pixel_scale,binary=False)
			sma=geom_start[-1]
		if rfinal in osi:
			rfinal=sma
			Print().status(f'rmax={rfinal} arcsec')
		else:
			rfinal=eval(rfinal)

		for j,p in enumerate(geom):
			if p in osi:
				geom[j]=geom_start[j]
			else:
				geom[j]=eval(p)
				
		# disk geometry
		[pa_g,inc_g,xc_g,yc_g]=geom
		
		if 0<inc_g<1: inc_g=eps_2_inc(inc_g)
		if vsys_g in osi :
			vsys_g=KC(self.mom1,xc_g,yc_g)
		else:
			vsys_g=eval(vsys_g)

		guess_prm=guess_vals(pa_g,inc_g,xc_g,yc_g,vsys_g,PHI_B )
		vary=np.array( [vary_pa_g,vary_inc_g,vary_XC,vary_YC,vary_vsys_g,vary_PHI] )

		self.osi=osi
		if self.survey not in self.osi :
			self.kin_params_table="%sana_kin_%s_model.%s.csv"%(self.outdir,self.vmode,self.survey)
		else:
			self.kin_params_table="%sana_kin_%s_model.%s.csv"%(self.outdir,self.vmode,self.galaxy)

		if "hrm_" in vmode:
			try:
				self.m_hrm=int(vmode[4:])
				if self.m_hrm == 0 :
					raise ValueError
			except(ValueError):
				print("XookSuut: provide a proper harmonic number different from zero, for example hrm_2")
				quit()
		# Print guess values
		self.P.guess_vals(galaxy,guess_prm,vmode)
		
		# Change inc_g to eps
		guess_prm=guess_vals(pa_g,inc_2_eps(inc_g),xc_g,yc_g,vsys_g,PHI_B*np.pi/180 )

		self.P.status("Starting Least Squares analysis",line=True)
		
		if "hrm" not in self.vmode:
			circ=Circular_model(self.vmode, galaxy, self.datacube, self.errcube, self.hdr_info, self.mom_obs, self.emoms, guess_prm, vary, n_it, rstart, rfinal, ring_space, frac_pixel, inner_interp, delta, bar_min_max, config, self.outdir,self.cube_class,self.psf_lsf)
			out=circ()
			[self.mod_cube,self.best_rings,self.best_vals,self.result]=out
		else:
			hrm=Harmonic_model(self.vmode, galaxy, self.datacube, self.hdr_info, self.mom_obs, self.emoms, guess_prm, vary, n_it, rstart, rfinal, ring_space, frac_pixel, inner_interp, delta, bar_min_max, config, self.outdir,self.cube_class,self.psf_lsf,self.m_hrm)
			out=hrm()
			[self.mod_cube,self.best_rings,best,self.result]=out
			self.best_vals,self.best_vels=best		

class XS_out(Run_models):

	def results(self):
	
		self.P.status("Best model found !")

		# plot momment maps and 1D profiles
		self.P.status("Plotting results")

		from .save_output import save_rings_fits
		save_rings_fits(self.galaxy, self.vmode, self.best_rings, self.result, self.psf_lsf, extra_header=None, out=self.outdir)


		mom_mod = self.cube_class.obs_mommaps(self.mod_cube)

		# remove zeros from moment maps		
		mom_mod=[zero2nan(mom_mod[k]) for k in range(3)]
		self.mom_obs=[zero2nan(self.mom_obs[k]) for k in range(3)]	

		# extract avg constant parameters							
		scalar_fields	= ["v_sys","inc","pa","x_center","y_center","phi_bar"]
		operator 		= [np.mean,circmean,circmean,np.mean,np.mean,circmean]
		const = {k : op(self.best_vals[k]) for k,op in zip(scalar_fields, operator)}
		const['rmax'] = np.max(self.best_vals['radius'])

		plot_rings_sky(self.galaxy,self.mom_obs,self.best_rings,const,self.vmode,self.psf_lsf,self.hdr_info,self.config,self.rms_cube,self.outdir)
		# save 1d models
		if 'hrm' in self.vmode:
			save_model_h(self.galaxy,self.vmode,const,self.best_vals,self.best_vels,self.result,self.m_hrm,self.outdir)		
		else:
			save_model(self.galaxy,self.vmode,const,self.best_vals,self.result,out=self.outdir)		
				
		self.P.status("Creating 0th, 1st and 2nd momment maps")
		
		plot_mommaps(self.galaxy,mom_mod,self.mom_obs,const,self.vmode,self.psf_lsf,self.hdr_info,self.config,self.outdir)		
		
		# save moment maps and cube model
		save_momments(self.galaxy,self.vmode,self.mom_obs,mom_mod,self.datacube,self.mod_cube,self.baselcube,self.hdr_ori,out=self.outdir)
		
		self.P.status("creating PVD maps")
		
		out_pvd=pv_array2(self.datacube,self.mod_cube,self.mom_obs,mom_mod,self.hdr_info,self.psf_lsf,self.rms_cube,const)

		plot_pvd(self.galaxy,out_pvd,self.best_vals,const,self.vmode,self.rms_cube,mom_mod,self.mom_obs,self.datacube,self.hdr_info,self.psf_lsf,self.config,self.outdir)

		save_pvds(self.galaxy,self.vmode,out_pvd,self.rms_cube,self.hdr_info,self.outdir)

		plot_channels(self.galaxy,self.datacube,self.mod_cube,const,self.vmode,self.hdr_info,self.psf_lsf,self.config,self.rms_cube,self.outdir)

		if "hrm" in self.vmode:
			plot_kin_models_h(self.galaxy,self.vmode,self.best_vals,self.best_vels,self.m_hrm,out=self.outdir)
		else:
			plot_kin_models(self.galaxy,self.vmode,const,self.best_vals,out=self.outdir)		
		   
		
		save_vrot_z_fits(self.galaxy, self.vmode, self.best_rings, z_values=0, filename='no', out = self.outdir)
		                    
		print("Done!. Check the XS3D directory")
		end_time=time.time()
		total_time=end_time-self.start_time
		t=strftime("%H:%M:%S", gmtime(total_time))
		ttime="Total time: "+t+" HMS"
		self.P.status(ttime)
		print("------------------------------------")


	def __call__(self):
		run=self.results()
		#return self.results()
