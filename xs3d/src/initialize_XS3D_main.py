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
from .convolve_cube import Cube_creation
from .momtools import mask_wave
from .pv import pv_array
from .plot_pv import plot_pvd
from .cube_stats import cstats,baselinecor,ecube,mask_cube
valid_strings_for_optional_inputs = ["", "-", ".", ",", "#","%", "&","None"]


def guess_vals(PA,INC,X0,Y0,VSYS,PHI_B ):
	# List of guess values
	guess = [PA,INC,X0,Y0,VSYS,PHI_B]
	return guess

class Run_models:

	def __init__(self, galaxy, datacube, msk_cube, VSYS, PA, INC, X0, Y0, PHI_B, n_it, vary_PA, vary_INC, vary_XC, vary_YC, vary_VSYS, vary_PHI, delta, rstart, rfinal, ring_space, frac_pixel, inner_interp, bar_min_max, vmode, survey, config, prefix, osi):

		#set time
		self.start_time = time.time()

		# print start messenge
		P=Print(); P()

		self.P=P
		self.outdir = direc_out(config)
		self.vmode = vmode
		self.galaxy = galaxy

		self.datacube, self.h=fits.getdata(datacube,header=True)
		if 'CRPIX3' not in self.h: self.h['CRPIX3']=1
		# Read header information
		self.hdr_info=Header_info(self.h,config)
		self.P.cubehdr(self.hdr_info)
		# Print
		self.P.out('V Doppler',self.hdr_info.vdoppler)
		self.P.configprint(self.h,config)
		# remove NaN values
		self.datacube[~np.isfinite(self.datacube)]=0

		#if cut wavelenghts apply here
		msk_w,h_tmp,cut_spec=mask_wave(self.h,config)
		if cut_spec:
			self.datacube=self.datacube[:,None][msk_w[:,None]]
			# Change header if axes are cut
			self.h=h_tmp
			# update
			self.hdr_info=Header_info(self.h,config)

		#baseline correction
		self.datacube,self.baselcube=baselinecor(self.datacube,config)


		# apply rms-based-mask to the cube
		if msk_cube in osi: msk_cube=None
		self.msk2d_cube=msk_cube
		rms3d,self.rms_cube,vpeak2D=mask_cube(self.datacube,config,self.hdr_info,msk_user=self.msk2d_cube)
		self.datacube=self.datacube*rms3d
		self.h['RMS_CUBE']=self.rms_cube

		# cube class
		cube_class=Cube_creation(self.datacube,self.h,[1]*3,config)

		# create error cube
		self.errcube=ecube(self.datacube,self.rms_cube)
		cube_class.eflux3d=self.errcube

		#create observed momemnt maps
		self.momaps=cube_class.obs_mommaps()
		# create random moments instead ?
		#self.momaps=cube_class.obs_mommaps_rnd(individual_run=1)

		[self.mom0,self.mom1,self.mom2]=self.momaps
		#if vpeak2D is not None: self.mom1=vpeak2D # vpeak from smoothed cube
		# create temporary error moment maps
		self.emoms=[np.ones_like(self.mom0),np.ones_like(self.mom1),np.ones_like(self.mom2)]


		self.vel_map=self.mom1
		self.pixel_scale=self.hdr_info.scale

		[ny,nx] = [self.hdr_info.ny,self.hdr_info.nx]
		self.e_ISM = 0

		self.PA_bar_mjr,self.PA_bar_mnr,self.PHI_BAR = 0,0,0
		self.survey = survey
		self.m_hrm = 3
		self.config=config

		geom=[PA,INC,X0,Y0]
		#check whether the disk geometry will be computed or not
		compute_geom=np.any([True if k in geom else False for k in osi])
		if compute_geom:
			# estimate geometric moments with mom0 map.
			geom_start=geom_moms(self.mom0,self.pixel_scale)
			sma=geom_start[-1]
		if rfinal in osi:
			rfinal = sma
			Print().status(f'rmax = {rfinal} arcsec')
		else:
			rfinal=eval(rfinal)

		for j,p in enumerate(geom):
			if p in osi:
				geom[j]=geom_start[j]
			else:
				geom[j]=eval(p)
		# disk geometry
		[PA,INC,X0,Y0]=geom
		if 0<INC<1: INC=eps_2_inc(INC)*180/np.pi
		if VSYS in osi :
			VSYS = KC(self.vel_map,X0,Y0)
		else:
			VSYS=eval(VSYS)

		guess0 = guess_vals(PA,INC,X0,Y0,VSYS,PHI_B )
		vary = np.array( [vary_PA,vary_INC,vary_XC,vary_YC,vary_VSYS,vary_PHI] )
		sigma = []
		self.ext = np.dot([-nx/2.,nx/2,-ny/2.,ny/2.], self.pixel_scale)
		self.osi = osi



		if self.survey not in self.osi :
			self.kin_params_table = "%sana_kin_%s_model.%s.csv"%(self.outdir,self.vmode,self.survey)
		else:
			self.kin_params_table = "%sana_kin_%s_model.%s.csv"%(self.outdir,self.vmode,self.galaxy)



		if "hrm_" in vmode:
			try:
				self.m_hrm = int(vmode[4:])
				if self.m_hrm == 0 :
					raise ValueError
			except(ValueError):
				print("XookSuut: provide a proper harmonic number different from zero, for example hrm_2")
				quit()
		# Print guess values
		self.P.guess_vals(galaxy,guess0,vmode)
		# Change INC to eps
		guess0 = guess_vals(PA,inc_2_eps(INC),X0,Y0,VSYS,PHI_B*np.pi/180 )


		self.P.status("Starting Least Squares analysis",line=True)
		if "hrm" not in self.vmode:
			circ = Circular_model(self.vmode, galaxy, self.datacube, self.errcube, self.h, self.momaps, self.emoms, guess0, vary, n_it, rstart, rfinal, ring_space, frac_pixel, inner_interp, delta, bar_min_max, config, self.outdir,cube_class)
			self.PA,self.EPS,self.XC,self.YC,self.VSYS,self.PHI_BAR,self.R,self.Disp,self.Vrot,self.Vrad,self.Vtan, self.kin_3D_mdls,self.PA_bar_mjr,self.PA_bar_mnr,self.bic_aic,self.errors_fit = circ()
		if "hrm" in self.vmode:
			hrm = Harmonic_model(galaxy, self.datacube, self.errcube, self.h, self.momaps, self.emoms, guess0, vary, n_it, rstart, rfinal, ring_space, frac_pixel, inner_interp, delta, bar_min_max, config, self.m_hrm, self.outdir,cube_class)
			self.PA,self.EPS,self.XC,self.YC,self.VSYS,self.R,self.Disp,self.Ck,self.Sk,self.kin_3D_mdls,self.bic_aic,self.errors_fit = hrm()
			self.Vrot=self.Ck[0]

		self.mom01d=self.kin_3D_mdls[0]
		self.ekin,self.econst, _ = self.errors_fit
		self.ePA,self.eEPS,self.eXC,self.eYC,self.eVSYS = self.econst[:5]
		if self.vmode == "bisymmetric":
			self.ePHI_BAR_deg = self.econst[5]*180/np.pi
		self.INC,self.eINC = eps_2_inc(self.EPS)*180/np.pi,e_eps2e_inc(self.EPS,self.eEPS)*180/np.pi
		self.PHI_BAR_deg = self.PHI_BAR*180/np.pi
		self.const=[self.PA,self.EPS,self.INC, self.XC,self.YC,self.VSYS,self.PHI_BAR_deg, np.nanmax(self.R)]
		self.redchi = self.bic_aic[-1]
		self.P.status("Best model found !")


class XS_out(Run_models):

	def results(self):
		if "hrm" not in self.vmode:
			e_Disp,e_Vrot,e_Vrad,e_Vtan = self.ekin
		else:
			e_Ck,e_Sk,e_Disp = self.ekin

		#
		## Write output into a table
		#
		if self.vmode in ["circular","radial","vertical", "ff"]:
			save_table(self.galaxy, self.vmode,self.R,self.Disp,self.Vrot,self.Vrad,self.Vtan,self.PA,self.EPS,self.XC,self.YC,self.VSYS,self.PHI_BAR,self.PA_bar_mjr,self.PA_bar_mnr,self.errors_fit,self.bic_aic, self.mom01d,out=self.outdir)

		if self.vmode in ["circular","radial","vertical", "ff"] or "hrm" in self.vmode:
			# write header of table
			if not path.exists(self.kin_params_table):
				hdr = ["object", "X0", "eX0", "Y0", "eY0", "PA_disk","ePA_disk", "INC", "eINC", "VSYS", "eVSYS", "redchi" ]
				write(hdr,self.kin_params_table,column = False)

			kin_params = [self.galaxy,self.XC,self.eXC,self.YC,self.eYC,self.PA,self.ePA,self.INC,self.eINC,self.VSYS,self.eVSYS,self.redchi]
			write(kin_params,self.kin_params_table,column = False)

		if self.vmode == "bisymmetric":
			# write header of table
			if not path.exists(self.kin_params_table):
				hdr = ["object", "X0", "eX0", "Y0", "PA_disk","INC", "eINC", "VSYS", "eVSYS", "PHI_BAR", "ePHI_BAR","PA_bar_mjr_sky","PA_bar_mnr_sky","redchi" ]
				write(hdr,self.kin_params_table,column = False)

			kin_params = [self.galaxy,self.XC,self.eXC,self.YC,self.eYC,self.PA,self.ePA,self.INC,self.eINC,self.VSYS,self.eVSYS,self.PHI_BAR_deg,self.ePHI_BAR_deg,self.PA_bar_mjr,self.PA_bar_mnr,self.redchi]
			write(kin_params,self.kin_params_table,column = False)



		# remove zeros from moment maps
		for k,mom in enumerate(self.momaps):
			mom[mom==0]=np.nan
			self.momaps[k]=mom


		# plot momment maps and 1D profiles
		self.P.status("Plotting results")
		plot_mommaps(self.galaxy,self.kin_3D_mdls,self.momaps,self.const,self.ext,self.vmode,self.h,self.config,self.pixel_scale,out=self.outdir)
		self.P.status("Saving 1D & 2D profiles")
		if 'hrm' not in self.vmode:
			# save 1d profiles
			s = save_model(self.galaxy, self.vmode,self.R,self.Disp,self.Vrot,self.Vrad,self.Vtan,self.PA,self.EPS,self.XC,self.YC,self.VSYS,self.PHI_BAR,self.PA_bar_mjr,self.PA_bar_mnr,self.errors_fit,self.bic_aic, self.e_ISM,out=self.outdir)
		else:
			s = save_model_h(self.galaxy, self.vmode,self.R,self.Disp,e_Disp,self.Ck,self.Sk,e_Ck,e_Sk,self.PA,self.EPS,self.XC,self.YC,self.VSYS,self.m_hrm,self.errors_fit,self.bic_aic,self.e_ISM,out=self.outdir)

		save_2d_models(self.galaxy,self.vmode,self.kin_3D_mdls,self.PA,self.INC,self.XC,self.YC,self.VSYS,self.m_hrm,out=self.outdir)

		# plot 1d models
		if "hrm" not in self.vmode:
			plot_kin_models(self.galaxy,self.vmode,self.kin_3D_mdls,self.R,self.Disp,e_Disp,self.Vrot,e_Vrot,self.Vrad,e_Vrad,self.Vtan,e_Vtan,self.VSYS,self.INC,self.ext,self.hdr_info,self.config,out=self.outdir)
		if "hrm" in self.vmode:
			plot_kin_models_h(self.galaxy,self.vmode,self.kin_3D_mdls,self.R,self.Disp,e_Disp,self.Ck,self.Sk,e_Ck,e_Sk,self.VSYS,self.ext,self.m_hrm,out=self.outdir)


		self.P.status("Creating 0th, 1st and 2nd momment maps")
		# save moment maps and cube model
		save_momments(self.galaxy,self.vmode,self.kin_3D_mdls,self.momaps,self.datacube,self.baselcube,self.h,out=self.outdir)
		self.P.status("creating PVD maps")
		out_pvd=pv_array(self.datacube,self.h,self.kin_3D_mdls,self.Vrot,self.R,self.PA,self.EPS,self.XC,self.YC,self.VSYS,self.pixel_scale,self.rms_cube,self.config)
		pvd_arr=out_pvd[0]
		# plot pvds
		plot_pvd(self.galaxy,out_pvd,self.Vrot,self.R,self.const,self.vmode,self.rms_cube,self.kin_3D_mdls,self.momaps,self.datacube,self.pixel_scale,self.h,self.hdr_info,self.config,self.outdir)
		# save pvds
		save_pvds(self.galaxy,self.vmode,out_pvd,self.rms_cube,self.hdr_info,self.outdir)

		self.P.status("creating residual cube")
		#create residual cube momement maps
		rescube=self.datacube-self.kin_3D_mdls[6]
		rescube[~np.isfinite(rescube)]=0

		plot_channels(self.galaxy,self.datacube,self.kin_3D_mdls,self.const,self.ext,self.vmode,self.h, self.hdr_info,self.config,self.rms_cube,self.pixel_scale,self.outdir)
		plot_rchannels(self.galaxy,self.datacube,rescube,self.const,self.ext,self.vmode,self.h, self.hdr_info,self.config,self.rms_cube,self.pixel_scale,self.outdir)

		print("Done!. Check the XS3D directory")
		end_time=time.time()
		total_time=end_time-self.start_time
		t=strftime("%H:%M:%S", gmtime(total_time))
		ttime="Total time: "+t+" HMS"
		self.P.status(ttime)
		print("------------------------------------")


	def __call__(self):
		run = self.results()
		#return self.results()
