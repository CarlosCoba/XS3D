import numpy as np
from .constants import __c__,__sigma_2_FWHM__,__FWHM_2_sigma__

class PsF_LsF:
	def __init__(self, cube_hdr, config):

		config_general	= config['general']

		header_xs3d		= config['header']

		config_clouds	= config['clouds']

		self.nthreads	= config_general.getint('nthreads',3)

		self.vary_disp	= config_general.getint('fit_disp',1)

		bmaj_hdr	= cube_hdr.bmaj

		bmin_hdr	= cube_hdr.bmin

		bpa_hdr		= cube_hdr.bpa

		self.pix_arcs	= cube_hdr.pix_arcs

		self.fwhm_psf_arc = bmaj_hdr if bmaj_hdr is not None else config_general.getfloat('psf_fwhm',self.pix_arcs)

		self.bmaj		=bmaj_hdr if bmaj_hdr is not None else config_general.getfloat('bmaj',self.fwhm_psf_arc)

		self.bmin		= bmin_hdr if bmin_hdr is not None else config_general.getfloat('bmin',self.bmaj)

		self.bpa		= bpa_hdr if bpa_hdr is not None else config_general.getfloat('bpa',0)
		
		self.fwhm_psf_pix = self.bmaj / self.pix_arcs		

		if self.fwhm_psf_arc is not None or self.bmaj is not None:
				self.fit_psf=True
		else:
				self.fit_psf=False

		self.cdelt3		= cube_hdr.cdelt3
		
		self.cdelt3_kms	= cube_hdr.cdelt3_kms		

		self.nz	= cube_hdr.nz

		self.ny	= cube_hdr.ny

		self.nx	= cube_hdr.nx

		self.eline_A =	config_general.getfloat('eline',None)
		
		fwhm_inst_kms=config_general.getfloat('fwhm_kms',None)
		
		if 	fwhm_inst_kms is not None:
			self.fwhm_inst_kms	= fwhm_inst_kms					
			self.sigma_inst_kms	= fwhm_inst_kms * __FWHM_2_sigma__
		else:
			fwhm_inst_A			= config_general.getfloat('fwhm_inst',2*self.cdelt3)
			self.fwhm_inst_kms 	= (fwhm_inst_A/self.eline_A) * __c__
			self.sigma_inst_kms = self.fwhm_inst_kms * __FWHM_2_sigma__

		
		self.A_beam_px=(np.pi*(self.fwhm_psf_pix)**2) / (4*np.log(2)) #px2
		self.sigma_inst_pix	= self.sigma_inst_kms / abs(self.cdelt3_kms)
		self.chanw_kms		= self.cdelt3_kms
		self.lsf_kms		= self.fwhm_inst_kms


		# radial_step is the spacing of the fine ring grid built internally by _interpolate_rings.
		# It controlls how densely the galaxy disk plane is sampled between anchor rings.
		self.radial_step = self.bmaj

		# vertical hight scale in arcseconds
		hz	= config_clouds.getfloat('z_scale',0.1)
		self.zscale	= np.max(hz,0)
		self.zscale_pix	= self.zscale	/ self.pix_arcs

		# slit width in arcsec
		self.slit_w = self.bmaj
