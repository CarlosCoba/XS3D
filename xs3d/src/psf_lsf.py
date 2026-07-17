import numpy as np
from .constants import __c__,__sigma_2_FWHM__,__FWHM_2_sigma__

class PsF_LsF:
	def __init__(self, cube_hdr, config):

		config_general = config['general']

		header_xs3d =config['header']

		config_clouds = config['clouds']

		self.nthreads = config_general.getint('nthreads',3)

		self.ctype3=header_xs3d.get('ctype3','wavelength')

		self.vary_disp=config_general.getint('fit_disp',1)

		self.fwhm_inst_A=config_general.getfloat('fwhm_inst',None)

		bmaj_hdr=cube_hdr.bmaj

		bmin_hdr=cube_hdr.bmin

		bpa_hdr=cube_hdr.bpa

		self.pix_arcs = cube_hdr.pix_arcs

		self.fwhm_psf_arc=bmaj_hdr if bmaj_hdr is not None else config_general.getfloat('psf_fwhm',None)

		self.fwhm_psf_pix=self.fwhm_psf_arc/self.pix_arcs

		self.bmaj=bmaj_hdr if bmaj_hdr is not None else config_general.getfloat('bmaj',self.fwhm_psf_arc)

		self.bmin=bmin_hdr if bmin_hdr is not None else config_general.getfloat('bmin',self.bmaj)

		self.bpa=bpa_hdr if bpa_hdr is not None else config_general.getfloat('bpa',0)

		if self.fwhm_psf_arc is not None or self.bmaj is not None:
				self.fit_psf=True
		else:
				self.fit_psf=False

		self.cdelt3 = cube_hdr.cdelt3

		self.nz=cube_hdr.nz

		self.ny=cube_hdr.ny

		self.nx=cube_hdr.nx

		self.eline_A=config_general.getfloat('eline',None)

		self.cdelt3_kms=__c__*(self.cdelt3/self.eline_A) if self.eline_A is not None else None

		self.fwhm_inst_A=config_general.getfloat('fwhm_inst',2*self.cdelt3)

		self.sigma_inst_A=self.fwhm_inst_A*__FWHM_2_sigma__ if self.fwhm_inst_A is not None else None

		self.sigma_inst_kms=(self.sigma_inst_A/self.eline_A)*__c__ if self.fwhm_inst_A is not None else None

		fwhm_inst_kms=config_general.getfloat('fwhm_kms',None)

		self.fwhm_inst_kms=None

		if fwhm_inst_kms is not None:
			self.fwhm_inst_kms=fwhm_inst_kms
		else:
			if self.sigma_inst_kms is not None:
				self.fwhm_inst_kms=self.sigma_inst_kms*__sigma_2_FWHM__

		if 'velocity' in self.ctype3:
			if 'kms' in self.ctype3:
				# start fwhm_A
				self.fwhm_inst_A=self.fwhm_inst_kms
				self.sigma_inst_kms=(self.fwhm_inst_kms*__FWHM_2_sigma__) if self.fwhm_inst_A is not None else None
				self.cdelt3_kms=self.cdelt3
				# sigma_inst_A must be in native units !
				self.sigma_inst_A=self.fwhm_inst_kms*__FWHM_2_sigma__
			elif 'ms' in self.ctype3:
				# start fwhm_A
				self.fwhm_inst_A=self.fwhm_inst_kms
				self.sigma_inst_kms=(self.fwhm_inst_kms*__FWHM_2_sigma__) if self.fwhm_inst_A is not None else None
				# sigma_inst_A must be in native units !
				self.sigma_inst_A=1e3*self.fwhm_inst_kms*__FWHM_2_sigma__
				self.cdelt3_kms=self.cdelt3/1e3
			else:
				print('XS3D: Not recognized velocity units in CTYPE3')
				quit()

		self.sigma_inst_pix=(self.sigma_inst_A/abs(self.cdelt3)) if self.fwhm_inst_A is not None else None

		if self.fwhm_inst_kms is not None:
			self.sigma_inst_kms=self.fwhm_inst_kms*__FWHM_2_sigma__
			self.sigma_inst_pix=(self.sigma_inst_kms/abs(self.cdelt3_kms))

		self.chanw_kms=self.cdelt3_kms
		self.lsf_kms=self.sigma_inst_kms*__sigma_2_FWHM__


		# radial_step is the spacing of the fine ring grid built internally by _interpolate_rings.
		# It controlls how densely the galaxy disk plane is sampled between anchor rings.
		self.radial_step = self.bmaj

		# vertical hight scale in arcseconds
		hz=config_clouds.getfloat('z_scale',0.1)
		self.zscale=np.max(hz,0)
		self.zscale_pix=self.zscale	/ self.pix_arcs

		# slit width in arcsec
		self.slit_w = self.bmaj
