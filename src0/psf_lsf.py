import numpy as np
from src0.constants import __c__,__sigma_2_FWHM__,__FWHM_2_sigma__

class PsF_LsF:
	def __init__(self,cube_hdr, config):
	
		config_general = config['general']
		header_xs3d =config['header']
		self.ctype3=header_xs3d.get('ctype3','wavelength')
		self.vary_disp=config_general.getboolean('fit_dispersion',False)
		self.fwhm_inst_A=config_general.getfloat('fwhm_inst',None)

		try:
			self.bmaj=cube_hdr['BMAJ']*3600
			self.bmin=cube_hdr['BMIN']*3600
			self.bpa=cube_hdr['BPA']
			self.fwhm_psf_arc=None			
			self.fit_psf=True
		except(KeyError):		
			self.fwhm_psf_arc=config_general.getfloat('psf_fwhm',None)
			self.bpa=config_general.getfloat('bpa',0)
			self.bmaj=config_general.getfloat('bmaj',self.fwhm_psf_arc)
			self.bmin=config_general.getfloat('bmin',self.bmaj)			
			if self.fwhm_psf_arc is not None or self.bmaj is not None:
				self.fit_psf=True
			else:
				self.fit_psf=False
		
		
		self.cdelt3=float(cube_hdr['CDELT3'])
		self.nz=cube_hdr['NAXIS3']		
		self.eline_A=config_general.getfloat('eline',None)
		self.fwhm_inst_A=config_general.getfloat('fwhm_inst',None)
		self.sigma_inst_A=self.fwhm_inst_A*__FWHM_2_sigma__ if self.fwhm_inst_A is not None else None
		self.sigma_inst_kms=(self.sigma_inst_A/self.eline_A)*__c__ if self.fwhm_inst_A is not None else None
		self.sigma_inst_pix=(self.sigma_inst_A/abs(self.cdelt3))*np.ones(self.nz) if self.fwhm_inst_A is not None else None
		
		if 'velocity' in self.ctype3:
			if 'kms' in self.ctype3:
				self.sigma_inst_kms=self.sigma_inst_A
			elif 'ms' in self.ctype3:
				self.sigma_inst_kms=self.sigma_inst_A/1e3 if self.fwhm_inst_A is not None else None			
			else:
				print('XS3D: Not recognized velocity units in CTYPE3')
				quit() 		
