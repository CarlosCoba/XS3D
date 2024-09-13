class PsF_LsF:
	def __init__(self,cube_hdr, config):
	
		config_general = config['general']
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
			
