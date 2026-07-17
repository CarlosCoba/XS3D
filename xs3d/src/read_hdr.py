from .constants import __c__
import numpy as np
import warnings

class Header_info:

	def __init__(self,hdr,config):

		self.naxis = hdr['NAXIS']

		self.nv = hdr['NAXIS3'] # <-- velocity channels

		self.nz = hdr['NAXIS3']

		self.ny = hdr['NAXIS2']

		self.nx = hdr['NAXIS1']

		self.crpix3 = hdr['CRPIX3'] if 'CRPIX3' in hdr else 1

		self.bmaj = hdr['BMAJ']*3600 if 'BMAJ' in hdr else None

		self.bmin = hdr['BMIN']*3600 if 'BMIN' in hdr else None

		self.bpa = hdr['BPA'] if 'BMIN' in hdr else None

		self.scale=0

		self.crval3=0

		self.bunit='Intensity'

		ctype3=hdr['CTYPE3'] if 'CTYPE3' in hdr else None

		cunit3=hdr['CUNIT3'] if 'CUNIT3' in hdr else None

		self.cunit1='deg'

		self.wavelength_wave=['angstrom','um','wavelength','wave','angstrom','um','micron','lambda']

		self.wavelength_frec=['freq','frequency','hz']

		self.wavelength_vel=['kms','km/s','kms-1','ms','m/s','ms-1']

		self.wave_types=self.wavelength_wave+self.wavelength_frec+self.wavelength_vel

		self.config=config
		general	= self.config['general']
		others	= self.config['others']
		highz	= self.config['high_z']
		header	= self.config['header']


		self.redshift=highz.getfloat('redshift',None)

		self.eline=general.getfloat('eline',None)

		self.vdoppler=others.get('vdoppler','opt')

		if self.naxis!=3:
			print(f'The input data is not a 3D datacube!.\
			Check your cube dimensions \
			NAXIS = {self.naxis} != 3')
			quit()
		try:
			self.crval3=hdr['CRVAL3']
		except(KeyError) as err:
			print(err,'No CRVAL3 found in header')
			quit()

		try:
			self.cdelt3=hdr['CDELT3']
		except(KeyError):
			try:
				self.cdelt3=hdr['CD3_3']
			except(KeyError) as err:
				print(err,'No CDELT3 or CD3_3 found in header')
				quit()

		try:
			self.scale=abs(hdr['CD1_1'])
		except(KeyError):
			try:
				self.scale=abs(hdr['CDELT1'])
			except(KeyError) as err:
				print(err,'No CDELT1 or CD1_1  found in header')
				quit()

		try:
			self.ctype3=(ctype3.replace(' ','')).casefold() if ctype3 is not None else ctype3
			self.cunit3=(cunit3.replace(' ','')).casefold() if cunit3 is not None else cunit3
			if self.ctype3 not in self.wave_types:
				if self.cunit3 not in self.wave_types:
					raise KeyError
		except(KeyError) as err:
			ctype=header.get('ctype3',None)
			self.ctype3=ctype
			if ctype is None:
				print(err, 'No CTYPE3 information was found in the header or the XS3D config file');quit()

		if self.scale>1:
			print('Warning!, the pixel scale seems to be too large !')


		self.pix_arcs = self.scale*3600
		self.scale=self.scale*3600 # from degree to arcsec

		# wave axis in original units:
		self.spec_axis = self.crval3 + self.cdelt3*(np.arange(self.nz) + 1 - self.crpix3)

		if (self.cunit3 in self.wavelength_wave) or (self.ctype3 in self.wavelength_wave):
			self.elinez=self.eline if self.redshift is None else self.eline*(1+self.redshift)
			wave_A2kms=__c__*(self.spec_axis-self.elinez)/self.elinez # wave in kms
			self.wave_kms=wave_A2kms
			self.cdelt3_kms=__c__*(self.cdelt3/self.elinez)
		elif (self.cunit3 in self.wavelength_frec) or (self.ctype3 in self.wavelength_frec):
			self.elinez=self.eline if self.redshift is None else self.eline/(1+self.redshift)
			if self.vdoppler == 'rad':
				wave_hz2kms=__c__*(self.elinez-self.spec_axis)/self.elinez # wave in kms radio definitiom
			if self.vdoppler == 'opt':
				wave_hz2kms=__c__*((self.elinez/self.spec_axis)-1) # wave in kms	optical definition
			self.wave_kms=wave_hz2kms
			self.cdelt3_kms=__c__*(self.cdelt3/self.elinez)
		elif any( [item in ['kms','km/s','kms-1'] for item in [self.cunit3,self.ctype3]] ):
			self.wave_kms=self.spec_axis
			self.cdelt3_kms=self.cdelt3
		elif any( [item in ['ms','m/s','ms-1'] for item in [self.cunit3,self.ctype3]] ):
				self.wave_kms=self.spec_axis/1e3
				self.cdelt3_kms=self.cdelt3/1e3
		else:
			print("No spectral information/units found in header or in config_file");quit()

		self.dx=self.pix_arcs		 #  arcsec/pixel
		self.dy=self.pix_arcs		 #  arcsec/pixel
		self.dv=self.cdelt3_kms
		self.v_min=self.wave_kms[0]
		self.rms = 1

	def cube_dims(self):
		return self.nz,self.ny,self.nx

	def spectral_axis(self):
		return 	self.spec_axis

	def cdelt_kms(self,eline):
		self.wave_axis_kms(eline)
		return self.cdelt3_kms

	def read_header(self):
		return self.crval3,self.cdelt3,self.scale
