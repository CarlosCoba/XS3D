from .constants import __c__
import numpy as np
import warnings

class Header_info:
	def __init__(self,hdr,config):
		self.config=config
		self.naxis = hdr['NAXIS']
		self.nz = hdr['NAXIS3']
		self.ny = hdr['NAXIS2']
		self.nx = hdr['NAXIS1']
		self.crpix3 = hdr['CRPIX3']
		self.scale=0
		self.crval3=0
		self.bunit='Intensity'
		self.ctype3=None
		self.cunit1='deg'
		#self.cdelt3_kms=0
		self.cunit3=''
		self.wavelength_wave=['Angstrom','angstrom','um','WAVELENGTH','WAVE','ANGSTROM','UM','micron','lambda','LAMBDA']
		self.wavelength_frec=['FREQ','Freq','freq','FREQUENCY','frequency','Hz','HZ']
		self.wave_types=self.wavelength_wave+self.wavelength_frec
		general=self.config['general']
		others=self.config['others']
		highz=self.config['high_z']
		self.redshift=highz.getfloat('redshift',None)
		self.eline=general.getfloat('eline',None)
		self.vdoppler=others.get('vdoppler','opt')

		if self.naxis!=3:
			print(f'The input data is not a datacube!.\
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
			ctype3=hdr['CTYPE3']
			self.ctype3=ctype3.replace(' ','')
			if self.ctype3 not in self.wave_types: raise KeyError
		except(KeyError) as err:
			header=self.config['header']
			ctype=header.get('ctype3',None)
			self.ctype3=ctype
			if ctype is None:
				print(err, 'No CTYPE3 information found in header or in XS3D config file');quit()
		try:
			self.cunit3=hdr['CUNIT3']
		except(KeyError):
			pass


		if self.scale>1:
			print('Warning!, the pixel scale seems to be too large !')

		self.scale=self.scale*3600 # from degree to arcsec
		self.spec_axis = self.crval3 + self.cdelt3*(np.arange(self.nz) + 1 - self.crpix3) # wave axis in original units
		#if self.spec_axis[0]>self.spec_axis[-1]:
		#	self.cdelt3=-1*self.cdelt3
		#	self.spec_axis = self.crval3 + self.cdelt3*(np.arange(self.nz) + 1 - self.crpix3) # wave axis in original units

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
		elif 'velocity' in self.ctype3:
			if 'kms' in self.ctype3:
				self.wave_kms=self.spec_axis
				self.cdelt3_kms=self.cdelt3
			elif 'ms' in self.ctype3:
				self.wave_kms=self.spec_axis/1e3
				self.cdelt3_kms=self.cdelt3/1e3
		else:
			print("No spectral information/units found in header or in config_file");quit()


	def cube_dims(self):
		return self.nz,self.ny,self.nx

	def spectral_axis(self):
		return 	self.spec_axis

	def cdelt_kms(self,eline):
		self.wave_axis_kms(eline)
		return self.cdelt3_kms

	def read_header(self):
		return self.crval3,self.cdelt3,self.scale
