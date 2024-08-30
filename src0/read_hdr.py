from src0.constants import __c__
import numpy as np
import warnings

class Header_info:
	def __init__(self,hdr,config):
		self.config=config
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
		self.wavelength_wave=['Angstrom','angstrom','um','WAVELENGTH','WAVE','ANGSTROM','UM','micron']
		self.wavelength_frec=['FREQ','Freq','freq','FREQUENCY','frequency','Hz','HZ']			
		self.wave_types=self.wavelength_wave+self.wavelength_frec
		general=self.config['general']
		self.eline=general.getfloat('eline',None)
		
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



		if (self.cunit3 in self.wavelength_wave) or (self.ctype3 in self.wavelength_wave):
			wave_A2kms=__c__*(self.spec_axis-self.eline)/self.eline # wave in kms
			self.wave_kms=wave_A2kms
			self.cdelt3_kms=__c__*(self.cdelt3/self.eline)		
		elif (self.cunit3 in self.wavelength_frec) or (self.ctype3 in self.wavelength_frec): 
			wave_hz2kms=__c__*(self.eline-self.spec_axis)/self.eline # wave in kms
			self.wave_kms=wave_hz2kms
			self.cdelt3_kms=__c__*(self.cdelt3/self.eline)
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


