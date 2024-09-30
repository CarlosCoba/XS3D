import numpy as np
np.set_printoptions(precision=4,suppress=True)
from src0.read_config import config_file
from src0.psf_lsf import PsF_LsF

class Print:
	def __init__(self):
		self.n=36 
		self.deli="-"*self.n
			
	def guess_vals(self,galaxy,guess,vmode):
		PA,INC,X0,Y0,VSYS,PHI_B = 	guess[0],guess[1],guess[2],guess[3], guess[4], guess[5] 
		print(self.deli)
		INC=round(INC,2)
		PA=round(PA,2)
		print ('{:<20}'.format(f'**Guess values for {galaxy}**'))		
		print ('{:<20} {:<15}'.format('P.A.:', f'{PA}'))
		print ('{:<20} {:<15}'.format('INC:', f'{INC}'))
		print ('{:<20} {:<15}'.format('X0,Y0:', '%s, %s'%(round(X0,2),round(Y0,2))))				
		print ('{:<20} {:<15}'.format('VSYS:', '%s km/s'%round(VSYS,2)))
		if vmode == "bisymmetric" :
			print ('{:<20} {:<15}'.format('PHI_BAR:', f'{PHI_B}'))
		print ('{:<20} {:<15}'.format('KIN MODEL:', f'{vmode}'))
		print(self.deli)		

	def __call__(self):
		print(self.deli)		
		print("-------------- XS3D ----------------")
		

	def out(self,hdr,value,tab="\t"):
		print(self.deli)		
		print ('{:<20} {:<15}'.format(f'{hdr}:', f'{value}'))
				
		
	def cubehdr(self,hdr):
		print(self.deli)
		cdelt3=round(hdr.cdelt3_kms,3)
		print ('{:<20} {:<15}'.format('Cube dims:', f'{hdr.nz}x{hdr.ny}x{hdr.nx} pix'))
		print ('{:<20} {:<15}'.format('Rest frame eline:', f'{hdr.eline}'))
		print ('{:<20} {:<15}'.format('Channel width:', f'{cdelt3} km/s'))
		
	def configprint(self,cube_hdr,config):
		#"""
		general=config['general']
		others=config['others']		
		#psf_fwhm=general.getfloat('psf_fwhm',0)
		#bmaj=general.getfloat('bmaj',0)	
		#bmin=general.getfloat('bmin',0)	
		#bpa=general.getfloat('bpa',0)			
		fwhm_inst=general.getfloat('fwhm_inst',0)
		vpeak=others.getboolean('vpeak',False)
		#"""
		psf_lsf= PsF_LsF(cube_hdr, config)
		fit_psf=psf_lsf.fit_psf
		bmaj=psf_lsf.bmaj 
		bmin=psf_lsf.bmin
		bpa= psf_lsf.bpa
		psf_fwhm=psf_lsf.fwhm_psf_arc						

		print(self.deli)						
		if psf_fwhm is not None:
				psf_fwhm=round(psf_fwhm,3)
				print ('{:<20} {:<15}'.format('PSF FWHM:', f'{psf_fwhm} arcsec'))
		if bmaj is not None and bmin is not None:				
				bmaj=round(bmaj,3)
				bmin=round(bmin,3)
				print ('{:<20} {:<15}'.format('BMAJ:', f'{bmaj} arcsec'))
				print ('{:<20} {:<15}'.format('BMIN:', f'{bmin} arcsec'))				
		if bpa!=0:
				bpa=round(bpa,1)				
				print ('{:<20} {:<15}'.format('BPA:', f'{bpa} deg'))										
		if fwhm_inst != 0:
				fwhm_inst=round(fwhm_inst,3)
				print ('{:<20} {:<15}'.format('Spec. broadening:', f'{fwhm_inst}'))
				
		if vpeak:
			print(self.deli)						
			print ('{:<20} {:<15}'.format('Vpeak:', f'{vpeak}'))
												
		print(self.deli)				
					
	def status(self,msn,line=False):
		print(self.deli)			
		m=len(msn)
		space=""
		if m < self.n:
			space="."*(self.n-m-1)
		print(f'{msn} {space}')	
		if line: print(self.deli)			


