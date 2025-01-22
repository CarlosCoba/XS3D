import io
import sys
import warnings
import itertools
import numpy as np
from copy import deepcopy as copy
from os.path import basename, isfile
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from astropy.io.fits.verify import VerifyWarning
from itertools import product
import matplotlib.pylab as plt

from src.constants import __c__, __sigma_2_FWHM__
from src.read_hdr import Header_info

def trapecium(flux,a,b,dx):
	n=len(flux)
	f_a=flux[0]
	f_b=flux[-1]
	integrl=(dx/2.)*(f_a+f_b+2*np.sum([flux[k] for k in range(1,n-1)]))
	return integrl

def trapecium3d(flux3d,dx,a=0,b=-1):
	[nz,ny,nx]=flux3d.shape
	f_a=flux3d[a]
	f_b=flux3d[b]
	#integrl=(dx/2.)*(f_a+f_b+2*np.nansum(flux3d[1:nz-1,:,:],axis=0))
	integrl=(dx/2.)*(f_a+f_b+2*(flux3d[1:nz-1,:,:]).sum(axis=0))	
	#msk=integrl!=0
	#integrl/=msk
	return integrl
	
def GaussProf(wave,lambda0,f0,sigma=None,fwhm=None):
	fi=1
	delta2=(wave-lambda0)**2
	if sigma!=None:
		sigma2=sigma**2
		fi=f0*np.exp(-0.5*delta2/sigma2 ) 
	if fwhm!=None:
		#sigma=fwhm/(2*np.sqrt(2*np.log(2)))
		#sigma2=fwhm2/(8*np.log(2))
		fwhm2=fwhm**2
		fi=f0*np.exp(-4*np.log(2)*delta2/fwhm2 ) 
		
	return fi


def GaussProf_V(wave_kms,V0,f0,sigma=None,fwhm=None):
	fi=1
	delta2=(wave_kms-V0)**2
	if sigma!=None:
		sigma2=sigma**2
		fi=f0*np.exp(-0.5*delta2/sigma2 ) 
	if fwhm!=None:
		#sigma=fwhm/(2*np.sqrt(2*np.log(2)))
		#sigma2=fwhm2/(8*np.log(2))
		fwhm2=fwhm**2
		fi=f0*np.exp(-4*np.log(2)*delta2/fwhm2 ) 
		
	return fi



def mask_wave(h,config):
	config_general = config['general']
	wmin,wmax=config_general.getfloat('wmin',None),config_general.getfloat('wmax',None)
	cut = False										
	if wmin is None and wmax is None:
		return None, None, False

	hdr=Header_info(h,config)
	[nz,ny,nx]=hdr.cube_dims()
	wave_cover=hdr.spectral_axis()

	msk=np.ones_like(wave_cover,dtype=bool)
	if wmin is not None:
		msk[:] = wave_cover >= wmin
	if wmax is not None:		
		wmax_i = wave_cover <= wmax
		for j,val in enumerate(wmax_i):
			if not val: msk[j]=False

	crval3=(wave_cover[msk])[0]
	cdelt3=wave_cover[1]-wave_cover[0]
	
	h['CRVAL3']=crval3
	h['CRPIX3']=1
	h['CDELT3']=cdelt3
	h['NAXIS3']=len(wave_cover[msk])			
	return msk,h,True

def mommaps(cube,h,config,rms=0):
	#[nz,ny,nx]=cube.shape
	hdr=Header_info(h)
	[nz,ny,nx]=hdr.cube_dims()
	
	config_general = config['general']
	eline=config_general.getfloat('eline',None)
	wmin,wmax=config_general.getfloat('wmin',None),config_general.getfloat('wmax',None)										
	crval,cdelt,pixel_scale=hdr.read_header()
				
	wave_cover=hdr.wave_kms
	msk_w,_=mask_wave(h,config)

	wave_cover=wave_cover[msk_w]		
	I0=np.zeros((ny,nx))
	I1=np.zeros((ny,nx))
	I2=np.zeros((ny,nx))
	
	for j,i in product(np.arange(ny),np.arange(nx)):
		flux_k = cube[:,j,i]
		if np.sum(flux_k)!=0:
			flux_k=flux_k[msk_w]
			x0,x1=wave_cover[0],wave_cover[-1]
			I0_k=trapecium(flux_k,x0,x1,cdelt)
			I1_k=trapecium(flux_k*wave_cover,x0,x1,cdelt)/I0_k
			I2_k=trapecium(flux_k*(wave_cover-I1_k)**2,x0,x1,cdelt)/I0_k
			
			if I0_k==0: I1_k,I2_k=0,0
			
			#plt.plot(wave_cover,flux_k);
			#plt.plot(I1_k,max(flux_k),'ko');plt.show()
						
			I0[j][i]=I0_k				
			I1[j][i]=I1_k				
			I2[j][i]=abs(I2_k)
			#print(I0_k,I1_k,I2_k)
	

	#I0[I0==0]=np.nan		
	#I1[I1==0]=np.nan		
	#I2[I2==0]=np.nan
	
	#I1=I1 if eline is None else __c__*(I1-eline)/eline
	I2=np.sqrt(I2)
	#I2=I2 if eline is None else __c__*(I2)/eline
	fwhm=I2*__sigma_2_FWHM__
	return I0,I1,I2,fwhm				

