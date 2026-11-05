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

from .constants import __c__, __sigma_2_FWHM__
from .read_hdr import Header_info
from .cube_selector import window
	
			
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
	integrl=(dx/2.)*(f_a+f_b+2*(flux3d[1:nz-1,:,:]).sum(axis=0))	
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
	wmin,wmax =config_general.getfloat('wmin',None),config_general.getfloat('wmax',None)
	crop 	  = False										
	if (wmin is None) and (wmax is None):
		return None, None, crop
	
	if wmin is not None and wmax is not None:
		pass
	elif wmin is None:
		wmin=-np.inf
	else:
		wmax=np.inf
	
	crop = True
	hdr			= Header_info(h,config)
	[nz,ny,nx]	= hdr.cube_dims()
	wave_cover	= hdr.spectral_axis()
		
	msk 	= (wave_cover >= wmin) * (wave_cover <= wmax)
	indices = (np.arange(nz).astype(int))[msk]	

	minz	= np.min(indices)
	maxz	= np.max(indices)+1
	minz	= np.clip(minz,0,None)
	maxz	= np.clip(maxz,None,nz)
	
	slices	= tuple([slice(minz,maxz,None),slice(None,None,None),slice(None,None,None)])
	
	crval3	= (wave_cover[msk]).min()
	cdelt3	= wave_cover[1]-wave_cover[0]

	# update header
	h['CRVAL3']=crval3
	h['CRPIX3']=1
	h['CDELT3']=cdelt3
	h['NAXIS3']=msk.sum()	

	return slices,h,crop


def mask_xyz(cube,h,config):
	"""
	config_general = config['general']
	wmin,wmax =config_general.getfloat('wmin',None),config_general.getfloat('wmax',None)
	crop 	  = False										
	if (wmin is None) and (wmax is None):
		return None, None, crop
	
	if wmin is not None and wmax is not None:
		pass
	elif wmin is None:
		wmin=-np.inf
	else:
		wmax=np.inf

	"""

	config_general	= config['general']
	crop			= config_general.getboolean('crop_xyz', False)
	
	if not crop:
		return None, None, None, None, crop

	hdr			= Header_info(h,config)
	[nz,ny,nx]	= hdr.cube_dims()
	wave_cover	= hdr.spectral_axis()

	xmin,xmax = 0,nx-1 
	ymin,ymax = 0,ny-1 
	zmin,zmax = 0,nz-1 
		
	results = window(cube,wave_cover)
	roi		= results['roi']
	wind_range = results['channel_ranges']
	
	# roi and wind_range provide indices
	if roi :
		(xmin,xmax,ymin,ymax) = roi
		xmax+=1	
		ymax+=1			
		crop = True
	
	if wind_range:
		[(zmin,zmax)] = wind_range
		zmax+=1			
		crop = True
	
	# `slice` last indice is exclusive.
	slices3D	= tuple([slice(zmin,zmax,None),slice(ymin,ymax,None),slice(xmin,xmax,None)])
	slices2D	= tuple([slice(ymin,ymax,None),slice(xmin,xmax,None)])
	slices2D	= tuple([slice(0,None,None),slice(0,None,None)])	

	cdelt3		= h['CDELT3']
	crval3_tmp	= h['CRVAL3']
	crpix3		= h['CRPIX3']
	crval3		= wave_cover[zmin]
						
	naxis3	= (zmax-zmin) 
	naxis1	= (xmax-xmin)	
	naxis2	= (ymax-ymin)		
	
	# update header
	h['CRVAL3']=crval3
	h['CRPIX3']=1
	h['CDELT3']=cdelt3
	h['NAXIS1']=naxis1
	h['NAXIS2']=naxis2
	h['NAXIS3']=naxis3	
	

	# Update the reference pixel coordinates	
	if 'CRPIX1' in h:
		h['CRPIX1']-=xmin		
		h['CRPIX2']-=ymin		

	xshift, yshift	= -xmin, -ymin
	#xshift, yshift	= xmin, ymin	
	xy_shift		= [xshift, yshift]
	
	
	return slices3D, slices2D, h, xy_shift, crop
				
	"""
		
	msk 	= (wave_cover >= wmin) * (wave_cover <= wmax)
	indices = (np.arange(nz).astype(int))[msk]	

	minz	= np.min(indices)
	maxz	= np.max(indices)+1
	minz	= np.clip(minz,0,None)
	maxz	= np.clip(maxz,None,nz)
	
	slices	= tuple([slice(minz,maxz,None),slice(None,None,None),slice(None,None,None)])
	
	crval3	= (wave_cover[msk]).min()
	cdelt3	= wave_cover[1]-wave_cover[0]

	# update header
	h['CRVAL3']=crval3
	h['CRPIX3']=1
	h['CDELT3']=cdelt3
	h['NAXIS3']=msk.sum()	

	return slices,h,crop
	"""
	return 0,0,0

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

