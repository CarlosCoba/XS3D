import numpy as np
from astropy.io import fits


def zero2nan(data):
	data[data==0]=np.nan
	return data


def save_pvds(galaxy,vmode,out_pvd, rms,hdr_info,out):
	pvd_arr=out_pvd[0]
	ext_pvds=out_pvd[2]
	cdelt3=hdr_info.cdelt3_kms

	[pvd_obs_major,pvd_obs_mnr,pvd_mdl_major,pvd_mdl_mnr]= pvd_arr
	
	# axis 1 corresponds to distance
	# axis 2 corresponds to velocity
	
	ext_pv=ext_pvds[0]
	pixelr=ext_pvds[2]
	
	crval1=ext_pv[0] # --> r. axis
	crval2=ext_pv[2] # --> vel. axis


	[ny,nx]=pvd_arr[0].shape
	data = np.zeros((4,ny,nx))
	data[0] = pvd_obs_major
	data[1] = pvd_mdl_major
	data[2] = pvd_obs_mnr	
	data[3] = pvd_mdl_mnr							
	RMS=rms if np.isfinite(rms) else 0
	
	hdu = fits.PrimaryHDU(data)
	hdu.header['CDELT1'] = pixelr
	hdu.header['CDELT2'] = cdelt3	
	hdu.header['CRVAL1'] = crval1	
	hdu.header['CRVAL2'] = crval2	
	hdu.header['CRPIX1'] = 1	
	hdu.header['CRPIX2'] = 1	
	hdu.header['CTYPE1'] = 'DISTANCE-ARCSEC'				
	hdu.header['CTYPE2'] = 'LOS-VELOCITY-KM/S'		
	hdu.header['BUNIT'] = 'INTENSITY'			
	hdu.header['RMS'] = RMS			
	hdu.header['NAME0'] = f'PVD major observed'
	hdu.header['NAME1'] = f'PVD major model'			
	hdu.header['NAME2'] = 'PVD minor observed'
	hdu.header['NAME3'] = 'PVD minor model'
	hdu.writeto("%smodels/%s.%s.pvd.fits.gz"%(out,galaxy,vmode),overwrite=True)
	
	return None
			

