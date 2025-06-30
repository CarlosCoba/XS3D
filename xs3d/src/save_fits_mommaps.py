import numpy as np
from astropy.io import fits


def zero2nan(data):
	data[data==0]=np.nan
	return data


def save_momments(galaxy,vmode,momms_mdls,momms_obs,datacube,baselcube,hdr,out):
	intens1d,mom0_axi,mom0_mdl,mom1_mdl,mom2_mdl_kms,mom2_mdl_A,cube_mdl,velmap_intr,sigmap_intr,twoDmodels= momms_mdls
	mom0,mom1,mom2=momms_obs
	mom0,mom1,mom2=zero2nan(mom0),zero2nan(mom1),zero2nan(mom2)
	mom0_mdl,mom1_mdl,mom2_mdl=zero2nan(mom0_mdl),zero2nan(mom1_mdl),zero2nan(mom2_mdl_kms)

	try:
		spec_axis_units=hdr['CUNIT3']
		spec_u=spec_axis_units.lower()
	except(KeyError):
		spec_u='AA'
	

	[ny,nx]=mom1.shape
	data = np.zeros((10,ny,nx))
	data[0] = mom0
	data[1] = mom0_mdl
	data[2] = mom0_axi	
	data[3] = mom1	
	data[4] = mom1_mdl	
	data[5] = velmap_intr
	data[6] = mom2		
	data[7] = mom2_mdl_A			
	data[8] = mom2_mdl_kms							
	data[9] = sigmap_intr							
	
	data=zero2nan(data)
	
	hdu = fits.PrimaryHDU(data)
	hdu.header['NAME0'] = f'observed mom0 (flux*km/s)'
	hdu.header['NAME1'] = f'model mom0 (flux*km/s)'			
	hdu.header['NAME2'] = f'model mom0 axisymmetric (flux*km/s)'				
	hdu.header['NAME3'] = 'observed mom1 (km/s)'
	hdu.header['NAME4'] = 'model mom1 (km/s)'
	hdu.header['NAME5'] = 'intrinsic circular velocity (km/s)'
	hdu.header['NAME6'] = 'observed mom2 (km/s)'
	hdu.header['NAME7'] = f'model mom2 (km/s)'
	hdu.header['NAME8'] = 'model mom2 (km/s)'
	hdu.header['NAME9'] = 'intrinsic dispersion(km/s)'
	
	hdu.writeto("%smodels/%s.%s.moments.fits.gz"%(out,galaxy,vmode),overwrite=True)
			

	cube_mdl=zero2nan(cube_mdl)
	hdu1 = fits.PrimaryHDU(cube_mdl)
	hdu1.header = hdr
	hdu1.writeto("%smodels/%s.%s.cube.fits.gz"%(out,galaxy,vmode),overwrite=True)
	
	
	residual=datacube-cube_mdl
	residual=zero2nan(residual)
	hdu1 = fits.PrimaryHDU(residual)
	hdu1.header['NAME0'] = 'Residual cube'
	hdu1.header = hdr
	hdu1.writeto("%smodels/%s.%s.res.cube.fits.gz"%(out,galaxy,vmode),overwrite=True)
	
	if baselcube is not None:
		hdu1 = fits.PrimaryHDU(baselcube)
		hdu1.header['NAME0'] = 'Baseline cube'
		hdu1.header = hdr
		hdu1.writeto("%smodels/%s.%s.baseline.cube.fits.gz"%(out,galaxy,vmode),overwrite=True)
	
	
def save_rmomments(galaxy,vmode,momms_obs,hdr,out):
	mom0,mom1,mom2=momms_obs
	mom0,mom1,mom2=zero2nan(mom0),zero2nan(mom1),zero2nan(mom2)

	try:
		spec_axis_units=hdr['CUNIT3']
		spec_u=spec_axis_units.lower()
	except(KeyError):
		spec_u='AA'
	

	[ny,nx]=mom1.shape
	data = np.zeros((3,ny,nx))
	data[0] = mom0
	data[1] = mom1	
	data[2] = mom2									
	
	data=zero2nan(data)
	
	hdu = fits.PrimaryHDU(data)
	hdu.header['NAME0'] = f'residual mom0 (flux*{spec_u})'
	hdu.header['NAME1'] = 'residual mom1 (km/s)'
	hdu.header['NAME2'] = 'residuk mom2 (km/s)'

	
	hdu.writeto("%smodels/%s.%s.resmoments.fits.gz"%(out,galaxy,vmode),overwrite=True)
			

