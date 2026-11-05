import numpy as np
from astropy.io import fits

def save_momments(galaxy,vmode,momms_obs,momms_mod,cube_obs,cube_mod,baselcube,hdr_ori,const,out):

	mom0_obs,mom1_obs,mom2_obs=momms_obs
	mom0_mod,mom1_mod,mom2_mod=momms_mod	
	msk=np.isfinite(mom0_obs)*np.isfinite(mom0_mod)

	[v_sys,inc,pa,x_center,y_center,phi_bar,rmax]=const['v_sys'],const['inc'],const['pa_NE'],const['x_center'],const['y_center'],const['phi_bar'],const['rmax']
	try:
		spec_axis_units=hdr_ori['CUNIT3']
		spec_u=spec_axis_units.lower()
	except(KeyError):
		spec_u='AA'
	

	[ny,nx]=mom0_obs.shape
	data = np.zeros((6,ny,nx))
	data[0] = mom0_obs
	data[1] = mom0_mod
	data[2] = mom1_obs	
	data[3] = mom1_mod	
	data[4] = mom2_obs		
	data[5] = mom2_mod							
	
	hdu = fits.PrimaryHDU(data)	
	hdu.header['NAME0'] = f'observed mom0 (flux*km/s)'
	hdu.header['NAME1'] = f'model mom0 (flux*km/s)'			
	hdu.header['NAME2'] = 'observed mom1 (km/s)'
	hdu.header['NAME3'] = 'model mom1 (km/s)'
	hdu.header['NAME4'] = 'observed mom2 (km/s)'
	hdu.header['NAME5'] = f'model mom2 (km/s)'
	hdu.header['pa'] 	= pa
	hdu.header['inc'] 	= inc
	hdu.header['xc'] 	= x_center
	hdu.header['yc'] 	= y_center
	hdu.header['v_sys'] = v_sys
	keys = ['CRPIX1','CRPIX2','CRVAL1','CRVAL2',
	'CDELT1','CDELT2','CD1_1','CD1_2','CD2_1','CD2_2',
	'BMAJ','BMIN','BPA','CTYPE1','CTYPE2','CUNIT1',
	'CUNIT2']
	for k in keys:
		if k in hdr_ori: hdu.header[k] = hdr_ori[k]		
	hdu.writeto("%smodels/%s.%s.moments.fits.gz"%(out,galaxy,vmode),overwrite=True)
			
	hdu1 = fits.PrimaryHDU(cube_mod*msk)
	hdu1.header = hdr_ori
	hdu1.writeto("%smodels/%s.%s.cube.fits.gz"%(out,galaxy,vmode),overwrite=True)
	
	
	residual=(cube_obs-cube_mod)*msk
	hdu1 = fits.PrimaryHDU(residual)
	hdu1.header['NAME0'] = 'Residual cube'
	hdu1.header = hdr_ori
	hdu1.writeto("%smodels/%s.%s.res.cube.fits.gz"%(out,galaxy,vmode),overwrite=True)
	
	if baselcube is not None:
		hdu1 = fits.PrimaryHDU(baselcube)
		hdu1.header['NAME0'] = 'Baseline cube'
		hdu1.header = hdr_ori
		hdu1.writeto("%smodels/%s.%s.baseline.cube.fits.gz"%(out,galaxy,vmode),overwrite=True)
	
	
def save_rmomments(galaxy,vmode,momms_obs,hdr_ori,out):
	mom0,mom1,mom2=momms_obs
	mom0,mom1,mom2=zero2nan(mom0),zero2nan(mom1),zero2nan(mom2)

	try:
		spec_axis_units=hdr_ori['CUNIT3']
		spec_u=spec_axis_units.lower()
	except(KeyError):
		spec_u='AA'
	
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
			

