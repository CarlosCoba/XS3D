import numpy as np
from astropy.io import fits
from .phi_bar_sky import error_pa_bar_sky
from .pixel_params import eps_2_inc,e_eps2e_inc,inc_2_eps

def save_model(galaxy,vmode,const,best,result,out):
	R = best['radius']
	nrings = len(R)	
	[v_sys,inc,pa,x_center,y_center,phi_bar,rmax]=const['v_sys'],const['inc'],const['pa'],const['x_center'],const['y_center'],const['phi_bar'],const['rmax']
	eps	= inc_2_eps(inc)
	scalar_fields = ["v_rot", "v_rad", "v_2t", "v_2r", "v_disp"]
	vels = {k:best[k] for k in scalar_fields}
	vrot = vels['v_rot']		


	prm_vel = ["v_rot", "v_rad", "v_2t", "v_2r", "v_disp"]	
	errorsv	={l:[] for l in prm_vel}			
	for k,pvel in enumerate(prm_vel):
		tmp = []
		for i in range(nrings):
			try:
				value = result.params[pvel+f'_r{i}'].stderr
				tmp.append(value if value is not None else 0)					
			except(KeyError):
				tmp.append(0)
		tmp = np.array(tmp)
		errorsv[pvel]=tmp
						
	
	scalar_fields = ['pa','inc','x_center','y_center','v_sys','phi_bar']	
	err_p = {l:0 for l in scalar_fields}
	for k,pscal in enumerate(scalar_fields):
		tmp = []
		for i in range(nrings):
			try:
				value = result.params[pscal+f'_r{i}'].stderr # is None by default
				tmp.append(value if value is not None else 0)				
			except(KeyError):
				tmp.append(0)
		tmp = np.array(tmp)
		err_p[pscal]=np.sqrt(np.sum(tmp**2))
			

	if vmode == 'circular':
		data = np.zeros((5,nrings))
		data[0][:] = R
		data[1][:] = vels['v_disp']
		data[2][:] = vels['v_rot']
		data[3][:] = errorsv['v_disp']
		data[4][:] = errorsv['v_rot']
		
	if vmode == 'radial':
		data = np.zeros((7,nrings))
		data[0][:] = R			
		data[1][:] = vels['v_disp']
		data[2][:] = vels['v_rot']
		data[3][:] = vels['v_rad']		
		data[4][:] = errorsv['v_disp']
		data[5][:] = errorsv['v_rot']	
		data[6][:] = errorsv['v_rad']	

	if vmode == 'bisymmetric':
		data = np.zeros((9,nrings))
		data[0][:] = R			
		data[1][:] = vels['v_disp']
		data[2][:] = vels['v_rot']
		data[3][:] = vels['v_2r']		
		data[4][:] = vels['v_2t']				
		data[5][:] = errorsv['v_disp']
		data[6][:] = errorsv['v_rot']		
		data[7][:] = errorsv['v_2r']	
		data[8][:] = errorsv['v_2t']	
		
	hdu = fits.PrimaryHDU(data)

	if vmode == "circular":
			hdu.header['NAME0'] = 'deprojected distance (arcsec)'
			hdu.header['NAME1'] = 'intrinsic dispersion (km/s)'
			hdu.header['NAME2'] = 'circular velocity (km/s)'
			hdu.header['NAME3'] = 'error velocity dispersion (km/s)'
			hdu.header['NAME4'] = 'error circular velocity (km/s)'
	if vmode == "ff":
			hdu.header['NAME0'] = 'deprojected distance (arcsec)'
			hdu.header['NAME1'] = 'intrinsic dispersion (km/s)'
			hdu.header['NAME2'] = 'circular velocity (km/s)'
			hdu.header['NAME3'] = 'error velocity dispersion (km/s)'
			hdu.header['NAME4'] = 'error circular velocity (km/s)'
	if vmode == "radial":
			hdu.header['NAME0'] = 'deprojected distance (arcsec)'
			hdu.header['NAME1'] = 'intrinsic dispersion (km/s)'
			hdu.header['NAME2'] = 'circular velocity (km/s)'
			hdu.header['NAME3'] = 'radial velocity (km/s)'
			hdu.header['NAME4'] = 'error velocity dispersion (km/s)'
			hdu.header['NAME5'] = 'error circular velocity (km/s)'
			hdu.header['NAME6'] = 'error radial velocity (km/s)'
	if vmode == "vertical":
			hdu.header['NAME0'] = 'deprojected distance (arcsec)'
			hdu.header['NAME1'] = 'intrinsic dispersion (km/s)'
			hdu.header['NAME2'] = 'circular velocity (km/s)'
			hdu.header['NAME3'] = 'vertical velocity (km/s)'
			hdu.header['NAME4'] = 'error velocity dispersion (km/s)'
			hdu.header['NAME5'] = 'error circular velocity (km/s)'
			hdu.header['NAME6'] = 'error vertical velocity (km/s)'
	if vmode == "bisymmetric":
			hdu.header['NAME0'] = 'deprojected distance (arcsec)'
			hdu.header['NAME1'] = 'intrinsic dispersion (km/s)'
			hdu.header['NAME2'] = 'circular velocity (km/s)'
			hdu.header['NAME3'] = 'radial velocity (km/s)'
			hdu.header['NAME4'] = 'tangencial velocity (km/s)'
			hdu.header['NAME5'] = 'error velocity dispersion (km/s)'
			hdu.header['NAME6'] = 'error circular velocity (km/s)'
			hdu.header['NAME7'] = 'error radial velocity (km/s)'
			hdu.header['NAME8'] = 'error tangencial velocity (km/s)'

	chi2=result.chisqr
	hdu.header['chi2r']	= chi2
	hdu.header['pa'] 	= pa
	hdu.header['e_pa'] 	= err_p['pa']
	hdu.header['eps'] 	= eps
	hdu.header['inc'] 	= inc
	hdu.header['e_inc'] = err_p['inc']
	hdu.header['v_sys'] = v_sys
	hdu.header['e_vsys']= err_p['v_sys']
	hdu.header['xc'] 	= x_center
	hdu.header['e_xc'] 	= err_p['x_center']
	hdu.header['yc'] 	= y_center
	hdu.header['e_yc'] 	= err_p['y_center']

	if vmode == "bisymmetric":
		hdu.header['HIERARCH phi_bar']		= phi_bar
		hdu.header['HIERARCH e_phi_bar']	= err_p['phi_bar']

	hdu.writeto(f"{out}models/{galaxy}.{vmode}.1D_model.fits.gz",overwrite=True)	
