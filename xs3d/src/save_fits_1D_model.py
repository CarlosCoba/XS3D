import numpy as np
from astropy.io import fits
from .phi_bar_sky import error_pa_bar_sky
from .pixel_params import eps_2_inc,e_eps2e_inc,inc_2_eps

def save_model(galaxy,vmode,const,best,result,out):
	R=best['radius']
	nrings=len(R)
	[v_sys,inc,pa,x_center,y_center,phi_bar,rmax]=const['v_sys'],const['inc'],const['pa'],const['x_center'],const['y_center'],const['phi_bar'],const['rmax']
	eps = inc_2_eps(inc)
	scalar_fields = ["v_rot", "v_rad", "v_2t", "v_2r", "v_disp"]
	vels = {k:best[k] for k in scalar_fields}
	vrot = vels['v_rot']

	if vmode == 'circular':
		data = np.zeros((5,nrings))
		data[0][:] = R
		data[1][:] = vels['v_disp']
		data[2][:] = vels['v_rot']
		data[3][:] = np.zeros_like(vrot)
		data[4][:] = np.zeros_like(vrot)

	if vmode == 'radial':
		data = np.zeros((7,nrings))
		data[0][:] = R
		data[1][:] = vels['v_disp']
		data[2][:] = vels['v_rot']
		data[3][:] = vels['v_rad']
		data[4][:] = np.zeros_like(vrot)
		data[5][:] = np.zeros_like(vrot)
		data[6][:] = np.zeros_like(vrot)

	if vmode == 'bisymmetric':
		data = np.zeros((9,nrings))
		data[0][:] = R
		data[1][:] = vels['v_disp']
		data[2][:] = vels['v_rot']
		data[3][:] = vels['v_2r']
		data[4][:] = vels['v_2t']
		data[5][:] = np.zeros_like(vrot)
		data[6][:] = np.zeros_like(vrot)
		data[7][:] = np.zeros_like(vrot)
		data[8][:] = np.zeros_like(vrot)

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
	hdu.header['chi2_red'] = chi2
	hdu.header['pa'] = pa
	hdu.header['e_pa'] = 0
	hdu.header['eps'] = eps
	hdu.header['e_pa'] = 0
	hdu.header['inc'] = inc
	hdu.header['e_inc'] = 0
	hdu.header['v_sys'] = v_sys
	hdu.header['e_vsys'] = 0
	hdu.header['xc'] = x_center
	hdu.header['e_xc'] = 0
	hdu.header['yc'] = y_center
	hdu.header['e_yc'] = 0

	if vmode == "bisymmetric":
		hdu.header['HIERARCH phi_bar'] = phi_bar
		hdu.header['HIERARCH e_phi_bar'] = 0

	hdu.writeto(f"{out}models/{galaxy}.{vmode}.1D_model.fits.gz",overwrite=True)
