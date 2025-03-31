import numpy as np
from astropy.io import fits
from .phi_bar_sky import error_pa_bar_sky
from .pixel_params import eps_2_inc,e_eps2e_inc

def save_model(galaxy,vmode,R,Disp,Vrot,Vrad,Vtan,PA,EPS,XC,YC,VSYS,THETA,PA_BAR_MAJOR,PA_BAR_MINOR,errors_fit,bic_aic, e_ISM, out):
	n = len(Vrot)
	e_PA,e_EPS,e_XC,e_YC,e_Vsys,e_theta  = errors_fit[1]
	e_Vrot,e_Vrad,e_Vtan,e_Disp  = errors_fit[0]
	INC, e_INC = eps_2_inc(EPS)*180/np.pi,e_eps2e_inc(EPS,e_EPS)*180/np.pi
	N_free, N_nvarys, N_data, bic, aic, redchi = bic_aic

        
	if vmode == "circular":
			data = np.zeros((5,n))
			data[0][:] = R
			data[1][:] = Disp
			data[2][:] = Vrot
			data[3][:] = e_Disp
			data[4][:] = e_Vrot
			
	if vmode == "ff":
			data = np.zeros((5,n))
			data[0][:] = R
			data[1][:] = Disp
			data[2][:] = Vrot
			data[3][:] = e_Disp
			data[4][:] = e_Vrot
			
	if vmode == "radial" or vmode == 'vertical':
			data = np.zeros((7,n))
			data[0][:] = R
			data[1][:] = Disp
			data[2][:] = Vrot
			data[3][:] = Vrad
			data[4][:] = e_Disp
			data[5][:] = e_Vrot
			data[6][:] = e_Vrad

	if vmode == "bisymmetric":
			data = np.zeros((9,n))
			data[0][:] = R
			data[1][:] = Disp
			data[2][:] = Vrot
			data[3][:] = Vrad
			data[4][:] = Vtan
			data[5][:] = e_Disp
			data[6][:] = e_Vrot
			data[7][:] = e_Vrad
			data[8][:] = e_Vtan



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

	hdu.header['redchisq'] = redchi
	hdu.header['Nfree'] = N_free
	hdu.header['Nvarys'] = N_nvarys
	hdu.header['Ndata'] = N_data
	hdu.header['BIC'] = bic
	hdu.header['AIC'] = aic
	hdu.header['PA'] = PA
	hdu.header['e_PA'] = e_PA
	hdu.header['EPS'] = EPS
	hdu.header['e_EPS'] = e_EPS
	hdu.header['INC'] = INC
	hdu.header['e_INC'] = e_INC
	hdu.header['VSYS'] = VSYS
	hdu.header['e_VSYS'] = e_Vsys
	hdu.header['XC'] = XC
	hdu.header['e_XC'] = e_XC
	hdu.header['YC'] = YC
	hdu.header['e_YC'] = e_YC

	if vmode == "bisymmetric":
			hdu.header['HIERARCH PHI_BAR'] = THETA*180/np.pi
			hdu.header['HIERARCH e_PHI_BAR'] = e_theta*180/np.pi
			hdu.header['HIERARCH PA_BAR_MAJOR'] = PA_BAR_MAJOR
			hdu.header['HIERARCH e_PA_BAR_MAJOR'] = error_pa_bar_sky(PA,EPS,THETA,e_PA,e_EPS,e_theta)
			hdu.header['HIERARCH PA_BAR_MINOR'] = PA_BAR_MINOR
			hdu.header['HIERARCH e_PA_BAR_MINOR'] = error_pa_bar_sky(PA,EPS,THETA-np.pi/2,e_PA,e_EPS,e_theta)
		
		hdu.writeto("%smodels/%s.%s.1D_model.fits.gz"%(out,galaxy,vmode),overwrite=True)
