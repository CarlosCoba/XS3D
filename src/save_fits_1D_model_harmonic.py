import numpy as np
from astropy.io import fits
from src.pixel_params import eps_2_inc


def save_model_h(galaxy,vmode,R,Disp,e_Disp, Ck,Sk,e_Ck,e_Sk,PA,EPS,XC,YC,VSYS,m_hrm,errors_fit,bic_aic,e_ISM,out):

	N_free, N_nvarys, N_data, bic, aic, redchi = bic_aic
	nx, ny = len(R), 4*m_hrm + 1 + 2
	n = (Ck)

	data = np.zeros((ny,nx))
	data[0][:] = R
	data[1][:] = Disp
	data[2][:] = e_Disp	
	for k in range(m_hrm):
		data[k+3][:] = Ck[k]
		data[m_hrm+3+k][:] = Sk[k]
		data[2*m_hrm+3+k][:] = e_Ck[k]
		data[3*m_hrm+3+k][:] = e_Sk[k]



	e_PA,e_EPS,e_XC,e_YC,e_Vsys,e_pa_bar  = errors_fit[1]
	INC, e_INC = eps_2_inc(EPS)*180/np.pi, eps_2_inc(e_EPS)*180/np.pi

	hdu = fits.PrimaryHDU(data)

	hdu.header['NAME0'] = 'deprojected distance (arcsec)'
	hdu.header['NAME1'] = 'intrinsinc dispersion (km/s)'
	hdu.header['NAME2'] = 'error intrinsinc dispersion (arcsec)'
	
	for k in range(1,m_hrm+1):
			kk=k+2
			hdu.header['NAME%s'%kk] = 'C%s deprojected velocity (km/s)'%k
	for k in range(1,m_hrm+1):
			kk=k+2
			hdu.header['NAME%s'%(kk+m_hrm)] = 'S%s deprojected velocity (km/s)'%k
	for k in range(1,m_hrm+1):
			kk=k+2
			hdu.header['NAME%s'%(kk+2*m_hrm)] = 'error C%s (km/s)'%k
	for k in range(1,m_hrm+1):
			kk=k+2
			hdu.header['NAME%s'%(kk+3*m_hrm)] = 'error S%s (km/s)'%k


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
	hdu.header['XC'] = XC
	hdu.header['e_XC'] = e_XC
	hdu.header['YC'] = YC
	hdu.header['e_YC'] = e_YC
	hdu.header['C0'] = VSYS
	hdu.header['e_C0'] = e_Vsys


	hdu.writeto("%smodels/%s.%s.1D_model.fits.gz"%(out,galaxy,vmode),overwrite=True)




