import numpy as np
from astropy.io import fits
from .phi_bar_sky import error_pa_bar_sky
from .pixel_params import eps_2_inc,e_eps2e_inc,inc_2_eps

def save_model_h(galaxy,vmode,const,best,best_vels,result,m_hrm,out):

	R=best['radius']
	nrings=len(R)	
	[v_sys,inc,pa,x_center,y_center,phi_bar,rmax]=const['v_sys'],const['inc'],const['pa'],const['x_center'],const['y_center'],const['phi_bar'],const['rmax']
	eps = inc_2_eps(inc)

	scalar_fields = ["v_disp"]
	vels = {k:best[k] for k in scalar_fields}
	v_disp = vels['v_disp']		
	
	Sk=[]
	Ck=[]
	for m in range(m_hrm):
		k = str(int(m+1))
		c_k = best_vels[f'c_m{k}']
		s_k = best_vels[f's_m{k}']
		Ck.append(c_k)		
		Sk.append(s_k)				
			
	nx, ny = len(R), 4*m_hrm + 1 + 2
	data = np.zeros((ny,nx))
	data[0][:] = R
	data[1][:] = v_disp
	data[2][:] = np.zeros_like(v_disp)
	for k in range(m_hrm):
		data[k+3][:] = Ck[k]
		data[m_hrm+3+k][:] = Sk[k]
		data[2*m_hrm+3+k][:] = np.zeros_like(v_disp)
		data[3*m_hrm+3+k][:] = np.zeros_like(v_disp)



	hdu = fits.PrimaryHDU(data)
	hdu.header['NAME0'] = 'Deprojected distance (arcsec)'
	hdu.header['NAME1'] = 'Intrinsinc dispersion (km/s)'
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


	chi2=result.chisqr
	hdu.header['chi2'] = chi2
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


	hdu.writeto("%smodels/%s.%s.1D_model.fits.gz"%(out,galaxy,vmode),overwrite=True)




