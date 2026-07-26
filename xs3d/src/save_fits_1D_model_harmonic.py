import numpy as np
from astropy.io import fits
import itertools

from .phi_bar_sky import error_pa_bar_sky
from .pixel_params import eps_2_inc,e_eps2e_inc,inc_2_eps

def save_model_h(galaxy,vmode,const,best,best_vels,result,m_hrm,out):

	R=best['radius']
	nrings=len(R)	
	[v_sys,inc,pa,x_center,y_center,phi_bar,rmax]=const['v_sys'],const['inc'],const['pa_NE'],const['x_center'],const['y_center'],const['phi_bar'],const['rmax']
	eps = inc_2_eps(inc)
		
	prm_vel_list = [[f'c_m{k}' for k in range(1,m_hrm+1)], [f's_m{k}' for k in range(1,m_hrm+1)], ['v_disp']]
	prm_vel = list(itertools.chain.from_iterable(prm_vel_list))


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

		
	errv	={l:[] for l in prm_vel}			
	for k,pvel in enumerate(prm_vel):
		tmp = []
		for i in range(nrings):
			try:
				value = result.params[pvel+f'_r{i}'].stderr
				tmp.append(value if value is not None else 0)					
			except(KeyError):
				tmp.append(0)
		tmp = np.array(tmp)
		if np.any([tmp==0]):
			tmp = np.ones_like(tmp)*tmp[0]
		errv[pvel]=tmp

	vhrm	={l:[] for l in prm_vel}			
	for k,pvel in enumerate(prm_vel):
		tmp = []
		for i in range(nrings):
			try:
				value = result.params[pvel+f'_r{i}'].value
				tmp.append(value if value is not None else 0)					
			except(KeyError):
				tmp.append(0)
		tmp = np.array(tmp)
		vhrm[pvel]=tmp		

	

	print(vhrm)		
	print(errv)	
			
	nx, ny = len(R), 4*m_hrm + 1 + 2
	data = np.zeros((ny,nx))
	data[0][:] = R
	data[1][:] = vhrm['v_disp']
	data[2*m_hrm+2][:] = errv['v_disp']
	edisp_i=2*m_hrm+2
	
	for k in range(m_hrm):
		j=k+1
		data[2*j][:] 		= vhrm[f'c_m{j}']
		data[2*j+1][:]		= vhrm[f's_m{j}']
		data[edisp_i+2*k+1][:]	= 	errv[f'c_m{j}']
		data[edisp_i+2*k+2][:]	= 	errv[f's_m{j}']	
		
	hdu = fits.PrimaryHDU(data)
	hdu.header['NAME0'] = 'Deprojected distance (arcsec)'
	hdu.header['NAME1'] = 'Intrinsinc dispersion (km/s)'
	hdu.header['NAME%s'%edisp_i] = 'error dispersion (km/s)'
		
	for k in range(m_hrm):
		j=k+1		
		hdu.header['NAME%s'%(2*j)] = 'C%s (km/s)'%(j)		
		hdu.header['NAME%s'%(2*j+1)] = 'S%s (km/s)'%(j)		
		hdu.header['NAME%s'%(edisp_i+2*k+1)] = 'error C%s (km/s)'%(j)		
		hdu.header['NAME%s'%(edisp_i+2*k+2)] = 'error S%s (km/s)'%(j)		
							

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


	hdu.writeto("%smodels/%s.%s.1D_model.fits.gz"%(out,galaxy,vmode),overwrite=True)




