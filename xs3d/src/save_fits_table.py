import numpy as np
from astropy.io import fits
from .phi_bar_sky import error_pa_bar_sky
from .pixel_params import eps_2_inc,e_eps2e_inc

def save_table(galaxy,vmode,R,Disp,Vrot,Vrad,Vtan,PA,EPS,XC,YC,VSYS,THETA,PA_BAR_MAJOR,PA_BAR_MINOR,errors_fit,bic_aic, e_ISM, out):
	n = len(Vrot)
	e_PA,e_EPS,e_XC,e_YC,e_VSYS,e_THETA  = errors_fit[1]
	e_Vrot,e_Vrad,e_Vtan,e_Disp  = errors_fit[0]
	INC, e_INC = eps_2_inc(EPS)*180/np.pi,e_eps2e_inc(EPS,e_EPS)*180/np.pi
	N_free, N_nvarys, N_data, bic, aic, redchi = bic_aic

	PA*=np.ones_like(R)
	EPS*=np.ones_like(R)
	XC*=np.ones_like(R)
	YC*=np.ones_like(R)
	VSYS*=np.ones_like(R)
	THETA*=np.ones_like(R)
	e_PA*=np.ones_like(R)
	e_EPS*=np.ones_like(R)
	e_XC*=np.ones_like(R)
	e_YC*=np.ones_like(R)
	e_VSYS*=np.ones_like(R)
	e_THETA*=np.ones_like(R)	
	
	INC, e_INC = eps_2_inc(EPS)*180/np.pi,e_eps2e_inc(EPS,e_EPS)*180/np.pi
	print('PA:',PA)	
	print('ePA:',e_PA)		
	print('INC:',INC)		
	print('eINC:',e_INC)		
	print('XC:',XC)		
	print('eXC:',e_XC)				
	print('YC:',YC)		
	print('eYC:',e_YC)
	print('VSYS:',VSYS)		
	print('eVSYS:',e_VSYS)						
	print(Vrot)
	print(e_Vrot)	
	print(Disp)
	print(e_Disp)
						
	col1=fits.Column(name='R', unit = 'arcsec', array=R, format = 'E')    
	col2=fits.Column(name='Sigma', unit = 'km/s', array=Disp, format = 'D') 
	col3=fits.Column(name='e_Sigma', unit = 'km/s', array=e_Disp, format = 'D')    	   
	col4=fits.Column(name='Vt', unit = 'km/s', array=Vrot, format = 'D')    	
	col5=fits.Column(name='e_Vt', unit = 'km/s', array=e_Vrot, format = 'D')    	
	col6=fits.Column(name='PA', unit = 'deg', array=PA, format = 'D') 
	col7=fits.Column(name='e_PA', unit = 'deg', array=e_PA, format = 'D') 	
	col8=fits.Column(name='EPS', array=EPS, format = 'D') 
	col9=fits.Column(name='e_EPS', array=e_EPS, format = 'D') 	
	col10=fits.Column(name='INC', unit =  'deg', array=INC, format = 'D') 
	col11=fits.Column(name='e_INC', unit =  'deg', array=e_INC, format = 'D') 	
	col12=fits.Column(name='X0', unit =  'pixel', array=XC, format = 'D') 
	col13=fits.Column(name='e_X0', unit =  'pixel', array=e_XC, format = 'D') 	
	col14=fits.Column(name='Y0', unit =  'pixel', array=YC, format = 'D') 	
	col15=fits.Column(name='e_Y0', unit =  'pixel', array=e_YC, format = 'D') 		
	col16=fits.Column(name='VSYS', unit =  'km/s', array=VSYS, format = 'D') 		
	col17=fits.Column(name='e_VSYS', unit =  'km/s', array=e_VSYS, format = 'D') 			
	
	
	if vmode == 'circular':		
		coldefs=fits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17])
		hdu=fits.BinTableHDU.from_columns(coldefs)		
	if vmode in ["radial",'vertical','bisymmetric']:
		name='V2r' if vmode == 'bisymmetric' else 'Vr'
		col18=fits.Column(name=name, unit = 'km/s', array=Vrad, format = 'D')    		
		col19=fits.Column(name=f'e_{name}', unit = 'km/s', array=e_Vrad, format = 'D')
		if vmode !=  'bisymmetric':   		
			coldefs=fits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17,col18,col19])
			hdu=fits.BinTableHDU.from_columns(coldefs)	
		else:
			col20=fits.Column(name='V2t', unit = 'km/s', array=Vrad, format = 'D')		
			col21=fits.Column(name='e_V2t', unit = 'km/s', array=e_Vrad, format = 'D')					
			coldefs=fits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17,col18,col19,col20,col21])
			hdu=fits.BinTableHDU.from_columns(coldefs)	
					
	hdu.writeto("%smodels/%s.%s.table.fits.gz"%(out,galaxy,vmode),overwrite=True)


