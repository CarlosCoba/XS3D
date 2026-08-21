import numpy as np
from astropy.io import fits
import matplotlib.pylab as plt

from .constants import __c__, __FWHM_2_sigma__
from .psf_lsf import PsF_LsF
from .read_hdr import Header_info

from .cloud_tilted_rings import TiltedRingModel, CubeConfig, Ring
from .cloud_fit_engine import (
	build_params, fit_rings, make_weight_map,
	residual_cube, rotation_curve,
	set_bounds, _print_params_summary,params_to_rings
)



from .params import Set_params
from .extract_prms import extractp

def arctan(r,rt,vflat):
	return (2/np.pi)*vflat*np.arctan(r/rt)

def mock(vsys_g, pa_g, inc_g, vary_pa_g, vary_inc_g,
			vary_XC, vary_YC, vary_vsys_g,
			delta, rstart, rfinal, ring_space, vmode, config):

	config_general	= config['general']
	config_clouds	= config['clouds']
	z_scale			= config_clouds.getfloat('z_scale',0.1)
	lagging			= config_clouds.getboolean('lagging',False)
	z_profile		= config_clouds.get('z_profile','sech2') 	 
		
	fwhm_inst_kms	= config_general.getfloat('fwhm_kms',None)
	sigma_inst_kms	= fwhm_inst_kms*__FWHM_2_sigma__
			
	
	vsys_g = eval(vsys_g)
	pa_g, inc_g = eval(pa_g), eval(inc_g)
	rfinal	= eval(rfinal)
	delta	= ring_space/2
							
	galaxy	= 'mock_califa3'
	pixel	= 1
	wwidth	= 1000 # km/s
	crval3	= vsys_g - 500 #  km/s
	cdelt3	= fwhm_inst_kms/2		  # step size km/s
	crpix3	= 1
	cunit3	= 'km/s'
	ctype3	= 'velocity'
	psf		= 2.5
	vel_axis = np.arange(crval3,crval3+wwidth,cdelt3)	
	nchannels = len(vel_axis)
	
	rmax 	= rfinal
	ny		= int( 2*(rmax+3*psf)/pixel)
	nx		= int( 2*(rmax+3*psf)/pixel) 
	
	xc_g = nx//2
	yc_g = ny//2	
		
	r 		= np.arange(rstart,rfinal,delta)
	vflat	= 200
	rt		= 6
	
	vrot_tab = arctan(r,rt,vflat) 
	disp_tab = 35*np.ones_like(vrot_tab)
	vrad_tab = vrot_tab*0
	vtan_tab = vrot_tab*0
	
	vels = [disp_tab,vrot_tab,vrad_tab,vtan_tab]
	

	vsys0	= vsys_g
	inc0	= 30
	pa0		= pa_g
	x0 		= xc_g
	y0 		= yc_g	
	z_scale	= z_scale
	z_profile	= z_profile
	lagging 	= lagging
	nclouds		= 5
	nsubclouds	= 20
	vmode		= 'circular'


	rwidth		= delta		
	vary=np.array( [vary_pa_g,vary_inc_g,vary_XC,vary_YC,vary_vsys_g,True] )


		
	# cube header
	data = np.zeros((nchannels,ny,nx))
	hdu = fits.PrimaryHDU(data)
	hdu.header['CDELT3']=cdelt3
	hdu.header['CRVAL3']=crval3	
	hdu.header['CRPIX3']=crpix3	
	hdu.header['CUNIT3']=cunit3
	hdu.header['CTYPE3']=ctype3

	hdu.header['NAXIS1']=nx
	hdu.header['NAXIS2']=ny			
	hdu.header['NAXIS3']=nchannels
			
	hdu.header['BMAJ']	= psf/3600
	hdu.header['BMIN']	= psf/3600
	hdu.header['BPA']	= 0

	hdu.header['CDELT1']=-pixel/3600
	hdu.header['CDELT2']= pixel/3600	
	
	hdu.header['CD1_1']= -pixel/3600		
	hdu.header['CD1_2']= 0			
	hdu.header['CD2_1']= 0		
	hdu.header['CD2_2']= pixel/3600		
			
	hdr_ori = hdu.header
	# Read header information
	hdr_info=Header_info(hdr_ori, config)

	hdr_info.object=galaxy	
		
	psf_lsf=PsF_LsF(hdr_info, config)
		
		
	hdr	=  hdr_info
	guess_common = dict(
				v_sys		= vsys0,
				inc			= inc0,
				pa			= pa0,
				x_center	= x0,
				y_center	= y0,
				z_scale		= z_scale,
				z_profile	= z_profile,
				vz_gradient	= lagging,
				n_clouds	= nclouds,
				n_subclouds	= nsubclouds,
				velocity_model = vmode,
				phi_bar		   = 45
	)


	R={'R_pos':r, 'R_nc': np.ones_like(r)}
			
	cnf_prms=Set_params(vmode, psf_lsf, R, ring_space, rwidth, vary, hdr,guess_common)
	guess_rings = cnf_prms.circular(vels)
	spec = cnf_prms.prms(vmode)
	
	params  = build_params(guess_rings, spec)				
	best_rings = params_to_rings(params, guess_rings)
	
	#print(params)
	#print(best_rings)		
		
	# ============================================================
	best_model	= TiltedRingModel(hdr, psf_lsf, seed=1234)
	mod_cube 	= best_model.build(best_rings, verbose=False)				

	mom0_obs = np.sum(mod_cube, axis = 0)
	mom0_obs[mom0_obs<0]=0
	
	# get the final mask
	rmax_px = np.max(r)/pixel
	W_cur= make_weight_map(mom0_obs, psf_lsf, best_rings, alpha=(0,0), r_max_px=rmax_px, n_sigma_z=5)
	msk = (W_cur !=0).astype(float)
	mod_cube*=msk
			
	hdu.data = mod_cube
	path = '/home/carlos/simcube'
	hdu.writeto(f'{path}/{galaxy}.cube.fits',overwrite=True)
	

	mom0_tmp = mod_cube.sum(axis=0)
	plt.imshow(mom0_tmp, origin = 'lower');plt.show()				
	quit()
	return None





