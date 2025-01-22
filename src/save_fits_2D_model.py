import numpy as np
from astropy.io import fits
from src.kin_components import AZIMUTHAL_ANGLE


def save_vlos_model(galaxy,vmode,vel_map,evel_map,vlos_2D_model, kin_2D_models,PA,INC,XC,YC,VSYS,m_hrm=None, theta = False, phi_bar_major = False, phi_bar_minor = False, out = False):


	if "hrm" in vmode:
		C_k, S_k, R_n =  kin_2D_models
		Vcirc_2D = C_k[0]
	else:
		Vcirc_2D, Vrad_2D, Vtan_2D, R_n =  kin_2D_models

	vel_map[vel_map == 0] = np.nan

	Ndata = len( vlos_2D_model[np.isfinite(vlos_2D_model)] )
	rms2 = np.nansum( (vel_map - vlos_2D_model)**2 ) / Ndata
	rms = np.sqrt(rms2)

	if True:

		if "hrm" not in vmode:
			VSYS_str = "VSYS"
			eVSYS_str = "eVSYS"

		else:
			VSYS_str = "C0"
			eVSYS_str = "eC0"


		# The best 2D model
		data = vlos_2D_model
		hdu = fits.PrimaryHDU(data)
		#hdu.data = vlos_2D_model

	
		hdu.header['NAME0'] = 'Two-dimensional vlos model'
    
		hdu.header['PA'] = PA
		hdu.header['INC'] = INC
		hdu.header['%s'%VSYS_str] = VSYS
		hdu.header['XC'] = XC
		hdu.header['YC'] = YC
		hdu.header['rms'] = rms
		if vmode == "bisymmetric":
			hdu.header['PHI_BAR'] = theta

		
		hdu.writeto("%smodels/%s.%s.2D_vlos_model.fits.gz"%(out,galaxy,vmode),overwrite=True)


		
		# Now the residual map
		data = vel_map - vlos_2D_model
		hdu = fits.PrimaryHDU(data)
		#hdu.data = vel_map- vlos_2D_model

	
		hdu.header['NAME0'] = 'residual map, data-model'
    
		hdu.header['PA'] = PA
		hdu.header['INC'] = INC
		hdu.header['%s'%VSYS_str] = VSYS
		hdu.header['XC'] = XC
		hdu.header['YC'] = YC
		hdu.header['rms'] = rms		
		hdu.writeto("%smodels/%s.%s.residual.fits.gz"%(out,galaxy,vmode),overwrite=True)


		
		# Now the chisquare map

		sigma = evel_map
		chisq = ( vel_map- vlos_2D_model )/sigma
		chisq_2 = chisq**2

		data = chisq_2
		hdu = fits.PrimaryHDU(data)
		#hdu.data = chisq_2

	
		hdu.header['NAME0'] = 'Chisquare map, (data-model)**2/sigma**2'
    
		hdu.header['PA'] = PA
		hdu.header['INC'] = INC
		hdu.header['%s'%VSYS_str] = VSYS
		hdu.header['XC'] = XC
		hdu.header['YC'] = YC
		hdu.header['rms'] = rms		
		hdu.writeto("%smodels/%s.%s.chisq.fits.gz"%(out,galaxy,vmode),overwrite=True)




		#
		# Save deprojected radius.
		#

		data = R_n
		hdu = fits.PrimaryHDU(data)
		#hdu.data = R_n

	
		hdu.header['NAME0'] = 'Two-dimensional deprojected radius'
		hdu.header['PA'] = PA
		hdu.header['INC'] = INC
		hdu.header['%s'%VSYS_str] = VSYS
		hdu.header['XC'] = XC
		hdu.header['YC'] = YC
		
		hdu.writeto("%smodels/%s.%s.2D_R.fits.gz"%(out,galaxy,vmode),overwrite=True)







		#
		# Save 2D kinematic models. Circular model is always saved.
		#
		
		# Circular velocity 2D model
		data = Vcirc_2D
		hdu = fits.PrimaryHDU(data)
		#hdu.data = Vcirc_2D


		if "hrm" not in vmode:	
			hdu.header['NAME0'] = 'Two-dimensional circular model'
		hdu.header['PA'] = PA
		hdu.header['INC'] = INC
		hdu.header['%s'%VSYS_str] = VSYS
		hdu.header['XC'] = XC
		hdu.header['YC'] = YC

		if "hrm" not in vmode:			
			hdu.writeto("%smodels/%s.%s.2D_circ_model.fits.gz"%(out,galaxy,vmode),overwrite=True)

		if vmode == "radial":
			# Radial velocity 2D model
			data = Vrad_2D
			hdu = fits.PrimaryHDU(data)
			#hdu.data = Vrad_2D

		
			hdu.header['NAME0'] = 'Two-dimensional radial model'
			hdu.header['PA'] = PA
			hdu.header['INC'] = INC
			hdu.header['%s'%VSYS_str] = VSYS
			hdu.header['XC'] = XC
			hdu.header['YC'] = YC
			
			hdu.writeto("%smodels/%s.%s.2D_rad_model.fits.gz"%(out,galaxy,vmode),overwrite=True)


		if vmode == "bisymmetric":
			# Radial velocity 2D model
			data = Vrad_2D
			hdu = fits.PrimaryHDU(data)
			#hdu.data = Vrad_2D

		
			hdu.header['NAME0'] = 'Two-dimensional V2r model'
			hdu.header['PA'] = PA
			hdu.header['INC'] = INC
			hdu.header['%s'%VSYS_str] = VSYS
			hdu.header['XC'] = XC
			hdu.header['YC'] = YC
			hdu.header['PHI_BAR'] = theta
			hdu.header['PA_BAR_MAJOR'] = phi_bar_major
			hdu.header['PA_BAR_MINOR'] = phi_bar_minor

			
			hdu.writeto("%smodels/%s.%s.2D_rad_model.fits.gz"%(out,galaxy,vmode),overwrite=True)

			# Tangential velocity 2D model
			data = Vtan_2D
			hdu = fits.PrimaryHDU(data)
			#hdu.data = Vtan_2D

		
			hdu.header['NAME0'] = 'Two-dimensional V2t model'
			hdu.header['PA'] = PA
			hdu.header['INC'] = INC
			hdu.header['%s'%VSYS_str] = VSYS
			hdu.header['XC'] = XC
			hdu.header['YC'] = YC
			hdu.header['PHI_BAR'] = theta
			hdu.header['PA_BAR_MAJOR'] = phi_bar_major
			hdu.header['PA_BAR_MINOR'] = phi_bar_minor
			
			hdu.writeto("%smodels/%s.%s.2D_tan_model.fits.gz"%(out,galaxy,vmode),overwrite=True)


		if "hrm" in vmode:

			for k in range(1,m_hrm+1):

				data = C_k[k-1]
				hdu = fits.PrimaryHDU(data)
				hdu.header['NAME0'] = 'Two-dimensional C%s model'%k
				hdu.header['PA'] = PA
				hdu.header['INC'] = INC
				hdu.header['%s'%VSYS_str] = VSYS
				hdu.header['XC'] = XC
				hdu.header['YC'] = YC			
				hdu.writeto("%smodels/%s.%s.2D_C%s_model.fits.gz"%(out,galaxy,vmode,k),overwrite=True)


				data = S_k[k-1]
				hdu = fits.PrimaryHDU(data)
				hdu.header['NAME0'] = 'Two-dimensional S%s model'%k
				hdu.header['PA'] = PA
				hdu.header['INC'] = INC
				hdu.header['%s'%VSYS_str] = VSYS
				hdu.header['XC'] = XC
				hdu.header['YC'] = YC			
				hdu.writeto("%smodels/%s.%s.2D_S%s_model.fits.gz"%(out,galaxy,vmode, k),overwrite=True)



		data = AZIMUTHAL_ANGLE(R_n.shape, PA, INC, XC, YC)
		data = data*vlos_2D_model/vlos_2D_model
		hdu = fits.PrimaryHDU(data)
		hdu.header['NAME0'] = 'Two-dimensional azimuthal angle'
		hdu.header['UNITS'] = 'Radians'
		hdu.header['PA'] = PA
		hdu.header['INC'] = INC
		hdu.header['%s'%VSYS_str] = VSYS
		hdu.header['XC'] = XC
		hdu.header['YC'] = YC
		hdu.writeto("%smodels/%s.%s.2D_theta.fits.gz"%(out,galaxy,vmode),overwrite=True)


	else: pass
