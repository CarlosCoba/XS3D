import numpy as np
from astropy.io import fits
from src.pixel_params import Rings
from src.constants import __c__
from src.read_hdr import Header_info
from src.psf_lsf import PsF_LsF

def slit(xy,pa,eps,x0,y0,width=5,pixel=1):
	
	x,y=xy
	# y = m(x-x0)+y0 --> y -mx +(mx0-y0)
	alpha=(pa+np.pi/2)
	m=np.tan(alpha)	 
	A,B,C=-m,1,m*x0-y0
	d = abs(A*x+B*y+C)/np.sqrt(A**2+B**2)
	darc=d*pixel
	msk = darc < width/2.
	
	return msk


def pv_array(datacube,hdr_cube,momms_mdls,vt,r,pa,eps,x0,y0,vsys,pixel,rms,config):
	mom0_mdl,mom1_mdl,mom2_mdl_kms,mom2_mdl_A,cube_mdl,velmap_intr,sigmap_intr,twoDmodels= momms_mdls
	[nz,ny,nx]=datacube.shape
	extimg=np.dot([-nx/2.,nx/2.,-ny/2.,ny/2.],pixel)
	

	psf_lsf=PsF_LsF(hdr_cube,config)
	bmaj_arc=psf_lsf.bmaj 
			
	wave_kms=Header_info(hdr_cube,config).wave_kms
	dv=wave_kms
	
	if wave_kms[1]-wave_kms[0] < 0: # radio velocities 
		datacube = datacube[::-1]	
		cube_mdl = cube_mdl[::-1]			
		wave_kms= wave_kms[::-1]

	#######################################################################
	# The following is to avoid plotting all channels from the cube.
	# Otherwise is time consuming and not all channels contain signal.
	#######################################################################
	meanflux_chan = np.array([np.mean(datacube[k], where=( (datacube[k]!=0) & (np.isfinite(datacube[k]))) )/rms for k in range(nz)])
	chan_sig=(meanflux_chan>0.01) & np.isfinite(meanflux_chan)
	vchan=wave_kms[chan_sig]
	vmin, vmax=np.min(vchan), np.max(vchan)			
	chan_msk=(wave_kms>=vmin) & (wave_kms<=vmax)
	datacube_tmp = np.copy(datacube)
	cube_mdl_tmp = np.copy(cube_mdl)
	datacube_tmp = datacube_tmp
	wave_kms_tmp = wave_kms[chan_msk]
	cube_mdl_tmp = cube_mdl_tmp[:,None][chan_msk[:,None]]
	datacube_tmp = datacube_tmp[:,None][chan_msk[:,None]]
	[nz,ny,nx]=datacube_tmp.shape
	dv=wave_kms[chan_msk]
	#######################################################################
			
	#dv+=vsys
	vmin,vmax=np.min(dv),np.max(dv)
	y,x=np.indices((ny,nx))


	msk_r = np.isfinite(mom0_mdl/mom0_mdl).astype(float)
	msk_r[msk_r==0]=np.nan
	pa_maj = pa % 360
	pa_min = (pa+90) % 360
	

	pa_maj=pa_maj*np.pi/180
	pa_min=pa_min*np.pi/180	
	pa=pa*np.pi/180
	
	# Upper quadrant if:
	if np.cos(pa_maj)>0:
		s=+1
	else:
		s=-1

	width=2*bmaj_arc if bmaj_arc is not None else 5 # arcsec
	# if the FOV is too small draw 5pixels width
	if nx*pixel//width <=2:
		width=5*pixel

	msk_slit_maj=slit((x,y),pa_maj,eps,x0,y0,width=width,pixel=pixel)
	msk_slit_min=slit((x,y),pa_min,eps,x0,y0,width=width,pixel=pixel)
	


	def signquad(pa0):
		rpix=Rings((x,y),pa_maj,eps,x0,y0)	
		cos_theta = (- (x-x0)*np.sin(pa0) + (y-y0)*np.cos(pa0))/rpix
		sign=np.sign(cos_theta)*s
		return sign*rpix	
	
	rarc=signquad(pa_maj)*msk_r*pixel		
	rpix=signquad(pa_maj)*msk_r
		
	#set pixel size of the x axis on pvd diagram
	pixel_pvd_arc=2*(pixel)
	
	r_norm = rarc // pixel_pvd_arc
	r_unique=np.unique(r_norm)
	r_norm_int = r_norm#.astype(int)
	
	rmin,rmax=int(np.nanmin(r_norm_int)),int(np.nanmax(r_norm_int))
	dr=(rmax-rmin)	
		
	pv_array_maj=np.zeros((len(dv),dr))
	pv_array_min=np.zeros((len(dv),dr))
	pv_array_maj_mdl=np.zeros((len(dv),dr))
	pv_array_min_mdl=np.zeros((len(dv),dr))

	ext_arc0=np.array([rmin*pixel_pvd_arc,rmax*pixel_pvd_arc,vmin,vmax]) 	
	ext_arc1=np.array([rmin*pixel_pvd_arc,rmax*pixel_pvd_arc,vmin,vmax])
	# Change scale to arcmin if necessary 	
	if np.max(abs(extimg)) > 80:
		ext_arc0[:2]=ext_arc0[:2]/60
		ext_arc1[:2]=ext_arc1[:2]/60
	
	m_mjr=msk_slit_maj*np.ones(nz)[:,None,None]	
	m_mnr=msk_slit_min*np.ones(nz)[:,None,None]		
	for ind,rval in enumerate(np.arange(rmin,rmax,1)):
		# loop for major axis	
		msk_R=r_norm_int==rval
		# Numer of spectra to coadd
		Npix=np.sum(msk_R*msk_slit_maj)
		if Npix==0: Npix=1
		masked_cube_maj_mdl=cube_mdl_tmp*(msk_R*m_mjr)
		masked_cube_maj=datacube_tmp*(msk_R*m_mjr)
		# the array[0][0] position contains the most blueshifted point, e.g., vmin-vsys.			
		pv_array_maj_mdl[:,ind]=np.nansum(np.nansum(masked_cube_maj_mdl,axis=2),axis=1)/Npix						
		pv_array_maj[:,ind]=np.nansum(np.nansum(masked_cube_maj,axis=2),axis=1)/Npix

		# loop for minor axis
		Npix=np.sum(msk_R*msk_slit_min)
		if Npix==0: Npix=1					
		#msk_R=r_norm_int==rval # already defined above.			
		masked_cube_min_mdl=cube_mdl_tmp*(msk_R*m_mnr)
		masked_cube_min=datacube_tmp*(msk_R*m_mnr)
		# the array[0][0] position contains the most blueshifted point, e.g., vmin-vsys.
		pv_array_min_mdl[:,ind]=np.nansum(np.nansum(masked_cube_min_mdl,axis=2),axis=1)/Npix
		pv_array_min[:,ind]=np.nansum(np.nansum(masked_cube_min,axis=2),axis=1)/Npix
	
	return [pv_array_maj,pv_array_min,pv_array_maj_mdl,pv_array_min_mdl], [msk_slit_maj,msk_slit_min], [ext_arc0,ext_arc1,pixel_pvd_arc]
	
