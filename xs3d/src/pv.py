import numpy as np
from astropy.io import fits
from .pixel_params import Rings, slits
from .constants import __c__
from .read_hdr import Header_info
from .psf_lsf import PsF_LsF

from .pixel_params import eps_2_inc,e_eps2e_inc,inc_2_eps
import matplotlib.pylab as plt			
	
def pv_array(datacube,hdr_cube,momms_mods,vt,r,pa,eps,x0,y0,vsys,pixel,rms,config):
	_,_,mom0_mod,mom1_mod,mom2_mod_kms,mom2_mod_A,cube_mod,velmap_intr,sigmap_intr,twoDmodels= momms_mods
	[nz,ny,nx]=datacube.shape
	extimg=np.dot([-nx/2.,nx/2.,-ny/2.,ny/2.],pixel)
	extimg=np.dot([-x0,nx-x0,-y0,ny-y0],pixel)


	psf_lsf=PsF_LsF(hdr_cube,config)
	bmaj_arc=psf_lsf.bmaj

	wave_kms=Header_info(hdr_cube,config).wave_kms
	dv=wave_kms

	if wave_kms[1]-wave_kms[0] < 0: # radio velocities
		datacube = datacube[::-1]
		cube_mod = cube_mod[::-1]
		wave_kms= wave_kms[::-1]

	#######################################################################
	# The following is to avoid plotting all channels from the cube.
	# Otherwise it is time consuming and not all channels contain signal.
	#######################################################################
	#meanflux_chan = np.array([np.mean(datacube[k], where=( (datacube[k]!=0) & (np.isfinite(datacube[k]))) )/rms for k in range(nz)])
	#chan_sig=(meanflux_chan>0.01) & np.isfinite(meanflux_chan)

	# first, select and average those channels with fluxes above 10% the rms noise
	meanflux_chan = np.array([np.mean(datacube[k], where=( (datacube[k]>=0.1*rms) & (np.isfinite(datacube[k]))) ) for k in range(nz)])
	# select those channels with fluxes above 10% the rms noise
	chan_sig=(meanflux_chan>=0.1*rms) & np.isfinite(meanflux_chan)
	vchan=wave_kms[chan_sig]
	vmin, vmax=np.min(vchan), np.max(vchan)
	chan_msk=(wave_kms>=vmin) & (wave_kms<=vmax)
	datacube_tmp = np.copy(datacube)
	cube_mod_tmp = np.copy(cube_mod)
	datacube_tmp = datacube_tmp
	wave_kms_tmp = wave_kms[chan_msk]
	cube_mod_tmp = cube_mod_tmp[:,None][chan_msk[:,None]]
	datacube_tmp = datacube_tmp[:,None][chan_msk[:,None]]
	[nz,ny,nx]=datacube_tmp.shape
	dv=wave_kms[chan_msk]
	#######################################################################

	#dv+=vsys
	vmin,vmax=np.min(dv),np.max(dv)
	y,x=np.indices((ny,nx))


	msk_r = np.isfinite(mom0_mod/mom0_mod).astype(float)
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

	msk_slit_maj=slits((x,y),pa_maj,eps,x0,y0,width=width,pixel=pixel)
	msk_slit_min=slits((x,y),pa_min,eps,x0,y0,width=width,pixel=pixel)



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
	pv_array_maj_mod=np.zeros((len(dv),dr))
	pv_array_min_mod=np.zeros((len(dv),dr))

	ext_arc0=np.array([rmin*pixel_pvd_arc,rmax*pixel_pvd_arc,vmin,vmax])
	ext_arc1=np.array([rmin*pixel_pvd_arc,rmax*pixel_pvd_arc,vmin,vmax])
	# Change scale to arcmin if necessary
	#if np.nanmax(rarc) > 80 and np.all(extimg>120):
	#	ext_arc0[:2]=ext_arc0[:2]/60
	#	ext_arc1[:2]=ext_arc1[:2]/60

	m_mjr=msk_slit_maj*np.ones(nz)[:,None,None]
	m_mnr=msk_slit_min*np.ones(nz)[:,None,None]
	for ind,rval in enumerate(np.arange(rmin,rmax,1)):
		# loop for major axis
		msk_R=r_norm_int==rval
		# Numer of spectra to coadd
		Npix=np.sum(msk_R*msk_slit_maj)
		if Npix==0: Npix=1
		masked_cube_maj_mod=cube_mod_tmp*(msk_R*m_mjr)
		masked_cube_maj=datacube_tmp*(msk_R*m_mjr)
		# the array[0][0] position contains the most blueshifted point, e.g., vmin-vsys.
		pv_array_maj_mod[:,ind]=np.nansum(np.nansum(masked_cube_maj_mod,axis=2),axis=1)/Npix
		pv_array_maj[:,ind]=np.nansum(np.nansum(masked_cube_maj,axis=2),axis=1)/Npix

		# loop for minor axis
		Npix=np.sum(msk_R*msk_slit_min)
		if Npix==0: Npix=1
		#msk_R=r_norm_int==rval # already defined above.
		masked_cube_min_mod=cube_mod_tmp*(msk_R*m_mnr)
		masked_cube_min=datacube_tmp*(msk_R*m_mnr)
		# the array[0][0] position contains the most blueshifted point, e.g., vmin-vsys.
		pv_array_min_mod[:,ind]=np.nansum(np.nansum(masked_cube_min_mod,axis=2),axis=1)/Npix
		pv_array_min[:,ind]=np.nansum(np.nansum(masked_cube_min,axis=2),axis=1)/Npix

	return [pv_array_maj,pv_array_min,pv_array_maj_mod,pv_array_min_mod], [msk_slit_maj,msk_slit_min], [ext_arc0,ext_arc1,pixel_pvd_arc]
	
	



def pv_array2(datacube,cube_mod,mom_obs,mom_mod,cube_config,psf_lsf,rms,const):

	[nz,ny,nx]	= datacube.shape
	bmaj_arc 	= psf_lsf.bmaj
	slit_w 		= psf_lsf.slit_w
	pixel=cube_config.pix_arcs
	cfg = cube_config
	mom0_obs,_,_ =	mom_obs
	mom0_mod,_,_ =	mom_mod	

	[v_sys,inc,pa,x_center,y_center,phi_bar,Rmax]=const['v_sys'],const['inc'],const['pa'],const['x_center'],const['y_center'],const['phi_bar'],const['rmax']
	eps=inc_2_eps(inc)
	x0=x_center
	y0=y_center

	wave_kms=cube_config.wave_kms
	dv=wave_kms

	if wave_kms[1]-wave_kms[0] < 0: # radio velocities
		datacube = datacube[::-1]
		cube_mod = cube_mod[::-1]
		wave_kms= wave_kms[::-1]

	#######################################################################
	# The following is to avoid plotting all channels from the cube.
	# Otherwise it is time consuming and not all channels contain signal.
	#######################################################################
	#meanflux_chan = np.array([np.mean(datacube[k], where=( (datacube[k]!=0) & (np.isfinite(datacube[k]))) )/rms for k in range(nz)])
	#chan_sig=(meanflux_chan>0.01) & np.isfinite(meanflux_chan)

	# first, select and average those channels with fluxes above 10% the rms noise
	meanflux_chan = np.array([np.mean(datacube[k], where=( (datacube[k]>=0.1*rms) & (np.isfinite(datacube[k]))) ) for k in range(nz)])
	# select those channels with fluxes above 50% the rms noise
	chan_sig=(meanflux_chan>=0.5*rms) & np.isfinite(meanflux_chan)
	vchan=wave_kms[chan_sig]
	vmin, vmax=np.min(vchan), np.max(vchan)
	chan_msk=(wave_kms>=vmin) & (wave_kms<=vmax)
	datacube_tmp = np.copy(datacube)
	cube_mod_tmp = np.copy(cube_mod)
	datacube_tmp = datacube_tmp
	wave_kms_tmp = wave_kms[chan_msk]
	cube_mod_tmp = cube_mod_tmp[:,None][chan_msk[:,None]]
	datacube_tmp = datacube_tmp[:,None][chan_msk[:,None]]
	[nz,ny,nx]=datacube_tmp.shape
	dv=wave_kms[chan_msk]
	#######################################################################

	#dv+=vsys
	vmin,vmax=np.min(dv),np.max(dv)
	y,x=np.indices((ny,nx))


	msk_r = np.isfinite(mom0_mod/mom0_mod).astype(float)
	msk_r[msk_r==0]=np.nan
	pa_maj = pa % 360
	pa_min = (pa+90) % 360


	pa_maj_rad = np.radians(pa_maj)
	pa_min_rad = np.radians(pa_min)

	# Upper quadrant if:
	if np.cos(pa_maj_rad)>0:
		s=+1
	else:
		s=-1

	slit_wpix = slit_w/pixel

	def signquad(pa0):
		rpix=Rings((x,y),pa_maj,eps,x0,y0)
		pa = np.radians(pa0)
		cos_theta = (- (x-x0)*np.sin(pa) + (y-y0)*np.cos(pa))/rpix
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
	rmin,rmax = -Rmax, Rmax
	dr = int(np.ceil(2*Rmax))
	rmin,rmax = -dr/2, dr/2

	pv_array_maj=np.zeros((len(dv),dr))
	pv_array_min=np.zeros((len(dv),dr))
	pv_array_maj_mod=np.zeros((len(dv),dr))
	pv_array_min_mod=np.zeros((len(dv),dr))

	ext_arc0=np.array([rmin*pixel_pvd_arc,rmax*pixel_pvd_arc,vmin,vmax])
	ext_arc1=np.array([rmin*pixel_pvd_arc,rmax*pixel_pvd_arc,vmin,vmax])
	
	# Change scale to arcmin if necessary
	#if np.nanmax(rarc) > 80 and np.all(extimg>120):
	#	ext_arc0[:2]=ext_arc0[:2]/60
	#	ext_arc1[:2]=ext_arc1[:2]/60

	def pv2(pa_rad, cube):
		cx, cy = x0, y0
		nv = len(dv)
	 
		sin_pa, cos_pa = np.sin(pa_rad), np.cos(pa_rad)
	 
		# --- CHANGE: limit offset range to [-r_max, +r_max] ---
		# Previously the slice always ran over the full cube half-diagonal
		# which included large amounts of empty sky beyond the galaxy.
		# Now r_max (the outermost ring radius) sets the physical boundary.
		
		r_max = Rmax / pixel
		
		half = int(np.ceil(r_max))
		# -------------------------------------------------------
	 
		offsets = np.arange(-half, half + 1)
		pv = np.zeros((len(dv),len(offsets)))
	 
		for k, off in enumerate(offsets):
			# Step along the slice direction: r_hat = (-sin(PA), cos(PA))
			xc = cx - off * sin_pa
			yc = cy + off * cos_pa
			total, count = np.zeros(nv), 0
			for w in range(-int(slit_wpix // 2), int(slit_wpix // 2 + 1)):
				# Step perpendicular: r_hat_perp = (-cos(PA), -sin(PA))
				xp = int(round(xc - w * cos_pa))
				yp = int(round(yc - w * sin_pa))
				if 0 <= xp < nx and 0 <= yp < ny:
					total += cube[:, yp, xp]
					count += 1
			if count:
				pv[:, k] = total / count

		return pv
		

	pv_array_maj 	 	= pv2(pa_maj_rad, datacube_tmp)
	pv_array_maj_mod	= pv2(pa_maj_rad, cube_mod_tmp)

	pv_array_min		= pv2(pa_min_rad, datacube_tmp)
	pv_array_min_mod	= pv2(pa_min_rad, cube_mod_tmp)


	ext_arc0=np.array([-rmax,rmax,vmin,vmax])
	ext_arc1=np.array([-rmax,rmax,vmin,vmax])


	return [pv_array_maj,pv_array_min,pv_array_maj_mod,pv_array_min_mod], [ext_arc0,ext_arc1,pixel_pvd_arc]

