import numpy as np
import matplotlib.pylab as plt
from .ellipse import drawellipse

def eps_2_inc(eps):
	cos_i = 1-eps
	inc_rad = np.arccos(cos_i)
	inc_deg = inc_rad*180/np.pi
	return inc_deg

def e_eps2e_inc(eps,deps):
	cos_i = 1-deps
	inc = np.arccos(cos_i)
	dinc=deps/np.sqrt(-(eps-2)*eps)
	return dinc


def inc_2_eps(inc_deg):
	inc_rad = np.radians(inc_deg)
	eps = 1-np.cos(inc_rad)
	return eps


def Rings(xy_mesh,pa_deg,eps,x0,y0,pixel_scale=1):

	(x,y) = xy_mesh
	
	# if eps > 1 then you passed the inclination angle, not eps.
	if eps>1:
		inc = eps
		eps = inc_2_eps(inc)
	
	pa_r = np.radians(pa_deg)
	cos_inc = 1 - eps
	
	XX = (x-x0)
	YY = (y-y0)
	
	# Rotate to kinematic frame (undo PA rotation)
	x_rot  =  -XX*np.sin(pa_r) + YY*np.cos(pa_r)
	y_rot  =  -XX*np.cos(pa_r) - YY*np.sin(pa_r)
	
	# Deproject minor axis: y_disk = y_rot / cos(inc)
	y_disk = y_rot / cos_inc
	r_disk = np.sqrt(x_rot**2 + y_disk**2)
			
	#X = (- (x-x0)*np.sin(pa) + (y-y0)*np.cos(pa))
	#Y = (- (x-x0)*np.cos(pa) - (y-y0)*np.sin(pa))
	#r_disk= np.sqrt(X**2+(Y/(1-eps))**2)

	return r_disk*pixel_scale


from scipy.ndimage import binary_dilation
def make_outer_mask_sky(xy_mesh,pa_deg,eps,x0,y0,rmax,offset,pixel_scale=1):
	# Step 1: projected galaxy footprint at r_max
	galaxy_mask = Rings(xy_mesh,pa_deg,eps,x0,y0,pixel_scale) <= rmax

	# Step 2: circular dilation structuring element
	# Radius = n_psf × PSF_FWHM in sky pixels
	radius_px = offset / pixel_scale
	r_ceil	= int(np.ceil(radius_px))
	yy, xx	= np.ogrid[-r_ceil:r_ceil+1, -r_ceil:r_ceil+1]
	footprint = (xx**2 + yy**2) <= radius_px**2

	# Step 3: dilate — include every sky pixel within radius_px
	# of any pixel in the galaxy footprint
	outer_mask = binary_dilation(galaxy_mask, structure=footprint)
	
	#plt.imshow(galaxy_mask, origin = 'lower');plt.show()	
	#outer_mask = outer_mask.astype(int) + galaxy_mask.astype(int)
	#plt.imshow(outer_mask, origin = 'lower');plt.show()
	
	return outer_mask


def v_interp(r, r2, r1, v2, v1 ):
	m = (v2 - v1) / (r2 - r1)
	v0 = m*(r-r1) + v1
	return v0

def slits(xy,pa_deg,eps,x0,y0,width=5,pixel=1, dist=False):

	x,y=xy
	# y = m(x-x0)+y0 --> y -mx +(mx0-y0)
	pa = np.radians(pa_deg)
	alpha=(pa+np.pi/2)
	m=np.tan(alpha)
	A,B,C=-m,1,m*x0-y0
	d = abs(A*x+B*y+C)/np.sqrt(A**2+B**2)
	darc=d*pixel
	msk = darc < width/2.
	if dist:
		return darc
	return msk


def slab(xy_mesh,pa_deg,xc, yc,pixel=1):
	(x,y) = xy_mesh
	
	XX = x-xc
	YY = y-yc

	pa = np.radians(pa_deg)					
	# Rotate sky coords to the disk major/minor axis frame
	x_rot =  -XX * np.sin(pa) + YY * np.cos(pa)   # along major axis
	y_rot =  -XX * np.cos(pa) - YY * np.sin(pa)   # along minor axis (sky)
	
	x_rot*=pixel
	y_rot*=pixel	
	plt.imshow(x_rot, origin = 'lower');plt.show()

def Cilinder(xy_mesh, pa_deg, eps, x0, y0, pixel_scale=1, width = 1):
	# Cilinder is the equivalend of the Ring function for tilted ring models,
	# but adding positive (redshifted) and negative (blueshifted) values to each major and minor axes sides.
	# This function creates slabs perpendicular to the major axis position angle.
	
	(x,y) = xy_mesh
	
	x = np.asarray(x)
	y = np.asarray(y)	

	if x.ndim == y.ndim == 2:		
		xx_yy = np.vstack((np.concatenate(x)-x0, np.concatenate(y)-y0))
	else:
		xx_yy = np.vstack((x-x0, y-y0))	

	pa = np.radians(pa_deg)
	t = np.pi/2
	cos_theta = np.cos(t*np.pi/180)
	sin_theta = np.sin(t*np.pi/180)
	R = np.array([[cos_theta, -sin_theta],
		[sin_theta,  cos_theta]])
			
	x_y_rot = np.dot(R, xx_yy) 
	xx = x_y_rot[0] + x0
	yy = x_y_rot[1]	+ y0

	x, y = xx.reshape(x.shape), yy.reshape(y.shape)		
	alpha = pa 
	m = np.tan(alpha)
	A,B,C = -m,1,m*x0-y0
	d_perp = (A*x+B*y+C)/np.sqrt(A**2+B**2)
	

	alpha = pa + np.pi/2
	m = np.tan(alpha)
	A,B,C = -m,1,m*x0-y0		
	d_para = (A*x+B*y+C)/np.sqrt(A**2+B**2)
	
	msk = abs(d_perp) < width / 2
	
	d_para = d_para*msk*pixel_scale
	

	#plt.imshow(d_para, cmap = 'rainbow', origin = 'lower');plt.show()
	return d_para
	


#######################################################3


def ring_pixels(xy_mesh,pa_deg,eps,x0,y0,ring,delta,pixel_scale):

	r_n = Rings(xy_mesh,pa_deg,eps,x0,y0,pixel_scale)
	a_k = ring

	mask = np.where( (r_n >= a_k - delta) & (r_n < a_k + delta) )
	r_n = r_n[mask]

	return mask


def pixels(shape,velmap,pa,eps,x0,y0,ring, delta=1,pixel_scale = 1):

	if eps> 0.90:
		eps = 0.9
	[ny,nx] = shape
	x = np.arange(0, nx, 1)
	y = np.arange(0, ny, 1)
	xy_mesh = np.meshgrid(x,y,sparse=True)

	#slab(xy_mesh, pa,x0,y0,pixel_scale)
	rxy_pixels_mask = ring_pixels(xy_mesh,pa,eps,x0,y0,ring,delta,pixel_scale)

	#Cilinder(xy_mesh, pa, eps, x0, y0, pixel_scale=pixel_scale)

	indices = np.indices((ny,nx))
	pix_y_indx =  indices[0]
	pix_x_indx =  indices[1]

	# pixels in each ring
	pix_x = pix_x_indx[rxy_pixels_mask]
	pix_y = pix_y_indx[rxy_pixels_mask]

	# lets check if the ring contains pixels along the semimajor axis
	width = delta
	pa_slit = pa 
	msk_slit = slits((pix_x_indx,pix_y_indx),pa_slit,eps,x0,y0,width=width,pixel=pixel_scale)
		
	# multiply two boolean masks
	pix_maj_axs = msk_slit * np.isfinite(velmap)
	# if npix_maj_axs!=0 then there is at least one pixel along the major axis
	npix_maj_axs = np.sum(pix_maj_axs[rxy_pixels_mask])

	####################################################
	# Now extract the same ring in a double size window
	####################################################
	nx2=nx*2
	ny2=ny*2
	xy_mesh0 = np.meshgrid(np.arange(0, nx2, 1),np.arange(0, ny2, 1),sparse=True)
	rxy_pixels_mask_twice = ring_pixels(xy_mesh0,pa,eps,nx2//2,ny2//2,ring,delta,pixel_scale)

	#pixels in the original image
	vel_pixesl = velmap[rxy_pixels_mask]

	#pixels in the new image
	npix_exp=len(rxy_pixels_mask_twice[0])


	good_vels = (np.isfinite(vel_pixesl)) & (vel_pixesl !=0)
	vel_good = vel_pixesl[good_vels]
	ngood = len(vel_good)

	f_pixel = ngood/(1.0*npix_exp) if npix_exp > 0 else 0
	# if there are pixels along the major axis accept the ring :
	if npix_maj_axs > 0:
		f_pixel = 1
	if npix_maj_axs == 0 and eps>=0.90:
		f_pixel = 0
		
	plot=0
	if plot:
		print('f_pixel=',f_pixel)
		plt.imshow(velmap, origin = "lower")
		plt.scatter(pix_x, pix_y, s=3, marker = 'x', color = 'k')

		elipse_mjr=drawellipse(x0, y0, bmajor=ring/pixel_scale, pa_deg=pa, eps=eps)
		x_,y_=elipse_mjr[0],elipse_mjr[1]
		plt.plot(x_, y_, linestyle='--', color = 'k',  lw=2)
		plt.show()

	return f_pixel
