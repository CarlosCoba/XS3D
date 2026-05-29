import numpy as np
import matplotlib.pylab as plt
from .ellipse import drawellipse

def eps_2_inc(eps):
	cos_i = 1-eps
	inc = np.arccos(cos_i)
	return inc

def e_eps2e_inc(eps,deps):
	cos_i = 1-deps
	inc = np.arccos(cos_i)
	dinc=deps/np.sqrt(-(eps-2)*eps)
	return dinc


def inc_2_eps(inc):
	inc = inc*np.pi/180
	eps = 1-np.cos(inc)
	return eps


def Rings(xy_mesh,pa,eps,x0,y0,pixel_scale=1):

	(x,y) = xy_mesh

	X = (- (x-x0)*np.sin(pa) + (y-y0)*np.cos(pa))
	Y = (- (x-x0)*np.cos(pa) - (y-y0)*np.sin(pa))

	R= np.sqrt(X**2+(Y/(1-eps))**2)

	return R*pixel_scale


def v_interp(r, r2, r1, v2, v1 ):
	m = (v2 - v1) / (r2 - r1)
	v0 = m*(r-r1) + v1
	return v0

def slits(xy,pa,eps,x0,y0,width=5,pixel=1):

	x,y=xy
	# y = m(x-x0)+y0 --> y -mx +(mx0-y0)
	alpha=(pa+np.pi/2)
	m=np.tan(alpha)
	A,B,C=-m,1,m*x0-y0
	d = abs(A*x+B*y+C)/np.sqrt(A**2+B**2)
	darc=d*pixel
	msk = darc < width/2.

	return msk


#######################################################3


def ring_pixels(xy_mesh,pa,eps,x0,y0,ring,delta,pixel_scale):
	pa=pa*np.pi/180

	r_n = Rings(xy_mesh,pa,eps,x0,y0,pixel_scale)
	a_k = ring


	mask = np.where( (r_n >= a_k - delta) & (r_n < a_k + delta) )
	r_n = r_n[mask]

	return mask


def pixels(shape,velmap,pa,eps,x0,y0,ring, delta=1,pixel_scale = 1):

	[ny,nx] = shape
	x = np.arange(0, nx, 1)
	y = np.arange(0, ny, 1)
	XY_mesh = np.meshgrid(x,y,sparse=True)
	rxy_pixels_mask = ring_pixels(XY_mesh,pa,eps,x0,y0,ring,delta,pixel_scale)

	indices = np.indices((ny,nx))
	pix_y_indx =  indices[0]
	pix_x_indx =  indices[1]

	# pixels in each ring
	pix_x = pix_x_indx[rxy_pixels_mask]
	pix_y = pix_y_indx[rxy_pixels_mask]

	# lets check if the ring contains pixels along the semimajor axis
	width = delta
	pa_slit = pa * np.pi/180
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
	XY_mesh0 = np.meshgrid(np.arange(0, nx2, 1),np.arange(0, ny2, 1),sparse=True)
	rxy_pixels_mask_twice = ring_pixels(XY_mesh0,pa,eps,nx2//2,ny2//2,ring,delta,pixel_scale)

	#pixels in the original image
	vel_pixesl = velmap[rxy_pixels_mask]

	#pixels in the new image
	npix_exp=len(rxy_pixels_mask_twice[0])


	good_vels = (np.isfinite(vel_pixesl)) & (vel_pixesl !=0)
	vel_good = vel_pixesl[good_vels]
	ngood = len(vel_good)


	#if npix_exp >0 and ngood >0 :
	#	f_pixel = ngood/(1.0*npix_exp)
	#else:
	#	f_pixel = 0

	f_pixel = ngood/(1.0*npix_exp)
	# if there are pixels along the major axis accept the ring :
	if npix_maj_axs > 0:
		f_pixel = 1

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
