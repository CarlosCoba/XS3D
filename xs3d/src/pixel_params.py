import numpy as np
import matplotlib.pylab as plt

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
	pix_y=  indices[0]
	pix_x=  indices[1]

	pix_x = pix_x[rxy_pixels_mask]
	pix_y = pix_y[rxy_pixels_mask]


	####################################################
	# Now extract the same ring in a double size window
	####################################################
	nx2=nx*2
	ny2=ny*2
	XY_mesh0 = np.meshgrid(np.arange(0, nx2, 1),np.arange(0, ny2, 1),sparse=True)
	rxy_pixels_mask_twice = ring_pixels(XY_mesh0,pa,eps,nx2//2,ny2//2,ring,delta,pixel_scale)



	#pixels in the original image
	vel_pixesl = velmap[rxy_pixels_mask]
	#npix_exp = len(pix_x)

	#pixels in the new image
	npix_exp=len(rxy_pixels_mask_twice[0])


	good_vels = (np.isfinite(vel_pixesl)) & (vel_pixesl !=0)
	vel_good = vel_pixesl[good_vels]
	ngood = len(vel_good)


	if npix_exp >0 and ngood >0 :
		f_pixel = ngood/(1.0*npix_exp)
	else:
		f_pixel = 0


	plot=0
	if plot:
		print('f_pixel=',f_pixel)
		plt.imshow(velmap, origin = "lower")
		plt.scatter(pix_x,pix_y,s=3, marker = 'x', color = 'k')
		plt.show()

	return f_pixel
