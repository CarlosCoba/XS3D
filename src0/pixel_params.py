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


def pixels(shape,vel,pa,eps,x0,y0,ring, delta=1,pixel_scale = 1):

	[ny,nx] = shape


	x = np.arange(0, nx, 1)
	y = np.arange(0, ny, 1)

	XY_mesh = np.meshgrid(x,y,sparse=True)
	r_pixels_mask = ring_pixels(XY_mesh,pa,eps,x0,y0,ring,delta,pixel_scale)	
	XY_mesh0 = np.meshgrid(np.arange(0, 2*nx, 1),np.arange(0, 2*ny, 1),sparse=True)	
	r_pixels_mask0 = ring_pixels(XY_mesh0,pa,eps,nx,ny,ring,delta,pixel_scale)
	mask = r_pixels_mask

	indices = np.indices((ny,nx))


	pix_y=  indices[0]
	pix_x=  indices[1]

	pix_x = pix_x[mask]
	pix_y = pix_y[mask]



	vel_pixesl = vel[mask]	
	pix_y =  np.asarray(pix_y)
	pix_x = np.asarray(pix_x)
	#npix_exp = len(pix_x)
	npix_exp=len(r_pixels_mask0[0])


	mask = np.isfinite(vel_pixesl) == True
	mask = (np.isfinite(vel_pixesl)) & (vel_pixesl !=0)
	vel_val = vel_pixesl[mask]
	len_vel = len(vel_val)


	"""
	plt.imshow(vel, origin = "lower")
	plt.plot(pix_x,pix_y,"ko")
	plt.show()
	"""
	if npix_exp >0 and len_vel >0 :
		f_pixel = len(vel_val)/(1.0*npix_exp)
	else:
		f_pixel = 0


	return f_pixel







