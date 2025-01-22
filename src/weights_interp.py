import numpy as np
from src.pixel_params import Rings


#v_new = v1*(r2-r)/(r2-r1) + v2*(r-r1)/(r2-r1)
#v_new = v1*w_v1 + v2*w_v2
def weigths_w(xy_mesh, pa, eps, x0, y0, r2, ring_space, pixel_scale, r1 = None):

	pa  = pa*np.pi/180  
	r_n = Rings(xy_mesh,pa,eps,x0,y0,pixel_scale)

	ring_space = float(ring_space)

	w_v1 =  ( r2 - r_n ) / ring_space
	if r1 == None: r1 = r2
	w_v2 = ( r_n - r1) / ring_space
	return np.ravel(w_v1),np.ravel(w_v2) 

