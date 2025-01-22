import numpy as np
from src.weights_interp import weigths_w
from src.pixel_params import Rings,eps_2_inc

def cs_k_remove_zeros(S_k,C_k,m_hrm):
	#C_k = [C1,C2, ...]
	#S_k = [S1,S2, ...]
	non_zero_s = S_k[0] != 0
	non_zero_c = C_k[0] != 0
	if m_hrm != 1:
		mask = [non_zero_c] + [non_zero_s]*(m_hrm - 1)
	else:
		mask = [non_zero_c] + [non_zero_s]


	# remove zeros from c_k and s_k
	C = np.array([C_k[k][mask[k]] for k in range(m_hrm)])
	S = np.array([S_k[k][mask[1]] for k in range(m_hrm)])
	n_noncirc = len(S[0])
	n_circ = len(C[0])

	return C, S, n_circ, n_noncirc


def cs_k_add_zeros(C_flat,S_flat,m_hrm,n_circ,n_noncirc):
	# C is flatten
	# S is flatten

	c1 = C_flat[:n_circ]; c1 = np.asarray(c1)
	c_k_flat = C_flat[n_circ:]
	s_k_flat = S_flat[:]

	s_k = np.array_split(s_k_flat, m_hrm)
	c_k = np.array_split(c_k_flat, m_hrm-1) if m_hrm !=1 else []

	
	c_k_zeros = [np.zeros(n_circ) for k in range(m_hrm)]
	s_k_zeros = [np.zeros(n_circ) for k in range(m_hrm)]

	c_k_zeros[0] = c1 
	if m_hrm ==1:
		s_k_zeros[0][0:n_noncirc] = s_k[0] 
	if m_hrm >1:
		for j in range(m_hrm):
			s_k_zeros[j][0:n_noncirc] = s_k[j]
		for j in range(1,m_hrm):
			c_k_zeros[j][0:n_noncirc] = c_k[j-1]


	return c_k_zeros, s_k_zeros


def myatan(x,y):
	m = np.pi*(1.0-0.5*(1+np.sign(x))*(1-np.sign(y**2))\
         -0.25*(2+np.sign(x))*np.sign(y))\
         -np.sign(x*y)*np.arctan((np.abs(x)-np.abs(y))/(np.abs(x)+np.abs(y)))

	m = np.pi/2. - m
	for i,j in enumerate(m):
		if m[i] < -np.pi : m[i] = m[i] + 2*np.pi
	return m


def myatan2D(x,y):
	m = np.pi*(1.0-0.5*(1+np.sign(x))*(1-np.sign(y**2))\
         -0.25*(2+np.sign(x))*np.sign(y))\
         -np.sign(x*y)*np.arctan((np.abs(x)-np.abs(y))/(np.abs(x)+np.abs(y)))

	m = np.pi/2. - m
	mask = m < -np.pi
	m[mask] = m[mask] +  2*np.pi
	return m

def AZIMUTHAL_ANGLE(shape,pa,eps,x0,y0):

	[ny,nx] = shape
	X = np.arange(0, nx, 1)
	Y = np.arange(0, ny, 1)
	xy_mesh = np.meshgrid(X,Y)
	(x,y) = xy_mesh

	pa,inc=pa*np.pi/180, eps_2_inc(eps)
	R  = Rings(xy_mesh,pa,eps,x0,y0)
	cos_theta = (- (x-x0)*np.sin(pa) + (y-y0)*np.cos(pa))/R
	sin_theta = (- (x-x0)*np.cos(pa) - (y-y0)*np.sin(pa))/((1-eps)*R)
	theta = myatan2D(sin_theta,cos_theta)
	return theta,cos_theta


def SIN_COS(xy_mesh,pa,eps,x0,y0):
	(x,y) = xy_mesh
	pa,inc=pa*np.pi/180, eps_2_inc(eps)
	R  = Rings(xy_mesh,pa,eps,x0,y0)
	cos_theta = (- (x-x0)*np.sin(pa) + (y-y0)*np.cos(pa))/R
	sin_theta = (- (x-x0)*np.cos(pa) - (y-y0)*np.sin(pa))/((1-eps)*R)
	return sin_theta,cos_theta



def CIRC_MODEL(xy_mesh,Vrot,pa,eps,x0,y0):
	(x,y) = xy_mesh
	pa,inc=pa*np.pi/180, eps_2_inc(eps)
	R  = Rings(xy_mesh,pa,eps,x0,y0)
	cos_theta = (- (x-x0)*np.sin(pa) + (y-y0)*np.cos(pa))/R
	sin_theta = (- (x-x0)*np.cos(pa) - (y-y0)*np.sin(pa))/((1-eps)*R)
	vlos = np.sin(inc)*(Vrot*cos_theta)
	return np.ravel(vlos)

def SIGMA_MODEL(xy_mesh,Sigma,pa,eps,x0,y0):
	(x,y) = xy_mesh
	pa,inc=pa*np.pi/180, eps_2_inc(eps)
	R  = Rings(xy_mesh,pa,eps,x0,y0)
	cos_theta = (- (x-x0)*np.sin(pa) + (y-y0)*np.cos(pa))/R
	sin_theta = (- (x-x0)*np.cos(pa) - (y-y0)*np.sin(pa))/((1-eps)*R)
	disp = Sigma*np.ones_like(cos_theta)
	return np.ravel(disp)


def RADIAL_MODEL(xy_mesh,Vrot,Vr2,pa,eps,x0,y0):
	(x,y) = xy_mesh
	pa,inc=pa*np.pi/180, eps_2_inc(eps)
	R  = Rings(xy_mesh,pa,eps,x0,y0)
	cos_theta = (- (x-x0)*np.sin(pa) + (y-y0)*np.cos(pa))/R
	sin_theta = (- (x-x0)*np.cos(pa) - (y-y0)*np.sin(pa))/((1-eps)*R)
	vlos = np.sin(inc)*(Vrot*cos_theta + Vr2*sin_theta)
	return np.ravel(vlos)


def BISYM_MODEL(xy_mesh,Vrot,Vrad,pa,eps,x0,y0,Vtan,phi_b):
	(x,y) = xy_mesh
	pa,inc=pa*np.pi/180, eps_2_inc(eps)
	R  = Rings(xy_mesh,pa,eps,x0,y0)
	cos_theta = (- (x-x0)*np.sin(pa) + (y-y0)*np.cos(pa))/R
	sin_theta = (- (x-x0)*np.cos(pa) - (y-y0)*np.sin(pa))/((1-eps)*R)

	m = 2
	theta = np.arctan(sin_theta/cos_theta)
	theta = myatan(sin_theta,cos_theta)
	theta_b = theta - phi_b
	
	vlos = np.sin(inc)*((Vrot*cos_theta) - Vtan*np.cos(m*theta_b)*cos_theta - Vrad*np.sin(m*theta_b)*sin_theta)

	return np.ravel(vlos)



def HARMONIC_MODEL(xy_mesh,c_k,s_k,pa,eps,x0,y0,m_hrm):
	(x,y) = xy_mesh
	pa,inc=pa*np.pi/180, eps_2_inc(eps)
	R  = Rings(xy_mesh,pa,eps,x0,y0)
	cos_theta = (- (x-x0)*np.sin(pa) + (y-y0)*np.cos(pa))/R
	sin_theta = (- (x-x0)*np.cos(pa) - (y-y0)*np.sin(pa))/((1-eps)*R)

	theta_c = np.arccos(cos_theta)
	theta_s = np.arcsin(sin_theta)


	v_k = 0
	for k in range(1,m_hrm+1):
		v_k = v_k + c_k[k-1]*np.sin(inc)*np.cos(k*theta_c) + s_k[k-1]*np.sin(inc)*np.sin(k*theta_s)

	return np.ravel(v_k)






