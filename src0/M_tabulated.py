import numpy as np
import matplotlib.pylab as plt
from src0.pixel_params import Rings,eps_2_inc
 


def weigths_w(xy_mesh,shape,pa,eps,x0,y0,ring,delta,k,pixel_scale):

	r_n = Rings(xy_mesh,pa,eps,x0,y0,pixel_scale)
	a_k = ring


	mask = np.where( (r_n >= a_k - delta) & (r_n < a_k + delta) ) 
	r_n = r_n[mask]

	w_k_n = (1 - (r_n -a_k)/delta)
	w_k_plus_1_n = (r_n -a_k)/delta


	(x,y) = xy_mesh


	return np.ravel(w_k_n),np.ravel(w_k_plus_1_n),mask


def myatan(x,y):
	m = np.pi*(1.0-0.5*(1+np.sign(x))*(1-np.sign(y**2))\
         -0.25*(2+np.sign(x))*np.sign(y))\
         -np.sign(x*y)*np.arctan((np.abs(x)-np.abs(y))/(np.abs(x)+np.abs(y)))

	m = np.pi/2. - m
	for i,j in enumerate(m):
		if m[i] < -np.pi : m[i] = m[i] + 2*np.pi
	return m



def cos_sin(xy_mesh,pa,eps,x0,y0,pixel_scale=1):
	(x,y) = xy_mesh
	R  = Rings(xy_mesh,pa,eps,x0,y0,pixel_scale)

	cos_tetha = (- (x-x0)*np.sin(pa) + (y-y0)*np.cos(pa))/R
	sin_tetha = (- (x-x0)*np.cos(pa) - (y-y0)*np.sin(pa))/((1-eps)*R)

	#return np.ravel(cos_tetha),np.ravel(sin_tetha)
	return cos_tetha,sin_tetha


def trigonometric_weights(xy_mesh,pa,eps,x0,y0,phi_b,mask,vmode="radial",pixel_scale=1, m_hrm = 1):
	inc = eps_2_inc(eps)
	cos,sin = cos_sin(xy_mesh,pa,eps,x0,y0)
	cos,sin = cos[mask],sin[mask]

	if vmode == "dispersion":
		w_rot = np.ones_like(cos)
		return np.ravel(w_rot)

	if vmode == "circular":
		w_rot = np.sin(inc)*cos
		return np.ravel(w_rot)

	if vmode == "radial":
		w_rot = np.sin(inc)*cos
		w_rad = np.sin(inc)*sin
		return np.ravel(w_rot), np.ravel(w_rad)

	if vmode == "vertical":
		w_rot = np.sin(inc)*cos
		w_z = np.cos(inc)*np.ones_like(cos)
		return np.ravel(w_rot), np.ravel(w_z)

	if vmode == "bisymmetric":

		theta = np.arctan(sin/cos)
		theta = myatan(sin,cos)		
		phi_b = phi_b
		phi_b = theta - phi_b

		w_rot = np.sin(inc)*cos
		w_rad = -np.sin(inc)*sin*np.sin(2*phi_b)
		w_tan = -np.sin(inc)*cos*np.cos(2*phi_b)
		return np.ravel(w_rot), np.ravel(w_rad), np.ravel(w_tan)

	if "hrm" in vmode:

		theta_c = np.arccos(cos) 
		theta_s = np.arcsin(sin) 
		c, s = [], []
		for i in range(1,m_hrm+1):
			globals()['w_cc%s' % i] = np.sin(inc)*np.cos(i*theta_c)
			globals()['w_ss%s' % i] = np.sin(inc)*np.sin(i*theta_s)
			c.append(globals()['w_cc%s' % i])
			s.append(globals()['w_ss%s' % i])

		return [ np.ravel(k) for k in c ],  [ np.ravel(k) for k in s ]



def M_tab(pa,eps,x0,y0,phi_b,rings, delta,k, shape, mommaps, emoms, pixel_scale=1,vmode = "radial", m_hrm = 1):

	[ny,nx] = shape

	pa = pa*np.pi/180
	X = np.arange(0, nx, 1)
	Y = np.arange(0, ny, 1)
	xy_mesh = np.meshgrid(X,Y)
	[mom0,mom1,mom2]=mommaps
	#for k in emoms: k = np.ones_like(mom0)	
	[emom0,emom1,emom2]=emoms
	vel_val=mom1
	disp=mom2

	weigths_k,weigths_j,mask = weigths_w(xy_mesh,shape,pa,eps,x0,y0,rings,delta,k,pixel_scale =pixel_scale)
	weigths_k,weigths_j = np.asarray(weigths_k),np.asarray(weigths_j)
	
	# dispersion
	w_rot = trigonometric_weights(xy_mesh,pa,eps,x0,y0,0,mask,'dispersion')
	sigma_v = emom2[mask]
	x11,x12 = w_rot**2/sigma_v**2,0/sigma_v**2
	x21,x22 = 0/sigma_v**2,0/sigma_v**2

	D = (disp[mask])
	y1 = (w_rot/sigma_v**2)*D
	y2 = (0/sigma_v**2)*D
	A = np.asarray([[np.nansum(x11),np.nansum(x12)],[np.nansum(x21),np.nansum(x22)]])
	B= np.asarray([np.nansum(y1),np.nansum(y2)])
	dispersion=np.nansum(y1)/np.nansum(x11)
	dispersion=abs(dispersion)

	if vmode == "circular":

		w_rot = trigonometric_weights(xy_mesh,pa,eps,x0,y0,0,mask,vmode)

		sigma_v = emom1[mask]
		x11,x12 = w_rot**2/sigma_v**2,0/sigma_v**2
		x21,x22 = 0/sigma_v**2,0/sigma_v**2

		D = (vel_val[mask])
		y1 = (w_rot/sigma_v**2)*D
		y2 = (0/sigma_v**2)*D

		A = np.asarray([[np.nansum(x11),np.nansum(x12)],[np.nansum(x21),np.nansum(x22)]])
		B= np.asarray([np.nansum(y1),np.nansum(y2)])
		vrot,vrad = np.nansum(y1)/np.nansum(x11), 0
		
		return dispersion,vrot,vrot*0,vrot*0

	if vmode == "radial":
		w_rot, w_rad = trigonometric_weights(xy_mesh,pa,eps,x0,y0,0,mask,vmode)
		sigma_v = emom1[mask]
		x11,x12 = w_rot**2/sigma_v**2,w_rot*w_rad/sigma_v**2
		x21,x22 = w_rot*w_rad/sigma_v**2,w_rad**2/sigma_v**2

		D = (vel_val[mask])
		y1 = (w_rot/sigma_v**2)*D
		y2 = (w_rad/sigma_v**2)*D

		A = np.asarray([[np.nansum(x11),np.nansum(x12)],[np.nansum(x21),np.nansum(x22)]])
		B= np.asarray([np.nansum(y1),np.nansum(y2)])

		x = np.linalg.solve(A, B)
		vrot,vrad = abs(x[0]),x[1]
		try:
			x = np.linalg.solve(A, B)
			vrot,vrad = abs(x[0]),x[1]
		except(np.linalg.LinAlgError):
			vrot,vrad = 0,0

		if np.isfinite(vrot) == False: vrot = 0
		if np.isfinite(vrad) == False: vrad = 0
		return dispersion,vrot,vrad,vrad*0


	if vmode == "vertical":
		w_rot, w_z = trigonometric_weights(xy_mesh,pa,eps,x0,y0,0,mask,vmode)
		sigma_v = emom1[mask]
		x11,x12 = w_rot**2/sigma_v**2,w_rot*w_z/sigma_v**2
		x21,x22 = w_rot*w_z/sigma_v**2,w_z**2/sigma_v**2

		D = (vel_val[mask])
		y1 = (w_rot/sigma_v**2)*D
		y2 = (w_z/sigma_v**2)*D

		A = np.asarray([[np.nansum(x11),np.nansum(x12)],[np.nansum(x21),np.nansum(x22)]])
		B= np.asarray([np.nansum(y1),np.nansum(y2)])
		try:
			x = np.linalg.solve(A, B)
			vrot,vz = abs(x[0]),x[1]
		except(np.linalg.LinAlgError):
			vrot,vz = 0,0
							
		if np.isfinite(vrot) == False: vrot = 0
		if np.isfinite(vz) == False: vz = 0
		return dispersion,vrot,vz,vz*0

	if vmode == "bisymmetric":
		w_rot, w_rad, w_tan = trigonometric_weights(xy_mesh,pa,eps,x0,y0,phi_b,mask,vmode)
		sigma_v = emom1[mask]
		x11,x12,x13 = w_rot**2/sigma_v**2,w_rot*w_rad/sigma_v**2,w_tan*w_rot/sigma_v**2
		x21,x22,x23 = w_rot*w_rad/sigma_v**2,w_rad**2/sigma_v**2,w_rad*w_tan/sigma_v**2
		x31,x32,x33 = w_rot*w_tan/sigma_v**2,w_rad*w_tan/sigma_v**2,w_tan**2/sigma_v**2


		D = (vel_val[mask])
		y1 = (w_rot/sigma_v**2)*D
		y2 = (w_rad/sigma_v**2)*D
		y3 = (w_tan/sigma_v**2)*D

		A = np.asarray([[np.nansum(x11),np.nansum(x12),np.nansum(x13)],[np.nansum(x21),np.nansum(x22),np.nansum(x23)],[np.nansum(x31),np.nansum(x32),np.nansum(x33)]])
		B= np.asarray([np.nansum(y1),np.nansum(y2),np.nansum(y3)])

		try:
			x = np.linalg.solve(A, B)
			vrot,vrad,vtan = x[0],x[1],x[2]
			if np.isfinite(vrot) == False: vrot = 50
			if np.isfinite(vrad) == False: vrad = 0
			if np.isfinite(vtan) == False: vtan = 0

		except(TypeError,np.linalg.LinAlgError):
			w_sys,w_rot,w_rad,w_tan,vrot,vrad,vsys,vtan =  0,0,0,0,0,0,0,0
		return dispersion,vrot,vrad,vtan


	if "hrm" in vmode:

		w_c, w_s = trigonometric_weights(xy_mesh,pa,eps,x0,y0,0,mask,vmode, m_hrm = m_hrm)
		w = w_c + w_s

		m = 2*m_hrm
		sigma_v = emom1[mask]
		X = []
		for j in range(1,m+1):
			for i in range(1,m+1):				
				globals()['x_%s%s' % (j,i)] = w[j-1]*w[i-1]/sigma_v**2
				X.append(globals()['x_%s%s' % (j,i)])

		D = (vel_val[mask])
		y = []
		for j in range(1,m_hrm+1):				
			globals()['y_c%s' % (j)] = (w_c[j-1]/sigma_v**2)*D
			y.append(globals()['y_c%s' % (j)])
		for j in range(1,m_hrm+1):				
			globals()['y_s%s' % (j)] = (w_s[j-1]/sigma_v**2)*D
			y.append(globals()['y_s%s' % (j)])

		A = np.zeros((m,m))
		k = 0
		for j in range(1,m+1):
			for i in range(1,m+1):
				A[j-1][i-1] = np.nansum(X[k])
				k = k + 1				

		B = np.zeros(m)
		for j in range(1,m+1):
			B[j-1] = np.nansum(y[j-1])	

		x = np.linalg.solve(A, B)
		#x = C, S
		C, S = x[:m_hrm],x[m_hrm:]
		return dispersion,C, S




