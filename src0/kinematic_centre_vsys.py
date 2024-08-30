import numpy as np
import matplotlib.pylab as plt



def KC(Vel_map,xc,yc,pixel_scale):

	vel_map_copy = np.copy(Vel_map)

	vel_map = vel_map_copy 
	x0,y0 = int(xc),int(yc)
	nans = np.isnan(vel_map)
	vel_map[nans]=0
	[ny,nx] =vel_map.shape
	delta = int(5./pixel_scale)
	if delta < 5:
		delta = 2*int(pixel_scale)


	# The gradient map around 5 arcsec around the optical nucleus:
	M = np.zeros((ny,nx))
	for i in range(x0-delta,x0+delta):
		for j in range(y0-delta,y0+delta):

			if x0-delta > 0 and x0+delta< nx and y0-delta > 0 and y0+delta < ny: 
				grad =[] 
				for k1 in range(-1,2):
					for k2 in range(-1,2):
						grad1 = abs(vel_map[j][i] - vel_map[j+k1][i+k2])
						if grad1!= 0:
							grad.append(grad1)
				#grad2 = np.nanmedian(grad)
				grad2 = np.nanmean(grad)
				M[j][i] = grad2


	M[M==0] = np.nan
	#Normalize
	M = M/np.nanmax(M)
	median_grad = np.nanmedian(M)
	#median_grad = np.nanmean(M)





	x = []
	y = []
	# Weights:
	w = []

	for j in range(ny):
		for i in range(nx):
			if M[j][i] > median_grad:
				# Weighted positions
				xi = i*M[j][i]
				ji = j*M[j][i]
				vel_grad = M[j][i]

				x.append(xi)
				y.append(ji)
				w.append(vel_grad)


	XK = np.nansum(x)/np.nansum(w)
	YK = np.nansum(y)/np.nansum(w)

	
	"""
	plt.imshow(M,cmap = "jet", origin = "l")
	plt.plot(XK,YK,"kx",markersize = 15)
	plt.plot(xc,yc,"rx",markersize = 15)
	plt.show()
	"""
	
	#YK = np.nan
	if np.isfinite(XK) == True and np.isfinite(YK) == True:
		xk_int,yc_int = int(XK), int(YK)
		x0,y0 = xk_int,yc_int
	else:
		XK,YK = xc,yc
	
	delta_psf = delta

	vel_map[vel_map == 0] = np.nan
	vsys_region = (vel_map[y0-delta_psf:y0+delta_psf,x0-delta_psf:x0+delta_psf])
	vsys_mean = np.nanmean(vsys_region)
	e_vsys = np.nanstd(vsys_region)
	#plt.imshow(vsys_region)
	#plt.show()
	#print (e_vsys,vsys_mean)
	#return XK,YK,vsys_mean,e_vsys
	return xc, yc, vsys_mean,e_vsys
