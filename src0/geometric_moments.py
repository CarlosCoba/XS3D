import numpy as np
import matplotlib.pylab as plt 

def ellipse(x0,y0,pa_deg,bmaj,eps):
	#pa_deg is the cartesian angle
	t=np.linspace(-2*np.pi,2*np.pi,100)	
	A=bmaj
	B=A*(1-eps)
	PA=pa_deg*np.pi/180
	x=x0+A*np.cos(PA)*np.cos(t)-B*np.sin(PA)*np.sin(t)
	y=y0+A*np.sin(PA)*np.cos(t)+B*np.cos(PA)*np.sin(t)
	return x,y

def geom_moms(image0,plot=False,binary=False):
	# compute the geometric moment maps
	img=np.copy(image0)
	img[~np.isfinite(img)]=0
	img[img<0]=0
	
	if binary:
		# work with a binary image
		img=img.astype(bool)

	[ny,nx]=img.shape
	indices = np.indices((ny,nx))
	pix_y= indices[0]+1
	pix_x= indices[1]+1
	
	F=np.nansum(img, where=np.isfinite(img))
	if F == 0: print('mom0 map is full with zeros ?');quit()
	# centroid of the image
	xcen=np.nansum(img*pix_x,where=np.isfinite(img))/F
	ycen=np.nansum(img*pix_y,where=np.isfinite(img))/F
	
	x2=np.nansum(img*(pix_x)**2,where=np.isfinite(img))/F - xcen**2
	y2=np.nansum(img*(pix_y)**2,where=np.isfinite(img))/F - ycen**2	
	xy=np.nansum(img*pix_x*pix_y,where=np.isfinite(img))/F - xcen*ycen
	
	bmaj2=0.5*(x2+y2) + np.sqrt( (0.5*(x2-y2))**2 + xy**2 )
	bmin2=0.5*(x2+y2) - np.sqrt( (0.5*(x2-y2))**2 + xy**2 )
	bmaj=np.sqrt(bmaj2)
	bmin=np.sqrt(bmin2)
	eps = 1-(bmin/bmaj)	
	
	tan_2theta=2*xy/(x2-y2)
	# there are two solutions theta1, theta2
	theta1=np.arctan(tan_2theta)/2
	theta2=theta1+np.pi/2
	theta_try=(x2-y2)*np.cos(2*theta1)+xy*np.sin(2*theta1)
	if theta_try>0:
		theta=theta1
	else:
		theta=theta2
		
	theta_deg=theta*180/np.pi
	inc_deg=np.arccos(1-eps)*180/np.pi	
	pa_astro_deg=theta_deg-90	
	
	if plot:
		print(pa_astro_deg,inc_deg, xcen, ycen)	
		plt.imshow(img*(img/img),origin='lower');
		x,y=ellipse(xcen,ycen,theta_deg,bmaj,eps)
		plt.scatter(xcen,ycen,s=50,c='r')
		plt.plot(x,y,'r-')
		plt.show()
	
	return pa_astro_deg,inc_deg, xcen, ycen
	

