import numpy as np

def geom_moms(image0):
	# compute the geometric moment maps
	img=np.copy(image0)
	img[~np.isfinite(img)]=0
	img[img<0]=0

	[ny,nx]=img.shape
	indices = np.indices((ny,nx))
	pix_y= indices[0]
	pix_x= indices[1]
	
	F=np.nansum(img)
	if F == 0: print('mom0 map is full with zeros ?');quit()
	# centroid of the image
	xcen=np.nansum(img*pix_x)/F
	ycen=np.nansum(img*pix_y)/F		
	
	x2=np.nansum(img*(pix_x)**2)/F - xcen**2
	y2=np.nansum(img*(pix_y)**2)/F - ycen**2	
	xy=np.nansum(img*pix_x*pix_y)/F - xcen*ycen
	
	bmaj2=0.5*(x2+y2) + np.sqrt( (0.5*(x2-y2))**2 + xy**2 )
	bmin2=0.5*(x2+y2) - np.sqrt( (0.5*(x2-y2))**2 + xy**2 )
	bmaj=np.sqrt(bmaj2)
	bmin=np.sqrt(bmin2)
	eps = 1-(bmin/bmaj)	
	
	tan_2theta=2*xy/(x2-y2)
	theta=np.arctan(tan_2theta)/2
	
	pa = (theta + np.pi/2)% (np.pi/2)
	pa_deg,inc_deg=pa*180/np.pi, np.arccos(1-eps)*180/np.pi
	return pa_deg,inc_deg, xcen, ycen
