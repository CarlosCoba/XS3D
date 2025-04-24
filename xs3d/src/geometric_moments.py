import numpy as np
import matplotlib.pylab as plt
from .start_messenge import Print

def ellipse(x0,y0,pa_deg,bmaj,eps):
	#pa_deg is the cartesian angle
	t=np.linspace(-2*np.pi,2*np.pi,100)
	A=bmaj
	B=A*(1-eps)
	PA=pa_deg*np.pi/180
	x=x0+A*np.cos(PA)*np.cos(t)-B*np.sin(PA)*np.sin(t)
	y=y0+A*np.sin(PA)*np.cos(t)+B*np.cos(PA)*np.sin(t)
	return x,y

def Rings(xy_mesh,pa,eps,x0,y0,pixel_scale=1):

	(x,y) = xy_mesh

	X = (- (x-x0)*np.sin(pa) + (y-y0)*np.cos(pa))
	Y = (- (x-x0)*np.cos(pa) - (y-y0)*np.sin(pa))

	R= np.sqrt(X**2+(Y/(1-eps))**2)

	return R*pixel_scale


def Gini(img_flat0):
	######################################
	# Florian 2016, 816:L23
	# DOI: 10.3847/2041-8205/816/2/L23
	#####################################

	img_flat0=abs(img_flat0)
	#order pixels within the given aperture
	img_flat=np.sort(img_flat0)
	N=len(img_flat)
	meanf=np.nanmean(img_flat)
	num=[(2*(i+1)-N-1)*img_flat[i] for i in range(N)]
	G=np.sum(num)/(meanf*N*(N-1))
	return G


ntries=0
def geom_moms(image0,pixel=1,plot=False,binary=True, nloops=15):
	# compute the geometric moment maps
	img=np.copy(image0)
	img[~np.isfinite(img)]=0
	img[img<0]=0

	if binary:
		# work with a binary image
		img=img.astype(bool)

	[ny,nx]=img.shape
	indices = np.indices((ny,nx))
	pix_y= indices[0]#+1
	pix_x= indices[1]#+1

	G_img=1
	G_ellipse=1
	Nloops=0

	Print().status('Estimating disk geometry')
	while G_img > 0.8 and G_ellipse > 0.7 and Nloops < nloops:
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
		pa_astro_deg = pa_astro_deg % 360
		pa_astro_rad=pa_astro_deg*np.pi/180
		rmax=2*(bmaj)*pixel



		x = np.arange(0, nx, 1)
		y = np.arange(0, ny, 1)
		XY_mesh = np.meshgrid(x,y,sparse=True)
		r = Rings(XY_mesh,pa_astro_rad,eps,xcen,ycen,pixel)


		msk_r=r<rmax
		imgb=img*(msk_r)

		img_flat=(img).flatten()
		ftot=len(img_flat[img_flat!=0])
		G_img=Gini(img_flat)

		img_flat0=imgb[msk_r]
		fellipse=len(img_flat0[img_flat0!=0])

		G_ellipse=Gini(img_flat0)
		G_ellipse_round=round(G_ellipse,3)

		f=fellipse/ftot

		Nloops+=1
		img=imgb

		Print().status(f'Redefining mask, Attempt #{Nloops}')
		Print().out('Current Gini-coefficient',G_ellipse_round)

		plot=0
		if plot:
			print(pa_astro_deg,inc_deg, xcen, ycen,rmax)
			plt.imshow(image0*(img/img),origin='lower');
			x,y=ellipse(xcen,ycen,theta_deg,bmaj*2,eps)
			plt.scatter(xcen,ycen,s=50,c='r')
			plt.plot(x,y,'r-')
			plt.show()
	rmax=(rmax//pixel)*pixel
	return pa_astro_deg,inc_deg, xcen, ycen, rmax
