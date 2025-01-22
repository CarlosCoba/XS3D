import numpy as np
from astropy.io import fits
import matplotlib.pylab as plt

from src.pixel_params import Rings
from src.constants import __c__



def pv_array(xy,pa_slit,pa,eps,x0,y0,width=5,pixel=1):
	
	x,y=xy
	pa=pa*np.pi/180	
	# y = m(x-x0)+y0 --> y -mx +(mx0-y0)
	alpha=(pa+np.pi/2)
	m=np.tan(alpha)	 
	A,B,C=-m,1,m*x0-y0
	d = abs(A*x+B*y+C)/np.sqrt(A**2+B**2)
	darc=d*pixel
	msk = darc < width/2.
	
	return msk


data=np.zeros((300,300))
y,x=np.indices(data.shape)
pv_array((x,y),10,pa=41,eps=0.5,x0=145,y0=145,pixel=1)




