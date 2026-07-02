import os
import numpy as np
from itertools import product
import matplotlib.pylab as plt
	
def set_threads(ncores=None):
	if ncores is None:
		ncores=1
	ncores=int(ncores)
	os.environ["MKL_NUM_THREADS"] = str(ncores)
	os.environ["NUMEXPR_NUM_THREADS"] = str(ncores) 
	os.environ["OMP_NUM_THREADS"] = str(ncores)	


def parabola(x,y):
	#(x1,y1),(x2,y2),(x3,y3)=p1,p2,p3
	x1,x2,x3=x[:]
	y1,y2,y3=y[:]
			
	# f(x)=A*x**2 + B*x +c	
	denom = (x1 - x2)*(x1 - x3)*(x2 - x3)
	A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
	B = (x3**2 * (y1 - y2) + x2**2 * (y3 - y1) + x1**2 * (y2 - y3)) / denom
	C = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom

	# the vertex:
	xmax = -B/(2*A)
	ymax =  C-(B**2)/(4*A)

	#print(A,B,C)	
	return xmax, ymax

def vparabola3D(datacube,wave_cover_kms):
	[nz,ny,nx]=datacube.shape
	vpeak2D=np.zeros((ny,nx))
	vmx_indx = np.argmax(datacube,axis=0).astype(int)		
	for i,j in product(np.arange(nx),np.arange(ny)):
		k=vmx_indx[j][i]
		if k!=0 and k<(nz-1):
			y_axis=datacube[k-1:k+2,j,i]
			x_axis=wave_cover_kms[k-1:k+2]				
			vpara,_=parabola(x_axis,y_axis)
			vpeak2D[j][i]=vpara
	return vpeak2D
	
	
def zero2nan(data):
	data[data==0]=np.nan
	return data
	
def nan2zero(data):
	data[~np.isfinite(data)]=0
	return data
	
def vmin_vmax(data,pmin=2,pmax=99.5,base=None,symmetric =False,mask=None):
	if mask is not None:
		data = data[mask]
	vmin,vmax=np.nanpercentile(np.unique(data),pmin),np.nanpercentile(np.unique(data),pmax)
	vsym = (vmax+abs(vmin))*0.5
	if symmetric: vmin,vmax=-1*vsym,vsym
	if base is not None:
		vmin,vmax=(vmin//base+1)*base,(vmax//base+1)*base
		if symmetric: vmin,vmax=-1*(vsym//base+1)*base,(vsym//base+1)*base
	return vmin,vmax



def circmean(data, high=360, low=0):
	data = np.array(data)
	# manual computation to circmean, similar to from scipy import stats.circmean

	data = data % high
	# Map data to a 0 to 2*pi radian scale
	rad = (np.array(data) - low) * 2 * np.pi / (high - low)

	# Average the coordinates on the unit circle
	sin_mean = np.mean(np.sin(rad))
	cos_mean = np.mean(np.cos(rad))
	
	# Convert back to the original range
	mean_rad = np.arctan2(sin_mean, cos_mean)
	mean_wrapped = (mean_rad * (high - low) / (2 * np.pi)) + low
	
	# Ensure the result stays within bounds
	return (mean_wrapped - low) % (high - low) + low

	
