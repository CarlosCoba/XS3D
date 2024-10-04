import numpy as np
from src0.pixel_params import Rings


def kincenter(mom1_map,xc,yc):

	vel_map = np.copy(mom1_map)
	vel_map[np.isnan(vel_map)]=0	
	x0,y0 = int(xc),int(yc)
	# first guess of systemic velocity
	vsys_x0y0=vel_map[y0][x0]
	
	[ny,nx]=vel_map.shape

	x = np.arange(0, nx, 1)
	y = np.arange(0, ny, 1)
	XY_mesh = np.meshgrid(x,y,sparse=True)	
		
	bmaj,bmin=nx//2,ny//2
	if bmin>bmaj:
		bmaj,bmin=ny//2,nx//2
		
	eps=1-bmin/bmaj
	r=Rings(XY_mesh,0,eps,x0,y0,1)
	rmax=bmaj/6
	rsearch=(r<rmax) & (vel_map!=0)
	vsearch=vel_map[rsearch]
	
	vsys=np.nanpercentile(np.unique(vsearch),50)
	if vsys==0:
		vsys=vsys_x0y0
		if vsys==0:
			vsys = np.nanmean(np.unique(vel_map))
			
	return vsys
