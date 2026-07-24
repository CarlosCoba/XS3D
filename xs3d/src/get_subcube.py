import numpy as np
import matplotlib.pylab as plt
from .start_messenge import Print


def sub_mask3D(cube, mom, config, f = 0.02, plot = False):

	P=Print()
			
	config_general = config['general']
	get_subcube = config_general.getboolean('subcube',1)

	xy_shift = [0,0]
	slices2D = tuple([slice(None,None) for k in range(2)])	
	slices3D = tuple([slice(None,None,None) for k in range(3)])	
	if not get_subcube:
		return cube,xy_shift,slices2D,slices3D

	shape_ori = cube.shape
	[nz, ny_ori, nx_ori] = shape_ori
	
	mask = mom != 0
	ntotx = ny_ori
	ntoty = nx_ori

	nx_frac = np.sum(mask, axis = 0) / ntotx
	ny_frac = np.sum(mask, axis = 1) / ntoty
	
	f1 = np.std(nx_frac)*0.5
	f2 = np.std(ny_frac)*0.5
	
	indx_x = np.argwhere(nx_frac>f1)
	indx_y = np.argwhere(ny_frac>f2)

	x1,x2 = min(indx_x)[0], max(indx_x)[0]
	y1,y2 = min(indx_y)[0], max(indx_y)[0]
	
	if np.any([x1==x2,y1==y2]):
		return cube,xy_shift,slices2D,slices3D
					
	newcube = cube[:,y1:y2, x1:x2]
	
	xshift, yshift = -x1, -y1

	xy_shift = [xshift, yshift]
		
	slices2D = tuple([slice(y1,y2), slice(x1,x2)])
	
	slices3D = tuple([slice(None,None,None),slice(y1,y2,None),slice(x1,x2,None)])
	
	Nori =  ny_ori * nx_ori
	Nnew = (y2-y1)*(x2-x1)
	
	f_new = round(100*(1 - (Nnew/Nori)),2)
	P.status(f'The input cube was reduced by {f_new}%')
	
	if plot:
		mom_t = mom[slices2D]
		fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
		ax1.imshow(np.log10(mom), origin = 'lower')
		ax2.imshow(np.log10(mom_t), origin = 'lower')
		ax1.set_title('Original')		
		ax2.set_title('Cropped')			
		plt.show()
	
	return newcube, xy_shift, slices2D, slices3D
