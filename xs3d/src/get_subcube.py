import numpy as np
import matplotlib.pylab as plt
from .start_messenge import Print


def sub_mask3D(cube, mom, config, psf_lsf, f = 0.02, plot = False):

	P=Print()
			
	config_general = config['general']
	f_trim_tmp	= config_general.getfloat('f_trim',0.5)
	f_trim		= np.clip(f_trim_tmp,0,1)
	psf_pix		= psf_lsf.fwhm_psf_pix

	xy_shift = [0,0]
	slices2D = tuple([slice(None,None) for k in range(2)])	
	slices3D = tuple([slice(None,None,None) for k in range(3)])	
	if f_trim == 0:
		return cube,xy_shift,slices2D,slices3D

	shape_ori = cube.shape
	[nz, ny_ori, nx_ori] = shape_ori
	
	mask = mom != 0
	ntotx = ny_ori
	ntoty = nx_ori

	nx_frac = np.sum(mask, axis = 0) / ntotx
	ny_frac = np.sum(mask, axis = 1) / ntoty
	
	f1 = np.std(nx_frac)*f_trim
	f2 = np.std(ny_frac)*f_trim
	
	indx_x = np.argwhere(nx_frac>f1)
	indx_y = np.argwhere(ny_frac>f2)

	pad	=	int(psf_pix)
	x1,x2 = int(min(indx_x)[0]-pad), int(max(indx_x)[0]+pad+1) # Index. We need to add 1 cause `slice` last indice is exclusive.
	y1,y2 = int(min(indx_y)[0]-pad), int(max(indx_y)[0]+pad+1) # Index. We need to add 1 cause `slice` last indice is exclusive.
	x1,x2 = np.clip(x1,0,None),np.clip(x2,None,nx_ori) # Index
	y1,y2 = np.clip(y1,0,None),np.clip(y2,None,ny_ori) # Index
	
	if np.any([x1==x2,y1==y2]):
		return cube,xy_shift,slices2D,slices3D

	new_slice = tuple([slice(None,None,None),slice(y1,y2,None), slice(x1,x2,None)])					
	newcube = cube[new_slice]
	msk 	= np.ones(newcube.shape[1:])
	pad0	= pad
	msk[0:pad0,:]=0;msk[-pad0:,:]=0;msk[:,0:pad0]=0;msk[:,-pad0:]=0
	
	#[_,nyy, nxx] = newcube.shape
	#plt.imshow(msk, origin='lower', extent=[0,nxx,0,nyy]);plt.grid(True);plt.show()
	newcube *= msk.astype(float)

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
		yc,xc = ny_ori/2, nx_ori/2
		fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
		ax1.imshow(np.log10(mom), origin = 'lower')
		ax1.plot(xc,yc, 'xk')
		ax2.imshow(np.log10(mom_t), origin = 'lower')
		ax2.plot(xc+xshift,yc+yshift, 'xk')		
		ax1.set_title('Original')		
		ax2.set_title('Cropped')			
		plt.show()
	
	return newcube, xy_shift, slices2D, slices3D
