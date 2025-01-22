import numpy as np
from astropy.io import fits
from src.pixel_params import Rings,eps_2_inc,inc_2_eps
from src.kin_components import AZIMUTHAL_ANGLE


def save_2d_models(galaxy,vmode,kin_3D_mdls,PA,INC,XC,YC,VSYS,m_hrm,out = False):
	mom0_mdl,mom1_mdl,mom2_mdl_kms,mom2_mdl_A,cube_mdl,velmap_intr,sigmap_intr,twoDmodels=kin_3D_mdls
			
	if "hrm" in vmode:
		C_k,S_k=[],[]
		for k in range(m_hrm):			
			ck,sk=twoDmodels[k],twoDmodels[k+m_hrm]
			C_k.append(ck)
			S_k.append(sk)			

	if "hrm" not in vmode:
		VSYS_str = "VSYS"
	else:
		VSYS_str = "C0"

	hdu = fits.PrimaryHDU()	
	hdu.header['PA'] = PA
	hdu.header['INC'] = INC
	hdu.header['XC'] = XC
	hdu.header['YC'] = YC	
	hdu.header['%s'%VSYS_str] = VSYS
	hdu.header['NAME0'] = '2D azimuthal angle / rad'	
	hdu.header['NAME1'] = '2D deprojected radius / arcs'


	[ny,nx]=mom0_mdl.shape
	X = np.arange(0, nx, 1)
	Y = np.arange(0, ny, 1)
	xy_mesh = np.meshgrid(X,Y)
	(x,y) = xy_mesh
	pa,eps=PA*np.pi/180,inc_2_eps(INC)
	Rn  = Rings(xy_mesh,pa,eps,XC,YC)
	
	theta,_ = AZIMUTHAL_ANGLE([ny,nx], PA, eps, XC, YC)
	nz=2+len(twoDmodels)
	data=np.zeros((nz,ny,nx))	
	data[0]=Rn
	data[1]=theta


	if 'hrm' not in vmode:
		hdu.header['NAME2'] = '2D circular velocity / km/s'
		data[2]=twoDmodels[0]		
		
		if vmode in ['radial','bisymmetric']:
			hdu.header['NAME3'] = '2D radial velocity / km/s'
			data[3]=twoDmodels[1]
		if vmode == 'bisymmetric':
			hdu.header['NAME4'] = '2D tangential velocity / km/s'	
			data[4]=twoDmodels[2]	
	
	else:		
		l=2
		for k in range(1,m_hrm+1):
			datas = C_k[k-1]
			hdu.header[f'NAME{l}'] = '2D C%s model'%k
			data[l]=datas
			l=l+1
		for k in range(1,m_hrm+1):
			datas = S_k[k-1]
			hdu.header[f'NAME{l}'] = '2D S%s model'%k
			data[l]=datas
			l=l+1			


	hdu.data=data
	hdu.writeto("%smodels/%s.%s.2D_depro.fits.gz"%(out,galaxy,vmode),overwrite=True)

	return None
