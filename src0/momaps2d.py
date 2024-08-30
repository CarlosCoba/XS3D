import numpy as np
from stropy.io import fits
from tertools import product
from omtools import trapecium
import matplotlib.pylab as plt
from constants import __c__, __sigma_to_FWHM__, __indices__, _INDICES_POS



def maps(cube,crval,cdelt,h,eline=None):
	header=Header_info(h)

	[nz,ny,nx]=cube.shape
	wave_cover = crval + cdelt*np.arange(nz)
	I0=np.zeros((ny,nx))
	I1=np.zeros((ny,nx))
	I2=np.zeros((ny,nx))
	for j,i in product(np.arange(ny),np.arange(nx)):
		flux_k = cube[:,j,i]
		x0,x1=wave_cover[0],wave_cover[-1]
		I0_k=trapecium(flux_k,x0,x1,cdelt)
		I1_k=trapecium(flux_k*wave_cover,x0,x1,cdelt)/I0_k
		I2_k=trapecium(flux_k*(wave_cover-I1_k)**2,x0,x1,cdelt)/I0_k
		
		I0[j][i]=I0_k				
		I1[j][i]=I1_k				
		I2[j][i]=I2_k
	
	I1=I1 if eline is None else __c__*(I1-eline)/eline
	I2=np.sqrt(I2)
	I2=I2 if eline is None else __c__*(I2)/eline
	fwhm=np.sqrt(8*np.log(2))*I2		
	return I0,I1,I2,fwhm				
	
	
data,h=fits.getdata("/home/carlos/Documents/Teacup_sartori/Teacup_sartori.halpha.cube.fits",header=True)
crval=h['CRVAL3']
cdelt=h['CDELT3']
[I0,I1,I2,fwhm]=maps(data,crval,cdelt,6562.68)


plt.imshow(I0,origin='lower')
plt.show()

plt.imshow(I2,origin='lower',vmin=50,vmax=300)
plt.show()


plt.imshow(I1,origin='lower',vmin=25200,vmax=25700)
plt.show()
