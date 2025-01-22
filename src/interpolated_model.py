import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pylab as plt
from src.pixel_params import eps_2_inc
import warnings
warnings.filterwarnings('ignore')


def model(vmode,nx,ny,pa,eps,Vsys,x0,y0,R,V,phi_b=0,pixel_scale=1):


	Vrot, Vrad, Vtan = V

	R = R/pixel_scale
	PA,inc=(pa)*np.pi/180,eps_2_inc(eps)

	X = np.arange(0, nx, 1)
	Y = np.arange(0, ny, 1)
	XY_mesh = np.meshgrid(X,Y,sparse=True)
	(x,y) = XY_mesh

	X = (- (x-x0)*np.sin(PA) + (y-y0)*np.cos(PA))
	Y = (- (x-x0)*np.cos(PA) - (y-y0)*np.sin(PA))



	R_mesh = np.sqrt(X**2+(Y/(1-eps))**2)
	R_mesh = R_mesh



	max_R = np.nanmax(R)
	R_mesh[R_mesh> max_R] = np.nan

	R_list = R.tolist()
	R_list.append(0)



	# We need to include R = 0, and V = 0

	if vmode == "circular":

		Vrot_list = Vrot.tolist()
		Vrot_list.append(0)

	if vmode == "radial":
		Vrot_list = Vrot.tolist()
		Vrad_list = Vrad.tolist()
		Vrot_list.append(0)
		Vrad_list.append(0)

	if vmode == "bisymmetric":
		Vrot_list = Vrot.tolist()
		Vrad_list = Vrad.tolist()
		Vtan_list = Vtan.tolist()
		Vrot_list.append(0)
		Vrad_list.append(0)
		Vtan_list.append(0)



	cos_tetha = (- (x-x0)*np.sin(PA) + (y-y0)*np.cos(PA))/R_mesh
	sin_tetha = (- (x-x0)*np.cos(PA) - (y-y0)*np.sin(PA))/((1-eps)*R_mesh)
	theta = np.arctan(sin_tetha/cos_tetha)


	# Interpolate velocities

	if vmode == "circular":
		f0 = interp1d(R_list, Vrot_list)
		Vrot_new = f0(R_mesh)


		vlos = 0*Vsys+np.sin(inc)*Vrot_new*cos_tetha
	if vmode == "radial":

		f0 = interp1d(R_list, Vrot_list)
		f1 = interp1d(R_list, Vrad_list)
		Vrot_new = f0(R_mesh)
		Vrad_new = f1(R_mesh)

		vlos = 0*Vsys+np.sin(inc)*(Vrot_new*cos_tetha + Vrad_new*sin_tetha)
	if vmode == "bisymmetric":

		f0 = interp1d(R_list, Vrot_list)
		f1 = interp1d(R_list, Vrad_list)
		Vrot_new = f0(R_mesh)
		Vrad_new = f1(R_mesh)
		f2 = interp1d(R_list, Vtan_list)
		Vtan_new = f2(R_mesh)

		#plt.scatter(R_mesh,Vrot_new, s =5)	
		#plt.scatter(R_mesh,Vrad_new, s =5)	
		#plt.scatter(R_mesh,Vtan_new, s =5)	
		#plt.show()


		m = 2
		theta_b = theta - phi_b
		vlos = 0*Vsys+np.sin(inc)*((Vrot_new*cos_tetha) - Vtan_new*np.cos(m*theta_b)*cos_tetha - Vrad_new*np.sin(m*theta_b)*sin_tetha)



	return vlos


