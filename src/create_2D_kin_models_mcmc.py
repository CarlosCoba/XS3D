import numpy as np
import matplotlib.pylab as plt
import scipy

from src.weights_interp import weigths_w
from src.kin_components import CIRC_MODEL
from src.kin_components import RADIAL_MODEL
from src.kin_components import BISYM_MODEL
from src.kin_components import HARMONIC_MODEL
from src.pixel_params import Rings, v_interp
from src.create_dataset import dataset_to_2D



class bidi_models:
	""" This class creates the 2D maps of the different kinematic componets. I performs a 1D linear interpolation between the best fit values """
	def __init__(self,vmode,shape,V_k, pa, inc, x0, y0, Vsys, ring_pos, ring_space,pixel_scale, inner_interp, m_hrm = 1, phi_b = None):


		self.vmode  =  vmode
		self.shape = shape
		self.pa, self.eps, self.x0, self.y0, self.Vsys =pa, inc, x0, y0, Vsys
		self.rings_pos = ring_pos
		self.ring_space = ring_space
		self.nrings = len(self.rings_pos)
		self.n_annulus = self.nrings - 1
		self.pixel_scale = pixel_scale
		self.phi_b = phi_b
		self.V = V_k
		self.m_hrm = m_hrm
		self.v_center = inner_interp

		if "hrm" not in self.vmode:
			self.Vrot, self.Vrad, self.Vtan = V_k
		else:
			self.m_hrm = int(m_hrm)
			self.m2_hrm = int(2*m_hrm)
			self.C_k, self.S_k = V_k

		[self.ny, self.nx] = shape
		X = np.arange(0, self.nx, 1)
		Y = np.arange(0, self.ny, 1)
		self.XY_mesh = np.meshgrid(X,Y)
		self.r_n = Rings(self.XY_mesh,self.pa*np.pi/180,self.eps,self.x0,self.y0,pixel_scale)


	def kinmdl_dataset(self, pars, V_i, xy_mesh, r_space, r_0 ):
		modl = V_i*np.ones(len(xy_mesh[0]),)*weigths_w(xy_mesh,self.pa,self.eps,self.x0,self.y0,r_0,r_space,pixel_scale=self.pixel_scale)
		return modl


	def mdl(self,V_i, v_t = False):

			interp_model = dataset_to_2D([self.ny,self.nx], self.n_annulus, self.rings_pos, self.r_n, self.XY_mesh, self.kinmdl_dataset, self.vmode, self.v_center, None, None, V_i = V_i)

			"""
			Analysis of the inner radius

			"""
			mask_inner = np.where( (self.r_n < self.rings_pos[0] ) )
			x_r0,y_r0 = self.XY_mesh[0][mask_inner], self.XY_mesh[1][mask_inner] 
			r_space_0 = self.rings_pos[0]

			# interpolation between rings requieres two velocities: v1 and v2
			#V_new 	= v1*(r2-r)/(r2-r1) + v2*(r-r1)/(r2-r1)
			#		= v1*(r2-r)/delta_R + v2*(r-r1)/delta_R
			#		= V(v1,r2) + V(v2,r1)

			#(a) velocity rises linearly from zero: r1 = 0, v1 = 0
			if self.v_center == 0 or (self.v_center != "extrapolate" and self.vmode != "circular"):
				r2, r1 = r_space_0, 0
				v1, v2 = 0, V_i[0]  
				V_xy_mdl = self.kinmdl_dataset(None, v2, (x_r0,y_r0), r_0 = r1, r_space = r_space_0)
				v_new_2 = V_xy_mdl[1]
				interp_model[mask_inner] = v_new_2

			else:

				r2,r1 = self.rings_pos[1],self.rings_pos[0]
				v2, v1 =  V_i[1], V_i[0]
				v0 = v_interp(r=0, r2=r2, r1=r1, v2=v2, v1=v1 )

				V_xy_mdl = self.kinmdl_dataset( None, v0, (x_r0,y_r0), r_0 = r1, r_space = r_space_0 )
				v_new_1 = V_xy_mdl[0]

				r0 = 0
				v1 = V_i[0]
				V_xy_mdl = self.kinmdl_dataset( None, v1, (x_r0,y_r0), r_0 = r0, r_space = r_space_0 )
				v_new_2 = V_xy_mdl[1]
				v_new = v_new_1 + v_new_2
				interp_model[mask_inner] = v_new

			interp_model[interp_model == 0]  = np.nan
			return interp_model





	def interp(self):
			if self.vmode == "circular":

				Vcirc_2D = self.mdl(self.Vrot, v_t = True)
				Vrad_2D = 0
				Vtan_2D = 0


			if self.vmode == "radial":

				Vcirc_2D = self.mdl(self.Vrot, v_t = True)
				Vrad_2D = self.mdl(self.Vrad)
				Vtan_2D = 0


			if self.vmode == "bisymmetric":

				Vcirc_2D = self.mdl(self.Vrot, v_t = True)
				Vrad_2D = self.mdl(self.Vrad)
				Vtan_2D = self.mdl(self.Vtan)


			if "hrm" in self.vmode:
				C = [self.mdl(self.C_k[k]) for k in range(self.m_hrm)]
				S = [self.mdl(self.S_k[k]) for k in range(self.m_hrm)]

			if "hrm" not in self.vmode:
				return Vcirc_2D, Vrad_2D, Vtan_2D, self.r_n
			else:
				return C, S, self.r_n








