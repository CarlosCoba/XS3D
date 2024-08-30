import numpy as np
import matplotlib.pylab as plt
import scipy


from src0.weights_interp import weigths_w
from src0.kin_components import CIRC_MODEL
from src0.kin_components import RADIAL_MODEL
from src0.kin_components import BISYM_MODEL
from src0.kin_components import HARMONIC_MODEL
from src0.kin_components import SIGMA_MODEL
from src0.pixel_params import Rings, v_interp
from src0.create_dataset import dataset_to_2D



class best_2d_model:
	def __init__(self,vmode,shape,V_k, pa, eps, x0, y0, Vsys, ring_pos, ring_space, pixel_scale, inner_interp, m_hrm = 1, phi_b = None):
		self.vmode  =  vmode
		self.shape = shape
		[self.ny,self.nx] = shape
		self.pa, self.eps, self.x0, self.y0, self.Vsys =pa, eps, x0, y0, Vsys
		self.rings_pos = ring_pos
		self.ring_space = ring_space
		self.nrings = len(self.rings_pos)
		self.n_annulus = self.nrings - 1
		self.pixel_scale = pixel_scale
		self.phi_b = phi_b
		self.V = V_k
		self.m_hrm = m_hrm
		self.v_center = inner_interp
		self.index_v0 = -1


		if "hrm" not in self.vmode:
			self.Sig, self.Vrot, self.Vrad, self.Vtan = V_k
			self.Vrot, self.Vrad, self.Vtan = np.append(self.Vrot, -1e4), np.append(self.Vrad, -1e4), np.append(self.Vtan, -1e4)
		else:
			self.m_hrm = int(m_hrm)
			self.m2_hrm = int(2*m_hrm)
			self.C_k, self.S_k = [ V_k[k] for k in range(self.m_hrm) ], [ V_k[k] for k in range(self.m_hrm,self.m2_hrm) ]


		[ny, nx] = shape
		X = np.arange(0, nx, 1)
		Y = np.arange(0, ny, 1)
		self.XY_mesh = np.meshgrid(X,Y)
		self.r_n = Rings(self.XY_mesh,self.pa*np.pi/180,self.eps,self.x0,self.y0,pixel_scale)
		self.interp_model = np.zeros((ny,nx))


	def kinmdl_dataset(self, pars, i, xy_mesh, r_space, r_0 = None):

		# For inner interpolation
		r1, r2 = self.rings_pos[0], self.rings_pos[1]
		if "hrm" not in self.vmode and self.v_center != 0 and i == self.index_v0:
			if self.v_center == "extrapolate":					
				v1, v2 = self.Vrot[0], self.Vrot[1]
				v_int =  v_interp(0, r2, r1, v2, v1 )
				# This only applies to circular rotation
				# I made up this index.
				self.Vrot[self.index_v0] = v_int
				if self.vmode == "radial" or self.vmode == "bisymmetric":
					v1, v2 = self.Vrad[0], self.Vrad[1]
					v_int =  v_interp(0, r2, r1, v2, v1 )
					self.Vrad[self.index_v0] = v_int
				if self.vmode == "bisymmetric":
					v1, v2 = self.Vtan[0], self.Vtan[1]
					v_int =  v_interp(0, r2, r1, v2, v1 )
					self.Vtan[self.index_v0] = v_int
			else:
				# This only applies to Vt component in circ
				# I made up this index.
				if self.vmode == "circular":
					self.Vrot[self.index_v0] = self.v_center

		if  "hrm" in self.vmode and self.v_center == "extrapolate" and i == self.index_v0:

				for k in range(self.m_hrm):
					self.C_k[k],self.S_k[k] = np.append(self.C_k[k],-1e4),np.append(self.S_k[k],-1e4)
					v1, v2 = self.C_k[k][0], self.C_k[k][1]
					v_int =  v_interp(0, r2, r1, v2, v1 )
					self.C_k[k][self.index_v0] = v_int

					v1, v2 = self.S_k[k][0], self.S_k[k][1]
					v_int =  v_interp(0, r2, r1, v2, v1 )
					self.S_k[k][self.index_v0] = v_int


		if "hrm" not in self.vmode:
			Vrot = self.Vrot[i]
		if self.vmode == "circular":
			modl1 = (CIRC_MODEL(xy_mesh,Vrot,self.pa,self.eps,self.x0,self.y0))*weigths_w(xy_mesh,self.pa,self.eps,self.x0,self.y0,r_0,r_space,pixel_scale=self.pixel_scale)
			modl0 = (SIGMA_MODEL(xy_mesh,Vrot,self.pa,self.eps,self.x0,self.y0))*weigths_w(xy_mesh,self.pa,self.eps,self.x0,self.y0,r_0,r_space,pixel_scale=self.pixel_scale)
			return modl0,modl1		
		if self.vmode == "radial":
			Vrad = self.Vrad[i]
			modl = (RADIAL_MODEL(xy_mesh,Vrot,Vrad,self.pa,self.eps,self.x0,self.y0))*weigths_w(xy_mesh,self.pa,self.eps,self.x0,self.y0,r_0,r_space,pixel_scale=self.pixel_scale)
		if self.vmode == "bisymmetric":
			Vrad = self.Vrad[i]
			Vtan = self.Vtan[i]
			if Vrad != 0 and Vtan != 0:
				phi_b = self.phi_b
				modl = (BISYM_MODEL(xy_mesh,Vrot,Vrad,self.pa,self.eps,self.x0,self.y0,Vtan,phi_b))*weigths_w(xy_mesh,self.pa,self.eps,self.x0,self.y0,r_0,r_space,pixel_scale=self.pixel_scale)
			else:
				modl = (BISYM_MODEL(xy_mesh,Vrot,0,self.pa,self.eps,self.x0,self.y0,0,0))*weigths_w(xy_mesh,self.pa,self.eps,self.x0,self.y0,r_0,r_space,pixel_scale=self.pixel_scale)
		if "hrm" in self.vmode:
			c_k, s_k  = [self.C_k[k][i] for k in range(self.m_hrm)] , [self.S_k[k][i] for k in range(self.m_hrm)]
			modl = (HARMONIC_MODEL(xy_mesh,c_k, s_k,self.pa,self.eps,self.x0,self.y0, self.m_hrm))*weigths_w(xy_mesh,self.pa,self.eps,self.x0,self.y0,r_0,r_space,pixel_scale=self.pixel_scale)

		return modl


	def model2D(self):

			self.interp_model = dataset_to_2D([self.ny,self.nx], self.n_annulus, self.rings_pos, self.r_n, self.XY_mesh, self.kinmdl_dataset, self.vmode, self.v_center, None, self.index_v0, nmodls=1)

			mask_inner = np.where( (self.r_n < self.rings_pos[0] ) )
			x_r0,y_r0 = self.XY_mesh[0][mask_inner], self.XY_mesh[1][mask_inner] 
			r_space_0 = self.rings_pos[0]


			"""
			Analysis of the inner radius

			"""
			#(a) velocity rises linearly from zero: r1 = 0, v1 = 0
			if self.v_center == 0 or (self.v_center != "extrapolate" and self.vmode != "circular"):
				V_xy_mdl = self.kinmdl_dataset(None, 0, (x_r0,y_r0), r_0 = 0, r_space = r_space_0)[1]
				v_new_2 = V_xy_mdl[1]
				self.interp_model[mask_inner] = v_new_2

			else:

				r2 = self.rings_pos[0] 		# ring posintion
				v1_index = self.index_v0	# index of velocity
				V_xy_mdl = self.kinmdl_dataset(None, v1_index, (x_r0,y_r0), r_0 = r2, r_space = r_space_0 )[1]
				v_new_1 = V_xy_mdl[0]

				r1 = 0 					# ring posintion
				v2_index = 0			# index of velocity
				V_xy_mdl = self.kinmdl_dataset(None, v2_index, (x_r0,y_r0), r_0 = r1, r_space = r_space_0 )[1]
				v_new_2 = V_xy_mdl[1]

				v_new = v_new_1 + v_new_2
				self.interp_model[mask_inner] = v_new

			self.interp_model[self.interp_model == 0]  = np.nan
			self.interp_model = self.interp_model + self.Vsys

			return self.interp_model



