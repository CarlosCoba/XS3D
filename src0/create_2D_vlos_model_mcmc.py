import numpy as np
import matplotlib.pylab as plt
import scipy


from src0.weights_interp import weigths_w
from src0.kin_components import CIRC_MODEL
from src0.kin_components import RADIAL_MODEL
from src0.kin_components import BISYM_MODEL
from src0.kin_components import HARMONIC_MODEL
from src0.pixel_params import Rings, v_interp
from src0.create_dataset import dataset_to_2D
from src0.kin_components import cs_k_add_zeros

class KinModel :
	def __init__(self, vel_map, evel_map, theta_ls, vmode, ring_pos, ring_space, pixel_scale, inner_interp, pdf, m_hrm, n_circ, n_noncirc, shape, only_model=False ):

		self.vel_map = vel_map
		self.evel_map = evel_map

		self.vmode  =  vmode
		self.shape = shape
		[self.ny,self.nx] = shape
		self.rings_pos = ring_pos
		self.ring_space = ring_space
		self.nrings = len(self.rings_pos)
		self.n_annulus = self.nrings - 1
		self.pixel_scale = pixel_scale
		self.v_center = inner_interp
		self.index_v0 = -1
		self.n_circ = n_circ 
		self.n_noncirc = n_noncirc 
		self.m = self.n_circ
		self.theta0 = np.copy(theta_ls)
		self.pdf = pdf
		self.m_hrm = int(m_hrm)
		self.only_model = only_model

		X = np.arange(0, self.nx, 1)
		Y = np.arange(0, self.ny, 1)
		self.XY_mesh = np.meshgrid(X,Y)

	def expand_theta(self, theta):
			vrot =  theta[:self.n_circ]
			self.Vrot = np.asarray(vrot)
			if self.v_center != 0: self.Vrot = np.append(self.Vrot, 0)
			
			if self.vmode == "circular":
				self.Vrad = self.Vrot*0
				self.Vtan = self.Vrot*0
				phi_b = 45

			if self.vmode == "radial":
				self.m = self.n_circ + 1*self.n_noncirc
				vrad = theta[self.n_circ:self.m]

				self.Vrad = self.Vrot*0
				self.Vrad[:self.n_noncirc] = vrad
				self.Vtan = self.Vrad*0
				phi_b = 45

			if self.vmode == "bisymmetric":
				self.m = self.n_circ + 2*self.n_noncirc
				vrad,vtan =  theta[self.n_circ:self.n_circ+self.n_noncirc],theta[self.n_circ+self.n_noncirc:self.m]
				self.Vrad = self.Vrot*0
				self.Vtan = self.Vrot*0
				self.Vrad[:self.n_noncirc] = vrad
				self.Vtan[:self.n_noncirc] = vtan


			self.pa, self.eps, self.x0, self.y0, self.Vsys = theta[self.m],theta[self.m+1],theta[self.m+2],theta[self.m+3],theta[self.m+4]
			if self.vmode == "bisymmetric": self.phi_b = theta[self.m+5]

			if "hrm" in self.vmode:
				self.m = self.n_circ + self.n_noncirc*(2*self.m_hrm-1)
				self.pa, self.eps, self.x0, self.y0, self.Vsys = theta[-5:]

			if "hrm" in self.vmode:
				# C_flat and S_flat samples
				C_flat,S_flat= theta[:self.n_circ+(self.m_hrm-1)*self.n_noncirc],theta[self.n_circ+(self.m_hrm-1)*self.n_noncirc:self.n_circ+(2*self.m_hrm-1)*self.n_noncirc]
				# Adding zeros to C, S
				self.C_k, self.S_k = cs_k_add_zeros(C_flat,S_flat,self.m_hrm,self.n_circ,self.n_noncirc)

			self.PA, self.EPS, self.XC, self.YC = (self.theta0)[self.m],(self.theta0)[self.m+1],(self.theta0)[self.m+2],(self.theta0)[self.m+3]
			#self.PA, self.EPS, self.XC, self.YC = self.pa, self.eps, self.x0, self.y0
			self.r_n = Rings(self.XY_mesh,self.PA*np.pi/180,self.EPS,self.XC,self.YC,self.pixel_scale)
	

	def kinmdl_dataset(self, pars, i, xy_mesh, r_space, r_0 = None):

		# For inner interpolation
		r1, r2 = self.rings_pos[0], self.rings_pos[1]
		#if "hrm" not in self.vmode and self.v_center == "extrapolate" and i == self.index_v0:
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
					v1, v2 = self.C_k[k][0], self.C_k[k][1]
					v_int =  v_interp(0, r2, r1, v2, v1 )
					self.C_k[k][self.index_v0] = v_int

					v1, v2 = self.S_k[k][0], self.S_k[k][1]
					v_int =  v_interp(0, r2, r1, v2, v1 )
					self.S_k[k][self.index_v0] = v_int


		if "hrm" not in self.vmode:
			Vrot = self.Vrot[i]
		if self.vmode == "circular":
			modl = (CIRC_MODEL(xy_mesh,Vrot,self.pa,self.eps,self.x0,self.y0))*weigths_w(xy_mesh,self.pa,self.eps,self.x0,self.y0,r_0,r_space,pixel_scale=self.pixel_scale)
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


	def interp_model(self, theta ):
			# update theta
			self.expand_theta(theta)
			"""
			Analysis of r > r0
			"""
			self.interp_model_r = dataset_to_2D([self.ny,self.nx], self.n_annulus, self.rings_pos, self.r_n, self.XY_mesh, self.kinmdl_dataset, self.vmode, self.v_center, None, self.index_v0)

			mask_inner = np.where( (self.r_n < self.rings_pos[0] ) )
			x_r0,y_r0 = self.XY_mesh[0][mask_inner], self.XY_mesh[1][mask_inner] 
			r_space_0 = self.rings_pos[0]


			"""
			Analysis of r < r0
			"""
			#(a) velocity rises linearly from zero: r1 = 0, v1 = 0
			if self.v_center == 0 or (self.v_center != "extrapolate" and self.vmode != "circular"):
				V_xy_mdl = self.kinmdl_dataset(None, 0, (x_r0,y_r0), r_0 = 0, r_space = r_space_0)
				v_new_2 = V_xy_mdl[1]
				self.interp_model_r[mask_inner] = v_new_2

			else:

				r2 = self.rings_pos[0] 		# ring posintion
				v1_index = self.index_v0	# index of velocity
				V_xy_mdl = self.kinmdl_dataset(None, v1_index, (x_r0,y_r0), r_0 = r2, r_space = r_space_0 )
				v_new_1 = V_xy_mdl[0]

				r1 = 0 					# ring posintion
				v2_index = 0			# index of velocity
				V_xy_mdl = self.kinmdl_dataset(None, v2_index, (x_r0,y_r0), r_0 = r1, r_space = r_space_0 )
				v_new_2 = V_xy_mdl[1]

				v_new = v_new_1 + v_new_2
				self.interp_model_r[mask_inner] = v_new

			#self.interp_model_r[self.interp_model_r == 0]  = np.nan
			#self.interp_model_r = self.interp_model_r + self.Vsys
			mask = self.interp_model_r == 0
			self.interp_model_r[~mask]  = self.interp_model_r[~mask] + self.Vsys

			if self.only_model:
				return self.interp_model_r

			residual = self.vel_map - self.interp_model_r
			mask_ones = residual/residual
			mask_finite = np.isfinite(mask_ones)
			mask_finite = mask_finite*(~mask)

			mdl = self.interp_model_r[mask_finite]
			obs = self.vel_map[mask_finite]
			err = self.evel_map[mask_finite]
			return mdl, obs, err
