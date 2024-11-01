import numpy as np
import matplotlib.pylab as plt
from itertools import product,chain

from src0.weights_interp import weigths_w
from src0.kin_components import CIRC_MODEL
from src0.kin_components import HARMONIC_MODEL
from src0.kin_components import SIGMA_MODEL
from src0.kin_components import AZIMUTHAL_ANGLE,SIN_COS
from src0.pixel_params import Rings, v_interp,eps_2_inc
from src0.create_dataset import dataset_to_2D
from src0.convolve_cube import Cube_creation

from src0.momtools import GaussProf,trapecium
from src0.constants import __c__
from src0.conv import conv2d,gkernel,gkernel1d
from src0.conv_spec1d import gaussian_filter1d,convolve_sigma


class best_3d_model:
	def __init__(self,mommaps_obs,cube,hdr,config,vmode,V_k, pa, eps, x0, y0, Vsys, ring_pos, ring_space, pixel_scale, inner_interp, m_hrm = 1, phi_b = None, nVk=2):
	
	
		[self.mom0,self.mom1,self.mom2]=mommaps_obs
		self.mommaps_obs=mommaps_obs	
		self.datacube=cube
		self.h=hdr
		self.vmode  =  vmode
		self.pa, self.eps, self.x0, self.y0, self.Vsys =pa, eps, x0, y0, Vsys
		self.rings_pos = ring_pos
		self.ring_space = ring_space
		self.nrings = len(self.rings_pos)
		self.n_annulus = self.nrings - 1
		self.pixel_scale = pixel_scale
		self.phi_b = phi_b
		self.alpha = phi_b
		self.V = V_k
		self.m_hrm = m_hrm
		self.v_center = inner_interp
		self.index_v0 = -1
		self.Vk=nVk


		if "hrm" not in self.vmode:
			self.Sig, self.Vrot, self.Vrad, self.Vtan = V_k
			self.Sig, self.Vrot, self.Vrad, self.Vtan = np.append(self.Sig, -1e4), np.append(self.Vrot, -1e4), np.append(self.Vrad, -1e4), np.append(self.Vtan, -1e4)
		else:
			self.m_hrm = int(m_hrm)
			self.m2_hrm = int(2*m_hrm)
			self.C_k, self.S_k = [ V_k[k] for k in range(self.m_hrm) ], [ V_k[k] for k in range(self.m_hrm,self.m2_hrm) ]
			self.Sig = V_k[-1]
			self.Sig = np.append(self.Sig, -1e4)

		[self.nz, self.ny, self.nx] = self.datacube.shape
		X = np.arange(0, self.nx, 1)
		Y = np.arange(0, self.ny, 1)
		self.XY_mesh = np.meshgrid(X,Y)
		self.r_n = Rings(self.XY_mesh,self.pa*np.pi/180,self.eps,self.x0,self.y0,pixel_scale)
		self.cube_modl = Cube_creation(self.datacube,self.h,self.mommaps_obs,config)
		
	def kinmdl_dataset(self, pars, i, xy_mesh, r_space, r_2 = None, disp = False):

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

		# Dispersion and Vz are always extrapolated
		s1, s2 = self.Sig[0], self.Sig[1]
		s_int =  v_interp(0, r2, r1, s2, s1 )
		self.Sig[self.index_v0] = s_int

		if self.vmode == 'vertical':	
			vz1, vz2 = self.Vrad[0], self.Vrad[1]
			vz_int =  v_interp(0, r2, r1, vz2, vz1 )
			self.Vrad[self.index_v0] = vz_int		

		Sig = self.Sig[i]
		modl0 = (SIGMA_MODEL(xy_mesh,Sig,self.pa,self.eps,self.x0,self.y0))*weigths_w(xy_mesh,self.pa,self.eps,self.x0,self.y0,r_2,r_space,pixel_scale=self.pixel_scale)
		if disp:
			return modl0
						

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
			v1 = (SIGMA_MODEL(xy_mesh,Vrot,self.pa,self.eps,self.x0,self.y0))*weigths_w(xy_mesh,self.pa,self.eps,self.x0,self.y0,r_2,r_space,pixel_scale=self.pixel_scale)
			return modl0,v1		
		if self.vmode == "radial" or self.vmode == 'vertical':
			Vrad = self.Vrad[i]
			v1 = (SIGMA_MODEL(xy_mesh,Vrot,self.pa,self.eps,self.x0,self.y0))*weigths_w(xy_mesh,self.pa,self.eps,self.x0,self.y0,r_2,r_space,pixel_scale=self.pixel_scale)			
			v2 = (SIGMA_MODEL(xy_mesh,Vrad,self.pa,self.eps,self.x0,self.y0))*weigths_w(xy_mesh,self.pa,self.eps,self.x0,self.y0,r_2,r_space,pixel_scale=self.pixel_scale)
			return modl0,v1,v2
		if self.vmode == 'ff':
			p = self.Vrad[i] !=0
			Vrad = Vrot*p
			v1 = (SIGMA_MODEL(xy_mesh,Vrot,self.pa,self.eps,self.x0,self.y0))*weigths_w(xy_mesh,self.pa,self.eps,self.x0,self.y0,r_2,r_space,pixel_scale=self.pixel_scale)			
			v2 = (SIGMA_MODEL(xy_mesh,Vrad,self.pa,self.eps,self.x0,self.y0))*weigths_w(xy_mesh,self.pa,self.eps,self.x0,self.y0,r_2,r_space,pixel_scale=self.pixel_scale)
			return modl0,v1,v2					
		if self.vmode == "bisymmetric":
			Vrad = self.Vrad[i]
			Vtan = self.Vtan[i]
			if Vrad != 0 and Vtan != 0:
				v1 = (SIGMA_MODEL(xy_mesh,Vrot,self.pa,self.eps,self.x0,self.y0))*weigths_w(xy_mesh,self.pa,self.eps,self.x0,self.y0,r_2,r_space,pixel_scale=self.pixel_scale)			
				v2 = (SIGMA_MODEL(xy_mesh,Vrad,self.pa,self.eps,self.x0,self.y0))*weigths_w(xy_mesh,self.pa,self.eps,self.x0,self.y0,r_2,r_space,pixel_scale=self.pixel_scale)				
				v3 = (SIGMA_MODEL(xy_mesh,Vtan,self.pa,self.eps,self.x0,self.y0))*weigths_w(xy_mesh,self.pa,self.eps,self.x0,self.y0,r_2,r_space,pixel_scale=self.pixel_scale)							
			else:
				v1 = (SIGMA_MODEL(xy_mesh,Vrot,self.pa,self.eps,self.x0,self.y0))*weigths_w(xy_mesh,self.pa,self.eps,self.x0,self.y0,r_2,r_space,pixel_scale=self.pixel_scale)
				v2=v1*0
				v3=v1*0								
			return modl0,v1,v2,v3
		if "hrm" in self.vmode:
			c_k, s_k  = [self.C_k[k][i] for k in range(self.m_hrm)] , [self.S_k[k][i] for k in range(self.m_hrm)]
			Ck = [(SIGMA_MODEL(xy_mesh,ck,self.pa,self.eps,self.x0,self.y0))*weigths_w(xy_mesh,self.pa,self.eps,self.x0,self.y0,r_2,r_space,pixel_scale=self.pixel_scale) for ck in c_k]
			Sk = [(SIGMA_MODEL(xy_mesh,sk,self.pa,self.eps,self.x0,self.y0))*weigths_w(xy_mesh,self.pa,self.eps,self.x0,self.y0,r_2,r_space,pixel_scale=self.pixel_scale) for sk in s_k]						
			vels=[Ck,Sk]
			flatCS=list(chain(*vels))
			return [modl0]+flatCS



	def model3D(self,twoD_only=False):

			twoDmdls= dataset_to_2D([self.ny,self.nx], self.n_annulus, self.rings_pos, self.r_n, self.XY_mesh, self.kinmdl_dataset, self.vmode, self.v_center, None, self.index_v0, nmodls=self.Vk)
			sigmap=twoDmdls[0]
			interp_model=twoDmdls[1:]
			
			"""
			Analysis of the inner radius

			"""
			mask_inner = np.where( (self.r_n < self.rings_pos[0] ) )
			x_r0,y_r0 = self.XY_mesh[0][mask_inner], self.XY_mesh[1][mask_inner] 
			r_space_0 = self.rings_pos[0]
						
			#(a) velocity rises linearly from zero: r1 = 0, v1 = 0
			if self.v_center == 0 or (self.v_center != "extrapolate" and self.vmode != "circular"):	
				VS_xy_mdl = self.kinmdl_dataset(None, 0, (x_r0,y_r0), r_2 = 0, r_space = r_space_0)
				S_xy_mdl=VS_xy_mdl[0]
				V_xy_mdl=VS_xy_mdl[1:]												
				for k,mdl2d in enumerate(interp_model):						
					v_new_2 = V_xy_mdl[k][1]
					mdl2d[mask_inner] = v_new_2
			else:
				r2 = self.rings_pos[0] 		# ring posintion
				v1_index = self.index_v0	# index of velocity
				#Velocity and Sigma
				VS_xy_mdl0 = self.kinmdl_dataset(None, v1_index, (x_r0,y_r0), r_2 = r2, r_space = r_space_0 )
				S_xy_mdl0=VS_xy_mdl0[0]
				V_xy_mdl0=VS_xy_mdl0[1:]
				
				r1 = 0 					# ring posintion
				v2_index = 0			# index of velocity
				#Velocity and Sigma				
				VS_xy_mdl1 = self.kinmdl_dataset(None, v2_index, (x_r0,y_r0), r_2 = r1, r_space = r_space_0 )
				S_xy_mdl1=VS_xy_mdl1[0]
				V_xy_mdl1=VS_xy_mdl1[1:]
						
				for k in range(len(interp_model)):								
					v_new_1=V_xy_mdl0[k][0]				
					v_new_2=V_xy_mdl1[k][1]
					v_new = v_new_1 + v_new_2
					(interp_model[k])[mask_inner] = v_new
								
			#"""	
			# Dispersion is always extrpolated:			
			r2 = self.rings_pos[0] 		# ring posintion
			v1_index = self.index_v0	# index of velocity
			V_xy_mdl = self.kinmdl_dataset(None, v1_index, (x_r0,y_r0), r_2 = r2, r_space = r_space_0, disp= True )
			v_new_1 = V_xy_mdl[0]

			r1 = 0 					# ring posintion
			v2_index = 0			# index of velocity
			V_xy_mdl = self.kinmdl_dataset(None, v2_index, (x_r0,y_r0), r_2 = r1, r_space = r_space_0, disp = True)
			v_new_2 = V_xy_mdl[1]

			v_new = v_new_1 + v_new_2
			sigmap[mask_inner] = v_new
			#"""

			#create a copy
			twoDmodels=np.copy(interp_model)

			theta,cos_theta0=AZIMUTHAL_ANGLE([self.ny,self.nx],self.pa,self.eps,self.x0,self.y0)
			sin,cos=SIN_COS(self.XY_mesh,self.pa,self.eps,self.x0,self.y0)			
			inc=eps_2_inc(self.eps)
			if self.vmode=='circular':
				vt=interp_model[0]
				vt*=cos*np.sin(inc)
				msk=vt!=0
				velmap=vt+msk*self.Vsys
			if self.vmode=='radial':
				[vt,vr]=interp_model
				vt*=np.sin(inc)*cos
				vr*=np.sin(inc)*sin				
				velsum=vt+vr
				msk=velsum!=0
				velmap=velsum+msk*self.Vsys
			if self.vmode=='ff':
				[vt,vr]=interp_model
				p=np.sqrt(2*(1-self.alpha**2))
				vt*=np.sin(inc)*cos
				vr*=-p*np.sin(inc)*sin				
				velsum=vt+vr
				msk=velsum!=0
				velmap=velsum+msk*self.Vsys																		
			if self.vmode=='vertical':
				[vt,vz]=interp_model
				vt*=np.sin(inc)*cos
				vz*=np.cos(inc)				
				velsum=vt+vz
				msk=velsum!=0
				velmap=velsum+msk*self.Vsys								
			if self.vmode=='bisymmetric':
				phi_b = self.phi_b % (2*np.pi)
				[vt,v2r,v2t]=interp_model
				vt*=np.sin(inc)*cos
				theta_b=theta-phi_b
				v2r*=-1*np.sin(inc)*sin*np.sin(2*theta_b)													
				v2t*=-1*np.sin(inc)*cos*np.cos(2*theta_b)
				velsum=vt+v2r+v2t
				msk=velsum!=0
				velmap=velsum+msk*self.Vsys
			if 'hrm' in self.vmode:
				velsum=0
				for k in range(self.m_hrm):
					CkSk=interp_model[k]*np.cos((k+1)*theta)*np.sin(inc)+interp_model[k+self.m_hrm]*np.sin((k+1)*theta)*np.sin(inc)
					velsum+=CkSk					
				msk=velsum!=0
				velmap=velsum+msk*self.Vsys

			# intrinsinc rotation and dispersion
			velmap_intr,sigmap_intr=velmap,sigmap
			
			# for bootstrap errors
			if twoD_only: return velmap_intr,sigmap_intr
			
			mom0_mdl,mom1_mdl,mom2_mdl_kms,mom2_mdl_A,cube_mdl=self.cube_modl.create_cube(velmap_intr,sigmap_intr)

			mom0_mdl[mom0_mdl==0]=np.nan
			mom1_mdl[mom1_mdl==0]=np.nan			
			mom2_mdl_kms[mom2_mdl_kms==0]=np.nan						
			mom2_mdl_A[mom2_mdl_A==0]=np.nan

			return mom0_mdl,mom1_mdl,mom2_mdl_kms,mom2_mdl_A,cube_mdl,velmap_intr,sigmap_intr,twoDmodels


