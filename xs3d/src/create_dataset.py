import numpy as np
import matplotlib.pylab as plt

def dataset_to_2D(shape, n_annulus, rings_pos, r_n, XY_mesh, kinmdl_dataset, vmode, v_center, pars, index_v0, V_i = [], nmodls=2, vconst = None ):

	[ny, nx] = shape	
	# make residual per data set
	#nmodls=2 # kinematics[1] and dispersion[2]
	mdls2D=[]
	for k in range(nmodls):
		interp_model = np.zeros((ny,nx))
		for Nring in range(n_annulus):

			# For r1 > rings_pos[0]
			v_new = 0
			r_space_k = rings_pos[Nring+1] - rings_pos[Nring]
			mask = np.where( (r_n >= rings_pos[Nring] ) & (r_n < rings_pos[Nring+1]) )
			x,y = XY_mesh[0][mask], XY_mesh[1][mask]


			# interpolation between rings requieres two velocities: v1 and v2
			#V_new 	= v1*(r2-r)/(r2-r1) + v2*(r-r1)/(r2-r1)
			#		= v1*(r2-r)/delta_R + v2*(r-r1)/delta_R
			#		= V(v1,r2) + V(v2,r1)

			r2 = rings_pos[ Nring + 1 ] # ring posintion
			v1_index = Nring		 # index of velocity
			if np.size(V_i) == 0:
				V_xy_mdl = kinmdl_dataset(pars, v1_index, (x,y), r_2 = r2, r_space = r_space_k)[k]
			else:
				v1 = V_i[v1_index]
				V_xy_mdl = kinmdl_dataset(pars, v1, (x,y), r_2 = r2, r_space = r_space_k)[k]

			v_new_1 = V_xy_mdl[0]

			r1 = rings_pos[ Nring ] 	# ring posintion
			v2_index =  Nring + 1		# index of velocity
			if np.size(V_i) == 0:
				V_xy_mdl = kinmdl_dataset(pars, v2_index, (x,y), r_2 = r1, r_space = r_space_k)[k]
			else:
				v2 = V_i[v2_index]
				V_xy_mdl = kinmdl_dataset(pars, v2, (x,y), r_2 = r1, r_space = r_space_k)[k]

			v_new_2 = V_xy_mdl[1]

			v_new = v_new_1 + v_new_2

			interp_model[mask] = v_new
		mdls2D.append(interp_model)
		
	return mdls2D
	


def dataset_to_2D_b(shape, n_annulus, rings_pos, r_n, XY_mesh, kinmdl_dataset, vmode, v_center, pars, index_v0, V_i = [], nmodls=2, vconst = None ):

	[ny, nx] = shape	
	# make residual per data set
	#nmodls=2 # kinematics[1] and dispersion[2]
	mdls2D=[]
	for k in range(nmodls):
		interp_model = np.zeros((ny,nx))
		for Nring in range(n_annulus):
			# For r1 > rings_pos[0]
			v_new = 0
			r_space_k = rings_pos[Nring+1] - rings_pos[Nring]
			mask = np.where( (r_n >= rings_pos[Nring] ) & (r_n < rings_pos[Nring+1]) )
			x,y = XY_mesh[0][mask], XY_mesh[1][mask]


			# interpolation between rings requieres two velocities: v1 and v2
			#V_new 	= v1*(r2-r)/(r2-r1) + v2*(r-r1)/(r2-r1)
			#		= v1*(r2-r)/delta_R + v2*(r-r1)/delta_R
			#		= V(v1,r2) + V(v2,r1)

			r2 = rings_pos[ Nring + 1 ] # ring posintion
			v1_index = Nring		 # index of velocity
			if np.size(V_i) == 0:
				V_xy_mdl_tmp = kinmdl_dataset(pars, v1_index, (x,y), r_2 = r2, r_space = r_space_k, vconst = vconst  )
				V_xy_mdl = V_xy_mdl_tmp if nmodls == 1 else  V_xy_mdl_tmp[k]
			else:
				v1 = V_i[v1_index]
				V_xy_mdl_tmp = kinmdl_dataset(pars, v1, (x,y), r_2 = r2, r_space = r_space_k, vconst = vconst  )
				V_xy_mdl = V_xy_mdl_tmp if nmodls == 1 else  V_xy_mdl_tmp[k]				

			v_new_1 = V_xy_mdl[0]

			r1 = rings_pos[ Nring ] 	# ring posintion
			v2_index =  Nring + 1		# index of velocity
			if np.size(V_i) == 0:
				V_xy_mdl_tmp = kinmdl_dataset(pars, v2_index, (x,y), r_2 = r1, r_space = r_space_k, vconst = vconst  )
				V_xy_mdl = V_xy_mdl_tmp if nmodls == 1 else  V_xy_mdl_tmp[k]				
			else:
				v2 = V_i[v2_index]
				V_xy_mdl_tmp = kinmdl_dataset(pars, v2, (x,y), r_2 = r1, r_space = r_space_k, vconst = vconst  )
				V_xy_mdl = V_xy_mdl_tmp if nmodls == 1 else  V_xy_mdl_tmp[k]				

			v_new_2 = V_xy_mdl[1]

			v_new = v_new_1 + v_new_2

			interp_model[mask] = v_new
		mdls2D.append(interp_model)
				
	return interp_model if nmodls == 1 else mdls2D
