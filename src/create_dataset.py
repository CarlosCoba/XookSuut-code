import numpy as np


def dataset_to_2D(shape, n_annulus, rings_pos, r_n, XY_mesh, kinmdl_dataset, vmode, v_center, pars, index_v0, V_i = [] ):

	[ny, nx] = shape
	interp_model = np.zeros((ny,nx))
	# make residual per data set
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
			V_xy_mdl = kinmdl_dataset(pars, v1_index, (x,y), r_0 = r2, r_space = r_space_k )
		else:
			v1 = V_i[v1_index]
			V_xy_mdl = kinmdl_dataset(pars, v1, (x,y), r_0 = r2, r_space = r_space_k )

		v_new_1 = V_xy_mdl[0]

		r1 = rings_pos[ Nring ] 	# ring posintion
		v2_index =  Nring + 1		# index of velocity
		if np.size(V_i) == 0:
			V_xy_mdl = kinmdl_dataset(pars, v2_index, (x,y), r_0 = r1, r_space = r_space_k )
		else:
			v2 = V_i[v2_index]
			V_xy_mdl = kinmdl_dataset(pars, v2, (x,y), r_0 = r1, r_space = r_space_k )

		v_new_2 = V_xy_mdl[1]

		v_new = v_new_1 + v_new_2
		interp_model[mask] = v_new


	return interp_model

