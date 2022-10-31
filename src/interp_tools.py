import numpy as np


class interp_loops:
	def __init__(self, interp_model, n_annulus, rings_pos, r_n, XY_mesh, inner_interp ):

		self.interp_model =  interp_model
		self.n_annulus = n_annulus
		self.rings_pos = rings_pos
		self.r_n = r_n
		self.XY_mesh = XY_mesh
		self.inner_interp = inner_interp

	# For circular, radial and bisymmetric 
	def mcmc_loop(self, eval_mdl):
		
			# make residual per data set
			for Nring in range(self.n_annulus):
				# For r1 > rings_pos[0]
				mdl_ev = 0
				r_space_k = self.rings_pos[Nring+1] - self.rings_pos[Nring] 
				mask = np.where( (self.r_n >= self.rings_pos[Nring] ) & (self.r_n < self.rings_pos[Nring+1]) )
				x,y = self.XY_mesh[0][mask], self.XY_mesh[1][mask]

				# For r1 < rings_pos[0]
				mdl_ev0 = 0
				mask_inner = np.where( (self.r_n < self.rings_pos[0] ) )
				x_r0,y_r0 = self.XY_mesh[0][mask_inner], self.XY_mesh[1][mask_inner] 
				r_space_0 = self.rings_pos[0]

				# interpolation between rings requieres two velocities: v1 and v2
				#v_new = v1*(r2-r)/(r2-r1) + v2*(r-r1)/(r2-r1)
				for kring in np.arange(2,0,-1):
					r2_1 = self.rings_pos[Nring + kring -1 ] # ring posintion
					v1_2_index = Nring + 3 - kring		 # index velocity starts in 1
					# For r > rings_pos[0]:
					Vxy,Vsys = eval_mdl( v1_2_index, (x,y), r_0 = r2_1, r_space = r_space_k )
					mdl_ev = mdl_ev + Vxy[2-kring]

					#"""
					# For r < rings_pos[0]:					
					# Inner interpolation
					#(a) velocity rises linearly from zero: r1 = 0, v1 = 0
					if self.inner_interp == False and Nring == 0 and kring == 2:
						Vxy,Vsys = eval_mdl( 1, (x_r0,y_r0), r_0 = 0, r_space = r_space_0 )
						mdl_ev0 = Vxy[1]
						self.interp_model[mask_inner] = mdl_ev0
					#(b) Extrapolate at the origin: r1 = 0, v1 != 0
					if self.inner_interp != False and Nring == 0:
						# we need to add a velocity at r1 = 0
						# index 0 is is reserved for the inner interpolation !
						r2_1 = [r_space_0,0]
						v1_2_index = [0,1]
						Vxy,Vsys = eval_mdl( v1_2_index[2-kring], (x_r0,y_r0), r_0 = r2_1[2-kring], r_space = r_space_0 )
						mdl_ev0 = mdl_ev0 + Vxy[2-kring]
						self.interp_model[mask_inner] = mdl_ev0

				self.interp_model[mask] = mdl_ev


	# For harmonic model
	def mcmc_hrm_loop(self, eval_mdl):


		# make residual per data set
		for Nring in range(self.n_annulus):
			# For r1 > rings_pos[0]
			mdl_ev = 0
			r_space_k = self.rings_pos[Nring+1] - self.rings_pos[Nring] 
			mask = np.where( (self.r_n >= self.rings_pos[Nring] ) & (self.r_n < self.rings_pos[Nring+1]) )
			x,y = self.XY_mesh[0][mask], self.XY_mesh[1][mask]

			# For r1 < rings_pos[0]
			mdl_ev0 = 0
			mask_inner = np.where( (self.r_n < self.rings_pos[0] ) )
			x_r0,y_r0 = self.XY_mesh[0][mask_inner], self.XY_mesh[1][mask_inner] 
			r_space_0 = self.rings_pos[0]

			# interpolation between rings requieres two velocities: v1 and v2
			#v_new = v1*(r2-r)/(r2-r1) + v2*(r-r1)/(r2-r1)
			for kring in np.arange(2,0,-1):
				r2_1 = self.rings_pos[Nring + kring -1 ] # ring posintion
				v1_2_index = Nring + 2 - kring		 # index velocity starts in 0
				# For r > rings_pos[0]:
				Vxy,Vsys = eval_mdl( v1_2_index, (x,y), r_0 = r2_1, r_space = r_space_k )
				mdl_ev = mdl_ev + Vxy[2-kring]


				# For r < rings_pos[0]:					
				# Inner interpolation
				#(a) velocity rises linearly from zero: r1 = 0, v1 = 0
				if self.inner_interp == False and Nring == 0 and kring == 2:
					Vxy,Vsys = eval_mdl( 1, (x_r0,y_r0), r_0 = 0, r_space = r_space_0 )
					mdl_ev0 = Vxy[1]
					self.interp_model[mask_inner] = mdl_ev0


			self.interp_model[mask] = mdl_ev


