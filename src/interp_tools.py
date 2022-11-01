import numpy as np

class interp_loops:
	def __init__(self, interp_model, n_annulus, rings_pos, r_n, XY_mesh, inner_interp ):

		self.interp_model =  interp_model
		self.n_annulus = n_annulus
		self.rings_pos = rings_pos
		self.r_n = r_n
		self.XY_mesh = XY_mesh
		self.inner_interp = inner_interp

		self.mask = [np.where( (self.r_n >= self.rings_pos[Nring] ) & (self.r_n < self.rings_pos[Nring+1]) ) for Nring in range(self.n_annulus)]

	# For circular, radial and bisymmetric 
	def mcmc_loop(self, eval_mdl):
		
			#t0 = time() # start time
			mask_inner = np.where( (self.r_n < self.rings_pos[0] ) )
			x_r0,y_r0 = self.XY_mesh[0][mask_inner], self.XY_mesh[1][mask_inner] 
			r_0 = self.rings_pos[0]
			r_space_0 = r_0


			for Nring in range(self.n_annulus):
				# For r1 > rings_pos[0]
				mdl_ev = 0
				r_space_k = self.rings_pos[Nring+1] - self.rings_pos[Nring] 
				#mask = np.where( (self.r_n >= self.rings_pos[Nring] ) & (self.r_n < self.rings_pos[Nring+1]) )
				x,y = self.XY_mesh[0][self.mask[Nring]], self.XY_mesh[1][self.mask[Nring]]

				# interpolation between rings requieres two velocities: v1 and v2
				#v_new = v1*(r2-r)/(r2-r1) + v2*(r-r1)/(r2-r1)
				
				#
				r2 = self.rings_pos[Nring + 1 ] # ring posintion
				v2_index = Nring + 2		 # index velocity starts in 1

				r1 = self.rings_pos[Nring] # ring posintion
				v1_index = Nring + 1		 # index velocity starts in 1

				Vxy = eval_mdl( (x,y), v2_index, r2, r_space_k, v1_index, r1  )
			
				mdl_ev = mdl_ev + Vxy[0] + Vxy[1]
				self.interp_model[self.mask[Nring]] = mdl_ev

			if self.inner_interp == False :
				Vxy = eval_mdl( (x_r0,y_r0), 1, r_0, r_space_0, None, 0 )
				self.interp_model[mask_inner] = Vxy[1]

			#(b) Extrapolate at the origin: r1 = 0, v1 != 0
			if self.inner_interp != False :
				r2, v2_index = r_0, 1
				r1, v1_index = 0, 0
				Vxy = eval_mdl( (x,y), v2_index, r2, r_0, v1_index, r1  )
				self.interp_model[mask_inner] = Vxy[0] + Vxy[1]




	# For harmonic model
	def mcmc_hrm_loop(self, eval_mdl):

		mask_inner = np.where( (self.r_n < self.rings_pos[0] ) )
		x_r0,y_r0 = self.XY_mesh[0][mask_inner], self.XY_mesh[1][mask_inner] 
		r_0 = self.rings_pos[0]
		r_space_0 = r_0

		# make residual per data set
		for Nring in range(self.n_annulus):
			# For r1 > rings_pos[0]
			mdl_ev = 0
			r_space_k = self.rings_pos[Nring+1] - self.rings_pos[Nring] 
			mask = np.where( (self.r_n >= self.rings_pos[Nring] ) & (self.r_n < self.rings_pos[Nring+1]) )
			x,y = self.XY_mesh[0][mask], self.XY_mesh[1][mask]

			# interpolation between rings requieres two velocities: v1 and v2
			#v_new = v1*(r2-r)/(r2-r1) + v2*(r-r1)/(r2-r1)


			# For r1 < rings_pos[0]
			mdl_ev0 = 0
			#mask_inner = np.where( (self.r_n < self.rings_pos[0] ) )
			#x_r0,y_r0 = self.XY_mesh[0][mask_inner], self.XY_mesh[1][mask_inner] 
			#r_space_0 = self.rings_pos[0]

			#"""
			r2 = self.rings_pos[Nring + 1 ] # ring posintion
			v1_index =  Nring  # index velocity starts in 0
			Vxy1 = eval_mdl( (x,y), v1_index, r_2 = r2, r_space = r_space_k ) 

			r1 = self.rings_pos[Nring]
			v2_index =  Nring + 1
			Vxy2 = eval_mdl( (x,y), v2_index, r_2 = r1, r_space = r_space_k) 

			#Vxy2 = eval_mdl( (x,y), v2_index, r_0 = r1, r_space = r_space_k ) 
			mdl_ev = Vxy1[0] + Vxy2[1]

			self.interp_model[mask] = mdl_ev
		 
		if self.inner_interp == False:
			#Vxy = eval_mdl( (x_r0,y_r0), 1 , r_2 = 0, r_space = r_space_0 )
			#Vxy = eval_mdl( (x_r0,y_r0), 0, r_2 = r_0, r_space = r_space_0)
			Vxy = eval_mdl( (x_r0,y_r0), 0, r_2 = r_0, r_space = r_0, ii =  None, r_1 = 0)
			mdl_ev0 = Vxy[1]
			self.interp_model[mask_inner] = mdl_ev0



			"""

			# interpolation between rings requieres two velocities: v1 and v2
			#v_new = v1*(r2-r)/(r2-r1) + v2*(r-r1)/(r2-r1)
			for kring in np.arange(2,0,-1):
				r2_1 = self.rings_pos[Nring + kring -1 ] # ring posintion
				v1_2_index = Nring + 2 - kring		 # index velocity starts in 0
				# For r > rings_pos[0]:
				Vxy,Vsys = eval_mdl( (x,y), v1_2_index, r_2 = r2_1, r_space = r_space_k )
				mdl_ev = mdl_ev + Vxy[2-kring]


				# For r < rings_pos[0]:					
				# Inner interpolation
				#(a) velocity rises linearly from zero: r1 = 0, v1 = 0
				#if self.inner_interp == False and Nring == 0 and kring == 2:
				#	Vxy,Vsys = eval_mdl( 1, (x_r0,y_r0), r_0 = 0, r_space = r_space_0 )
				#	mdl_ev0 = Vxy[1]
				#	self.interp_model[mask_inner] = mdl_ev0
			"""




