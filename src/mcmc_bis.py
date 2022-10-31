import numpy as np

from weights_interp import weigths_w
from kin_components import CIRC_MODEL
from kin_components import RADIAL_MODEL
from kin_components import BISYM_MODEL

from interp_tools import interp_loops
 
def Rings(xy_mesh,pa,inc,x0,y0,pixel_scale):
	(x,y) = xy_mesh
	X = (- (x-x0)*np.sin(pa) + (y-y0)*np.cos(pa))
	Y = (- (x-x0)*np.cos(pa) - (y-y0)*np.sin(pa))
	R= np.sqrt(X**2+(Y/np.cos(inc))**2)
	return R*pixel_scale

def v_interp(r, r2, r1, v2, v1 ):
	m = (v2 - v1) / (r2 - r1)
	v0 = m*(r-r1) + v1
	return v0




class KinModel :
	"""
	This method defines the obseved array for computing the likelihood
	"""

	def __init__(self, vel_map, evel_map, theta0, vmode, rings_pos, ring_space, pixel_scale, inner_interp, lnsigma_int):


		self.vel_map = vel_map
		self.evel_map = evel_map
		[self.ny, self.nx] = vel_map.shape
		self.pixel_scale = pixel_scale

		self.vrot0,self.vrad0,self.vtan0,self.pa0,self.inc0,self.xc0,self.yc0,self.vsys0,self.phi0 = theta0

		self.n_circ = len(self.vrot0)
		self.rings_pos = rings_pos
		self.ring_space = ring_space


		self.non_zero = self.vrad0 != 0
		vrad0_ = self.vrad0[self.non_zero]
		self.n_noncirc = len(vrad0_)
		self.m = self.n_circ + 2*self.n_noncirc
		self.lnsigma_int = lnsigma_int

		self.vmode = vmode
		nrings = len(self.rings_pos)
		self.n_annulus = nrings - 1
		X = np.arange(0, self.nx, 1)
		Y = np.arange(0, self.ny, 1)
		self.XY_mesh = np.meshgrid(X,Y)
		self.inner_interp = inner_interp
		self.r_n = Rings(self.XY_mesh,self.pa0*np.pi/180,self.inc0*np.pi/180,self.xc0,self.yc0,pixel_scale)


	def interp_model(self, theta ):

		VROT,VRAD,VTAN,PA,INC,X0,Y0,VSYS,PHI_B = theta[:self.n_circ],theta[self.n_circ:self.n_circ+self.n_noncirc],theta[self.n_circ+self.n_noncirc:self.m],theta[self.m],theta[self.m+1],theta[self.m+2],theta[self.m+3],theta[self.m+4],theta[self.m+5]

		nrings = len(self.rings_pos)
		n_annulus = nrings - 1

		VROT_dic, VRAD_dic, VTAN_dic = {},{},{}
		for j in range(1,self.n_circ+1):
		    VROT_dic["VROT{0}".format(j)] = VROT[j-1]

		for j in range(1,self.n_noncirc+1):
		    VRAD_dic["VRAD{0}".format(j)] = VRAD[j-1]
		    VTAN_dic["VTAN{0}".format(j)] = VTAN[j-1]


		model_array = np.array([])
		obs_array = np.array([])
		error_array = np.array([])
		Xpos,Ypos = np.array([]),np.array([])
		interp_model = np.zeros((self.ny,self.nx))

		def eval_mdl( ii, xy_mesh, r_0 , r_space ):

			# For inner interpolation
			r1, r2 = self.rings_pos[0], self.rings_pos[1]
			if "hrm" not in self.vmode and self.inner_interp != False:
				v1, v2 = VROT_dic["VROT1"], VROT_dic["VROT2"]
				v_int =  v_interp(0, r2, r1, v2, v1 )
				# This only applies to circular rotation
				if self.inner_interp != True : v_int = self.inner_interp 
				# I made up this index.
				VROT_dic["VROT0"] = v_int
				if self.vmode == "radial" or self.vmode == "bisymmetric":
					v1, v2 = VRAD_dic["VRAD1"], VRAD_dic["VRAD2"]
					v_int =  v_interp(0, r2, r1, v2, v1 )
					VRAD_dic["VRAD0"] = v_int
				if self.vmode == "bisymmetric":
					v1, v2 = VTAN_dic["VTAN1"], VTAN_dic["VTAN2"]
					v_int =  v_interp(0, r2, r1, v2, v1 )
					VTAN_dic["VTAN0"] = v_int

			VROT_ = VROT_dic["VROT%s"%ii]
			if ii  <= self.n_noncirc:
				VRAD_ = VRAD_dic["VRAD%s"%ii]
				VTAN_ = VTAN_dic["VTAN%s"%ii]

				bisymm = BISYM_MODEL(xy_mesh,VROT_,VRAD_,PA, INC, X0, Y0, VTAN_, PHI_B)
			else:
				bisymm = BISYM_MODEL(xy_mesh, VROT_, 0, PA, INC, X0, Y0, 0, 0)


			W = weigths_w(xy_mesh, PA, INC, X0, Y0, r_0, r_space, pixel_scale = self.pixel_scale)
			mdl = bisymm*W

			return mdl,VSYS


		"""
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

				# For r < rings_pos[0]:					
				# Inner interpolation
				#(a) velocity rises linearly from zero: r1 = 0, v1 = 0
				if self.inner_interp == False and Nring == 0 and kring == 2:
					Vxy,Vsys = eval_mdl( 1, (x_r0,y_r0), r_0 = 0, r_space = r_space_0 )
					mdl_ev0 = Vxy[1]
					interp_model[mask_inner] = mdl_ev0
				#(b) Extrapolate at the origin: r1 = 0, v1 != 0
				if self.inner_interp != False and Nring == 0:
					# we need to add a velocity at r1 = 0
					# index 0 is is reserved for the inner interpolation !
					r2_1 = [r_space_0,0]
					v1_2_index = [0,1]
					Vxy,Vsys = eval_mdl( v1_2_index[2-kring], (x_r0,y_r0), r_0 = r2_1[2-kring], r_space = r_space_0 )
					mdl_ev0 = mdl_ev0 + Vxy[2-kring]
					interp_model[mask_inner] = mdl_ev0


			interp_model[mask] = mdl_ev
		"""

		# create the 2D model
		x = interp_loops(interp_model, self.n_annulus, self.rings_pos, self.r_n, self.XY_mesh, self.inner_interp)
		interpmdl = x.mcmc_loop(eval_mdl)

		interp_model = interp_model + VSYS

		residual = self.vel_map - interp_model
		w_residual = residual/self.evel_map
		w_residual[abs(w_residual) > 100] = np.nan
		mask_ones = w_residual/w_residual
		mask_finite = np.isfinite(mask_ones)

		mdl = interp_model[mask_finite]
		obs = self.vel_map[mask_finite]
		err = self.evel_map[mask_finite]

		return mdl, obs, err


from chain_bis import chain_res_mcm
from metropolis_hastings_algo import run_metropolis_hastings
from posterior_probs import set_like
ln_prob = set_like("bisymmetric",KinModel)

import emcee
prng =  np.random.RandomState(12345)

def Metropolis(data, model_params, mcmc_config, lnsigma_int, inner_interp):

	galaxy, vel_map, evel_map, theta0  = data
	[sampler,steps,thin,burnin,nwalkers,save_plots], step_size = mcmc_config
	vmode, rings_pos, ring_space, pixel_scale, r_bar_min, r_bar_max = model_params

	theta_ = np.hstack(theta0.flatten())
	sigmas = np.hstack(step_size)

	if lnsigma_int == False:
		theta = theta0
	else:
		theta = theta0[:-1]

	scatter_model = ln_prob(vel_map, evel_map, theta, vmode, rings_pos, ring_space, pixel_scale, inner_interp, lnsigma_int)
	accept_rate = 0

	if sampler not in [2,3,4]:
		print("XookSuut: choose a proper sampling method")
		quit()

	if sampler in [2,3]:
		ndim = len(theta_)
		if nwalkers < ndim : nwalkers = int(2.5*ndim); print("XookSuut: Warning!, too few walkers, I have increased nwalkers by 2.5*ndimensions")
		# covariance of the proposal distribution.
		cov = sigmas**2*np.eye(len(theta_))
		pos = np.empty((nwalkers, ndim))
		for k in range(nwalkers):
			theta_prop = prng.multivariate_normal(theta_, cov)
			pos[k] =  theta_prop

		sampler = emcee.EnsembleSampler(nwalkers, ndim, scatter_model )
		sampler.run_mcmc(pos, steps, progress=True, skip_initial_state_check = True);

		chain = sampler.get_chain(discard=burnin, thin=thin, flat = True)
		acceptance = sampler.acceptance_fraction
		#There is one acceptance per walker, thus we take the mean
		acc_frac = np.nanmean(acceptance)
		#This does not have sense here
		accept_rate = np.ones(steps,)*acc_frac


	else:

			
		chain,_,acc_frac,accept_rate = run_metropolis_hastings(theta_, n_steps=int(steps), model=scatter_model, proposal_sigmas=sigmas)
	if acc_frac < 0.1:
		print("XookSuut: Something went really bad, either because the priors are not the good ones or the stepsize is too small/large")
		quit()
	print("accept_rate = ", round(acc_frac,3))
	best = chain_res_mcm(galaxy, vmode, sampler, theta0, chain, step_size, steps, thin, burnin, accept_rate, vel_map.shape, save_plots, rings_pos, ring_space, pixel_scale, inner_interp, lnsigma_int)
	return best


