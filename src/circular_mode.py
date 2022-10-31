import numpy as np
from eval_tab_model import tab_mod_vels
from mcmc_circular import Metropolis
from fit_params import Fit_kin_mdls as fit
from bootstrap import Fit_kin_mdls as refit
prng =  np.random.RandomState(12345)


#first_guess_it = []


class Circular_model:
	def __init__(self, galaxy, vel, evel, guess0, vary, n_it, rstart, rfinal, ring_space, frac_pixel, inner_interp, delta, pixel_scale, bar_min_max, errors_method, config, e_ISM, fitmethod, use_mcmc, stepsize):


		self.galaxy=galaxy
		self.vel=vel
		self.evel=evel
		self.guess0=guess0
		self.vary=vary
		self.n_it=n_it
		self.rstart=rstart
		self.rfinal=rfinal
		self.ring_space=ring_space
		self.frac_pixel=frac_pixel
		self.inner_interp=inner_interp
		self.delta=delta
		self.pixel_scale=pixel_scale
		self.bar_min_max=bar_min_max
		self.errors_method=errors_method
		self.config=config
		self.e_ISM=e_ISM
		self.fitmethod=fitmethod
		self.use_mcmc=use_mcmc
		self.stepsize=stepsize


		self.rings = np.arange(self.rstart, self.rfinal, self.ring_space)
		self.nrings = len(self.rings)

		esize = np.size(errors_method) 
		if  esize == 1 : self.emethod = 0
		if  esize != 1 : self.emethod = errors_method[0]
		self.use_best_mcmc = False

		self.pa0,self.inc0,self.x0,self.y0,self.vsys0,self.theta_b = self.guess0
		self.vmode = "circular"
		[ny,nx] = vel.shape
		self.shape = [ny,nx]


		#outs
		self.PA,self.INC,self.XC,self.YC,self.VSYS,self.THETA = 0,0,0,0,0,0
		self.GUESS = []
		self.Vrot,self.Vrad,self.Vtan = [],[],[]
		self.chisq_global = 1e10
		self.aic_bic = 0
		self.best_vlos_2D_model = 0
		self.best_kin_2D_models = 0
		self.Rings = 0
		self.std_errors = 0
		"""

		 					CIRCULAR MODEL


		"""



	def lsq(self):

		vrad_it, vtan_it = np.zeros(100,), np.zeros(100,)
		vrot_tab_it, vrad_tab_it, vtan_tab_it = np.zeros(self.nrings,), np.zeros(self.nrings,), np.zeros(self.nrings,)

		for it in np.arange(self.n_it):
				
			if int(self.pa0) == 360: self.pa0 = 0
			if int(self.pa0) == 0: self.pa0 = 1
			if int(self.pa0) == 180: self.pa0 = 181

			# Here we create the tabulated model			
			vrot_tab, vrad_tab, vtan_tab, R_pos = tab_mod_vels(self.rings,self.vel, self.evel, self.pa0,self.inc0,self.x0,self.y0,self.vsys0,self.theta_b,self.delta,self.pixel_scale,self.vmode,self.shape,self.frac_pixel,0, 0)
			vrot_tab[abs(vrot_tab) > 400] = np.nanmedian(vrot_tab) 
			guess = [vrot_tab,vrot_tab*0,vrot_tab*0,self.pa0,self.inc0,self.x0,self.y0,self.vsys0,self.theta_b]
			if it == 0: first_guess_it = guess

			# Minimization
			fitting = fit(self.vel, self.evel, guess, self.vary, self.vmode, self.config, R_pos, self.ring_space, self.fitmethod, self.e_ISM, self.pixel_scale, self.frac_pixel, self.inner_interp)
			v_2D_mdl, kin_2D_modls, Vk , self.pa0, self.inc0, self.x0, self.y0, self.vsys0, self.theta_b, out_data, Errors, true_rings = fitting.results()
			xi_sq = out_data[-1]
			#Unpack velocities 
			vrot, vrad, vtan = Vk

			if np.nanmean(vrot) < 0 :
				self.pa0 = self.pa0 - 180
				vrot = abs(np.asarray(vrot))
				if self.pa0 < 0 : self.pa0 = self.pa0 + 360

			# Keep the best fit 
			if xi_sq < self.chisq_global:

				self.PA,self.INC,self.XC,self.YC,self.VSYS,self.THETA = self.pa0, self.inc0, self.x0, self.y0,self.vsys0,self.theta_b
				self.Vrot = np.asarray(vrot)
				self.chisq_global = xi_sq
				self.aic_bic = out_data
				self.best_vlos_2D_model = v_2D_mdl
				self.best_kin_2D_models = kin_2D_modls
				self.Rings = true_rings
				self.std_errors = Errors
				self.GUESS = [self.Vrot, self.Vrot*0, self.Vrot*0, self.PA, self.INC, self.XC, self.YC, self.VSYS, 0]




	""" Following, the error computation.
	"""



	def boots(self):

		lsq = self.lsq()
		n_boot =  self.errors_method[1]
		if self.use_best_mcmc == True :
			n_boot = 5


		print("starting bootstrap analysis ..")
		CONSTANT_PARAMS = [self.PA, self.INC, self.XC,self.YC,self.VSYS,self.THETA]
		BEST_KIN = np.concatenate([self.Vrot,self.Vrad,self.Vtan])
		res = self.vel - self.best_vlos_2D_model
		mdl = self.best_vlos_2D_model

		from resample import resampling
		bootstrap_contstant_prms = np.empty((n_boot, 6))
		nkin = len(self.Vrot) 
		bootstrap_kin = np.empty((n_boot, nkin))

		for k in range(n_boot):

			if (k+1) % 5 == 0 : print("%s/%s bootstraps" %((k+1),n_boot))

			vary = np.ones(len(self.vary),)
			new_vel = resampling(mdl,res,self.Rings,self.delta,self.PA,self.INC,self.XC,self.YC,self.pixel_scale)
			fn_bootting = fit(new_vel, self.evel, self.GUESS, vary, self.vmode, self.config, self.Rings, self.ring_space, self.fitmethod, self.e_ISM, self.pixel_scale, 0, self.inner_interp)
			v_2D_mdl, kin_2D_modls, Vk , self.pa0, self.inc0, self.x0, self.y0, self.vsys0, self.theta_b, out_data, Errors, true_rings = fn_bootting.results()
			bootstrap_contstant_prms[k,:] = np.array ([ self.pa0, self.inc0, self.x0, self.y0, self.vsys0, self.theta_b ] )
			N = len(Vk[0])
			if N == nkin:
				bootstrap_kin[k,:] = np.array(Vk[0])
			else:
				bootstrap_kin[k,0:N] = np.array(Vk[0])
				
			res = self.vel - v_2D_mdl
			vrot_tab, vrad_tab, vtan_tab, R_pos = tab_mod_vels(self.rings,new_vel, self.evel, self.pa0,self.inc0,self.x0,self.y0,self.vsys0,self.theta_b,self.delta,self.pixel_scale,self.vmode,self.shape,self.frac_pixel,0, 0)


		rms_constant_prms = [ np.sqrt( np.sum( (bootstrap_contstant_prms[:,k] - CONSTANT_PARAMS[k]) )**2/(n_boot-1) ) for k in range(len(CONSTANT_PARAMS)) ]
		rms_kin = [ np.sqrt( np.sum( (bootstrap_kin[:,k] - self.Vrot[k]) )**2/(n_boot) ) for k in range(len(self.Vrot)) ]
		#rms_kin = [ np.sqrt( np.sum( (bootstrap_kin[:,k])**2/n_boot ) ) for k in range(len(self.Vrot)) ]

		self.std_errors = [np.asarray(rms_kin),0,0] + rms_constant_prms
		self.pa0, self.inc0, self.x0, self.y0, self.vsys0,self.theta_b  = rms_constant_prms

	def mcmc(self):

		lsq = self.lsq()

		if self.emethod in [2, 3] :
			steps,thin,burnin,walkers,save_plots = self.errors_method[1:]
		if self.emethod == 4:
			steps,thin,burnin,walkers,save_plots = self.errors_method[1:]
			walkers = 0


		print("starting MCMC analysis ..")

		theta0 = np.asarray([self.Vrot, self.PA, self.INC, self.XC, self.YC, self.VSYS])
		n_circ = len(self.Vrot)
		#Covariance of the proposal distribution
		if self.stepsize.size == 0:
			#default
			sigmas = np.array([np.ones(n_circ)*10,10,10,1,1,10])*1e-1
			sigmas = np.array([np.ones(n_circ),1,1,1,1,1])*1e-3

		else:
			#custom
			sigmas = np.array([np.ones(n_circ),1,1,1,1,1])*self.stepsize

		lnsigma_int = 0
		if lnsigma_int == True:
			sigma_0 = 0.1
			lsigma = np.log(sigma_0)
			theta0 = np.append(theta0, lsigma)
			sigmas = np.append(sigmas, 0.1)

				
		r_bar_min, r_bar_max = 0, 0

		data = [self.galaxy, self.vel, self.evel, theta0]
		mcmc_config = [self.errors_method,sigmas] 
		model_params = [self.vmode, self.Rings, self.ring_space, self.pixel_scale, r_bar_min, r_bar_max]


		#I only keep the errors
		v_2D_mdl_, kin_2D_models_, Vk_,  PA_, INC_ , XC_, YC_, Vsys_, THETA_, self.std_errors = Metropolis(data, model_params, mcmc_config, lnsigma_int, self.inner_interp )
		#Unpack velocities 
		Vrot_, Vrad_, Vtan_ = Vk_


		if self.use_mcmc == 1 or self.use_best_mcmc == True:
			self.PA, self.INC, self.XC,self.YC,self.VSYS,self.THETA = PA_, INC_, XC_,YC_,Vsys_,THETA_
			self.Vrot = Vrot_
			self.best_vlos_2D_model = v_2D_mdl_
			self.best_kin_2D_models = kin_2D_models_


	def output(self):
		if self.emethod ==  0:
			ecovar = self.lsq()
		if self.emethod == 1:
			eboots = self.boots()
		if self.emethod == 2 or self.emethod == 4: 
			emcmc = self.mcmc()
		if self.emethod == 3:
			self.use_best_mcmc = True
			emcmc = self.mcmc()
			eboots = self.boots()

	def __call__(self):
		out = self.output()
		return self.PA,self.INC,self.XC,self.YC,self.VSYS,0,self.Rings,self.Vrot,0*self.Vrot,0*self.Vrot,self.best_vlos_2D_model,self.best_kin_2D_models,self.aic_bic,self.std_errors

