import numpy as np
from eval_tab_model import tab_mod_vels
from mcmc_hrm import Metropolis
from fit_params import Fit_kin_mdls as fit
from bootstrap import Fit_kin_mdls as refit

class Harmonic_model:
	def __init__(self, galaxy, vel, evel, guess0, vary, n_it, rstart, rfinal, ring_space, frac_pixel, inner_interp, delta, pixel_scale,  bar_min_max, errors_method, config, e_ISM, fitmethod, use_mcmc, stepsize, m_hrm):

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
		self.m_hrm = m_hrm

		self.rings = np.arange(self.rstart, self.rfinal, self.ring_space)
		self.nrings = len(self.rings)

		self.r_bar_min, self.r_bar_max = self.bar_min_max

		esize = np.size(errors_method) 
		if  esize == 1 : self.emethod = 0
		if  esize != 1 : self.emethod = errors_method[0]
		self.use_best_mcmc = False

		self.pa0,self.inc0,self.x0,self.y0,self.vsys0,self.theta_b = self.guess0
		self.vmode = "hrm"
		[ny,nx] = vel.shape
		self.shape = [ny,nx]




		#outs
		self.PA,self.INC,self.XC,self.YC,self.VSYS,self.THETA = 0,0,0,0,0,0
		self.GUESS = []
		self.C_k, self.S_k = [],[]
		self.chisq_global = 1e10
		self.aic_bic = 0
		self.best_vlos_2D_model = 0
		self.best_kin_2D_models = 0
		self.Rings = 0
		self.std_errors = 0
		self.GUESS = 0
		self.n_circ = 0
		self.n_noncirc = 0


		"""

		 					Harmonic model


		"""


	def lsq(self):

		c1_tab_it, c3_tab_it, s1_tab_it, s3_tab_it = np.zeros(self.nrings,), np.zeros(self.nrings,), np.zeros(self.nrings,),np.zeros(self.nrings,)
		for it in np.arange(self.n_it):
			# Here we create the tabulated model
			c_tab, s_tab, R_pos = tab_mod_vels(self.rings,self.vel, self.evel, self.pa0,self.inc0,self.x0,self.y0,self.vsys0,self.theta_b,self.delta,self.pixel_scale,self.vmode,self.shape,self.frac_pixel,self.r_bar_min, self.r_bar_max, self.m_hrm)
			c1_tab = c_tab[0]
			c1_tab[abs(c1_tab) > 400] = np.nanmedian(c1_tab) 	
			guess = [c_tab,s_tab,self.pa0,self.inc0,self.x0,self.y0,self.vsys0]

			# Minimization
			fitting = fit(self.vel, self.evel, guess, self.vary, self.vmode, self.config, R_pos, self.ring_space, self.fitmethod, self.e_ISM, self.pixel_scale, self.frac_pixel, self.inner_interp, self.m_hrm)
			v_2D_mdl, kin_2D_modls,  v_k , self.pa0, self.inc0, self.x0, self.y0, self.vsys0, self.phi_b0, out_data, Errors, true_rings = fitting.results()
			xi_sq = out_data[-1]

			#c_k, s_k = [ np.asarray(v_k[k]) for k in range(self.m_hrm) ], [ np.asarray(v_k[k]) for k in range(self.m_hrm,2*self.m_hrm) ]
			c_k, s_k = np.array([ np.asarray(v_k[k]) for k in range(self.m_hrm) ]), np.array([  np.asarray(v_k[k]) for k in range(self.m_hrm,2*self.m_hrm) ])

			# The first circular and first radial components
			c1 = c_k[0]
			s1 = s_k[0]

			if np.nanmean(c1) < 0 :
				self.pa0 = self.pa0 - 180
				if self.pa0 < 0 : self.pa0 = self.pa0 + 360
				c1 = abs(np.asarray(c1))
				self.xi_sq = 1e10

			# Keep the best fit 
			if xi_sq < self.chisq_global:

				self.PA,self.INC,self.XC,self.YC,self.VSYS = self.pa0, self.inc0, self.x0, self.y0, self.vsys0
				self.C_k, self.S_k = c_k, s_k 
				self.chisq_global = xi_sq
				self.aic_bic = out_data
				self.best_vlos_2D_model = v_2D_mdl
				self.best_kin_2D_models = kin_2D_modls
				self.Rings = true_rings
				self.std_errors = Errors
				self.GUESS = [self.C_k, self.S_k, self.PA, self.INC, self.XC, self.YC, self.VSYS]
				self.n_circ = len(self.C_k[0])
				self.n_noncirc = len((self.S_k[0])[self.S_k[0]!=0])


	""" Following, the error computation.
	"""



	def boots(self):


		lsq = self.lsq()
		n_boot =  self.errors_method[1]
		if self.use_best_mcmc == True :
			n_boot = 5



		print("starting bootstrap analysis ..")
		CONSTANT_PARAMS = [self.PA, self.INC, self.XC, self.YC, self.VSYS]
		BEST_KIN_CON = np.concatenate([self.C_k, self.S_k])
		BEST_KIN = np.asarray([item for sublist in BEST_KIN_CON for item in sublist])			
		res = self.vel - self.best_vlos_2D_model
		mdl = self.best_vlos_2D_model

		from resample import resampling
		bootstrap_contstant_prms = np.empty((n_boot, 6))
		#total number of velocities
		nkin = len(BEST_KIN)
		bootstrap_kin = np.empty((n_boot, nkin))

		for k in range(n_boot):
			if (k+1) % 5 == 0 : print("%s/%s bootstraps" %((k+1),n_boot))
				
			vary = np.ones(len(self.vary),)
			# resample the best vlos model:
			new_vel = resampling(mdl,res,self.Rings,self.delta,self.PA,self.INC,self.XC,self.YC,self.pixel_scale)
			# fit the resampled model:
			boots_fit = refit(new_vel, self.evel, self.GUESS, vary, self.vmode, self.config, self.Rings, self.ring_space, self.fitmethod, self.e_ISM, self.pixel_scale, 0, self.inner_interp, self.m_hrm)
			v_2D_mdl, kin_2D_modls, Vk , self.pa0, self.inc0, self.x0, self.y0, self.vsys0, self.theta_b, out_data, Errors, true_rings = boots_fit.results()
			bootstrap_contstant_prms[k,:] = np.array ([ self.pa0, self.inc0, self.x0, self.y0, self.vsys0, self.theta_b ] )


			c_k, s_k = np.array([ np.asarray(Vk[j]) for j in range(self.m_hrm) ]), np.array([  np.asarray(Vk[j]) for j in range(self.m_hrm,2*self.m_hrm) ])
			best_kin_con = np.concatenate([c_k, s_k])
			best_kins = np.asarray([item for sublist in best_kin_con for item in sublist])

			bootstrap_kin[k,:] = best_kins


		rms_constant_prms = [ np.sqrt( np.sum( (bootstrap_contstant_prms[:,t] - CONSTANT_PARAMS[t]) )**2/(n_boot-1) ) for t in range(len(CONSTANT_PARAMS)) ]

		rms_kin = [ np.sqrt( np.sum( (bootstrap_kin[:,k] - BEST_KIN[k]) )**2/(n_boot) ) for k in range(nkin) ]
		mask_c = [0] + [self.n_circ] + [self.n_circ + self.n_circ*(k+1) for k in range(2*self.m_hrm-1)]
		self.std_errors = [np.asarray(rms_kin[mask_c[t-1]:mask_c[t]]) for t in range(1,len(mask_c)) ] + rms_constant_prms
		self.pa0, self.inc0, self.x0, self.y0, self.vsys0  = rms_constant_prms





	def mcmc(self):

		lsq = self.lsq()

		if self.emethod in [2, 3] :
			steps,thin,burnin,walkers,save_plots = self.errors_method[1:]
		if self.emethod == 4:
			steps,thin,burnin,walkers,save_plots = self.errors_method[1:]
			walkers = 0


		print("starting MCMC analysis ..")

		non_zero_s = self.S_k[0] != 0
		non_zero_c = self.C_k[0] != 0
		if self.m_hrm != 1:
			mask = [non_zero_c] + [non_zero_s]*(self.m_hrm - 1)
		else:
			mask = [non_zero_c] + [non_zero_s]


		# remove zeros from c_k and s_k
		C = np.array([self.C_k[k][mask[k]] for k in range(self.m_hrm)])
		S = np.array([self.S_k[k][mask[1]] for k in range(self.m_hrm)])
		n_noncirc = len(S[0])

		#priors:
		theta_list = [np.hstack(C), np.hstack(S), self.PA, self.INC, self.XC, self.YC, self.VSYS ]
		theta0 = np.hstack(theta_list)

		#Covariance of the proposal distribution
		if self.stepsize.size == 0:
			#default
			sigmas = np.divide( theta0,theta0)*1e-2								
		else:
			#custom
			sigma_circ = np.ones(self.n_circ)*self.stepsize[0]
			sigma_nonc = np.ones(n_noncirc)*self.stepsize[1]
			sigmas_cs = np.array([sigma_nonc for k in range(2*self.m_hrm-1)])
			sigmas = np.array([np.hstack(sigma_circ),np.hstack(sigmas_cs),1,1,1,1,1])*self.stepsize
			sigmas = np.hstack(sigmas)
			sigmas[-5:] = sigmas[-5:]*self.stepsize[-5:] 

		lnsigma_int = False
		if lnsigma_int == True:
			theta0 = np.append(theta0, 0.01)
			sigmas = np.append(sigmas, 0.01)

				
		data = [self.galaxy, self.vel, self.evel, theta0]
		mcmc_config = [self.errors_method,sigmas] 
		model_params = [self.vmode, self.Rings, self.ring_space, self.pixel_scale, self.r_bar_min, self.r_bar_max]


		v_2D_mdl_, kin_2D_models_,V_k, PA_, INC_ , XC_, YC_, Vsys_, std_errors = Metropolis(data, model_params, mcmc_config, lnsigma_int, self.m_hrm, self.n_circ, n_noncirc, self.inner_interp )

		c_k, s_k = np.array([ np.asarray(V_k[k]) for k in range(self.m_hrm) ]), np.array([  np.asarray(V_k[k]) for k in range(self.m_hrm,2*self.m_hrm) ])

		if self.use_mcmc == 1 or self.use_best_mcmc == True:
			self.PA, self.INC, self.XC, self.YC, self.VSYS = PA_, INC_, XC_,YC_,Vsys_
			self.C_k, self.S_k = c_k, s_k 
			self.best_vlos_2D_model = v_2D_mdl_
			self.best_kin_2D_models = kin_2D_models_




	def output(self):
		#least
		if self.emethod ==  0:
			ecovar = self.lsq()
		#bootstrap
		if self.emethod == 1:
			eboots = self.boots()
		#emcee and MH
		if self.emethod == 2 or self.emethod == 4: 
			emcmc = self.mcmc()
		if self.emethod == 3:
			self.use_best_mcmc = True
			emcmc = self.mcmc()
			eboots = self.boots()

	def __call__(self):
		out = self.output()
		return self.PA,self.INC,self.XC,self.YC,self.VSYS,self.Rings,self.C_k,self.S_k,self.best_vlos_2D_model,self.best_kin_2D_models,self.aic_bic,self.std_errors



