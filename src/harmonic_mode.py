import numpy as np

from src.eval_tab_model import tab_mod_vels
from src.fit_params import Fit_kin_mdls as fit
from src.resample import resampling
from src.prepare_mcmc import Metropolis as MP
from src.chain_hrm import chain_res_mcmc
from src.tools_fits import array_2_fits

class Harmonic_model:
	def __init__(self, galaxy, vel, evel, guess0, vary, n_it, rstart, rfinal, ring_space, frac_pixel, inner_interp, delta, pixel_scale,  bar_min_max, config, e_ISM, fitmethod, m_hrm, outdir):

		self.galaxy=galaxy
		self.vel=vel
		self.evel=evel
		self.guess0=guess0
		self.vary=vary
		self.n_it,self.n_it0=n_it,n_it
		self.rstart=rstart
		self.rfinal=rfinal
		self.ring_space=ring_space
		self.frac_pixel=frac_pixel
		self.inner_interp=inner_interp
		self.delta=delta
		self.pixel_scale=pixel_scale
		self.bar_min_max=bar_min_max
		self.config=config
		self.e_ISM=e_ISM
		self.fitmethod=fitmethod
		self.m_hrm = m_hrm
		if self.n_it == 0: self.n_it = 1
		self.rings = np.arange(self.rstart, self.rfinal, self.ring_space)
		self.nrings = len(self.rings)

		self.r_bar_min, self.r_bar_max = self.bar_min_max

		self.pa0,self.inc0,self.x0,self.y0,self.vsys0,self.theta_b = self.guess0
		self.vmode = "hrm"
		[ny,nx] = vel.shape
		self.shape = [ny,nx]




		#outs
		self.PA,self.INC,self.XC,self.YC,self.VSYS,self.THETA = 0,0,0,0,0,0
		self.GUESS = []
		self.C_k, self.S_k = [],[]
		self.chisq_global = np.inf
		self.aic_bic = 0
		self.best_vlos_2D_model = 0
		self.best_kin_2D_models = 0
		self.Rings = 0
		self.std_errors = 0
		self.GUESS = 0
		self.n_circ = 0
		self.n_noncirc = 0

		config_mcmc = config['mcmc']
		config_boots = config['bootstrap']

		self.boots_ana = config_boots.getboolean('boots_ana', False)
		self.n_boot = config_boots.getint('Nboots', 5)


		self.config_mcmc = config_mcmc
		self.mcmc_ana = config_mcmc.getboolean('mcmc_ana', False)
		self.PropDist = config_mcmc.get('PropDist',"G")
		self.use_best_mcmc = config_mcmc.getboolean('use_mcmc_vals', True)
		self.save_chain = config_mcmc.getboolean('save_chain', False)
		self.plot_chain = config_mcmc.getboolean('plot_chain', False)

		self.outdir = outdir



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
			guess = [c_tab,s_tab,self.pa0,self.inc0,self.x0,self.y0,self.vsys0,self.theta_b]

			# Minimization
			fitting = fit(self.vel, self.evel, guess, self.vary, self.vmode, self.config, R_pos, self.ring_space, self.fitmethod, self.e_ISM, self.pixel_scale, self.frac_pixel, self.inner_interp, self.m_hrm, N_it = self.n_it0)
			v_2D_mdl, kin_2D_modls,  Vk_ , self.pa0, self.inc0, self.x0, self.y0, self.vsys0, self.phi_b0, out_data, Errors, true_rings = fitting.results()
			xi_sq = out_data[-1]

			#c_k, s_k = [ np.asarray(Vk_[k]) for k in range(self.m_hrm) ], [ np.asarray(Vk_[k]) for k in range(self.m_hrm,2*self.m_hrm) ]
			c_k, s_k = np.array([ np.asarray(Vk_[k]) for k in range(self.m_hrm) ]), np.array([  np.asarray(Vk_[k]) for k in range(self.m_hrm,2*self.m_hrm) ])

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
		n_boot = self.n_boot


		print("starting bootstrap analysis ..")
		CONSTANT_PARAMS = [self.PA, self.INC, self.XC, self.YC, self.VSYS]
		BEST_KIN_CON = np.concatenate([self.C_k, self.S_k])
		BEST_KIN = np.asarray([item for sublist in BEST_KIN_CON for item in sublist])			
		res = self.vel - self.best_vlos_2D_model
		mdl = self.best_vlos_2D_model

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
		sigmas = np.divide( theta0,theta0)*1e-2

		# C stands for a Cauchy distribution
		if self.PropDist == "C":
			sigma_0 = 0.5
			theta0 = np.append(theta0, sigma_0)
			sigmas = np.append(sigmas, 0.1)
		# G stands for a Gaussian distribution
		else:
			self.PropDist = "G"
								
				
		data = [self.galaxy, self.vel, self.evel, theta0]
		mcmc_config = [self.config_mcmc,sigmas] 
		model_params = [self.vmode, self.Rings, self.ring_space, self.pixel_scale, self.r_bar_min, self.r_bar_max]

		from src.create_2D_vlos_model_mcmc import KinModel
		#MCMC RESULTS
		chain, acc_frac, steps, thin, burnin, nwalkers, post_dist, ndim = MP(KinModel, data, model_params, mcmc_config, self.inner_interp, self.n_circ, self.n_noncirc, self.m_hrm )
		mcmc_params = [steps,thin,burnin,nwalkers,post_dist,self.plot_chain]
		v_2D_mdl_, kin_2D_models_, Vk_,  PA_, INC_ , XC_, YC_, Vsys_, THETA_, self.std_errors = chain_res_mcmc(self.galaxy, self.vmode, theta0, chain, mcmc_params, acc_frac, self.shape, self.Rings, self.ring_space, self.pixel_scale, self.inner_interp,outdir = self.outdir, m_hrm=self.m_hrm, n_circ=self.n_circ, n_noncirc=self.n_noncirc )


		#
		# TO DO ...
		#

		if self.save_chain:
			print("Saving chain ..")
			header0 = {"0":["CHAIN_SHAPE","[[NWALKERS],[NSTEPS],[Ck,Sk,PA,INC,XC,YC,C0]]"],"1":["ACCEPT_F",acc_frac],"2":["STEPS", steps],"3":["WALKERS",nwalkers],"4":["BURNIN",burnin], "5":["DIM",ndim],"6":["C1_DIMS",self.n_circ],"7":["S1_DIMS",self.n_noncirc]}
			vmode2 = self.vmode + "_%s"%self.m_hrm
			array_2_fits(chain, self.outdir, self.galaxy, vmode2, header0)

		c_k, s_k = np.array([ np.asarray(Vk_[k]) for k in range(self.m_hrm) ]), np.array([  np.asarray(Vk_[k]) for k in range(self.m_hrm,2*self.m_hrm) ])

		if self.use_best_mcmc:
			self.PA, self.INC, self.XC, self.YC, self.VSYS = PA_, INC_, XC_,YC_,Vsys_
			self.C_k, self.S_k = c_k, s_k 
			self.best_vlos_2D_model = v_2D_mdl_
			self.best_kin_2D_models = kin_2D_models_

	def output(self):
		#least
		ecovar = self.lsq()
		#bootstrap
		if self.boots_ana:
			eboots = self.boots()
		#emcee
		if self.mcmc_ana: 
			emcmc = self.mcmc()
		if self.boots_ana and self.mcmc_ana:
			self.use_best_mcmc = True
			emcmc = self.mcmc()
			eboots = self.boots()



	def __call__(self):
		out = self.output()
		return self.PA,self.INC,self.XC,self.YC,self.VSYS,self.Rings,self.C_k,self.S_k,self.best_vlos_2D_model,self.best_kin_2D_models,self.aic_bic,self.std_errors



