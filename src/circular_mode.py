import numpy as np

from src.eval_tab_model import tab_mod_vels
from src.fit_params import Fit_kin_mdls as fit
from src.resample import resampling
from src.prepare_mcmc import Metropolis as MP
from src.chain_circ import chain_res_mcmc
from src.tools_fits import array_2_fits
prng =  np.random.RandomState(12345)


#first_guess_it = []


class Circular_model:
	def __init__(self, galaxy, vel, evel, guess0, vary, n_it, rstart, rfinal, ring_space, frac_pixel, inner_interp, delta, pixel_scale, bar_min_max, config, e_ISM, fitmethod, outdir):


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
		if self.n_it == 0: self.n_it =1


		self.rings = np.arange(self.rstart, self.rfinal, self.ring_space)
		self.nrings = len(self.rings)

		self.pa0,self.inc0,self.x0,self.y0,self.vsys0,self.theta_b = self.guess0
		self.vmode = "circular"
		[ny,nx] = vel.shape
		self.shape = [ny,nx]


		#outs
		self.PA,self.INC,self.XC,self.YC,self.VSYS,self.THETA = 0,0,0,0,0,0
		self.GUESS = []
		self.Vrot,self.Vrad,self.Vtan = [],[],[]
		self.chisq_global = np.inf
		self.aic_bic = 0
		self.best_vlos_2D_model = 0
		self.best_kin_2D_models = 0
		self.Rings = 0
		self.std_errors = 0

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


		#self.mcmc_parms = [self.Nsteps, self.thin, self.burnin, self.Nwalkers, self.PropDist, self.Parallel, self.plot_chain,self.PropDist]
		
		self.outdir = outdir

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
			fitting = fit(self.vel, self.evel, guess, self.vary, self.vmode, self.config, R_pos, self.ring_space, self.fitmethod, self.e_ISM, self.pixel_scale, self.frac_pixel, self.inner_interp,N_it=self.n_it0)
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
		n_boot = self.n_boot

		print("starting bootstrap analysis ..")
		CONSTANT_PARAMS = [self.PA, self.INC, self.XC,self.YC,self.VSYS,self.THETA]
		BEST_KIN = np.concatenate([self.Vrot,self.Vrad,self.Vtan])
		res = self.vel - self.best_vlos_2D_model
		mdl = self.best_vlos_2D_model

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
				
			bootstrap_kin[k,:] = np.array(Vk[0])
			res = self.vel - v_2D_mdl
			vrot_tab, vrad_tab, vtan_tab, R_pos = tab_mod_vels(self.rings,new_vel, self.evel, self.pa0,self.inc0,self.x0,self.y0,self.vsys0,self.theta_b,self.delta,self.pixel_scale,self.vmode,self.shape,self.frac_pixel,0, 0)


		rms_constant_prms = [ np.sqrt( np.sum( (bootstrap_contstant_prms[:,k] - CONSTANT_PARAMS[k]) )**2/(n_boot-1) ) for k in range(len(CONSTANT_PARAMS)) ]
		rms_kin = [ np.sqrt( np.sum( (bootstrap_kin[:,k] - self.Vrot[k]) )**2/(n_boot) ) for k in range(len(self.Vrot)) ]
		#rms_kin = [ np.sqrt( np.sum( (bootstrap_kin[:,k])**2/n_boot ) ) for k in range(len(self.Vrot)) ]

		self.std_errors = [np.asarray(rms_kin),0,0] + rms_constant_prms
		self.pa0, self.inc0, self.x0, self.y0, self.vsys0,self.theta_b  = rms_constant_prms

	def mcmc(self):

		lsq = self.lsq()

		print("starting MCMC analysis ..")

		theta0 = np.asarray([self.Vrot, self.PA, self.INC, self.XC, self.YC, self.VSYS])
		n_circ, n_noncirc = len(self.Vrot),0
		#Covariance of the proposal distribution		
		sigmas = np.array([np.ones(n_circ),1,1,1,1,1])

		# C stands for a Cauchy distribution
		if self.PropDist == "C":
			sigma_0 = 0.5
			theta0 = np.append(theta0, sigma_0)
			sigmas = np.append(sigmas, 0.1)
		# G stands for a Gaussian distribution
		else:
			self.PropDist = "G"

		r_bar_min, r_bar_max = 0, 0
		data = [self.galaxy, self.vel, self.evel, theta0]
		mcmc_config = [self.config_mcmc,sigmas] 
		model_params = [ self.vmode, self.Rings, self.ring_space, self.pixel_scale, r_bar_min, r_bar_max ]

		from src.create_2D_vlos_model_mcmc import KinModel
		#MCMC RESULTS
		chain, acc_frac, steps, thin, burnin, nwalkers, post_dist, ndim = MP(KinModel, data, model_params, mcmc_config, self.inner_interp, n_circ, n_noncirc )
		mcmc_params = [steps,thin,burnin,nwalkers,post_dist,self.plot_chain]
		v_2D_mdl_, kin_2D_models_, Vk_,  PA_, INC_ , XC_, YC_, Vsys_, THETA_, self.std_errors = chain_res_mcmc(self.galaxy, self.vmode, theta0, chain, mcmc_params, acc_frac, self.shape, self.Rings, self.ring_space, self.pixel_scale, self.inner_interp,outdir = self.outdir)
		#Unpack velocities 
		Vrot_, Vrad_, Vtan_ = Vk_

		if self.save_chain:
			header0 = {"0":["CHAIN_SHAPE","[[NWALKERS],[NSTEPS],[VROT,PA,INC,XC,YC,VSYS]]"],"1":["ACCEPT_F",acc_frac],"2":["STEPS", steps],"3":["WALKERS",nwalkers],"4":["BURNIN",burnin], "5":["DIM",ndim],"6":["VROT_DIMS",len(self.Vrot)]}
			array_2_fits(chain, self.outdir, self.galaxy, self.vmode, header0)
			

		if self.use_best_mcmc:
			self.PA, self.INC, self.XC,self.YC,self.VSYS,self.THETA = PA_, INC_, XC_,YC_,Vsys_,THETA_
			self.Vrot = Vrot_
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
		return self.PA,self.INC,self.XC,self.YC,self.VSYS,0,self.Rings,self.Vrot,0*self.Vrot,0*self.Vrot,self.best_vlos_2D_model,self.best_kin_2D_models,self.aic_bic,self.std_errors

