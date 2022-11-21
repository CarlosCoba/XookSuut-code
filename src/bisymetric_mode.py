import numpy as np
import time
from src.eval_tab_model import tab_mod_vels
from src.phi_bar_sky import pa_bar_sky
from src.fit_params import Fit_kin_mdls as fit
from src.resample import resampling
from src.prepare_mcmc import Metropolis as MP
from src.chain_bis import chain_res_mcmc
from src.tools_fits import array_2_fits

prng =  np.random.RandomState(12345)


#first_guess_it = []


class Bisymmetric_model:
	def __init__(self, galaxy, vel, evel, guess0, vary, n_it, rstart, rfinal, ring_space, frac_pixel, inner_interp, delta, pixel_scale, bar_min_max, config, e_ISM, fitmethod, outdir ):


		self.galaxy=galaxy
		self.vel_copy=np.copy(vel)
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

		self.r_bar_min, self.r_bar_max = self.bar_min_max

		self.pa0,self.inc0,self.x0,self.y0,self.vsys0,self.theta_b = self.guess0
		self.vmode = "bisymmetric"
		[ny,nx] = vel.shape
		self.shape = [ny,nx]


		#outs
		self.PA,self.INC,self.XC,self.YC,self.VSYS,self.THETA = 100,0,0,0,0,0
		self.GUESS = []
		self.Vrot,self.Vrad,self.Vtan = [],[],[]
		self.chisq_global = np.inf
		self.aic_bic = 0
		self.best_vlos_2D_model = 0
		self.best_kin_2D_models = 0
		self.Rings = 0
		self.std_errors = 0


		config_mcmc = config['mcmc']
		self.config_psf = config['convolve']
		config_boots = config['bootstrap']

		#bootstrap
		self.boots_ana = config_boots.getboolean('boots_ana', False)
		self.n_boot = config_boots.getint('Nboots', 5)
		self.use_bootstrap = config_boots.getboolean('use_bootstrap', False)
		self.parallel = config_boots.getboolean('parallelize', False)

		self.bootstrap_contstant_prms = np.zeros((self.n_boot, 6))
		self.bootstrap_kin = 0

		self.config_mcmc = config_mcmc
		self.mcmc_ana = config_mcmc.getboolean('mcmc_ana', False)
		self.PropDist = config_mcmc.get('PropDist',"G")
		self.use_best_mcmc = config_mcmc.getboolean('use_mcmc_vals', True)
		self.save_chain = config_mcmc.getboolean('save_chain', False)
		self.plot_chain = config_mcmc.getboolean('plot_chain', False)

		self.outdir = outdir

		"""

		 					BISYMMETRIC MODEL


		"""



	def lsq(self):

		vrad_it, vtan_it = np.zeros(100,), np.zeros(100,)
		vrot_tab_it, vrad_tab_it, vtan_tab_it = np.zeros(self.nrings,), np.zeros(self.nrings,), np.zeros(self.nrings,)

		for it in np.arange(self.n_it):
				
			if int(self.pa0) == 360: self.pa0 = 0
			if int(self.pa0) == 0: self.pa0 = 1
			if int(self.pa0) == 180: self.pa0 = 181

			# Here we create the tabulated model			
			vrot_tab, vrad_tab, vtan_tab, R_pos = tab_mod_vels(self.rings,self.vel, self.evel, self.pa0,self.inc0,self.x0,self.y0,self.vsys0,self.theta_b,self.delta,self.pixel_scale,self.vmode,self.shape,self.frac_pixel,self.r_bar_min, self.r_bar_max)
			vrot_tab[abs(vrot_tab) > 400] = np.nanmedian(vrot_tab) 
			guess = [vrot_tab,vrad_tab,vtan_tab,self.pa0,self.inc0,self.x0,self.y0,self.vsys0,self.theta_b]
			if it == 0: first_guess_it = guess

			# Minimization
			fitting = fit(self.vel, self.evel, guess, self.vary, self.vmode, self.config, R_pos, self.ring_space, self.fitmethod, self.e_ISM, self.pixel_scale, self.frac_pixel, self.inner_interp,N_it=self.n_it0)
			self.v_2D_mdl, kin_2D_modls, Vk , self.pa0, self.inc0, self.x0, self.y0, self.vsys0, self.theta_b, out_data, Errors, true_rings = fitting.results()
			xi_sq = out_data[-1]
			#Unpack velocities 
			vrot, vrad, vtan = Vk
			self.vrot, self.vrad, self.vtan=vrot, vrad, vtan 

			if np.nanmean(vrot) < 0 :
				self.pa0 = self.pa0 - 180
				vrot = abs(np.asarray(vrot))
				if self.pa0 < 0 : self.pa0 = self.pa0 + 360

			# Keep the best fit 
			if xi_sq < self.chisq_global:

				self.PA,self.INC,self.XC,self.YC,self.VSYS,self.THETA = self.pa0, self.inc0, self.x0, self.y0,self.vsys0,self.theta_b
				self.Vrot = np.asarray(vrot);self.n_circ = len(vrot)
				self.Vrad = np.asarray(vrad)
				self.Vtan = np.asarray(vtan)
				self.chisq_global = xi_sq
				self.aic_bic = out_data
				self.best_vlos_2D_model = self.v_2D_mdl
				self.best_kin_2D_models = kin_2D_modls
				self.Rings = true_rings
				self.std_errors = Errors
				self.GUESS = [self.Vrot, self.Vrad, self.Vtan, self.PA, self.INC, self.XC, self.YC, self.VSYS, self.THETA]

				self.bootstrap_kin = np.zeros((self.n_boot, 3*self.n_circ))


	""" Following, the error computation.
	"""


	import time
	def boots(self,individual_run=0):
		print("starting bootstrap analysis ..")

		self.frac_pixel = 0
		self.n_it = 1
		runs = np.arange(0,self.n_boot)
		if self.parallel: runs = [individual_run]
		for k in runs:
			seed0 = int(time.time());#print("seed0",seed0)
			# setting chisq to -inf will preserve the leastsquare results
			self.chisq_global = -np.inf
			if (k+1) % 5 == 0 : print("%s/%s bootstraps" %((k+1),self.n_boot))

			mdl_old = self.v_2D_mdl
			res = self.vel_copy - mdl_old
			# Inject a different seed per process !
			new_vel = resampling(mdl_old,res,self.Rings,self.delta,self.PA,self.INC,self.XC,self.YC,self.pixel_scale,seed=seed0)
			mdl_zero =  np.isfinite(new_vel)
			# sum two arrays containing nans
			new_vel_map = np.nansum(np.dstack((new_vel*mdl_zero,~mdl_zero*self.vel_copy)),2) ; new_vel_map[new_vel_map==0]=np.nan
			self.vel = new_vel_map
			lsq = self.lsq()
			self.bootstrap_contstant_prms[k,:] = np.array ([ self.pa0, self.inc0, self.x0, self.y0, self.vsys0, self.theta_b ] )
			self.bootstrap_kin[k,:] = np.concatenate([self.vrot,self.vrad,self.vtan])
			if self.parallel: return([[ self.pa0, self.inc0, self.x0, self.y0, self.vsys0, self.theta_b ], np.concatenate([self.vrot,self.vrad,self.vtan])])

		if self.parallel == False:
			std_kin = np.nanstd(self.bootstrap_kin,axis=0)
			self.eVrot, self.eVrad, self.eVtan = std_kin[:self.n_circ],std_kin[self.n_circ:2*self.n_circ],std_kin[2*self.n_circ:]		
			self.std_errors = [[self.eVrot, self.eVrad, self.eVtan],np.nanstd(self.bootstrap_contstant_prms,axis=0)]

	def run_boost_para(self):
		from multiprocessing import Pool, cpu_count
		ncpu = cpu_count()
		with Pool(ncpu-1) as pool:
			# Parallelize the loop
			result=pool.map(self.boots,np.arange(self.n_boot))
		for k in range(self.n_boot):
			self.bootstrap_kin[k,:] = result[k][1]
			self.bootstrap_contstant_prms[k,:] = result[k][0]

		std_kin = np.nanstd(self.bootstrap_kin,axis=0)
		self.eVrot, self.eVrad, self.eVtan = std_kin[:self.n_circ],std_kin[self.n_circ:2*self.n_circ],std_kin[2*self.n_circ:]		
		self.std_errors = [[self.eVrot, self.eVrad, self.eVtan],np.nanstd(self.bootstrap_contstant_prms,axis=0)]


	def mcmc(self):

		print("starting MCMC analysis ..")

		n_circ = len(self.Vrot)
		non_zero = self.Vrad != 0
		Vrad_nonzero = self.Vrad[non_zero]
		Vtan_nonzero = self.Vtan[non_zero]
		n_noncirc = len(Vrad_nonzero)

		theta0 = np.asarray([self.Vrot, Vrad_nonzero, Vtan_nonzero, self.PA, self.INC, self.XC, self.YC, self.VSYS, self.THETA])
		n_circ = len(self.Vrot)
		#Covariance of the proposal distribution
		sigmas = np.array([np.ones(n_circ),np.ones(n_noncirc),np.ones(n_noncirc),1,1,1,1,1,1])*1e-1

		# For the intrinsic scatter
		sigma_0 = 0.5
		theta0 = np.append(theta0, sigma_0)
		sigmas = np.append(sigmas, 0.1)

		
		data = [self.galaxy, self.vel, self.evel, theta0]
		mcmc_config = [self.config_mcmc,sigmas] 
		model_params = [self.vmode, self.Rings, self.ring_space, self.pixel_scale, self.r_bar_min, self.r_bar_max]

		#MCMC RESULTS
		from src.create_2D_vlos_model_mcmc import KinModel
		chain, acc_frac, steps, thin, burnin, nwalkers, post_dist, ndim = MP(KinModel, data, model_params, mcmc_config, self.config_psf, self.inner_interp, n_circ, n_noncirc )
		mcmc_params = [steps,thin,burnin,nwalkers,post_dist,self.plot_chain]
		v_2D_mdl_, kin_2D_models_, Vk_,  PA_, INC_ , XC_, YC_, Vsys_, THETA_, self.std_errors = chain_res_mcmc(self.galaxy, self.vmode, theta0, chain, mcmc_params, acc_frac, self.shape, self.Rings, self.ring_space, self.pixel_scale, self.inner_interp,outdir = self.outdir, config_psf = self.config_psf)
		#Unpack velocities 
		Vrot_, Vrad_, Vtan_ = Vk_

		if self.save_chain:
			header0 = {"0":["CHAIN_SHAPE","[[NWALKERS],[NSTEPS],[Vk,PA,INC,XC,YC,VSYS,PHI_BAR]]"],"1":["ACCEPT_F",acc_frac],"2":["STEPS", steps],"3":["WALKERS",nwalkers],"4":["BURNIN",burnin], "5":["DIM",ndim],"6":["VROT_DIMS",len(self.Vrot)],"7":["V2R_DIMS",n_noncirc],"8":["V2T_DIMS",n_noncirc]}
			array_2_fits(chain, self.outdir, self.galaxy, self.vmode, header0)


		if self.use_best_mcmc:
			self.PA, self.INC, self.XC,self.YC,self.VSYS,self.THETA = PA_, INC_, XC_,YC_,Vsys_,THETA_
			self.Vrot, self.Vrad, self.Vtan = Vrot_, Vrad_, Vtan_
			self.best_vlos_2D_model = v_2D_mdl_
			self.best_kin_2D_models = kin_2D_models_


	def output(self):

		#least
		ecovar = self.lsq()
		#bootstrap
		if self.boots_ana:
			eboots= self.run_boost_para() if self.parallel else self.boots() 

			if self.use_bootstrap:
				mean_kin = np.nanmean(self.bootstrap_kin,axis=0)
				self.PA,self.INC,self.XC,self.YC,self.VSYS,self.THETA = np.nanmean(self.bootstrap_contstant_prms,axis=0)
				self.Vrot, self.Vrad, self.Vtan = mean_kin[0:self.n_circ],mean_kin[self.n_circ:2*self.n_circ],mean_kin[2*self.n_circ:]

		#emcee
		if self.mcmc_ana:
			emcmc = self.mcmc()
		if self.boots_ana and self.mcmc_ana:
			self.use_best_mcmc = True
			emcmc = self.mcmc()
			eboots = self.boots()

	def __call__(self):
		out = self.output()
		# Propagation of errors on the sky bar position angle
		PA_bar_major = pa_bar_sky(self.PA,self.INC,self.THETA)
		PA_bar_minor = pa_bar_sky(self.PA,self.INC,self.THETA-90)
		return self.PA,self.INC,self.XC,self.YC,self.VSYS,self.THETA,self.Rings,self.Vrot,self.Vrad,self.Vtan,self.best_vlos_2D_model,self.best_kin_2D_models,PA_bar_major,PA_bar_minor,self.aic_bic,self.std_errors

