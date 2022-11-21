import numpy as np
import emcee
from scipy import stats
import multiprocessing 
from multiprocessing import Pool
from src.interp_tools import interp_loops
from src.posterior_probs import set_likelihood

import zeus
prng =  np.random.RandomState(12345)

def Metropolis(KinModel, data, model_params, mcmc_params, config_psf, inner_interp, n_circ, n_noncirc, m_hrm = 0):

	galaxy, vel_map, evel_map, theta0  = data
	shape = vel_map.shape
	config_mcmc, step_size = mcmc_params


	Nsteps = config_mcmc.getint('Nsteps', 1000)
	thin = config_mcmc.getint('thin', 1)
	burnin = config_mcmc.getfloat('burnin', int(0.1*Nsteps))
	if burnin < 1:
			burnin = int(burnin*Nsteps)
	else:
			burnin = int(burnin)

	Nwalkers = config_mcmc.getint('Nwalkers')
	PropDist = config_mcmc.get('PropDist',"G")
	Parallel = config_mcmc.getboolean('Parallelize', False)
	Ncpus = config_mcmc.get('Ncpus', False)
	mcmc_sampler = config_mcmc.get('mcmc_sampler', "emcee")
	int_scatter = config_mcmc.getboolean('sigma_int', False)
	if mcmc_sampler not in ["emcee", "zeus"]: print("XS: Choose a valid MCMC sampler !");quit()

	# Real number of steps
	steps = Nsteps - burnin


	if Ncpus ==False:
		Ncpus = 1
	else:
		Ncpus=multiprocessing.cpu_count()-1
	
	vmode, rings_pos, ring_space, pixel_scale, r_bar_min, r_bar_max = model_params

	theta_ = np.hstack(theta0.flatten())
	sigmas = np.hstack(step_size)
	if PropDist == "G":
		if not int_scatter :
			theta_, sigmas = theta_[:-1],sigmas[:-1]

	kinmodel = KinModel( vel_map, evel_map, theta_, vmode, rings_pos, ring_space, pixel_scale, inner_interp, PropDist, m_hrm, n_circ, n_noncirc, shape, config_psf)
	#log_posterior = set_likelihood(vmode, shape, PropDist, n_circ, n_noncirc, kinmodel)
	set_L = set_likelihood(vmode, shape, PropDist, n_circ, n_noncirc, kinmodel,theta_,int_scatter)
	log_posterior = set_L.ln_posterior

	ndim =  len(theta_)
	if Nwalkers == None:
		Nwalkers = int(2*ndim)
	# covariance of proposal distribution.
	cov = sigmas*np.eye(len(theta_))
	cov = np.eye(len(theta_))
	pos = np.empty((Nwalkers, ndim))
	for k in range(Nwalkers):
		theta_prop = prng.multivariate_normal(theta_, cov)
		pos[k] =  theta_prop

	########
	#from src.flemming import set_approxposterior
	#a=set_approxposterior(vmode, shape, PropDist, n_circ, n_noncirc, set_L, theta_, cov)
	#chain,steps,thin,burnin = a.set_ap(Nwalkers, Nsteps)
	#acc_frac = 1
	########	

	#"""

	# Tune emcee moves
	moves=[
		(emcee.moves.DESnookerMove(), 0.1),
		(emcee.moves.DEMove(), 0.9 * 0.9),
		(emcee.moves.DEMove(gamma0=1.0), 0.9 * 0.1),
	]

	# Tune zeus moves
	#moves_z = zeus.moves.GlobalMove()


	print("N_steps:\t %s"%Nsteps)
	print("N_walkers:\t %s"%Nwalkers)
	print("burnin:\t\t %s"%burnin)
	print("thin:\t\t %s"%thin)
	print("parallelize:\t %s"%bool(Parallel))
	if Parallel:
		print("ncpu:\t\t %s"%int(Ncpus))
	print("############################")

	with Pool(Ncpus) as pool:

		if Parallel :
			sampler = emcee.EnsembleSampler(Nwalkers, ndim, log_posterior, moves = moves, pool = pool ) if mcmc_sampler == "emcee" else \
			zeus.EnsembleSampler(Nwalkers, ndim, log_posterior, pool = pool, light_mode = True)

		else:
			sampler = emcee.EnsembleSampler(Nwalkers, ndim, log_posterior, moves = moves) if mcmc_sampler == "emcee" else \
			zeus.EnsembleSampler(Nwalkers, ndim, log_posterior, mu = 1e3, light_mode = True)

		if mcmc_sampler == "emcee":
			pos0, prob, state = sampler.run_mcmc(pos, burnin)	
			sampler.reset()
			sampler.run_mcmc(pos0, steps, progress=True)


		if mcmc_sampler == "zeus":
			sampler.run_mcmc(pos, burnin)
			pos0=sampler.get_last_sample()
			sampler.reset()

			cb0 = zeus.callbacks.AutocorrelationCallback(ncheck=100, nact=10, discard=0.5)
			sampler.run_mcmc(pos0, steps,callbacks=[cb0])
			sampler.summary # Print summary diagnostics
			acc_frac = 1




	if mcmc_sampler == "emcee":
		#autocorrelation time
		act = sampler.get_autocorr_time(quiet = True)[0]
		if np.size(act) == 1 and np.isfinite(act) == False: act = 1e8
		max_act = int(np.max(act))
		print("Autocorrelation time: {0:.2f} steps".format(act), max_act)
		acceptance = sampler.acceptance_fraction
		#There is one acceptance per walker, thus we take the mean
		acc_frac = np.nanmean(acceptance)


	# get the chain
	chain = sampler.get_chain()

	if acc_frac < 0.1:
		print("XookSuut: you got a very low acceptance rate ! ")

	print("accept_rate = ", round(acc_frac,3))
	#best = chain_res_mcmc(galaxy, vmode, sampler, theta0, chain, step_size, steps, thin, burnin, Nwalkers, accept_rate, vel_map.shape, save_plots, rings_pos, ring_space, pixel_scale, inner_interp, PropDist)
	#return best
	return chain, acc_frac, steps, thin, burnin, Nwalkers, PropDist, ndim


