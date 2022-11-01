import numpy as np
import emcee
from scipy import stats
import multiprocessing 
from multiprocessing import Pool
from src.interp_tools import interp_loops
from src.posterior_probs import set_likelihood


prng =  np.random.RandomState(12345)

def Metropolis(KinModel, data, model_params, mcmc_params, inner_interp, n_circ, n_noncirc, m_hrm = 0):

	galaxy, vel_map, evel_map, theta0  = data
	shape = vel_map.shape
	config_mcmc, step_size = mcmc_params


	Nsteps = config_mcmc.getint('Nsteps', 1000)
	thin = config_mcmc.getint('thin', 1)
	burnin = config_mcmc.getint('burnin', int(0.1*Nsteps))
	Nwalkers = config_mcmc.getint('Nwalkers')
	PropDist = config_mcmc.get('PropDist',"G")
	Parallel = config_mcmc.getboolean('Parallelize', False)
	Ncpus = config_mcmc.get('Ncpus', False)
	if Ncpus ==False:
		Ncpus = 1
	else:
		Ncpus=multiprocessing.cpu_count()-1
	
	vmode, rings_pos, ring_space, pixel_scale, r_bar_min, r_bar_max = model_params

	theta_ = np.hstack(theta0.flatten())
	sigmas = np.hstack(step_size)

	kinmodel = KinModel( vel_map, evel_map, theta_, vmode, rings_pos, ring_space, pixel_scale, inner_interp, PropDist, m_hrm, n_circ, n_noncirc, shape)
	log_posterior = set_likelihood(vmode, shape, PropDist, n_circ, n_noncirc, kinmodel)

	ndim =  len(theta_)
	if Nwalkers == None:
		Nwalkers = int(2*ndim)
	# covariance of proposal distribution.
	cov = sigmas*np.eye(len(theta_))
	pos = np.empty((Nwalkers, ndim))
	for k in range(Nwalkers):
			theta_prop = prng.multivariate_normal(theta_, cov)
			#theta_prop = theta_ + stats.cauchy.rvs(loc=0, scale=0.5, size=ndim)
			pos[k] =  theta_prop

	# Tune the moves
	moves=[
		(emcee.moves.DESnookerMove(), 0.1),
		(emcee.moves.DEMove(), 0.9 * 0.9),
		(emcee.moves.DEMove(gamma0=1.0), 0.9 * 0.1),
	]


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
			sampler = emcee.EnsembleSampler(Nwalkers, ndim, log_posterior, moves = moves, pool = pool )
		else:
			sampler = emcee.EnsembleSampler(Nwalkers, ndim, log_posterior, moves = moves)

		pos0, prob, state= sampler.run_mcmc(pos, burnin)	
		sampler.reset()
		# Real number of steps
		steps = Nsteps - burnin
		pos, prob, state = sampler.run_mcmc(pos0, steps, progress=True)


	#autocorrelation time
	act = sampler.get_autocorr_time(quiet = True)[0]
	if np.size(act) == 1 and np.isfinite(act) == False: act = 1e8
	max_act = int(np.max(act))
	#thin = max_act

	print("Autocorrelation time: {0:.2f} steps".format(act), max_act)

	#chain = sampler.get_chain(discard=burnin, thin=thin, flat = True)
	chain = sampler.chain
	acceptance = sampler.acceptance_fraction
	#There is one acceptance per walker, thus we take the mean
	acc_frac = np.nanmean(acceptance)
	#This does not have sense here
	accept_rate = np.ones(steps,)*acc_frac

	if acc_frac < 0.1:
		print("XookSuut: you got a very low acceptance rate ! ")

	print("accept_rate = ", round(acc_frac,3))
	#best = chain_res_mcmc(galaxy, vmode, sampler, theta0, chain, step_size, steps, thin, burnin, Nwalkers, accept_rate, vel_map.shape, save_plots, rings_pos, ring_space, pixel_scale, inner_interp, PropDist)
	#return best
	return chain, acc_frac, steps, thin, burnin, Nwalkers, PropDist, ndim


