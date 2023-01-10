import numpy as np
import emcee
from scipy import stats
import multiprocessing 
from multiprocessing import Pool
from src.interp_tools import interp_loops
from src.posterior_probs import set_likelihood
import time
import zeus
import dynesty
from dynesty import NestedSampler,DynamicNestedSampler
from dynesty.utils import resample_equal
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc
from src.dynesty_plots import dplots
import matplotlib.pylab as plt


# chain shape: (nsteps, nwalkers, ndim)
def Metropolis(KinModel, data, model_params, mcmc_params, config_psf, inner_interp, n_circ, n_noncirc, m_hrm = 0):

	galaxy, vel_map, evel_map, theta0  = data
	shape = vel_map.shape
	config_mcmc, step_size = mcmc_params

	theta_ = np.hstack(theta0.flatten())
	sigmas = np.hstack(step_size)

	Nsteps = config_mcmc.getint('Nsteps', 1000)
	thin = config_mcmc.getint('thin', 1)
	burnin = config_mcmc.getfloat('burnin', int(0.1*Nsteps))
	if burnin < 1:
			burnin = int(burnin*Nsteps)
	else:
			burnin = int(burnin)

	# emcee and zeus configurations
	Nwalkers = config_mcmc.getint('Nwalkers')
	PropDist = config_mcmc.get('PropDist',"G")
	Parallel = config_mcmc.getboolean('parallelize', False)
	Ncpus = config_mcmc.getint('Ncpus', False)
	mcmc_sampler = config_mcmc.get('mcmc_sampler', "emcee")
	int_scatter = config_mcmc.getboolean('sigma_int', False)

	# Dynesty configurations
	dlogz_init = config_mcmc.getfloat('dlogz_init', 0.1)
	maxiter  = config_mcmc.getint('maxiter',25e3)
	maxbatch = config_mcmc.getint('maxbatch',5)
	nlive = config_mcmc.getint('nlive',10*len(theta_))
	sample = config_mcmc.get('sample','auto')
	priors_dynesty = config_mcmc.getboolean('priors',1)
	bound = config_mcmc.get('bound',"single")
	periodic, reflective = None, None

	if mcmc_sampler not in ["emcee", "zeus", "dynesty"]: print("XS: Choose a valid MCMC sampler !");quit()

	# Real number of steps
	steps = Nsteps - burnin

	check_ncpus = multiprocessing.cpu_count()
	if Ncpus == False:
		Ncpus = check_ncpus-1 if check_ncpus >=2 else 1   
	
	vmode, rings_pos, ring_space, pixel_scale, r_bar_min, r_bar_max = model_params

	if PropDist == "G":
		if not int_scatter :
			theta_, sigmas = theta_[:-1],sigmas[:-1]
			#theta0 = theta0[:-1]

	if vmode == "bisymmetric":
		if ((PropDist == "G") and (int_scatter!=True)): bound_pabar = -1 
		if PropDist == "C" or int_scatter: bound_pabar = -2 
		periodic = [bound_pabar] if not priors_dynesty else periodic

	kinmodel = KinModel( vel_map, evel_map, theta_, vmode, rings_pos, ring_space, pixel_scale, inner_interp, PropDist, m_hrm, n_circ, n_noncirc, shape, config_psf)
	#log_posterior = set_likelihood(vmode, shape, PropDist, n_circ, n_noncirc, kinmodel)
	set_L = set_likelihood(vmode, shape, PropDist, n_circ, n_noncirc, kinmodel,theta_,int_scatter, pixel_scale, m_hrm, priors_dynesty, mcmc_sampler)
	log_likelihood = set_L.ln_likelihood
	prior_transform = set_L.prior_transform
	log_posterior = set_L.ln_posterior

	ndim =  len(theta_)
	if Nwalkers == None:
		Nwalkers = int(2*ndim)
	# covariance of proposal distribution.
	cov = sigmas*np.eye(len(theta_))
	#cov = np.eye(len(theta_))
	pos = np.empty((Nwalkers, ndim))

	seed0 = int(time.time())
	pnrg = np.random.RandomState(seed0)
	for k in range(Nwalkers):
		theta_prop = pnrg.multivariate_normal(theta_, cov)
		#theta_prop = pnrg.normal(theta_, 0.001)
		pos[k] =  theta_prop

	#"""

	# Tune emcee moves
	moves=[
		(emcee.moves.DESnookerMove(), 0.1),
		(emcee.moves.DEMove(), 0.9 * 0.9),
		(emcee.moves.DEMove(gamma0=1.0), 0.9 * 0.1),
	]

	# Tune zeus moves
	moves_z = [(zeus.moves.DifferentialMove(), 0.1), (zeus.moves.GaussianMove(), 0.9)]
	print("sampler:\t %s"%mcmc_sampler)
	if mcmc_sampler in ["emcee","zeus"]:
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
			if mcmc_sampler == "emcee":
				sampler = emcee.EnsembleSampler(Nwalkers, ndim, log_posterior, moves = moves, pool = pool )
			if mcmc_sampler == "zeus":
				sampler = zeus.EnsembleSampler(Nwalkers, ndim, log_posterior, light_mode=True, pool = pool )
			if mcmc_sampler == "dynesty":
				priors = "Truncated-gaussians" if priors_dynesty else "Uniform"
				print("Dynesty parameters")
				print(" bound:\t\t %s \n sample:\t %s \n dlogz_init:\t %s \n maxiter:\t %s \n maxbatch:\t %s \n nlive:\t\t %s \n priors:\t %s"\
				%(bound,sample,dlogz_init,maxiter,maxbatch,nlive,priors))
				print("############################")
				sampler = DynamicNestedSampler(log_likelihood, prior_transform, ndim, bound=bound, sample=sample, \
				pool=pool, queue_size=Ncpus, periodic = periodic)

		else:
			if mcmc_sampler == "dynesty": print("XS: Dynesty can only be runned in Parallel");quit()
			sampler = emcee.EnsembleSampler(Nwalkers, ndim, log_posterior, moves = moves) if mcmc_sampler == "emcee" else \
			zeus.EnsembleSampler(Nwalkers, ndim, log_posterior, mu = 1e3)

		if mcmc_sampler == "dynesty":
			sampler.run_nested(dlogz_init=dlogz_init, nlive_init=nlive, nlive_batch=100, maxiter=maxiter, maxbatch=maxbatch, use_stop=False) 

			res = sampler.results # get results dictionary from sampler
			# Print out a summary of the results.
			res.summary()			
			weights = res.importance_weights()
			samples = res.samples # samples
			chain0= res.samples_equal() # These are the samples we are interested in

			nwalkers, nparams = chain0.shape
			nwalkers, steps, ndim = nwalkers, 1, nparams
			chain = np.zeros((nwalkers, steps,ndim))
			chain[:,0,:] = chain0
			dplots(res, theta_, vmode, galaxy, PropDist, int_scatter, n_circ, n_noncirc)


		if mcmc_sampler == "emcee":
			print("running burnin period ..")
			pos0, prob, state = sampler.run_mcmc(pos, burnin, progress=True)
			sampler.reset()
			print("running post-burnin chains ..")
			sampler.run_mcmc(pos0, steps, progress=True)

		if mcmc_sampler == "zeus":
			print("running burnin period ..")
			sampler.run_mcmc(pos, burnin)
			pos0=sampler.get_last_sample()
			sampler.reset()
			print("running post-burnin chains ..")
			sampler.run_mcmc(pos0, steps)
			sampler.summary # Print summary diagnostics



	# get the chain
	if mcmc_sampler in ["emcee", "zeus"]: chain = sampler.get_chain()

	if mcmc_sampler == "emcee":
		#autocorrelation time
		act = sampler.get_autocorr_time(quiet = True)[0]
		acceptance = sampler.acceptance_fraction
		#There is one acceptance per walker, thus we take the mean
		acc_frac = np.nanmean(acceptance)
	if mcmc_sampler == "zeus":
		act = zeus.AutoCorrTime(chain)
		acc_frac = 1
	if mcmc_sampler == "dynesty":
		acc_frac = 1
		act = [1]*ndim

	if np.size(act) == 1 and np.isfinite(act) == False: act = 1e8
	max_act = int(np.nanmean(act)) if int(np.mean(act)) !=0 else 1e8
	print("Autocorrelation time: %s steps"%max_act)
	print("The chain contains %s times the autocorrelation time"%int(steps/max_act))
	print("accept_rate = ", round(acc_frac,2))

	if acc_frac < 0.1:
		print("XookSuut: you got a very low acceptance rate ! ")

	## NOTE ! if sampler is emcee or zeus we need to transform -1<phi_b<1 --> radians
	if all( [ (PropDist == "C" or int_scatter), vmode == "bisymmetric", mcmc_sampler in ["emcee","zeus"] ] ):
		chain[:,:,-2] = np.arccos(chain[:,:,-2]); theta0[-2] = np.arccos(theta0[-2])
	if all( [ (PropDist == "G" and not int_scatter), vmode == "bisymmetric", mcmc_sampler in ["emcee","zeus"] ] ):
		chain[:,:,-1] = np.arccos(chain[:,:,-1]); theta0[-2] = np.arccos(theta0[-2])

	return chain, acc_frac, steps, thin, burnin, Nwalkers, PropDist, ndim, max_act, theta0


