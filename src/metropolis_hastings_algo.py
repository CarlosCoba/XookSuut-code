import numpy as np
from scipy.stats import multivariate_normal
from tune_sigmas import tune_step
prng =  np.random.RandomState(12345)
  


def run_metropolis_hastings(theta, n_steps, model, proposal_sigmas):
	"""
	Run a Metropolis-Hastings MCMC sampler to generate samples from the input
	log-posterior function, starting from some initial parameter vector.
	
	Parameters
	----------
	theta : iterable
		Initial parameter vector.
	n_steps : int
		Number of steps to run the sampler for.
	model : set_posterior instance (or subclass)
		A callable object that takes a parameter vector and computes 
		the log of the posterior pdf.
	proposal_sigmas : list, array
		A list of standard-deviations passed to the sample_proposal 
		function. These are like step sizes in each of the parameters.
	"""
	theta = np.array(theta)
	if len(proposal_sigmas) != len(theta):
		raise ValueError("theta must have the same dimmesions as the sigmas proposal.")

	
	# the objects we'll fill and return:
	chain = np.zeros((n_steps, len(theta))) # parameter values at each step
	ln_probs = np.zeros(n_steps) # log-probability values at each step
	
	# we'll keep track of how many steps we accept to compute the acceptance fraction						
	n_accept = 0 
	
	# evaluate the log-posterior at the initial position and store starting position in chain
	ln_probs[0] = model(theta)
	chain[0] = theta

	accept_rate = np.zeros(n_steps)
	rejected = []
	accepted = []
	ndim = len(theta)
	# proposal covariance
	cov = proposal_sigmas**2*np.eye(theta.size)
	# loop through the number of steps requested and run MCMC
	for i in range(1,n_steps):
		# proposed new parameters
		tune = True
		if tune == True:
			new_theta, cov = tune_step(theta, proposal_sigmas, chain[:i-1], prng, i, cov)
		#else:
		#	new_theta = prng.normal(theta, proposal_sigmas)

		# compute log-posterior at new parameter values
		new_ln_prob = model(new_theta)

		# log of the ratio of the new log-posterior to the previous log-posterior value
		ln_prob_ratio = new_ln_prob - ln_probs[i-1]

		#if (ln_prob_ratio > 0) or (ln_prob_ratio > np.log(np.random.uniform())):
		if (ln_prob_ratio > 0) or (ln_prob_ratio > np.log(prng.uniform())):
			chain[i] = new_theta
			ln_probs[i] = new_ln_prob
			n_accept += 1
			accepted.append(new_theta)
			
		else:
			chain[i] = chain[i-1]
			ln_probs[i] = ln_probs[i-1]

		theta = chain[i]
		accept_rate[i] = n_accept / i
	
	acc_frac = n_accept / n_steps

	accepted = np.asarray(accepted)
	n_accepted = len(accepted)
	accepted_chain = accepted.reshape((n_accepted,ndim))


	return chain, ln_probs, acc_frac, accept_rate
	#return accepted_chain, ln_probs, acc_frac, accept_rate






