import numpy as np
import corner
import matplotlib.pylab as plt
import george
from approxposterior import approx, gpUtils as gpu

prng =  np.random.RandomState(12345)


class set_approxposterior:


	def __init__(self, vmode, shape, pdf, n_circ, n_noncirc, set_L, theta, cov):

			self.vmode = vmode
			self.ny, self.nx = shape
			self.pdf = pdf
			self.n_circ = n_circ
			if vmode == "cicular": k = 0
			if vmode == "radial": k = 1
			if vmode == "bisymmetric": k = 2

			self.n_noncirc = k*n_noncirc
			self.set_L = set_L
			self.m = 0
			self.theta = theta
			self.cov = cov

			self.ndim = len(theta)
			if vmode != "bisymmetric":
				self.vsys = theta[-1]
			else:
				self.vsys = theta[-2]


	def sampleFunction(self,n):
		pos = np.empty((n, self.ndim))
		for k in range(n):
			theta_prop = prng.multivariate_normal(self.theta, self.cov)
			pos[k] =  theta_prop

		#return np.array([m,b]).T
		return pos




	def bounds_vmode(self):
		kin_bonds = (-450, 450)
		pa_bonds = [(-360, 360)]
		inc_bonds = [(0, 90)]
		xc_bonds = [(0, self.nx)]
		yc_bonds = [(0, self.ny)]
		vsys_bonds = [(self.vsys-500,self.vsys+500)]
		vsys_bonds = [(0, np.inf)]
		dip_bonds = [(-10,10)]
		bonds = [kin_bonds for k in np.arange(self.n_circ + self.n_noncirc)]
		bonds = bonds + pa_bonds + inc_bonds + xc_bonds + yc_bonds + vsys_bonds# + dip_bonds
		if self.vmode == "bisymmetric": bonds = bonds + [(-180,180)]
		return bonds




	def set_ap(self, walkers, steps):

		# Define algorithm parameters
		m0 = 10                           # Initial size of training set
		m = 10                            # Number of new points to find each iteration
		nmax = 2                          # Maximum number of iterations
		bounds = self.bounds_vmode()			  # Prior bounds
		algorithm = "bape"                # Use the Kandasamy et al. (2017) formalism

		# emcee MCMC parameters: Use the same MCMC parameters as the emcee-only analysis
		samplerKwargs = {"nwalkers" :  walkers}  # emcee.EnsembleSampler parameters
		mcmcKwargs = {"iterations" : steps} # emcee.EnsembleSampler.run_mcmc parameters

		# Data and uncertainties that we use to condition our model
		#args = (x, obs, obserr)

		# Create a training set to condition the GP

		# Randomly sample initial conditions from the prior
		#theta = sampleFunction(m0)
		theta_sample = self.sampleFunction(m0)

		# Evaluate forward model to compute log likelihood + lnprior for each theta
		y = list()
		logPost = (self.set_L).ln_posterior
		logPrior = (self.set_L).ln_prior
		logLikelihood = (self.set_L).ln_likelihood

		for ii in range(len(theta_sample)):
			y.append(logPost(theta_sample[ii]))
		y = np.array(y)


		# We'll create the initial GP using approxposterior's built-in default
		# initialization.  This default typically works well in many applications.
		gp = gpu.defaultGP(theta_sample, y)


		ap = approx.ApproxPosterior(theta=theta_sample,            # Initial model parameters for inputs
				                    y=y,                           # Logprobability of each input
				                    gp=gp,                         # Initialize Gaussian Process
				                    lnprior=logPrior,              # logprior function
				                    lnlike=logLikelihood,          # loglikelihood function
				                    priorSample=self.sampleFunction,    # Prior sample function
				                    algorithm=algorithm,           # bape, agp, or alternate
				                    bounds=bounds)                 # Parameter bounds





		# Run!
		ap.run(m=m, nmax=nmax,estBurnin=True, mcmcKwargs=mcmcKwargs, cache=False,
			   samplerKwargs=samplerKwargs, verbose=False, onlyLastMCMC=True,
			   )
		

		chain = ap.sampler.get_chain()
		burn, thin = ap.iburns[-1], ap.ithins[-1]
		print(chain.shape, burn, thin)

		samples = ap.sampler.get_chain(discard=ap.iburns[-1], flat=True, thin=ap.ithins[-1])

		"""
		fig = corner.corner(samples, quantiles=[0.16, 0.5, 0.84], scale_hist=True,
				            plot_contours=True );
		plt.savefig("test.png")
		"""
		return chain,steps,thin,burn 





