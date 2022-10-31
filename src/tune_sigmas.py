import numpy as np
from scipy.stats import multivariate_normal


def tune_step(theta, sigmas, chain, prng, i, cov0, delay = 0, period = 50):
		theta,sigmas = np.asarray(theta), np.asarray(sigmas)

		"""
		Adaptive Metropolis algorithm

		"""
		naccepted = np.unique(chain[:, 0]).size
		ntheta = theta.size
		[ny,nx] = chain.shape

		# scaling factor
		s_d = (2.38**2/ntheta)
		epsilon = 1e-2

		#if naccepted > ntheta*(ntheta + 1.)/2.:
		#if naccepted > ntheta:
		if naccepted > delay and i % period == 0:


			diag = np.diag((sigmas*epsilon)**2)
			cov0 = s_d*(np.cov(chain.T) + diag)
			theta = prng.multivariate_normal(theta, cov0)


			#Haario et al. (2001) Method
			#Covariance matrix proposal distribution
			#cov0 = s_d * ( np.cov(chain.T) +  epsilon*np.eye(nx) )
			#theta = prng.multivariate_normal(theta, cov0)


			#Jeffrey S. Rosentha +2008 Method
			#Covariance matrix proposal distribution
			#s1 = s_d * ( np.cov(chain.T) ) 
			#s2 = epsilon*sigmas**2*np.eye(nx)/ntheta
			#theta = (1-0.05)*prng.multivariate_normal(theta, s1) + 0.05*prng.multivariate_normal(theta, s2)


		else:

			theta = prng.multivariate_normal(theta, cov0)
			#theta = prng.normal(theta, sigmas)
		return theta, cov0

