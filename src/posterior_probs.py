import numpy as np
import matplotlib.pylab as plt

class set_likelihood:


	def __init__(self, vmode, shape, pdf, n_circ, n_noncirc, kinmodel, theta_lm, int_scatter):
			self.vmode = vmode
			self.ny, self.nx = shape
			self.pdf = pdf
			self.n_circ = n_circ
			self.n_noncirc = n_noncirc
			self.kinmodel = kinmodel
			self.theta_LM = theta_lm
			self.sigma_int = int_scatter
			self.m = 0

	def g_priors(self,param,mu,sigma):
		# gaussian priors
		lp = -0.5*((param-mu)/sigma)**2
		return lp

	def ln_prior(self, theta):
			""" The prior only depends on the parameters """
			if self.vmode == "circular":
				self.m = self.n_circ
				vrot, pa, inc, x0, y0, vsys =  theta[:self.n_circ],theta[self.m],theta[self.m+1],theta[self.m+2],theta[self.m+3],theta[self.m+4]
				vrot_arr = np.asarray(vrot)
				vrad_arr = vrot_arr*0
				vtan_arr = vrot_arr*0
				phi_b = 45

			if self.vmode == "radial":
				self.m = self.n_circ + 1*self.n_noncirc
				vrot, vrad, pa, inc, x0, y0, vsys =  theta[:self.n_circ],theta[self.n_circ:self.m],theta[self.m],theta[self.m+1],theta[self.m+2],theta[self.m+3],theta[self.m+4]
				vrot_arr = np.asarray(vrot)
				vrad_arr = np.asarray(vrad)
				vtan_arr = vrad_arr*0
				phi_b = 45


			if self.vmode == "bisymmetric":
				self.m = self.n_circ + 2*self.n_noncirc
				vrot, vrad, vtan, pa, inc, x0, y0, vsys, phi_b =  theta[:self.n_circ],theta[self.n_circ:self.n_circ+self.n_noncirc],theta[self.n_circ+self.n_noncirc:self.m],theta[self.m],theta[self.m+1],theta[self.m+2],theta[self.m+3],theta[self.m+4],theta[self.m+5]
				vrot_arr = np.asarray(vrot)
				vrad_arr = np.asarray(vrad)
				vtan_arr = np.asarray(vtan)


			if self.vmode != "hrm":
				mask_1 = list ( abs(vrot_arr) < 450 )
				mask_2 = list ( abs(vrad_arr) < 450 )
				mask_3 = list ( abs(vtan_arr) < 450 )
				mask_product = all( ( mask_1, mask_2, mask_3 ) )# mask_2*mask_3


			if "hrm" in self.vmode:
				if self.pdf == "C" or self.sigma_int:
					pa, inc, x0, y0, vsys = theta[-6:-1]
					V_k = theta[:-6]
				else:
					pa, inc, x0, y0, vsys = theta[-5:]
					V_k = theta[:-5]

				Vk_arr = np.asarray(V_k)
				phi_b = 45

				mask_1 = list ( abs(Vk_arr) < 450 )
				mask_product = all(mask_1)

		    # set prior to 1 (log prior to 0) if in the range and zero (-inf) outside the range
			lp = 0
			lnsigma2 =  1

			if self.pdf == "C":
				lnsigma2 =  theta[-1]
				lp = 0 if -10<lnsigma2<10 else -1*np.inf
			if self.pdf == "G":
				if self.sigma_int:
					lnsigma2 =  theta[-1]
					lp = 0 if -10<lnsigma2<10 else -1*np.inf

			# Uniform priors on parameters
			if all([-360 <= pa <= 360, 0 < inc < 90, 0 < x0 < self.nx, 0 < y0 < self.ny, -180 < phi_b < 180, mask_product]):
				lp = 0 + lp
			else:
				lp = -1*np.inf

			return lp


	def ln_posterior(self, theta):
			""" 
			Up to a normalization constant, the log of the posterior pdf is just 
			the sum of the log likelihood plus the log prior.
			"""
			l_prior = self.ln_prior(theta)
			if np.isinf(l_prior): 
				return l_prior

			l_likelihood = self.ln_likelihood(theta)
			l_post = l_prior + l_likelihood
			if np.isnan(l_post):
				return -1*np.inf

			return l_post


	def ln_likelihood(self, theta):
			if self.pdf == "C":
				beta =  theta[-1]
				theta_mdls = theta[:-1]
			if self.pdf == "G":
				if self.sigma_int:
					beta =  theta[-1]
					theta_mdls = theta[:-1]
				else:
					theta_mdls = theta


			""" The likelihood function evaluation requires a particular set of model parameters and the data """
			model = (self.kinmodel).interp_model
			mdl, obs, error = model(theta_mdls)
			dy = obs - mdl
			N = len(dy)

			if self.pdf == "G":
				# Gaussian
				if self.sigma_int:
					sigma2 = error**2 + np.exp(2*beta)
					log_L = -0.5*np.sum(dy**2/sigma2 +  np.log(sigma2) )
				else:
					resid = dy / error
					log_L = -0.5*resid@resid - np.sum( np.log(error) )

			if self.pdf == "C":
				# Cauchy
				log_L = -(N + 1) * np.log(beta) - np.sum(np.log(1.0 + ( dy / beta )**2) )

			return log_L


	#def __call__(self, theta):
	#		return self.ln_posterior(theta)

