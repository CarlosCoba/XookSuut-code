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
			self.pa, self.inc, self.x0, self.y0, self.phi_b = 0, 0, 0, 0, 45

			model = (self.kinmodel).interp_model
			self.mdl_LM, obs, error = model(theta_lm)



	def g_priors(self,param,mu,sigma):
		# gaussian priors
		lp = -0.5*((param-mu)/sigma)**2
		return lp

	def ln_prior(self, theta):
			""" The prior only depends on the parameters """
			if self.vmode == "circular":
				self.m = self.n_circ
				vrot, pa, inc, x0, y0, vsys =  theta[:self.n_circ],theta[self.m],theta[self.m+1],theta[self.m+2],theta[self.m+3],theta[self.m+4]
				self.pa, self.inc, self.x0, self.y0, self.vsys = self.theta_LM[self.m],self.theta_LM[self.m+1],self.theta_LM[self.m+2],self.theta_LM[self.m+3],self.theta_LM[self.m+4]
				vrot_arr = np.asarray(vrot)
				vrad_arr = vrot_arr*0
				vtan_arr = vrot_arr*0
				phi_b = 45

			if self.vmode == "radial":
				self.m = self.n_circ + 1*self.n_noncirc
				vrot, vrad, pa, inc, x0, y0, vsys =  theta[:self.n_circ],theta[self.n_circ:self.m],theta[self.m],theta[self.m+1],theta[self.m+2],theta[self.m+3],theta[self.m+4]
				self.pa, self.inc, self.x0, self.y0, self.vsys = self.theta_LM[self.m],self.theta_LM[self.m+1],self.theta_LM[self.m+2],self.theta_LM[self.m+3],self.theta_LM[self.m+4]
				vrot_arr = np.asarray(vrot)
				vrad_arr = np.asarray(vrad)
				vtan_arr = vrad_arr*0
				phi_b = 45


			if self.vmode == "bisymmetric":
				self.m = self.n_circ + 2*self.n_noncirc
				vrot, vrad, vtan, pa, inc, x0, y0, vsys, phi_b =  theta[:self.n_circ],theta[self.n_circ:self.n_circ+self.n_noncirc],theta[self.n_circ+self.n_noncirc:self.m],theta[self.m],theta[self.m+1],theta[self.m+2],theta[self.m+3],theta[self.m+4],theta[self.m+5]

				self.pa, self.inc, self.x0, self.y0, self.vsys, self.phi_b = self.theta_LM[self.m],self.theta_LM[self.m+1],self.theta_LM[self.m+2],self.theta_LM[self.m+3],self.theta_LM[self.m+4],self.theta_LM[self.m+5]


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
					self.pa, self.inc, self.x0, self.y0, self.vsys = self.theta_LM[-6:-1]
					V_k = theta[:-6]
				else:
					pa, inc, x0, y0, vsys = theta[-5:]
					self.pa, self.inc, self.x0, self.y0, self.vsys = self.theta_LM[-5:]
					V_k = theta[:-5]				

				Vk_arr = np.asarray(V_k)
				phi_b = 45
				mask_1 = list ( abs(Vk_arr) < 450 )
				mask_product = all(mask_1)
				# Accept any velocity ?
				#mask_product = True

			#trick the chain ?
			#pa, inc, x0, y0  = self.pa, self.inc, self.x0, self.y0

		    # set prior to 1 (log prior to 0) if in the range and zero (-inf) outside the range
			lp = 0
			lnsigma2 =  1

			if self.pdf == "C":
				lnsigma2 =  theta[-1]
				lp = 0 if 0<lnsigma2<100 else -1*np.inf
			if self.pdf == "G":
				if self.sigma_int:
					lnsigma2 =  theta[-1]
					lp = 0 if -10<lnsigma2<10 else -1*np.inf

			sin_25, sin_80 = np.sin(25*np.pi/180), np.sin(80*np.pi/180)
			# Uniform priors on parameters
			if all([-1 <= np.sin(pa*np.pi/180) <= 1, sin_25 < np.sin(inc*np.pi/180) < sin_80, 0 < x0 < self.nx, \
			0 < y0 < self.ny, -1 < np.sin(phi_b*np.pi/180) < 1, mask_product, self.vsys*(1-300/self.vsys) < vsys < self.vsys*(1+300/self.vsys) ]):
			# Gaussian priors ?
				lp = 0 + lp
				lp += self.g_priors(self.x0,x0,5)
				lp += self.g_priors(self.y0,y0,5)
				lp += self.g_priors(self.inc,inc,10)
				lp += self.g_priors(self.pa,pa,10)
				lp += self.g_priors(self.vsys,vsys,50)
				lp += self.g_priors(self.phi_b,phi_b,30)

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
				l_post = -1*np.inf
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
				# Gaussian with intrinsic dispersion
				if self.sigma_int:
					sigma2 = error**2 + np.exp(beta)
					log_L = -0.5*np.sum(dy**2/sigma2 +  np.log(sigma2)) -0.5*N*np.log(2*np.pi)
				else:
				# Only Gaussian model
					resid = dy / error
					log_L = -0.5*resid@resid - np.sum( np.log(error) ) -0.5*N*np.log(2*np.pi)

			if self.pdf == "C":
				# Cauchy
				log_L = -(N + 1) * np.log(beta) - np.sum(np.log(1.0 + ( dy / beta )**2) )
			if log_L ==0:log_L = -np.inf
			return log_L


	#def __call__(self, theta):
	#		return self.ln_posterior(theta)

