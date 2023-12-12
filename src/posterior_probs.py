import numpy as np
import matplotlib.pylab as plt
from scipy.special import ndtri
from scipy.stats import cauchy
from scipy.stats import truncnorm

class set_likelihood:


	def __init__(self, vmode, shape, pdf, n_circ, n_noncirc, kinmodel, theta_lm, int_scatter, pixel_scale, m_hrm, priors, mcmc_sampler):
			self.vmode = vmode
			self.ny, self.nx = shape
			self.pdf = pdf
			self.n_circ = n_circ
			self.n_noncirc = n_noncirc
			self.kinmodel = kinmodel
			self.theta_LM = theta_lm
			self.sigma_int = int_scatter
			self.pa, self.eps, self.x0, self.y0, self.phi_b = 0, 0, 0, 0, np.pi/4.
			self.vrot = 0
			self.vrad = self.vrot*0
			self.vtan = self.vrot*0
			self.v_k = 0
			self.pixel_scale = pixel_scale
			self.m_hrm = m_hrm
			self.priorvol = priors
			self.mcmc_sampler = mcmc_sampler

			model = (self.kinmodel).interp_model
			self.mdl_LM, obs, error = model(theta_lm)

			if self.vmode == "circular":
				self.m = self.n_circ
			if self.vmode == "bisymmetric":
				self.m = self.n_circ + 2*self.n_noncirc
			if self.vmode == "radial":
				self.m = self.n_circ + 1*self.n_noncirc
			if "hrm" in self.vmode:
				self.m = self.n_circ + self.n_noncirc*(2*self.m_hrm-1) 

	def prior_transform(self,theta_u):
		"""
		Prior transformation for Dynesty.
		Transforms the uniform random variable `u ~ Unif[0., 1.)`
		to the parameter of interest."""

		# Here will be saved the transformed values
		theta = theta_u*0
		self.vrot = self.theta_LM[:self.n_circ]
		self.pa, self.eps, self.x0, self.y0, self.vsys = self.theta_LM[self.m],self.theta_LM[self.m+1],self.theta_LM[self.m+2],self.theta_LM[self.m+3],self.theta_LM[self.m+4]
		vrot_u, pa_u, eps_u, x0_u, y0_u, vsys_u =  theta_u[:self.n_circ],theta_u[self.m],theta_u[self.m+1],theta_u[self.m+2],theta_u[self.m+3],theta_u[self.m+4]

		# Define the priors for nesting sample: Gaussian or Uniform
		def tn(uparam,mean,sigma,p_min,p_max):
			# Truncated Normal priors
			low_n, high_n = (p_min - mean) / sigma, (p_max - mean) / sigma # standarize
			n = truncnorm.ppf(uparam, low_n,  high_n, loc = mean, scale = sigma)
			return n
		def up(uparam, p_min, p_max ):
			# Uniform priors
			m = (p_max-p_min)
			p=m*uparam + p_min
			return p

		vrot = np.zeros_like(vrot_u)
		for i in range(len(self.vrot)):
			# Truncated Normal
			vmin, vmax = self.vrot[i]-150, self.vrot[i]+150
			vrot[i] = tn(vrot_u[i], self.vrot[i], 50, vmin, vmax) if self.priorvol else up(vrot_u[i],0, 400)

		if self.vmode == "radial":
			vrad_u,self.vrad = theta_u[self.n_circ:self.m], self.theta_LM[self.n_circ:self.m]
			vrad = np.zeros_like(vrad_u)
			for i in range(len(self.vrad)):
				# Truncated Normal
				vmin, vmax = self.vrad[i]-75, self.vrad[i]+75
				vrad[i] = tn(vrad_u[i], self.vrad[i], 50, vmin, vmax) if self.priorvol else up(vrad_u[i],-250, 250)

		if self.vmode == "bisymmetric":
			self.phi_b, phi_b_u = self.theta_LM[self.m+5], theta_u[self.m+5]
			vrad_u,self.vrad = theta_u[self.n_circ:self.n_circ+self.n_noncirc], self.theta_LM[self.n_circ:self.n_circ+self.n_noncirc]
			vtan_u,self.vtan = theta_u[self.n_circ+self.n_noncirc:self.m], self.theta_LM[self.n_circ+self.n_noncirc:self.m]

			vrad = np.zeros_like(vrad_u)
			vtan = np.zeros_like(vtan_u)
			for i in range(len(self.vrad)):
				# Truncated Normal
				vmin, vmax = self.vrad[i]-75, self.vrad[i]+75
				vrad[i] = tn(vrad_u[i], self.vrad[i], 50, vmin, vmax) if self.priorvol else up(vrad_u[i],-250, 250)
				# Truncated Normal
				vmin, vmax = self.vtan[i]-75, self.vtan[i]+75
				vtan[i] = tn(vtan_u[i], self.vtan[i], 50, vmin, vmax) if self.priorvol else up(vtan_u[i],-250, 250)

			min_phib, max_phib = self.phi_b-np.pi/2. , self.phi_b+np.pi/2.
			phi_b = tn(phi_b_u, self.phi_b, 20*np.pi/180., min_phib, max_phib) if self.priorvol else (phi_b_u % 1.)*2*np.pi #np.arccos(2*phi_b_u - 1)

		if "hrm" in self.vmode:
			v_k_u = theta_u[:self.m]
			v_k = np.zeros_like(v_k_u)
			self.v_k = self.theta_LM[:self.m]
			for i in range(len(self.v_k)):
				vmin, vmax = self.v_k[i]-75, self.v_k[i]+75
				v_k[i] =  tn(v_k_u[i], self.v_k[i], 5, vmin, vmax) if self.priorvol else up(v_k_u[i],-400, 400)

		if self.pdf == "C":
			lnsigma2_u = theta_u[-1]
			lnsigma2 = self.theta_LM[-1]
			#uniform priors for lnsigma
			lnsigma_min , lnsigma_max =1e-5, 100
			lnsigma = lnsigma_max*lnsigma2_u +  lnsigma_min
			theta[-1] = lnsigma
		if self.sigma_int and self.pdf == "G":
			lnsigma2_u = theta_u[-1]
			lnsigma2 = self.theta_LM[-1]
			#uniform priors for lnsigma
			lnsigma_min , lnsigma_max =-10, 10
			lnsigma = 2*lnsigma_max*lnsigma2_u +  lnsigma_min
			theta[-1] = lnsigma

		#priors for pa
		pa_min , pa_max = self.pa-45, self.pa+45
		pa = tn(pa_u, self.pa, 15, pa_min , pa_max) #if self.priorvol else np.arccos(2*pa_u - 1)*360/np.pi

		#priors for xc
		rarc = 10./self.pixel_scale # search around 10 arcsecs from the kinematic centre
		xc_min, xc_max = self.x0-rarc,self.x0+rarc
		x0 = tn(x0_u, self.x0, 2./self.pixel_scale, xc_min, xc_max) #if self.priorvol else up(x0_u,xc_min, xc_max)

		#priors for yc
		yc_min, yc_max = self.y0-rarc,self.y0+rarc
		y0 = tn(y0_u, self.y0, 2./self.pixel_scale, yc_min, yc_max) #if self.priorvol else up(y0_u, yc_min, yc_max)

		#priors for eps
		eps_min,eps_max = 1-np.cos(15*np.pi/180), 1-np.cos(80*np.pi/180)
		eps = tn(eps_u, self.eps, 0.05, eps_min,eps_max) #if self.priorvol else up(eps_u, eps_min,eps_max)

		#priors for vsys
		vsys0=self.theta_LM[self.m+4]
		vsys_min, vsys_max = vsys0*(1-100/vsys0), vsys0*(1+100/vsys0)
		vsys = tn(vsys_u, vsys0, 50, vsys_min, vsys_max) if self.priorvol else up(vsys_u, vsys_min, vsys_max)


		if self.vmode == "circular":
			theta[:self.n_circ],theta[self.m],theta[self.m+1],theta[self.m+2],theta[self.m+3],theta[self.m+4] = \
			vrot, pa, eps, x0, y0, vsys
		if self.vmode == "radial":
			theta[:self.n_circ],theta[self.n_circ:self.m],theta[self.m],theta[self.m+1],theta[self.m+2],theta[self.m+3],theta[self.m+4] = \
			vrot, vrad, pa, eps, x0, y0, vsys
		if self.vmode == "bisymmetric":
			theta[:self.n_circ],theta[self.n_circ:self.n_circ+self.n_noncirc], theta[self.n_circ+self.n_noncirc:self.m], theta[self.m],theta[self.m+1],theta[self.m+2],theta[self.m+3],theta[self.m+4], theta[self.m+5] = vrot, vrad, vtan, pa, eps, x0, y0, vsys, phi_b
		if "hrm" in self.vmode:
			theta[:self.m],theta[self.m],theta[self.m+1],theta[self.m+2],theta[self.m+3],theta[self.m+4] = \
			v_k, pa, eps, x0, y0, vsys
		return theta

	
	# priors for MCMC analysis
	def g_priors(self,param,mu,sigma):
		# gaussian priors
		lp = -0.5*((param-mu)/sigma)**2
		return lp

	def ln_prior(self, theta):

			""" The prior only depends on the parameters """
			if self.vmode == "circular":
				vrot, pa, eps, x0, y0, vsys =  theta[:self.n_circ],theta[self.m],theta[self.m+1],theta[self.m+2],theta[self.m+3],theta[self.m+4]
				self.pa, self.eps, self.x0, self.y0, self.vsys = self.theta_LM[self.m],self.theta_LM[self.m+1],self.theta_LM[self.m+2],self.theta_LM[self.m+3],self.theta_LM[self.m+4]
				vrot_arr,vrad_arr,vtan_arr = np.asarray(vrot),np.asarray(vrot)*0,np.asarray(vrot)*0
				self.vrot = self.theta_LM[:self.n_circ]
				self.vrad = self.vrot*0
				self.vtan = self.vrot*0
				phi_b = np.pi/4.

			if self.vmode == "radial":
				vrot, vrad, pa, eps, x0, y0, vsys =  theta[:self.n_circ],theta[self.n_circ:self.m],theta[self.m],theta[self.m+1],theta[self.m+2],theta[self.m+3],theta[self.m+4]
				self.pa, self.eps, self.x0, self.y0, self.vsys = self.theta_LM[self.m],self.theta_LM[self.m+1],self.theta_LM[self.m+2],self.theta_LM[self.m+3],self.theta_LM[self.m+4]
				self.vrot = self.theta_LM[:self.n_circ]
				self.vrad = self.theta_LM[self.n_circ:self.m]
				self.vtan = self.vrot*0
				vrot_arr = np.asarray(vrot)
				vrad_arr = np.asarray(vrad)
				vtan_arr = vrad_arr*0
				phi_b = np.pi/4.


			if self.vmode == "bisymmetric":
				vrot, vrad, vtan, pa, eps, x0, y0, vsys, phi_b =  theta[:self.n_circ],theta[self.n_circ:self.n_circ+self.n_noncirc],theta[self.n_circ+self.n_noncirc:self.m],theta[self.m],theta[self.m+1],theta[self.m+2],theta[self.m+3],theta[self.m+4],theta[self.m+5]

				self.pa, self.eps, self.x0, self.y0, self.vsys, self.phi_b = self.theta_LM[self.m],self.theta_LM[self.m+1],self.theta_LM[self.m+2],self.theta_LM[self.m+3],self.theta_LM[self.m+4],self.theta_LM[self.m+5]


				self.vrot,self.vrad,self.vtan = self.theta_LM[:self.n_circ],self.theta_LM[self.n_circ:self.n_circ+self.n_noncirc],self.theta_LM[self.n_circ+self.n_noncirc:self.m]
				vrot_arr = np.asarray(vrot)
				vrad_arr = np.asarray(vrad)
				vtan_arr = np.asarray(vtan)


			if self.vmode != "hrm":
				mask_1 = list( self.vrot[i]-75 < x < self.vrot[i]+75 for i,x in enumerate(vrot_arr) )
				mask_2 = list( self.vrad[i]-100 < x < self.vrad[i]+100 for i,x in enumerate(vrad_arr) )
				mask_3 = list( self.vtan[i]-100 < x < self.vtan[i]+100 for i,x in enumerate(vtan_arr) )
				mask_product = all( ( mask_1, mask_2, mask_3 ) )# mask_2*mask_3


			if "hrm" in self.vmode:
				self.v_k= self.theta_LM[:self.m]
				if self.pdf == "C" or self.sigma_int:
					pa, eps, x0, y0, vsys = theta[-6:-1]
					self.pa, self.eps, self.x0, self.y0, self.vsys = self.theta_LM[-6:-1]
					V_k = theta[:-6]
				else:
					pa, eps, x0, y0, vsys = theta[-5:]
					self.pa, self.eps, self.x0, self.y0, self.vsys = self.theta_LM[-5:]
					V_k = theta[:-5]
				v_k_arr = np.asarray(V_k)
				mask_1 = list( self.v_k[i]-75 < x < self.v_k[i]+75 for i,x in enumerate(V_k) )				

				Vk_arr = np.asarray(V_k)
				phi_b = np.pi/4.
				mask_product = all(mask_1)
				# Accept any velocity ?
				#mask_product = True

			#trick the chain ?
			#pa, eps, x0, y0  = self.pa, self.eps, self.x0, self.y0

		    # set prior to 1 (log prior to 0) if in the range and zero (-inf) outside the range
			lp = 0
			lnsigma2 =  1

			if self.pdf == "C":
				lnsigma2 =  theta[-1]
				lp = 0 if 0<lnsigma2<100 else -1*np.inf
			if self.pdf == "G" and self.sigma_int:
				lnsigma2 =  theta[-1]
				lp = 0 if -10<lnsigma2<10 else -1*np.inf

			# search around a fixed distance of 10arcsec from (x0,y0)
			rsearch = 10 #arcsec
			r = np.sqrt((self.x0-x0)**2 + (self.y0-y0)**2 )*self.pixel_scale # r is in arcsec
			eps_min, eps_max = 1-np.cos(25*np.pi/180),1-np.cos(80*np.pi/180)
			# Uniform priors on parameters
			# note 1: for emcee and zeus: phi_bar goes from (-1,1) so that its arccos value goes from (0,2*np.pi)
			# note 2: vsys is searched around +-300 km/s from guess value.
			if all([-2*np.pi <= pa*np.pi/180 <= 2*np.pi, eps_min < eps < eps_max, r < rsearch, \
			-1 < phi_b < 1, mask_product, self.vsys*(1-300/self.vsys) < vsys < self.vsys*(1+300/self.vsys) ]):
			# Gaussian priors ?
				lp = 0 + lp
				lp += self.g_priors(self.x0,x0,5)
				lp += self.g_priors(self.y0,y0,5)
				lp += self.g_priors(self.eps,eps,1e-2)
				lp += self.g_priors(self.pa,pa,10)
				lp += self.g_priors(self.vsys,vsys,50)

				# angles close the minor/major axis are not excluded, but less likely.				
				# 85deg,5deg
				d85,d5 = np.cos(85*np.pi/180),np.cos(5*np.pi/180)
				if not (abs(phi_b) > d85) & (abs(phi_b) < d5):
					lp = -1e6

			else:
				lp = -1*np.inf
			return lp


	def ln_posterior(self, theta):
			"""
			For emcee and zeus: 
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

			# theta_mdls includes only parameters describing the kinematic models (i.e., not lnsigma2 nor gamma) :
			if self.pdf == "C" or self.sigma_int:
				beta =  theta[-1]
				theta_mdls = theta[:-1]
			if self.pdf == "G" and not self.sigma_int:
				theta_mdls = theta

			# For emcee and zeus: Need to transform -1<phi_b<1 to radians for model evaluation
			if self.mcmc_sampler in ['emcee','zeus'] and self.vmode == 'bisymmetric': theta_mdls[-1] = np.arccos(theta_mdls[-1])
			""" The model evaluation depends only on the parameters"""
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
				#log_L = -(N + 1) * np.log(beta) - np.sum(np.log(1.0 + ( dy / beta )**2) )
				log_L = -N * np.log(np.pi*beta) - np.sum(np.log(1.0 + ( dy / beta )**2) )

			if log_L ==0:log_L = -np.inf
			return log_L


	#def __call__(self, theta):
	#		return self.ln_posterior(theta)

