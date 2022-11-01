import numpy as np
import matplotlib.pylab as plt

class set_likelihood:


	def __init__(self, vmode, shape, pdf, n_circ, n_noncirc, kinmodel):
			self.vmode = vmode
			self.ny, self.nx = shape
			self.pdf = pdf
			self.n_circ = n_circ
			self.n_noncirc = n_noncirc
			self.kinmodel = kinmodel
			self.m = 0

	def ln_prior(self, theta): 
			""" The prior only depends on the parameters """
			if self.vmode == "circular":

				vrot, pa, inc, x0, y0, vsys =  theta[:self.n_circ],theta[self.n_circ],theta[self.n_circ+1],theta[self.n_circ+2],theta[self.n_circ+3],theta[self.n_circ+4]
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
				#mask_1 = (vrot_arr > -450) & (vrot_arr < 450) 
				#mask_2 = (vrad_arr > -450) & (vrad_arr < 450)
				#mask_3 = (vtan_arr > -450) & (vtan_arr < 450)
				mask_1 = list ( abs(vrot_arr) < 450 )
				mask_2 = list ( abs(vrad_arr) < 450 )
				mask_3 = list ( abs(vtan_arr) < 450 )

				#mask_product = mask_2*mask_3
				#print([mask_1, mask_2, mask_3])
				mask_product = all( ( mask_1, mask_2, mask_3 ) )# mask_2*mask_3



			if "hrm" in self.vmode:
				if self.pdf == "C":
					pa, inc, x0, y0, vsys = theta[-6:-1]
					V_k = theta[:-6]
				else:
					pa, inc, x0, y0, vsys = theta[-5:]
					V_k = theta[:-5]


				Vk_arr = np.asarray(V_k)
				phi_b = 45

				#mask_1 = (Vk_arr > -450) & (Vk_arr < 450)
				#mask_product = mask_1


				mask_1 = list ( abs(Vk_arr) < 450 )
				mask_product = all(mask_1)

			if self.pdf == "C":
				lnsigma2 =  theta[-1]
			else:
				lnsigma2 =  1

			if all([-360 <= pa <= 360, 0 < inc < 90, 0 < x0 < self.nx, 0 < y0 < self.ny, -180 < phi_b < 180, lnsigma2 > 0, mask_product]) : return 0.0
			else: return -1*np.inf


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
				theta = theta[:-1]
			
			""" The likelihood function evaluation requires a particular set of model parameters and the data """
			model = (self.kinmodel).interp_model
			mdl, obs, error = model(theta)
			dy = obs - mdl
			N = len(dy)

			if self.pdf == "G":
				# Gaussian
				resid = dy / error
				log_L = -0.5*resid@resid - np.sum( np.log(error) )

			if self.pdf == "C":
				# Cauchy
				log_L = -(N + 1) * np.log(beta) - np.sum(np.log(1.0 + ( dy / beta )**2) )



			return log_L


	def __call__(self, theta):
			return self.ln_posterior(theta)

