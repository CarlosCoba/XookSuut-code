import numpy as np
import matplotlib.pylab as plt
def set_like(vmode, clase):

	class set_likelihood(clase):


		def ln_prior(self, theta): 
			""" The prior only depends on the parameters """


			if vmode == "circular":

				vrot, pa, inc, x0, y0, vsys =  theta[:self.n_circ],theta[self.n_circ],theta[self.n_circ+1],theta[self.n_circ+2],theta[self.n_circ+3],theta[self.n_circ+4]
				vrot_arr = np.asarray(vrot)
				vrad_arr = vrot_arr*0
				vtan_arr = vrot_arr*0
				phi_b = 45

			if vmode == "radial":

				vrot, vrad, pa, inc, x0, y0, vsys =  theta[:self.n_circ],theta[self.n_circ:self.m],theta[self.m],theta[self.m+1],theta[self.m+2],theta[self.m+3],theta[self.m+4]
				vrot_arr = np.asarray(vrot)
				vrad_arr = np.asarray(vrad)
				vtan_arr = vrad_arr*0
				phi_b = 45


			if vmode == "bisymmetric":

				vrot, vrad, vtan, pa, inc, x0, y0, vsys, phi_b =  theta[:self.n_circ],theta[self.n_circ:self.n_circ+self.n_noncirc],theta[self.n_circ+self.n_noncirc:self.m],theta[self.m],theta[self.m+1],theta[self.m+2],theta[self.m+3],theta[self.m+4],theta[self.m+5]
				vrot_arr = np.asarray(vrot)
				vrad_arr = np.asarray(vrad)
				vtan_arr = np.asarray(vtan)


			if vmode != "hrm":
				mask_1 = (vrot_arr > -450) & (vrot_arr < 450) 
				mask_2 = (vrad_arr > -450) & (vrad_arr < 450)
				mask_3 = (vtan_arr > -450) & (vtan_arr < 450)
				mask_product = mask_2*mask_3



			if "hrm" in vmode:
				if self.lnsigma_int == True:
					pa, inc, x0, y0, vsys = theta[-6:-1]
					V_k = theta[:-6]
				else:
					pa, inc, x0, y0, vsys = theta[-5:]
					V_k = theta[:-5]


				Vk_arr = np.asarray(V_k)
				phi_b = 45

				mask_1 = (Vk_arr > -450) & (Vk_arr < 450)
				mask_product = mask_1

			if self.lnsigma_int == True:
				lnsigma2 =  theta[-1]
			else:
				lnsigma2 =  1
			#if  0 <= pa <= 360 and 0 < inc < 90 and 0 < x0 < self.nx and 0 < y0 < self.ny and 0 < vsys < 3e5 and 0 < phi_b < 180 and  -100 < lnsigma2 <100:
			if  0 <= pa <= 360 and 0 < inc < 90 and 0 < x0 < self.nx and 0 < y0 < self.ny and 0 < phi_b < 180 and  -100 < lnsigma2 <100:

				if False in mask_1 or False in mask_product:
					return -np.inf
				else:  
					return 0.0
			else:
				return -np.inf


		def ln_posterior(self, theta):
			""" 
			Up to a normalization constant, the log of the posterior pdf is just 
			the sum of the log likelihood plus the log prior.
			"""
			lnp = self.ln_prior(theta)
			if np.isinf(lnp): 
				return lnp

			lnL = self.ln_likelihood(theta)
			lnprob = lnp + lnL

			if np.isnan(lnprob):
				return -np.inf

			return lnprob


		def ln_likelihood(self, theta):
			""" The likelihood function evaluation requires a particular set of model parameters and the data """

			mdl, obs, error = self.interp_model(theta)
			mdl, obs, error = np.asarray(mdl), np.asarray(obs), np.asarray(error)
			dy = obs - mdl
			resid = dy / error
			N = len(dy)

			if self.lnsigma_int == True:
				lnsigma2 =  theta[-1]
				error2 = ( error**2 + np.exp(lnsigma2) )
				log_L = -0.5 * (np.sum(dy**2 / error2)  -0.5*np.sum(np.log(error2)) )

			else:
				# Gaussian
				log_L = -0.5*resid@resid - np.sum( np.log(error) )
				# T-student with nu = 3
				#log_L = -1*np.sum( np.log(1+resid*resid) ) - np.sum( np.log(error) )
			return log_L


		def __call__(self, theta):
			return self.ln_posterior(theta)

	return set_likelihood

