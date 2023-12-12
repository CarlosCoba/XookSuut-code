import numpy as np
import matplotlib.pylab as plt
import scipy
import sys
from lmfit import Model, Parameters, fit_report, minimize
from matplotlib.gridspec import GridSpec
from astropy.convolution import convolve,convolve_fft
import configparser
import random


def Rings(xy_mesh,pa,eps,x0,y0,pixel_scale):
	(x,y) = xy_mesh

	X = (- (x-x0)*np.sin(pa) + (y-y0)*np.cos(pa))
	Y = (- (x-x0)*np.cos(pa) - (y-y0)*np.sin(pa))

	R= np.sqrt(X**2+(Y/(1-eps))**2)

	return R*pixel_scale


from src.kin_components import CIRC_MODEL
from src.kin_components import RADIAL_MODEL
from src.kin_components import BISYM_MODEL
from src.kin_components import HARMONIC_MODEL
from src.pixel_params import pixels, v_interp
from src.weights_interp import weigths_w
from src.create_2D_kin_models import bidi_models
from src.create_2D_vlos_model import best_2d_model
from src.create_dataset import dataset_to_2D


class Least_square_fit:
	def __init__(self,vel_map, e_vel_map, guess, vary, vmode, config, rings_pos, ring_space, fit_method, e_ISM, pixel_scale, frac_pixel, v_center, m_hrm = 1, N_it = 1):

		"""
		vary = [Vrot,Vrad,Vtan,PA,INC,XC,YC,VSYS,theta]
		"""

		self.N_it = N_it
		if self.N_it == 0:
			vary,self.vary_kin = vary*0,0
			vary,self.vary_kin = vary,1
		else:
			self.vary_kin = 1
		if "hrm" in vmode:
			self.c_k0, self.s_k0,self.pa0,self.eps0,self.xc0,self.yc0,self.vsys0,self.phi_bar = guess
			constant_params = [self.pa0,self.eps0,self.xc0,self.yc0,self.vsys0]
			self.vary_pa,self.vary_eps,self.vary_xc,self.vary_yc,self.vary_vsys,self.vary_phib = vary
			self.vary_sk, self.vary_ck = True, True

		
		else:
			self.vrot0,self.vrad0,self.vtan0,self.pa0,self.eps0,self.xc0,self.yc0,self.vsys0,self.phi_bar = guess
			n_circ,n_noncirc = len(self.vrot0),len(self.vrad0[self.vrad0!=0])
			if n_noncirc !=n_circ:
				self.vrad0[n_noncirc],self.vtan0[n_noncirc]=1e-3,1e-3
			constant_params = [self.pa0,self.eps0,self.xc0,self.yc0,self.vsys0,self.phi_bar]
			self.vary_pa,self.vary_eps,self.vary_xc,self.vary_yc,self.vary_vsys,self.vary_phib = vary
			self.vary_vrot, self.vary_vrad, self.vary_vtan = 1*self.vary_kin, 1*self.vary_kin, 1*self.vary_kin

		self.m_hrm = m_hrm
		self.ny,self.nx = vel_map.shape
		self.rings_pos = rings_pos
		self.nrings = len(self.rings_pos)
		self.n_annulus = self.nrings - 1
		self.vel_map = vel_map
		#self.e_vel_map = np.sqrt(abs(vel_map-self.vsys0)) if np.nanmean(e_vel_map) == 1 else e_vel_map
		self.e_vel_map = e_vel_map
		self.vmode = vmode
		self.ring_space = ring_space
		self.fit_method = fit_method
		self.config = config
		self.constant_params = constant_params
		self.osi = ["-", ".", ",", "#","%", "&", ""]

		# args to pass to minimize function
		self.kwargs = {}
		if self.N_it == 0:
			self.kwargs = {"ftol":1e8}

		X = np.arange(0, self.nx, 1)
		Y = np.arange(0, self.ny, 1)
		self.XY_mesh = np.meshgrid(X,Y)

		if self.xc0 % int(self.xc0) == 0 or self.yc0 % int(self.yc0) == 0 : 
			self.xc0, self.yc0 =  self.xc0 + 1e-5, self.yc0 + 1e-5
		self.r_n = Rings(self.XY_mesh,self.pa0*np.pi/180,self.eps0,self.xc0,self.yc0,pixel_scale)
		self.r_n = np.asarray(self.r_n, dtype = np.longdouble)
		self.pixel_scale = pixel_scale
		self.frac_pixel = frac_pixel
		self.v_center = v_center

		self.e_ISM = e_ISM
		interp_model = np.zeros((self.ny,self.nx))
		self.index_v0 = 123456
		if vmode == "circular": self.Vk = 1
		if vmode == "radial": self.Vk = 2
		if vmode == "bisymmetric": self.Vk = 3
		

		self.Vrot, self.Vrad, self.Vtan = 0,0,0
		if "hrm" not in self.vmode:
			self.V_k = [self.Vrot, self.Vrad, self.Vtan] 
			self.V_k_std = [0,0,0]
		else: 
			self.V_k = [0]*(2*self.m_hrm) 
			self.V_k_std = [0]*(2*self.m_hrm)
			self.Vk = 2*self.m_hrm


		config_const = config['constant_params']
		self.Vmin, self.Vmax = -450, 450
		eps_min, eps_max = 1-np.cos(20*np.pi/180),1-np.cos(80*np.pi/180)
		self.PAmin,self.PAmax,self.vary_pa = config_const.getfloat('MIN_PA', 0), config_const.getfloat('MAX_PA', 360),config_const.getboolean('FIT_PA', self.vary_pa)
		self.INCmin,self.INCmax,self.vary_eps = config_const.getfloat('MIN_INC', eps_min), config_const.getfloat('MAX_INC', eps_max),config_const.getboolean('FIT_INC', self.vary_eps)
		# To change input INCmin (in deg) and INCmax (in deg) values from the config file to eps 
		if self.INCmin >1:  self.INCmin = 1-np.cos(self.INCmin*np.pi/180)
		if self.INCmax >1:  self.INCmax = 1-np.cos(self.INCmax*np.pi/180)
		self.X0min,self.X0max,self.vary_xc = config_const.getfloat('MIN_X0', 0), config_const.getfloat('MAX_X0', self.nx),config_const.getboolean('FIT_X0', self.vary_xc)
		self.Y0min,self.Y0max,self.vary_yc = config_const.getfloat('MIN_Y0', 0), config_const.getfloat('MAX_Y0', self.ny),config_const.getboolean('FIT_Y0', self.vary_yc)
		self.VSYSmin,self.VSYSmax,self.vary_vsys = config_const.getfloat('MIN_VSYS', 0), config_const.getfloat('MAX_VSYS', np.inf),config_const.getboolean('FIT_VSYS', self.vary_vsys)
		self.PAbarmin,self.PAbarmax,self.vary_phib = config_const.getfloat('MIN_PHI_BAR', -np.pi), config_const.getfloat('MAX_PHI_BAR', np.pi),config_const.getboolean('FIT_PHI_BAR', self.vary_phib)

		config_general = config['general']
		outliers = config_general.getboolean('outliers', False)
		if outliers: self.kwargs["loss"]="cauchy"

		# Rename to capital letters
		self.PA = self.pa0
		self.EPS =  self.eps0 
		self.X0 = self.xc0
		self.Y0 = self.yc0
		self.VSYS = self.vsys0
		self.PHI_BAR = self.phi_bar

class Config_params(Least_square_fit):

		def assign_constpars(self,pars):

			#if self.config in self.osi:
			pars.add('Vsys', value=self.VSYS, vary = self.vary_vsys, min = self.VSYSmin)
			pars.add('pa', value=self.PA, vary = self.vary_pa, min = self.PAmin, max = self.PAmax)
			pars.add('eps', value=self.EPS, vary = self.vary_eps, min = self.INCmin, max = self.INCmax)
			pars.add('x0', value=self.X0, vary = self.vary_xc,  min = self.X0min, max = self.X0max)
			pars.add('y0', value=self.Y0, vary = self.vary_yc, min = self.Y0min, max = self.Y0max)
			if self.vmode == "bisymmetric":
				pars.add('phi_b', value=self.PHI_BAR, vary = self.vary_phib, min = self.PAbarmin , max = self.PAbarmax)


		def tune_velocities(self,pars,iy):
				if "hrm" not in self.vmode:
					if self.vmode == "radial":
						if self.vrad0[iy] == 0:
							self.vary_vrad = False
						else:
							self.vary_vrad = True*self.vary_kin
					if self.vmode == "bisymmetric":
						if self.vrad0[iy] == 0  and self.vtan0[iy] ==0:
							self.vary_vrad = False
							self.vary_vtan = False
						else:
							self.vary_vrad = True*self.vary_kin
							self.vary_vtan = True*self.vary_kin
				else:
					if self.s_k0[0][iy] == 0:
						self.vary_sk = False
						self.vary_ck = False
					else:
						self.vary_sk = True
						self.vary_ck = True



		def assign_vels(self,pars):

			for iy in range(self.nrings):
				if "hrm" not in self.vmode:
					pars.add('Vrot_%i' % (iy),value=self.vrot0[iy], vary = self.vary_vrot, min = self.Vmin, max = self.Vmax)

					#if self.vrad0[iy] == 0 and self.vtan0[iy] ==0:
					#	self.vary_vrad = False
					#	self.vary_vtan = False

					if self.vmode == "radial":
						self.tune_velocities(pars,iy)
						pars.add('Vrad_%i' % (iy), value=self.vrad0[iy], vary = self.vary_vrad, min = self.Vmin, max = self.Vmax)

					if self.vmode == "bisymmetric":
						self.tune_velocities(pars,iy)
						pars.add('Vrad_%i' % (iy), value=self.vrad0[iy], vary = self.vary_vrad, min = self.Vmin, max = self.Vmax)
						pars.add('Vtan_%i' % (iy), value=self.vtan0[iy], vary = self.vary_vtan, min = self.Vmin, max = self.Vmax)

				if "hrm" in self.vmode:
					pars.add('C1_%i' % (iy),value=self.c_k0[0][iy], vary = True, min = self.Vmin, max = self.Vmax)
					self.tune_velocities(pars,iy)
					k = 1
					for j in range(1,self.m_hrm+1):
						if k != self.m_hrm and self.m_hrm != 1:
							pars.add('C%s_%i' % (k+1,iy), value=self.c_k0[k][iy], vary = self.vary_ck, min = self.Vmin, max = self.Vmax)
						pars.add('S%s_%i' % (j,iy), value=self.s_k0[j-1][iy], vary = self.vary_sk, min = self.Vmin, max = self.Vmax)
						k = k + 1




class Models(Config_params):

			def kinmdl_dataset(self, pars, i, xy_mesh, r_space, r_0 = None):

				pars = pars.valuesdict()
				pa = pars['pa']
				eps = pars['eps']
				x0,y0 = pars['x0'],pars['y0']

				# For inner interpolation
				r1, r2 = self.rings_pos[0], self.rings_pos[1]
				if "hrm" not in self.vmode and self.v_center != 0:
					if self.v_center == "extrapolate":					
						v1, v2 = pars["Vrot_0"], pars["Vrot_1"]
						v_int =  v_interp(0, r2, r1, v2, v1 )
						pars["Vrot_%i" % (self.index_v0)] = v_int
						if self.vmode == "radial" or self.vmode == "bisymmetric":
							v1, v2 = pars["Vrad_0"], pars["Vrad_1"]
							v_int =  v_interp(0, r2, r1, v2, v1 )
							pars["Vrad_%i" % (self.index_v0)] = v_int
						if self.vmode == "bisymmetric":
							v1, v2 = pars["Vtan_0"], pars["Vtan_1"]
							v_int =  v_interp(0, r2, r1, v2, v1 )
							pars["Vtan_%i" % (self.index_v0)] = v_int
					else:
						# This only applies to Vt component in circ
						# I made up this index.
						if self.vmode == "circular":
							pars["Vrot_%i" % (self.index_v0)] = self.v_center


				
				if  "hrm" in self.vmode and self.v_center == "extrapolate":

					for k in range(1,self.m_hrm+1) :

						v1, v2 = pars['C%s_%i'% (k,0)], pars['C%s_%i'% (k,1)]
						v_int =  v_interp(0, r2, r1, v2, v1 )
						pars["C%s_%i" % (k, self.index_v0)] = v_int

						v1, v2 = pars['S%s_%i'% (k,0)], pars['S%s_%i'% (k,1)]
						v_int =  v_interp(0, r2, r1, v2, v1 )
						pars["S%s_%i" % (k, self.index_v0)] = v_int


				if "hrm" not in self.vmode:
					Vrot = pars['Vrot_%i'% i]

				if self.vmode == "circular":
					modl = (CIRC_MODEL(xy_mesh,Vrot,pa,eps,x0,y0))*weigths_w(xy_mesh,pa,eps,x0,y0,r_0,r_space,pixel_scale=self.pixel_scale)
				if self.vmode == "radial":
					Vrad = pars['Vrad_%i'% i]
					modl = (RADIAL_MODEL(xy_mesh,Vrot,Vrad,pa,eps,x0,y0))*weigths_w(xy_mesh,pa,eps,x0,y0,r_0,r_space,pixel_scale=self.pixel_scale)
				if self.vmode == "bisymmetric":
					Vrad = pars['Vrad_%i'% i]
					Vtan = pars['Vtan_%i'% i]
					if Vrad != 0 and Vtan != 0:
						phi_b = pars['phi_b']
						modl = (BISYM_MODEL(xy_mesh,Vrot,Vrad,pa,eps,x0,y0,Vtan,phi_b))*weigths_w(xy_mesh,pa,eps,x0,y0,r_0,r_space,pixel_scale=self.pixel_scale)
					else:
						modl = (CIRC_MODEL(xy_mesh,Vrot,pa,eps,x0,y0))*weigths_w(xy_mesh,pa,eps,x0,y0,r_0,r_space,pixel_scale=self.pixel_scale)
				if "hrm" in self.vmode:
					C_k, S_k  = [pars['C%s_%i'% (k,i)] for k in range(1,self.m_hrm+1)], [pars['S%s_%i'% (k,i)] for k in range(1,self.m_hrm+1)]
					modl = (HARMONIC_MODEL(xy_mesh,C_k, S_k,pa,eps,x0,y0, m_hrm = self.m_hrm))*weigths_w(xy_mesh,pa,eps,x0,y0,r_0,r_space,pixel_scale=self.pixel_scale)
				return modl




class Fit_kin_mdls(Models):
		def residual(self, pars):

			Vsys = pars['Vsys']
			interp_model = np.zeros((self.ny,self.nx))
			interp_model = dataset_to_2D([self.ny,self.nx], self.n_annulus, self.rings_pos, self.r_n, self.XY_mesh, self.kinmdl_dataset, self.vmode, self.v_center, pars, self.index_v0)
			mask_inner = np.where( (self.r_n < self.rings_pos[0] ) )
			x_r0,y_r0 = self.XY_mesh[0][mask_inner], self.XY_mesh[1][mask_inner] 
			r_space_0 = self.rings_pos[0]

			#(a) velocity rises linearly from zero: r1 = 0, v1 = 0
			if self.v_center == 0 or (self.v_center != "extrapolate" and self.vmode != "circular"):

				V_xy_mdl = self.kinmdl_dataset(pars, 0, (x_r0,y_r0), r_0 = 0, r_space = r_space_0)
				v_new_2 = V_xy_mdl[1]
				interp_model[mask_inner] = v_new_2

			else:

				r2 = self.rings_pos[0] 		# ring posintion
				v1_index = self.index_v0	# index of velocity
				V_xy_mdl = self.kinmdl_dataset(pars, v1_index, (x_r0,y_r0), r_0 = r2, r_space = r_space_0 )
				v_new_1 = V_xy_mdl[0]

				r1 = 0 					# ring posintion
				v2_index = 0			# index of velocity
				V_xy_mdl = self.kinmdl_dataset(pars, v2_index, (x_r0,y_r0), r_0 = r1, r_space = r_space_0 )
				v_new_2 = V_xy_mdl[1]

				v_new = v_new_1 + v_new_2
				interp_model[mask_inner] = v_new

			
			sigma = np.sqrt(self.e_vel_map**2 + self.e_ISM**2)
			interp_model[interp_model == 0] = np.nan
			interp_model = interp_model + Vsys
			residual = self.vel_map - interp_model
			w_residual = residual/sigma
			flat = w_residual.flatten()
			return flat



		def run_mdl(self):
			pars = Parameters()
			self.assign_vels(pars)
			self.assign_constpars(pars)
			res = self.residual(pars)
			out = minimize(self.residual, pars, method = self.fit_method, nan_policy = "omit", **self.kwargs)
			return out

		def results(self):
			out = self.run_mdl()
			best = out.params
			N_free = out.nfree
			N_nvarys = out.nvarys
			N_data = out.ndata
			bic, aic = out.bic, out.aic
			red_chi = out.redchi


			constant_parms = np.asarray( [best["pa"].value, best["eps"].value, best["x0"].value,best["y0"].value, best["Vsys"].value, 45] )
			e_constant_parms =  [ best["pa"].stderr, best["eps"].stderr, best["x0"].stderr, best["y0"].stderr, best["Vsys"].stderr, 0] 

			pa, eps, x0, y0, Vsys, phi_b = constant_parms
			std_pa, std_eps, std_x0, std_y0, std_Vsys, std_phi_b = list(e_constant_parms)

			if "hrm" not in self.vmode:
				v_kin = ["Vrot","Vrad", "Vtan"]
				for i in range(self.Vk):	
					self.V_k[i] = [ best["%s_%s"%(v_kin[i],iy)].value for iy in range(self.nrings) ]
					self.V_k_std[i] = [ best["%s_%s"%(v_kin[i],iy)].stderr for iy in range(self.nrings) ]
					# In case something goes wrong with errors:
					if None in self.V_k_std[i] :  self.V_k_std[i] = len(self.V_k[i])*[1e-3]

				if self.vmode == "bisymmetric": 
					phi_b, std_phi_b = best["phi_b"].value, best["phi_b"].stderr
					constant_parms[-1], e_constant_parms[-1] = phi_b, std_phi_b

				#errors = self.V_k_std + e_constant_parms

			else:
				v_kin = ["C","S"]
				k = 0
				for j in range(len(v_kin)):
					for i in range(self.m_hrm):
						self.V_k[k] = [ best["%s%s_%s"%(v_kin[j],i+1,iy)].value for iy in range(self.nrings) ]
						self.V_k_std[k] = [ best["%s%s_%s"%(v_kin[j],i+1,iy)].stderr for iy in range(self.nrings) ]
						# In case something goes wrong with errors:
						if None in self.V_k_std[k] :  self.V_k_std[k] = len(self.V_k[k])*[1e-3]
						k = k + 1

			if None in e_constant_parms:  e_constant_parms = [1e-3]*len(constant_parms)

			create_2D = best_2d_model(self.vmode, [self.ny,self.nx], self.V_k, pa, eps, x0, y0, Vsys, self.rings_pos, self.ring_space, self.pixel_scale, self.v_center , self.m_hrm, phi_b)
			vlos_2D_model = create_2D.model2D()

			#"""
			#We need to re-compute chisquare with the best model !
			
			delta = 0.5*(self.rings_pos[1] - self.rings_pos[0])
			# Recompute fraction of pixels per ring with the best values
			fpix = np.array([pixels([self.ny,self.nx],self.vel_map,pa,eps,x0,y0,k, delta=delta,pixel_scale = self.pixel_scale) for k in self.rings_pos])
			mask_fpix = fpix > self.frac_pixel
			# This should be the real number of rings
			true_rings = self.rings_pos[mask_fpix]
			Rn = Rings(self.XY_mesh,pa*np.pi/180,eps,x0,y0,self.pixel_scale)
			Rn[Rn > true_rings[-1]]= np.nan
			# A simple way for applying a mask
			vlos_2D_model = vlos_2D_model*Rn/Rn
			# Compute the residuals
			res = ( vlos_2D_model - self.vel_map) / self.e_vel_map
			# Residual sum of squares
			rss = np.nansum( ( vlos_2D_model - self.vel_map)**2 )
			N_data = len(vlos_2D_model[np.isfinite(vlos_2D_model)])
			N_free = N_data - N_nvarys
			# Compute reduced chisquare
			chisq = np.nansum(res**2)
			red_chi = np.nansum(res**2)/ (N_free)
			# Akaike Information Criterion
			aic = N_data*np.log(rss/N_data) + 2*N_nvarys
			#Bayesian Information Criterion
			bic = N_data*np.log(rss/N_data) + np.log(N_data)*N_nvarys

			for k in range(self.Vk):
				self.V_k[k] = np.asarray(self.V_k[k])[mask_fpix]
				self.V_k_std[k] = np.asarray(self.V_k_std[k])[mask_fpix]
			
			errors = [[],[]]
			if "hrm" not in self.vmode:
				errors[0],errors[1] = self.V_k_std,e_constant_parms
			else:
				errors[0],errors[1] = [self.V_k_std[0:self.m_hrm],self.V_k_std[self.m_hrm:]],e_constant_parms[:-1]

			self.rings_pos = true_rings
			#"""
			true_rings = self.rings_pos

			if len(self.V_k) != len(self.V_k_std)  : self.V_k_std = [1e-3]*len(self.V_k)
			##########
			interp_mdl =  bidi_models(self.vmode, [self.ny,self.nx], self.V_k, pa, eps, x0, y0, Vsys, self.rings_pos, self.ring_space, self.pixel_scale, self.v_center, self.m_hrm, phi_b) 
			kin_2D_models = interp_mdl.interp()
			##########
			out_data = [N_free, N_nvarys, N_data, bic, aic, red_chi]
			return vlos_2D_model, kin_2D_models, self.V_k, pa, eps , x0, y0, Vsys, phi_b, out_data, errors, true_rings





