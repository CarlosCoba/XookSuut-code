import numpy as np
import matplotlib.pylab as plt
import scipy
import sys
from lmfit import Model, Parameters, fit_report, minimize
from matplotlib.gridspec import GridSpec
from pixel_params import pixels
from pixel_params import pixels
from read_config import config_file
from weights_interp import weigths_w

from astropy.convolution import convolve,convolve_fft
import random



 
def Rings(xy_mesh,pa,inc,x0,y0,pixel_scale):
	(x,y) = xy_mesh

	X = (- (x-x0)*np.sin(pa) + (y-y0)*np.cos(pa))
	Y = (- (x-x0)*np.cos(pa) - (y-y0)*np.sin(pa))

	R= np.sqrt(X**2+(Y/np.cos(inc))**2)

	return R*pixel_scale


def v_interp(r, r2, r1, v2, v1 ):
	m = (v2 - v1) / (r2 - r1)
	v0 = m*(r-r1) + v1
	return v0


from kin_components import CIRC_MODEL
from kin_components import RADIAL_MODEL
from kin_components import BISYM_MODEL
from kin_components import HARMONIC_MODEL


class Least_square_fit:
	def __init__(self,vel_map, e_vel_map, guess, vary, vmode, config, rings_pos, ring_space, fit_method, e_ISM, pixel_scale, frac_pixel, inner_interp, m_hrm = 1):

		"""
		vary = [Vrot,Vrad,Vtan,PA,INC,XC,YC,VSYS,theta]
		"""


		if fit_method == "LM": fit_method = "leastsq"
		if fit_method == "Powell": fit_method = "Powell"
		if fit_method == "powell": fit_method = "Powell"
		if fit_method == "POWELL": fit_method = "Powell"

		if fit_method not in ["leastsq", "Powell"] :
			print("XookSuut: choose an appropiate fitting method: LM or Powell")
			quit()



		if "hrm" in vmode:
			self.c_k0, self.s_k0,self.pa0,self.inc0,self.xc0,self.yc0,self.vsys0 = guess
			constant_params = [self.pa0,self.inc0,self.xc0,self.yc0,self.vsys0]
			self.vary_pa,self.vary_inc,self.vary_xc,self.vary_yc,self.vary_vsys = vary[:-1]
			self.vary_sk, self.vary_ck = True, True

		
		else:
			self.vrot0,self.vrad0,self.vtan0,self.pa0,self.inc0,self.xc0,self.yc0,self.vsys0,self.phi_bar = guess
			constant_params = [self.pa0,self.inc0,self.xc0,self.yc0,self.vsys0,self.phi_bar]
			self.vary_pa,self.vary_inc,self.vary_xc,self.vary_yc,self.vary_vsys,self.vary_theta = vary
			self.vary_vrot, self.vary_vrad, self.vary_vtan = 1, 1, 1
		
		self.m_hrm = m_hrm
		self.ny,self.nx = vel_map.shape
		self.rings_pos = rings_pos
		self.nrings = len(self.rings_pos)
		self.n_annulus = self.nrings - 1
		self.vel_map = vel_map
		self.e_vel_map = e_vel_map
		self.vmode = vmode
		self.ring_space = ring_space
		self.fit_method = fit_method
		self.config = config
		self.constant_params = constant_params

		X = np.arange(0, self.nx, 1)
		Y = np.arange(0, self.ny, 1)
		self.XY_mesh = np.meshgrid(X,Y)

		if self.xc0 % int(self.xc0) == 0 or self.yc0 % int(self.yc0) == 0 : 
			self.xc0, self.yc0 =  self.xc0 + 1e-5, self.yc0 + 1e-5
		self.r_n = Rings(self.XY_mesh,self.pa0*np.pi/180,self.inc0*np.pi/180,self.xc0,self.yc0,pixel_scale)
		self.r_n = np.asarray(self.r_n, dtype = np.longdouble)
		self.pixel_scale = pixel_scale
		self.frac_pixel = frac_pixel
		self.inner_interp = inner_interp
		
		self.e_ISM = 0
		self.interp_model = np.zeros((self.ny,self.nx))
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

class Config_params(Least_square_fit):

		def constant(self,pars):

			if self.config == "":
				pars.add('Vsys', value=self.vsys0, vary = self.vary_vsys, min = 0, max = 3e5)
				pars.add('pa', value=self.pa0, vary = self.vary_pa, min = 0, max = 360)
				pars.add('inc', value=self.inc0, vary = self.vary_inc, min = 5, max = 85)
				pars.add('x0', value=self.xc0, vary = self.vary_xc,  min = 0, max = self.nx)
				pars.add('y0', value=self.yc0, vary = self.vary_yc, min = 0, max = self.ny)

			else:

				for res in config_file(self.config):
					param0, fit_param, val, vmin, vmax = str(res["param"]), bool(float(res["fit"])), eval(res["val"]), eval(res["min"]), eval(res["max"])
					if self.vmode != "bisymmetric" and param0 != "phi_b":  
						if fit_param == False:
							pars.add(param0, value = val, vary = fit_param)
						else:
							pars.add(param0, value = val, vary = fit_param, min = vmin, max = vmax)
					else:
						if fit_param == False:
							pars.add(param0, value = val, vary = False)
						else:
							pars.add(param0, value = val, vary = fit_param, min = vmin, max = vmax)
			

		def tune_velocities(self,pars,iy):
				if "hrm" not in self.vmode:
					if self.vrad0[iy] == 0 and self.vmode == "radial":
						self.vary_vrad = False
					else:
						self.vary_vrad = True
					if self.vrad0[iy] == 0  or self.vtan0[iy] ==0 and self.vmode == "bisymmetric":
						self.vary_vrad = False
						self.vary_vtan = False
					else:
						self.vary_vrad = True
						self.vary_vtan = True
						if self.vmode == "bisymmetric" and self.config == "":
							pars.add('phi_b', value=self.phi_bar, vary = self.vary_theta, min = 0 , max = 180)
				else:
					if self.s_k0[0][iy] == 0:
						self.vary_sk = False
						self.vary_ck = False
					else:
						self.vary_sk = True
						self.vary_ck = True



		def mdls_diskfit(self,pars):

			self.constant(pars)
			for iy in range(self.nrings):
				if "hrm" not in self.vmode:
					pars.add('Vrot_%i' % (iy),value=self.vrot0[iy], vary = self.vary_vrot, min = -400, max = 400)

					if self.vrad0[iy] == 0 and self.vtan0[iy] ==0:
						self.vary_vrad = False
						self.vary_vtan = False

					if self.vmode == "radial":
						self.tune_velocities(pars,iy)
						pars.add('Vrad_%i' % (iy), value=self.vrad0[iy], vary = self.vary_vrad, min = -300, max = 300)

					if self.vmode == "bisymmetric":
						self.tune_velocities(pars,iy)
						pars.add('Vrad_%i' % (iy), value=self.vrad0[iy], vary = self.vary_vrad, min = -300, max = 300)
						pars.add('Vtan_%i' % (iy), value=self.vtan0[iy], vary = self.vary_vtan, min = -300, max = 300)

				if "hrm" in self.vmode:
					pars.add('C1_%i' % (iy),value=self.c_k0[0][iy], vary = True, min = -400, max = 400)
					self.tune_velocities(pars,iy)
					k = 1
					for j in range(1,self.m_hrm+1):
						if k != self.m_hrm and self.m_hrm != 1:
							pars.add('C%s_%i' % (k+1,iy), value=self.c_k0[k][iy], vary = self.vary_ck, min = -300, max = 300)
						pars.add('S%s_%i' % (j,iy), value=self.s_k0[j-1][iy], vary = self.vary_sk, min = -300, max = 300)
						k = k + 1




class Models(Config_params):

			def vmodel_dataset(self, pars, i, xy_mesh, r_space, r_0 = None):

				parvals = pars.valuesdict()
				pa = parvals['pa']
				inc = parvals['inc']
				Vsys = parvals['Vsys']
				x0,y0 = parvals['x0'],parvals['y0']

				# For inner interpolation
				r1, r2 = self.rings_pos[0], self.rings_pos[1]
				if "hrm" not in self.vmode and self.inner_interp != False:
					v1, v2 = pars["Vrot_0"], pars["Vrot_1"]
					v_int =  v_interp(0, r2, r1, v2, v1 )
					# This only applies to circular rotation
					if self.inner_interp != True : v_int = self.inner_interp 
					# I made up this index.
					parvals["Vrot_%i" % (self.n_annulus+10)] = v_int
					if self.vmode == "radial" or self.vmode == "bisymmetric":
						v1, v2 = pars["Vrad_0"], pars["Vrad_1"]
						v_int =  v_interp(0, r2, r1, v2, v1 )
						parvals["Vrad_%i" % (self.n_annulus+10)] = v_int
					if self.vmode == "bisymmetric":
						v1, v2 = pars["Vtan_0"], pars["Vtan_1"]
						v_int =  v_interp(0, r2, r1, v2, v1 )
						parvals["Vtan_%i" % (self.n_annulus+10)] = v_int
				
				if  "hrm" in self.vmode and self.inner_interp != False:
					for k in range(1,self.m_hrm+1) :

						v1, v2 = parvals['C%s_%i'% (k,0)], parvals['C%s_%i'% (k,1)]
						v_int =  v_interp(0, r2, r1, v2, v1 )
						parvals["C%s_%i" % (k, self.n_annulus+10)] = v_int

						v1, v2 = parvals['S%s_%i'% (k,0)], parvals['S%s_%i'% (k,1)]
						v_int =  v_interp(0, r2, r1, v2, v1 )
						parvals["S%s_%i" % (k, self.n_annulus+10)] = v_int


				if "hrm" not in self.vmode:
					Vrot = parvals['Vrot_%i'% i]

				if self.vmode == "circular":
					modl = (CIRC_MODEL(xy_mesh,Vrot,pa,inc,x0,y0))*weigths_w(xy_mesh,pa,inc,x0,y0,r_0,r_space,pixel_scale=self.pixel_scale)
				if self.vmode == "radial":
					Vrad = parvals['Vrad_%i'% i]
					modl = (RADIAL_MODEL(xy_mesh,Vrot,Vrad,pa,inc,x0,y0))*weigths_w(xy_mesh,pa,inc,x0,y0,r_0,r_space,pixel_scale=self.pixel_scale)
				if self.vmode == "bisymmetric":
					Vrad = parvals['Vrad_%i'% i]
					Vtan = parvals['Vtan_%i'% i]
					if Vrad != 0 and Vtan != 0:
						phi_b = parvals['phi_b']
						modl = (BISYM_MODEL(xy_mesh,Vrot,Vrad,pa,inc,x0,y0,Vtan,phi_b))*weigths_w(xy_mesh,pa,inc,x0,y0,r_0,r_space,pixel_scale=self.pixel_scale)
					else:
						modl = (BISYM_MODEL(xy_mesh,Vrot,0,pa,inc,x0,y0,0,0))*weigths_w(xy_mesh,pa,inc,x0,y0,r_0,r_space,pixel_scale=self.pixel_scale)
				if "hrm" in self.vmode:
					C_k, S_k  = [parvals['C%s_%i'% (k,i)] for k in range(1,self.m_hrm+1)], [parvals['S%s_%i'% (k,i)] for k in range(1,self.m_hrm+1)]
					modl = (HARMONIC_MODEL(xy_mesh,C_k, S_k,pa,inc,x0,y0, m_hrm = self.m_hrm))*weigths_w(xy_mesh,pa,inc,x0,y0,r_0,r_space,pixel_scale=self.pixel_scale)

				return modl,Vsys




class Fit_kin_mdls(Models):

		def residual(self, pars):
			"""Calculate total residual for fits of VMODELS to several data sets."""

			# make residual per data set
			for Nring in range(self.n_annulus):
				# For r1 > rings_pos[0]
				mdl_ev = 0
				r_space_k = self.rings_pos[Nring+1] - self.rings_pos[Nring] 
				mask = np.where( (self.r_n >= self.rings_pos[Nring] ) & (self.r_n < self.rings_pos[Nring+1]) )
				x,y = self.XY_mesh[0][mask], self.XY_mesh[1][mask]

				# For r1 < rings_pos[0]
				mdl_ev0 = 0
				mask_inner = np.where( (self.r_n < self.rings_pos[0] ) )
				x_r0,y_r0 = self.XY_mesh[0][mask_inner], self.XY_mesh[1][mask_inner] 
				r_space_0 = self.rings_pos[0]

				# interpolation between rings requieres two velocities: v1 and v2
				#v_new = v1*(r2-r)/(r2-r1) + v2*(r-r1)/(r2-r1)
				for kring in np.arange(2,0,-1):
					r2_1 = self.rings_pos[Nring + kring -1 ] # ring posintion
					v1_2_index = Nring + 2 - kring		 # index of velocity
					# For r > rings_pos[0]:					
					Vxy,Vsys = self.vmodel_dataset(pars, v1_2_index, (x,y), r_0 = r2_1, r_space = r_space_k )
					mdl_ev = mdl_ev + Vxy[2-kring]

					#"""
					# For r < rings_pos[0]:					
					# Inner interpolation
					#(a) velocity rises linearly from zero: r1 = 0, v1 = 0
					if self.inner_interp == False and Nring == 0 and kring == 2:
						Vxy,Vsys = self.vmodel_dataset(pars, 0, (x_r0,y_r0), r_0 = 0, r_space = r_space_0)
						mdl_ev0 = Vxy[1]
						self.interp_model[mask_inner] = mdl_ev0
					#(b) Extrapolate at the origin: r1 = 0, v1 != 0
					if self.inner_interp != False and Nring == 0 and "hrm":
						# we need to add a velocity at r1 = 0
						# index self.n_annulus + 10 is not in the velocity dictonary. I made up !
						r2_1 = [r_space_0,0]
						v1_2_index = [self.n_annulus+10,0]

						Vxy,Vsys = self.vmodel_dataset(pars, v1_2_index[2-kring], (x_r0,y_r0), r_0 = r2_1[2-kring], r_space = r_space_0 )
						mdl_ev0 = mdl_ev0 + Vxy[2-kring]
						self.interp_model[mask_inner] = mdl_ev0
					#"""

				self.interp_model[mask] = mdl_ev
				#self.interp_model[mask_inner] = mdl_ev0 + Vsys

			
			sigma = np.sqrt(self.e_vel_map**2 + 0*self.e_ISM**2)
			self.interp_model[self.interp_model == 0] = np.nan
			self.interp_model = self.interp_model + Vsys
			residual = self.vel_map - self.interp_model

			w_residual = residual/sigma
			#w_residual[abs(w_residual) > 500] = np.nan
			flat = w_residual.flatten()
			return flat



		def run_mdl(self):
			pars = Parameters()
			self.mdls_diskfit(pars)
			res = self.residual(pars)
			out = minimize(self.residual, pars, method = self.fit_method, nan_policy = "omit")
			return out

		def results(self):
			out = self.run_mdl()
			best = out.params
			N_free = out.nfree
			N_nvarys = out.nvarys
			N_data = out.ndata
			bic, aic = out.bic, out.aic
			red_chi = out.redchi


			constant_parms = np.asarray( [best["pa"].value, best["inc"].value, best["x0"].value,best["y0"].value, best["Vsys"].value, 45] )
			e_constant_parms =  [ best["pa"].stderr, best["inc"].stderr, best["x0"].stderr, best["y0"].stderr, best["Vsys"].stderr, 0] 

			pa, inc, x0, y0, Vsys, phi_b = constant_parms
			std_pa, std_inc, std_x0, std_y0, std_Vsys, std_phi_b = list(e_constant_parms)
			if None in e_constant_parms:  e_constant_parms = [0]*len(constant_parms)

			if "hrm" not in self.vmode:
				v_kin = ["Vrot","Vrad", "Vtan"]
				for i in range(self.Vk):
					self.V_k[i] = [ best["%s_%s"%(v_kin[i],iy)].value for iy in range(self.nrings) ]
					self.V_k_std[i] = [ best["%s_%s"%(v_kin[i],iy)].stderr for iy in range(self.nrings) ]

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
						k = k + 1

				#errors = self.V_k_std + e_constant_parms[:-1]

			from create_2D_vlos_model import best_2d_model
			create_2D = best_2d_model(self.vmode, [self.ny,self.nx], self.V_k, pa, inc, x0, y0, Vsys, self.rings_pos, self.ring_space, self.pixel_scale, self.inner_interp , self.m_hrm, phi_b)
			vlos_2D_model = create_2D.model2D()

			#"""
			#We need to re-compute chisquare with the best model !
			
			delta = self.rings_pos[1] - self.rings_pos[0]
			# Recompute fraction of pixels per ring with the best values
			fpix = np.array([pixels([self.ny,self.nx],self.vel_map,pa,inc,x0,y0,k, delta=delta,pixel_scale = self.pixel_scale) for k in self.rings_pos])
			mask_fpix = fpix > self.frac_pixel
			# This should be the real number of rings
			true_rings = self.rings_pos[mask_fpix]
			Rn = Rings(self.XY_mesh,pa*np.pi/180,inc*np.pi/180,x0,y0,self.pixel_scale)
			Rn[Rn > true_rings[-1]]= np.nan
			# A simple way for applying a mask
			vlos_2D_model = vlos_2D_model*Rn/Rn
			# Compute the residuals
			res = ( vlos_2D_model - self.vel_map) / self.e_vel_map
			N_data = len(vlos_2D_model[np.isfinite(vlos_2D_model)])
			N_free = N_data - N_nvarys
			# Compute reduced chisquare
			red_chi = np.nansum(res**2)/ (N_free)

			for k in range(self.Vk):
				self.V_k[k] = np.asarray(self.V_k[k])[mask_fpix]
				self.V_k_std[k] = np.asarray(self.V_k_std[k])[mask_fpix]

			if "hrm" not in self.vmode:
				errors = self.V_k_std + e_constant_parms
			else:
				errors = self.V_k_std + e_constant_parms[:-1]

			self.rings_pos = true_rings
			#"""
			true_rings = self.rings_pos

			##########
			from create_2D_kin_models import bidi_models
			interp_mdl =  bidi_models(self.vmode, [self.ny,self.nx], self.V_k, pa, inc, x0, y0, Vsys, self.rings_pos, self.ring_space, self.pixel_scale, self.inner_interp, self.m_hrm, phi_b) 
			kin_2D_models = interp_mdl.interp()
			##########

			out_data = [N_free, N_nvarys, N_data, bic, aic, red_chi]
			return vlos_2D_model, kin_2D_models, self.V_k, pa, inc , x0, y0, Vsys, phi_b, out_data, errors, true_rings





