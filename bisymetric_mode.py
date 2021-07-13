import numpy as np
import matplotlib.pylab as plt
import lmfit
import sys
import matplotlib
from lmfit import Model
from lmfit import Parameters, fit_report, minimize
from astropy.stats import sigma_clip
from scipy.interpolate import interp1d
from matplotlib.colors import Normalize
#from numpy.linalg import LinAlgError

from poly import legendre
import fit_params
from fit_params import fit
from phi_bar_sky import pa_bar_sky


from M_tabulated import M_tab
from eval_tab_model import tab_mod_vels

def bisym_mod(galaxy, vel, evel, guess0, vary, n_it, rstart, rfinal, ring_space, frac_pixel, delta, pixel_scale,bar_min_max,errors, config, e_ISM, steps, method, use_metropolis):


		vrot0,vr20,pa0,inc0,x0,y0,vsys0,vtan,theta_b0 = guess0
		vmode = "bisymmetric"
		[ny,nx] = vel.shape
		shape = [ny,nx]
		r_bar_min, r_bar_max = bar_min_max


		"""

		 					BYSIMETRIC MODEL


		"""

		chisq_global = 1e10
		PA, INC, XC,YC,VSYS = 0,0,0,0,0
		Vrot, Vrad, Vsys,Vtan = [],[],[],[]
		R = 0
		best_xy_pix = []

		vrad_it, vtan_it = np.zeros(100,), np.zeros(100,)
		rings = np.arange(rstart, rfinal, ring_space)
		nrings = len(rings)
		vrot_tab_it, vrad_tab_it, vtan_tab_it = np.zeros(nrings,), np.zeros(nrings,), np.zeros(nrings,)
		for it in np.arange(n_it):

			if np.mean(vrad_tab_it) < 0 or np.mean(vtan_tab_it) <0 :
				theta_b0 = theta_b - 90
				if theta_b0 < 0: theta_b0 = 180 + theta_b0


			if it == 100:
				vrot_tab, vrad_tab, vtan_tab, R_pos = tab_mod_vels(rings,vel, evel, pa0,inc0,x0,y0,vsys0,theta_b0,delta,pixel_scale,"circular",shape,frac_pixel,r_bar_min, r_bar_max)
				vrad_tab, vtan_tab = vrad_tab+ 1e-3, vtan_tab + 1e-3
			else:
				vrot_tab, vrad_tab, vtan_tab, R_pos = tab_mod_vels(rings,vel, evel, pa0,inc0,x0,y0,vsys0,theta_b0,delta,pixel_scale,vmode,shape,frac_pixel,r_bar_min, r_bar_max)
				vrot_tab_it, vrad_tab_it, vtan_tab_it = vrot_tab, vrad_tab, vtan_tab
					

			vrot_tab[abs(vrot_tab) > 400] = np.nanmedian(vrot_tab)
			guess = [vrot_tab,vrad_tab,pa0,inc0,x0,y0,vsys0, vtan_tab,theta_b0]

			v_2D_mdl,  kin_2D_modls,  vrot , vrad, vsys0,  pa0, inc0, x0, y0, vtan,theta_b, xi_sq, n_data, Errors = fit(shape, vel, evel, guess, vary, vmode, config, R_pos, fit_method = method, e_ISM = e_ISM, pixel_scale = pixel_scale, ring_space = ring_space )

			if np.nanmean(vrot) < 0 :
				pa0 = pa0 - 180
				if pa0 < 0 : pa0 = pa0 + 360
				vrot = abs(np.asarray(vrot))


			theta_b0 = theta_b

			if xi_sq < chisq_global:

				PA, INC, XC,YC,VSYS,THETA = pa0, inc0, x0, y0,vsys0,theta_b
				Vrot = np.asarray(vrot)
				Vrad = np.asarray(vrad)
				Vtan = np.asarray(vtan)

				chisq_global = xi_sq
				best_vlos_2D_model = v_2D_mdl
				best_kin_2D_models = kin_2D_modls

				Rings = R_pos
				std_errors = Errors
				


		if errors == 1:

			print("starting Metropolis-Hastings analysis")

			from metropolis_hastings_bisym import bayesian_mcmc
			# Recompute the tabulated model ?
			#vrot_tab, vrad_tab, vtan_tab, R_pos = tab_mod_vels(Rings,vel, evel, PA,INC,XC,YC,VSYS,THETA,delta,pixel_scale,vmode,shape,frac_pixel,r_bar_min, r_bar_max)
			#GUESS = [vrot_tab, vrad_tab, PA, INC, XC, YC, VSYS, vtan_tab, THETA]

			GUESS = [Vrot, Vrad, PA, INC, XC, YC, VSYS, Vtan, THETA]

			chain,v_2D_mdl_, kin_2D_models_, Vrot_, Vrad_, Vsys_,  PA_, INC_ , XC_, YC_, Vtan_, THETA_, std_errors = bayesian_mcmc(galaxy, shape, vel, evel, GUESS, vmode, config, Rings, e_ISM = e_ISM, pixel_scale = pixel_scale, ring_space = ring_space, r_bar_min = r_bar_min, r_bar_max = r_bar_max, steps = steps)

			if use_metropolis == 1 or len(Vrot_) != len(Vrot):
				PA, INC, XC,YC,VSYS,THETA = PA_, INC_, XC_,YC_,Vsys_,THETA_
				Vrot, Vrad, Vtan = Vrot_, Vrad_, Vtan_
				best_vlos_2D_model = v_2D_mdl_
				best_kin_2D_models = kin_2D_models_
				#Rings = R_pos


		std_Vrot,std_Vrad,std_pa, std_inc, std_x0, std_y0, std_Vsys, std_theta, std_Vtan = std_errors

		Vtan = np.asarray(Vtan)
		Vrad = np.asarray(Vrad)
		Vrot = np.asarray(Vrot)
		PA_bar_major = pa_bar_sky(PA,INC,THETA)
		PA_bar_minor = pa_bar_sky(PA,INC,THETA-90)



		return PA,INC,XC,YC,VSYS,THETA,Rings,Vrot,Vrad,Vtan,best_vlos_2D_model,best_kin_2D_models,PA_bar_major,PA_bar_minor,chisq_global,std_errors
