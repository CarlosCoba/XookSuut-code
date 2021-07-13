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
from eval_tab_model import tab_mod_vels


def rad_mod(galaxy, vel, evel, guess0, vary, n_it, rstart, rfinal, ring_space, frac_pixel, delta, pixel_scale, bar_min_max, errors, config, e_ISM, steps, method, use_metropolis ):


		vrot0,vr20,pa0,inc0,x0,y0,vsys0,vtan,theta_b = guess0
		vmode = "radial"
		[ny,nx] = vel.shape
		shape = [ny,nx]




		"""

		 					RADIAL MODEL


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

			vrot_tab, vrad_tab, vtan_tab, R_pos = tab_mod_vels(rings,vel, evel, pa0,inc0,x0,y0,vsys0,theta_b,delta,pixel_scale,vmode,shape,frac_pixel,0,0)

			vrot_tab[abs(vrot_tab) > 400] = np.nanmedian(vrot_tab) 	
			guess = [vrot_tab,vrad_tab,pa0,inc0,x0,y0,vsys0, vtan_tab,theta_b]


			v_2D_mdl, kin_2D_modls, vrot , vrad, vsys0,  pa0, inc0, x0, y0, xi_sq, n_data, Errors = fit(shape, vel, evel, guess, vary, vmode, config, R_pos, fit_method = method, e_ISM = e_ISM, pixel_scale = pixel_scale, ring_space = ring_space  )

			if np.nanmean(vrot) < 0 :
				pa0 = pa0 - 180
				if pa0 < 0 : pa0 = pa0 + 360
				vrot = abs(np.asarray(vrot))

			if xi_sq < chisq_global:

				PA, INC, XC,YC,VSYS,THETA = pa0, inc0, x0, y0,vsys0,theta_b
				Vrot = np.asarray(vrot)
				Vrad = np.asarray(vrad)
				chisq_global = xi_sq
				best_vlos_2D_model = v_2D_mdl
				best_kin_2D_models = kin_2D_modls
				Rings = R_pos
				std_errors = Errors
				GUESS = [Vrot, 0, PA, INC, XC, YC, VSYS, 0, 0]


		if errors == 1:

			print("starting Metropolis-Hastings analysis")
			from metropolis_hastings_rad import bayesian_mcmc
			
			#vrot_tab, vrad_tab, vtan_tab, R_pos = tab_mod_vels(Rings,vel, evel, PA,INC,XC,YC,VSYS,THETA,delta,pixel_scale,vmode,shape,frac_pixel,0,0)
			#GUESS = [vrot_tab, vrad_tab, PA, INC, XC, YC, VSYS, vtan_tab, THETA]

			GUESS = [Vrot, Vrad, PA, INC, XC, YC, VSYS, Vtan, THETA]

			chain,v_2D_mdl_, kin_2D_models_, Vrot_, Vrad_, Vsys_,  PA_, INC_ , XC_, YC_, Vtan_, THETA_, std_errors = bayesian_mcmc(galaxy, shape, vel, evel, GUESS, vmode, config, Rings, e_ISM = e_ISM, pixel_scale = pixel_scale, ring_space = ring_space, r_bar_min = 0, r_bar_max = 1e5, steps = steps)

			if use_metropolis == 1  or len(Vrot_) != len(Vrot):
				PA, INC, XC,YC,VSYS,THETA = PA_, INC_, XC_,YC_,Vsys_,THETA_
				Vrot, Vrad, Vtan = Vrot_, Vrad_, Vtan_
				best_vlos_2D_model = v_2D_mdl_
				best_kin_2D_models = kin_2D_models_
				#Rings = R_pos




		std_Vrot,std_Vrad,std_pa, std_inc, std_x0, std_y0, std_Vsys, std_theta, std_Vtan = std_errors

		Vrot = np.array(Vrot)
		Vrad = np.array(Vrad)

		return PA,INC,XC,YC,VSYS,0,Rings,Vrot,Vrad,0*Vrot,best_vlos_2D_model,best_kin_2D_models,chisq_global,std_errors




