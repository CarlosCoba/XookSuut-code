import numpy as np
import matplotlib.pylab as plt
import random


from circular_mode import circ_mod
from radial_mode import rad_mod
from bisymetric_mode import bisym_mod

#from emcee_fit import EMCEE
def init_models(galaxy,vel,evel,guess0,vary,n_it,sigma, delta,  rstart, rfinal, ring_space, frac_pixel, bar_min_max, pixel_scale, vmode, errors, config, e_ISM, steps, method , use_metropolis):

	vrot0,vr20,pa0,inc0,x0,y0,vsys0,theta0 = guess0
	guess0 = [vrot0,vr20,pa0,inc0,x0,y0,vsys0,0,theta0]


	if vmode == "circular":

		PA,INC,XC,YC,VSYS,THETA,R,Vrot,Vrad,Vtan,best_vlos_2D_mdl, best_kin_2D_mdls,CHISQ_r,res_MCMC  = circ_mod(galaxy, vel, evel, guess0, vary, n_it, rstart =  rstart, rfinal = rfinal, ring_space = ring_space, frac_pixel = frac_pixel, delta = delta, pixel_scale = pixel_scale, bar_min_max = bar_min_max, errors = errors, config = config, e_ISM = e_ISM, steps = steps, method = method, use_metropolis = use_metropolis )

		return PA,INC,XC,YC,VSYS,0,R,Vrot,0*Vrot,0*Vrot,best_vlos_2D_mdl, best_kin_2D_mdls,CHISQ_r,res_MCMC



	if vmode == "radial":
		PA,INC,XC,YC,VSYS,THETA,R,Vrot,Vrad,Vtan,best_vlos_2D_mdl, best_kin_2D_mdls,CHISQ_r,res_MCMC = rad_mod(galaxy, vel, evel, guess0, vary, n_it, rstart =  rstart, rfinal = rfinal, ring_space = ring_space, frac_pixel = frac_pixel, delta = delta, pixel_scale = pixel_scale, bar_min_max = bar_min_max,errors = errors,  config = config, e_ISM = e_ISM, steps = steps, method = method, use_metropolis = use_metropolis)
		return PA,INC,XC,YC,VSYS,0,R,Vrot,Vrad,0*Vrot,best_vlos_2D_mdl, best_kin_2D_mdls,CHISQ_r,res_MCMC



	
	if vmode == "bisymmetric":
		PA,INC,XC,YC,VSYS,THETA,R,Vrot,Vrad,Vtan,best_vlos_2D_mdl, best_kin_2D_mdls,PA_BAR_MAJOR,PA_BAR_MINOR,CHISQ_r,res_MCMC  = bisym_mod(galaxy, vel, evel, guess0, vary, n_it, rstart =  rstart, rfinal = rfinal, ring_space = ring_space, frac_pixel = frac_pixel, delta = delta, pixel_scale = pixel_scale, bar_min_max = bar_min_max, errors = errors,  config = config, e_ISM = e_ISM, steps = steps, method = method, use_metropolis = use_metropolis)

		return PA,INC,XC,YC,VSYS,THETA,R,Vrot,Vrad,Vtan,best_vlos_2D_mdl, best_kin_2D_mdls,PA_BAR_MAJOR,PA_BAR_MINOR,CHISQ_r,res_MCMC







