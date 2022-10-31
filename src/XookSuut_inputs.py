#!/usr/bin/env python3
import sys
import numpy as np
nargs=len(sys.argv)
from initialize_XS_main import XS_out

"""
#################################################################################
# 				The XookSuut-code.				#
#				version v.1.0.0					#
# 				C. Lopez-Coba					#
#################################################################################

"""


class input_params:
	def __init__(self):

		if (nargs < 23 or nargs > 32):

			print ("USE: XookSuut name vel_map.fits [error_map.fits,SN] pixel_scale PA INC X0 Y0 [VSYS] vary_PA vary_INC vary_X0 vary_Y0 vary_VSYS ring_space [delta] Rstart,Rfinal frac_pixel inner_interp  kin_model fit_method N_it [R_bar_min,R_bar_max] [errors] [stepsize_mcmc] [use_mcmc_res=0,1] [plot_chain_mcmc=0,1] [e_ISM] [survey] [config_file] [prefix]" )

			exit()

		#object name
		galaxy = sys.argv[1]

		#FITS information
		vel_map = sys.argv[2]
		evel_map_SN = sys.argv[3]
		pixel_scale = float(sys.argv[4])

		# Geometrical parameters
		PA = float(sys.argv[5])
		INC =float(sys.argv[6])
		X0 = float(sys.argv[7])
		Y0 = float(sys.argv[8])
		VSYS = sys.argv[9]
		vary_PA = bool(float(sys.argv[10]))
		vary_INC = bool(float(sys.argv[11]))
		vary_XC = bool(float(sys.argv[12]))
		vary_YC = bool(float(sys.argv[13]))
		vary_VSYS = bool(float(sys.argv[14]))

		# Rings configuration
		ring_space = float(sys.argv[15])
		delta = sys.argv[16]
		rstart, rfinal =  eval(sys.argv[17])
		frac_pixel = eval(sys.argv[18])
		inner_interp = float(sys.argv[19])

		# Kinematic model, minimization method and iterations
		vmode = sys.argv[20]
		fit_method = sys.argv[21]
		n_it = int(sys.argv[22])


		#valid optional-string-inputs (osi):
		osi = ["-", ".", ",", "#","%", "&", ""]

		r_bar_min_max,errors,stepsize,use_mcmc,save_plots,e_ISM,survey,config,prefix = "","","","","","","","",""

		try:
			if sys.argv[23] not in osi: r_bar_min_max =  eval(sys.argv[23])
			if sys.argv[24] not in osi: errors = eval(sys.argv[24])
			if sys.argv[25] not in osi: stepsize=np.asarray(eval(sys.argv[25]))
			if sys.argv[26] not in osi: use_mcmc =  bool(sys.argv[26])
			if sys.argv[27] not in osi: save_plots = bool(sys.argv[27])
			if sys.argv[28] not in osi: e_ISM = float(sys.argv[28])
			if sys.argv[29] not in osi: survey = sys.argv[29]
			if sys.argv[30] not in osi: config = sys.argv[30]
			if sys.argv[31] not in osi: prefix = sys.argv[31]
		except(IndexError): pass


		if evel_map_SN not in osi:
			evel_map_SN =  evel_map_SN.split(",")
			evel_map, SN = evel_map_SN[0], float(evel_map_SN[1])
		else:
			evel_map, SN = "", 1e6	

		if VSYS not in osi: VSYS = eval(sys.argv[9])
		#if stepsize in osi: stepsize = np.asarray([])
		if delta in osi:
			delta = ring_space/2. 
		else:
			delta = float(delta)

		if e_ISM in osi: e_ISM = 0
		if save_plots in osi: save_plots = 0
		if use_mcmc in osi: use_mcmc = 0
		if r_bar_min_max in osi: r_bar_min_max = np.inf
		if errors in osi : errors = 0

		if errors not in osi:
			if errors == 0: errors = [0]
			if errors[0] == 1 and len(errors) !=2: print("XookSuut: Invalid dimension for bootstrap errors [1,N_boots]") ;quit()
			if errors[0] in [2,3] and len(errors) !=5: print("XookSuut: Invalid dimension for emcee  [2,steps,thin,burnin,N_walkers]") ;quit()
			if errors[0] == 4 and len(errors) !=4: print("XookSuut: Invalid dimension for M-H errors [4,steps,thin,burnin]") ;quit()
			nwalkers_mh = -9999
			if errors[0] == 4: errors.extend([nwalkers_mh])
			if errors[0] in [2,3,4]:
				errors.append(save_plots)
		else:
			errors = 0 
	
		vary_PHI = 1
		if vmode not in ["circular", "radial", "bisymmetric"] and ("hrm_" in vmode) == False:
			print("XookSuut: choose a model between circular, radial, bisymmetric or hrm_m ")
			quit() 
		if stepsize not in osi :
			if vmode == "circular": 
				if stepsize.size  != 6 : print("XookSuut: stepsize dims. is different from 1 + 5 Ex: [sigmaVrot,sigmaPA,sigmaINC,sigmaXC,sigmaYC,sigmaVsys]") ; quit()
			if vmode == "radial": 
				if stepsize.size  != 7 : print("XookSuut: stepsize dims. is different from 2 + 5 Ex: [sigmaVrot,sigmaVrad,sigmaPA,sigmaINC,sigmaXC,sigmaYC,sigmaVsys]") ; quit()
			if vmode == "bisymmetric": 
				if stepsize.size  != 9 : print("XookSuut: stepsize dims. is different from 3 + 6 Ex: [sigmaVrot,sigmaVrad,sigmaVtan,sigmaINC,sigmaPA,sigmaXC,sigmaYC,sigmaVsys,sigma_phi]") ; quit()
			if "hrm" in vmode:
				if stepsize.size  != (5 + 2 ) : print("XookSuut: stepsize dims. is different from 2 + 5 Ex: [sigma_ck,sigma_sk,sigmaPA,sigmaINC,sigmaXC,sigmaYC,sigma_c0]") ; quit()

		if stepsize in osi:
			stepsize = np.asarray([])


		if type(r_bar_min_max)  == tuple:
			bar_min_max = [r_bar_min_max[0], r_bar_min_max[1] ]
		else:

			bar_min_max = [rstart, r_bar_min_max ]

		if prefix != "": galaxy = "%s-%s"%(galaxy,prefix)			

		x = XS_out(galaxy, vel_map, evel_map, SN, VSYS, PA, INC, X0, Y0, n_it, pixel_scale, vary_PA, vary_INC, vary_XC, vary_YC, vary_VSYS, vary_PHI, delta, rstart, rfinal, ring_space, frac_pixel, inner_interp, bar_min_max, vmode, errors, survey, config, e_ISM, fit_method , use_mcmc, save_plots, stepsize, prefix, osi  )
		print("saving plots ..")
		out_xs = x()

if __name__ == "__main__":

	init = input_params()
