#!/usr/bin/env python3
import sys
import numpy as np
nargs=len(sys.argv)
import configparser
from src.initialize_XS_main import XS_out




"""
#################################################################################
# 				XookSuut						#
#				version v.2.0.0					#
# 				C. Lopez-Coba					#
#################################################################################

"""


class input_params:
	def __init__(self):
		if (nargs < 22 or nargs > 25):

			print ("USE: XookSuut name vel_map.fits [error_map.fits,SN] pixel_scale PA INC X0 Y0 [VSYS] vary_PA vary_INC vary_X0 vary_Y0 vary_VSYS ring_space [delta] Rstart,Rfinal cover kin_model fit_method N_it [R_bar_min,R_bar_max] [config_file] [prefix]" )

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
		vary_PHIB = 1

		# Rings configuration
		ring_space = float(sys.argv[15])
		delta = sys.argv[16]
		rstart, rfinal =  eval(sys.argv[17])
		frac_pixel = eval(sys.argv[18])

		# Kinematic model, minimization method and iterations
		vmode = sys.argv[19]
		fit_method = sys.argv[20]
		n_it = int(sys.argv[21])


		#valid optional-string-inputs (osi):
		osi = ["-", ".", "#","%", "&", ""]

		r_bar_min_max,config_file,prefix = "","",""
		C, G = "C", "G"
		try:
			if sys.argv[22] not in osi: r_bar_min_max =  eval(sys.argv[22])
			if sys.argv[23] not in osi: config_file = sys.argv[23]
			if sys.argv[24] not in osi: prefix = sys.argv[24]
		except(IndexError): pass

		if config_file in osi:
			config_file = "../src/xs_conf.ini"
			print("XookSuut: No config file has been passed. Using default configuration file ..")


		if evel_map_SN not in osi:
			evel_map_SN =  evel_map_SN.split(",")
			evel_map, SN = evel_map_SN[0], float(evel_map_SN[1])
		else:
			evel_map, SN = "", 1e6	

		if VSYS not in osi: VSYS = eval(sys.argv[9])
		if delta in osi:
			delta = ring_space/2. 
		else:
			delta = float(delta)

		if r_bar_min_max in osi: r_bar_min_max = np.inf
		if vmode not in ["circular","radial","bisymmetric"] and "hrm_" not in vmode: print("XookSuut: choose a proper kinematic model !"); quit()
	

		if type(r_bar_min_max)  == tuple:
			bar_min_max = [r_bar_min_max[0], r_bar_min_max[1] ]
		else:

			bar_min_max = [rstart, r_bar_min_max ]

		if prefix != "": galaxy = "%s-%s"%(galaxy,prefix)	


		if fit_method == "LM": fit_method = "least_squares"
		if fit_method in ["Powell","powell","POWELL"]: fit_method = "Powell"

		if fit_method not in ["leastsq", "Powell","least_squares"] :
			print("XookSuut: choose an appropiate fitting method: LM or Powell")
			quit()


		input_config = configparser.ConfigParser(
			# allow a variables be set without value
			allow_no_value=True,
			# allows duplicated keys in different sections
			strict=False,
			# deals with variables inside configuratio file
			interpolation=configparser.ExtendedInterpolation())
		input_config.read(config_file)

		# Shortcuts to the different configuration sections variables.
		config_const = input_config['constant_params']
		config_general = input_config['general']

		PA = config_const.getfloat('PA', PA)
		INC = config_const.getfloat('INC', INC)
		X0 = config_const.getfloat('X0', X0)
		Y0 = config_const.getfloat('Y0', Y0)
		VSYS = config_const.getfloat('VSYS', VSYS)
		PHI_BAR = config_const.getfloat('PHI_BAR', 45)

		vary_PA = config_const.getboolean('FIT_PA', vary_PA)
		vary_INC = config_const.getboolean('FIT_INC', vary_INC)
		vary_XC = config_const.getboolean('FIT_X0', vary_XC)
		vary_YC = config_const.getboolean('FIT_Y0', vary_YC)
		vary_VSYS = config_const.getboolean('FIT_VSYS', vary_VSYS)
		vary_PHIB = config_const.getboolean('FIT_PHI_BAR', vary_PHIB)

		e_centroid = config_general.getfloat("e_centroid", 5)
		v_center = config_general.get("v_center", 0)  
		survey = config_general.get("dataset", "-")
		try:
			v_center = float(v_center)
		except (ValueError): pass

		if type(v_center) == str and v_center != "extrapolate":
			print("XookSuut: v_center: %s, did you mean ´extrapolate´ ?"%v_center)
			quit()


		config = input_config

		x = XS_out(galaxy, vel_map, evel_map, SN, VSYS, PA, INC, X0, Y0, PHI_BAR, n_it, pixel_scale, vary_PA, vary_INC, vary_XC, vary_YC, vary_VSYS, vary_PHIB, delta, rstart, rfinal, ring_space, frac_pixel, v_center, bar_min_max, vmode, survey, config, e_centroid, fit_method , prefix, osi  )
		print("saving plots ..")
		out_xs = x()

if __name__ == "__main__":
	init = input_params()


