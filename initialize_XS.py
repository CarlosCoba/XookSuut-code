import numpy as np
import matplotlib.pylab as plt
from astropy.io import fits
np.warnings.filterwarnings('ignore')
import os.path
from os import path

from axis import AXIS 
from fit_rotcurve import Rturn_Vmax


from read_config import config_file
from xs import init_models 

from kinematic_centre_vsys import KC
from CBAR import colorbar as cb

from isolated_pixels import filter_isolated_cells

from write_table import write

from filter_pixels import filter_SN
c = 3e5

def SN(flux,eflux,sn):
	A = np.divide(flux,eflux)
	A[A<sn] = np.nan
	return np.divide(A,A)


def rotcur(galaxy, vel_map2D,evel_map,SN,VSYS,PA,INC,X0,Y0,n_it =5, pixel_scale=1,vary_PA=True,vary_INC=True,vary_XC=True,vary_YC=True,vary_VSYS=True,vary_PHI=True,delta=1, rstart=2, rfinal=50, ring_space=2, frac_pixel=2/3., r_bar_max=20,vmode="circular", save_plots = 1, errors = 1,survey = "", config = "", e_ISM = 5,steps = 1e4, method = "Powell", use_metropolis = 0):
	"""
	vary = [Vrot,Vrad,PA,INC,XC,YC,VSYS,theta,Vtan]
	"""



	vel_map = fits.getdata(vel_map2D)


	[ny,nx] = vel_map.shape
	ext = [nx/2., -nx/2,-ny/2.,ny/2.]
	ext = np.dot([nx/2., -nx/2,-ny/2.,ny/2.], pixel_scale)

	
	if evel_map == "":
		
		evel_map = np.ones((ny,nx))
		SN = 1e5
	else:
		evel_map = fits.getdata(evel_map)
		#evel_map[evel_map>SN]=np.nan

	
	evel_map[evel_map == 0] = np.nan
	evel_map[abs(evel_map)>1e3] = np.nan
	evel_map[~np.isfinite(evel_map)] = np.nan

	evel_map_copy = np.copy(evel_map)
	evel_map[evel_map>SN]=np.nan

	mask_vel=np.divide(evel_map,evel_map)

	vel_ha = filter_SN(vel_map,evel_map, SN)

	evel_map = evel_map_copy
	#vel_ha = vel_map*mask_vel
	vel_ha[vel_ha==0]=np.nan
	vel_ha[vel_ha<0] = np.nan 

	#plt.imshow(vel_ha)
	#plt.show()

	if VSYS == "":
		XC,YC,VSYS,e_vsys = KC(vel_map,X0,Y0,pixel_scale)


	PHI_B = 45
	# List of guess values
	if config == "":
		guess = [50,0,PA,INC,X0,Y0,VSYS,PHI_B]
	else:
		guess = [50,0]

		for res in config_file(config):
				param, fit, val, vmin, vmax = str(res["param"]), bool(float(res["fit"])), eval(res["val"]), eval(res["min"]), eval(res["max"]) 
				guess.append(val)



	# Starting messenge
	from start_messenge import start
	start(galaxy,guess,vmode,config)


	# Maybe apply a filter to remove isolates pixels in the velocity map?	
	vel_ha = filter_isolated_cells(vel_ha,struct=np.ones((3,3)))


	def run_code():


		vary = [True,True,vary_PA,vary_INC,vary_XC,vary_YC,vary_VSYS,vary_PHI,True]
		sigma = []
		if vmode == "bisymmetric":
			PA,INC,XC,YC,VSYS,THETA,R,Vrot,Vrad,Vtan,VLOS_2D_MODEL, KIN_2D_MODELS,PA_BAR_MAJOR,PA_BAR_MINOR,CHISQ_r,errors_fit = init_models(galaxy,vel_ha,evel_map,guess,vary, n_it, sigma, delta, rstart, rfinal, ring_space, frac_pixel,r_bar_max,pixel_scale,vmode,errors, config, e_ISM, steps, method , use_metropolis)
		else:

			PA,INC,XC,YC,VSYS,THETA,R,Vrot,Vrad,Vtan,VLOS_2D_MODEL, KIN_2D_MODELS,CHISQ_r,errors_fit = init_models(galaxy,vel_ha,evel_map,guess,vary, n_it, sigma, delta, rstart, rfinal, ring_space, frac_pixel, r_bar_max,pixel_scale,vmode,errors, config, e_ISM, steps, method , use_metropolis)
			PA_BAR_MAJOR,PA_BAR_MINOR,THETA = 0,0,0

		return PA,INC,XC,YC,VSYS,THETA,R,Vrot,Vrad,Vtan,VLOS_2D_MODEL, KIN_2D_MODELS,PA_BAR_MAJOR,PA_BAR_MINOR,CHISQ_r,errors_fit




	PA,INC,XC,YC,VSYS,THETA,R,Vrot,Vrad,Vtan,VLOS_2D_MODEL, KIN_2D_MODELS,PA_BAR_MAJOR,PA_BAR_MINOR,CHISQ_r,errors_fit = run_code()



	#Extract Errors
	
	e_Vrot,e_Vrad,e_PA,e_INC,XC_e,YC_e,e_Vsys,e_theta,e_Vtan = errors_fit	


	if np.nanmean(Vrot) < 0: Vrot = abs(Vrot)

	if None in sum(e_Vrot, []) or None in Vrot: 
		if survey != "":
			error_table = "BAD_FIT_kin_%s_model.%s.csv"%(vmode,survey)
		else:
			error_table = "BAD_FIT_kin_%s_model.%s.csv"%(vmode,galaxy)

		if path.exists(error_table) == True:
			pass
		else:

			hdr = ["object", "error messege!" ]
			write(hdr,error_table,column = False)
		flag = "Initial guess values are probably too far from the real ones" 
		tab_err = [galaxy,flag]
		write(tab_err,error_table,column = False)
		print(flag)
		print("############################")
		quit()


	#
	## Write output into a table
	#

	if vmode == "circular" or vmode == "radial":
		if survey != "":
			kin_params_table = "ana_kin_%s_model.%s.csv"%(vmode,survey)
		else:
			kin_params_table = "ana_kin_%s_model.%s.csv"%(vmode,galaxy)

		# write header of table
		if path.exists(kin_params_table) == True:
			pass
		else:
			hdr = ["object", "XC", "e_XC_l", "e_XC_u", "YC", "e_YC_l", "e_YC_u", "PA", "e_PA_l", "e_PA_u", "INC", "e_INC_l", "e_INC_u", "VSYS", "e_VSYS_l", "e_VSYS_u", "CHISQ_r" ]
			write(hdr,kin_params_table,column = False)


		kin_params = [galaxy,XC,XC_e,XC_e,YC,YC_e,YC_e,PA,e_PA,e_PA,INC,e_INC,e_INC,VSYS,e_Vsys,e_Vsys,CHISQ_r]

		write(kin_params,kin_params_table,column = False)

	if vmode == "bisymmetric":
		if survey != "":
			kin_params_table = "ana_kin_%s_model.%s.csv"%(vmode,survey)
		else:
			kin_params_table = "ana_kin_%s_model.%s.csv"%(vmode,galaxy)


		# write header of table
		if path.exists(kin_params_table) == True:
			pass
		else:
			hdr = ["object", "XC", "e_XC_l", "e_XC_u", "YC", "e_YC_l", "e_YC_u", "PA", "e_PA_l", "e_PA_u", "INC", "e_INC_l", "e_INC_u", "VSYS", "e_VSYS_l", "e_VSYS_u", "PHI_BAR", "e_PHI_BAR_l","e_PHI_BAR_u","PA_BAR_MAJOR_sky","PA_BAR_MINOR_sky","CHISQ_r" ]
			write(hdr,kin_params_table,column = False)

		kin_params = [galaxy,XC,XC_e,XC_e,YC,YC_e,YC_e,PA,e_PA,e_PA,INC,e_INC,e_INC,VSYS,e_Vsys,e_Vsys,THETA,e_theta,e_theta,PA_BAR_MAJOR,PA_BAR_MINOR,CHISQ_r]

		write(kin_params,kin_params_table,column = False)


	save_file = save_plots


	from plot_models import plot_kin_models
	plot_kin_models(galaxy,vmode,vel_ha,R,Vrot,e_Vrot,Vrad,e_Vrad,Vtan,e_Vtan, VSYS, VLOS_2D_MODEL, ext,plot = 0, save = save_file)


	from save_fits_1D_model import save_model
	s = save_model(galaxy,vmode,R,Vrot,e_Vrot,Vrad,e_Vrad,Vtan,e_Vtan,PA,INC,XC,YC,VSYS,THETA,PA_BAR_MAJOR,PA_BAR_MINOR,save=save_file)


	from save_fits_2D_model import save_vlos_model
	if vmode == "bisymmetric":
		save_vlos_model(galaxy,vmode,vel_map,evel_map_copy,VLOS_2D_MODEL,KIN_2D_MODELS,PA,INC,XC,YC,VSYS,save = save_file,theta = THETA, phi_bar_major = PA_BAR_MAJOR, phi_bar_minor = PA_BAR_MINOR)
	else:
		save_vlos_model(galaxy,vmode,vel_map,evel_map_copy,VLOS_2D_MODEL,KIN_2D_MODELS,PA,INC,XC,YC,VSYS,save = save_file)


	from save_plot_vmax_rturn import fit_rotcur
	Vmax,Rturn = fit_rotcur(galaxy,vmode,e_Vrot,survey)


	print("Done!")
	print("############################")




 
