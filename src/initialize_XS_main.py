import numpy as np
import matplotlib.pylab as plt
from astropy.io import fits
np.warnings.filterwarnings('ignore')
import os.path
from os import path

from axis import AXIS 
from read_config import config_file

from kinematic_centre_vsys import KC
from CBAR import colorbar as cb

from isolated_pixels import filter_isolated_cells

from write_table import write

from filter_pixels import filter_SN
c = 3e5


from circular_mode import Circular_model
from radial_mode import Radial_model
from bisymetric_mode import Bisymmetric_model
from harmonic_mode import Harmonic_model
import create_directories

valid_strings_for_optional_inputs = ["", "-", ".", ",", "#","%", "&"]


def clean_vel_map(vel_map2D, evel_map, SN, osi):


	vel_map = fits.getdata(vel_map2D)
	[ny,nx] = vel_map.shape

	if evel_map in osi:
		
		evel_map = np.ones((ny,nx))
		SN = 1e5
	else:
		evel_map = fits.getdata(evel_map)


	evel_map[evel_map == 0] = np.nan
	evel_map[abs(evel_map)>1e3] = np.nan
	evel_map[~np.isfinite(evel_map)] = np.nan

	evel_map_copy = np.copy(evel_map)
	evel_map[evel_map>SN]=np.nan

	mask_vel=np.divide(evel_map,evel_map)
	vel_ha = filter_SN(vel_map,evel_map, SN)

	evel_map = evel_map_copy
	vel_ha[vel_ha==0]=np.nan


	# Maybe apply a filter to remove isolates pixels in the velocity map?	
	vel_ha = filter_isolated_cells(vel_ha,struct=np.ones((3,3)))

	return vel_ha, evel_map






def guess_vals(config,PA,INC,X0,Y0,VSYS,PHI_B ):

	# List of guess values
	if config == "":
		guess = [PA,INC,X0,Y0,VSYS,PHI_B]
	else:
		guess = []

		for res in config_file(config):
				param, fit, val, vmin, vmax = str(res["param"]), bool(float(res["fit"])), eval(res["val"]), eval(res["min"]), eval(res["max"]) 
				guess.append(val)
	return guess







class Run_models:

	def __init__(self, galaxy, vel_map, evel_map, SN, VSYS, PA, INC, X0, Y0, n_it, pixel_scale, vary_PA, vary_INC, vary_XC, vary_YC, vary_VSYS, vary_PHI, delta, rstart, rfinal, ring_space, frac_pixel, inner_interp, bar_min_max, vmode, errors, survey, config, e_ISM, fit_method , use_mcmc, save_plots, stepsize, prefix, osi):


		PHI_B = 45
		self.vmode = vmode
		self.galaxy = galaxy
		self.vel_map, self.evel_map = clean_vel_map(vel_map, evel_map, SN, osi)

		input_vel_map = self.vel_map
		input_evel_map = self.evel_map
		[ny,nx] = self.vel_map.shape

		self.PA_bar_mjr,self.PA_bar_mnr,self.PHI_BAR = 0,0,0
		self.survey = survey
		self.m_hrm = 3
		
		if VSYS in osi :
			X0,Y0,VSYS,e_vsys = KC(self.vel_map,X0,Y0,pixel_scale)

		guess0 = guess_vals(config,PA,INC,X0,Y0,VSYS,PHI_B )
		vary = [vary_PA,vary_INC,vary_XC,vary_YC,vary_VSYS,vary_PHI]
		sigma = []
		self.ext = np.dot([nx/2., -nx/2,-ny/2.,ny/2.], pixel_scale)
		self.osi = osi

		if "hrm_" in vmode:
			try:
				self.m_hrm = int(vmode[4:])
				if self.m_hrm == 0 : 
					raise ValueError
			except(ValueError):
				print("XookSuut: provide a proper harmonic number different from zero, for example hrm_2")
				quit()

		# Print starting messenge
		from start_messenge import start
		start(galaxy,guess0,vmode,config)
		print("starting Least Squares analysis ..")
		if self.vmode == "circular": 
			#self.PA,self.INC,self.XC,self.YC,self.VSYS,self.PHI_BAR,self.R,self.Vrot,self.Vrad,self.Vtan,self.vlos_2D_mdl, self.kin_2D_mdls,self.bic_aic,self.errors_fit  = circ_mod(galaxy, self.vel_map, self.evel_map, guess0, vary, n_it, rstart, rfinal, ring_space, frac_pixel, inner_interp, delta, pixel_scale,bar_min_max, errors, config, e_ISM, fit_method, use_mcmc,  stepsize)
			circ = Circular_model(galaxy, self.vel_map, self.evel_map, guess0, vary, n_it, rstart, rfinal, ring_space, frac_pixel, inner_interp, delta, pixel_scale,bar_min_max, errors, config, e_ISM, fit_method, use_mcmc,  stepsize)
			self.PA,self.INC,self.XC,self.YC,self.VSYS,self.PHI_BAR,self.R,self.Vrot,self.Vrad,self.Vtan,self.vlos_2D_mdl, self.kin_2D_mdls,self.bic_aic,self.errors_fit = circ()

		if self.vmode == "radial":
			#self.PA,self.INC,self.XC,self.YC,self.VSYS,self.PHI_BAR,self.R,self.Vrot,self.Vrad,self.Vtan,self.vlos_2D_mdl, self.kin_2D_mdls,self.bic_aic,self.errors_fit = rad_mod(galaxy, self.vel_map, self.evel_map, guess0, vary, n_it, rstart, rfinal, ring_space, frac_pixel, inner_interp, delta, pixel_scale,bar_min_max, errors, config, e_ISM, fit_method, use_mcmc,  stepsize)
			rad = Radial_model(galaxy, self.vel_map, self.evel_map, guess0, vary, n_it, rstart, rfinal, ring_space, frac_pixel, inner_interp, delta, pixel_scale,bar_min_max, errors, config, e_ISM, fit_method, use_mcmc,  stepsize)
			self.PA,self.INC,self.XC,self.YC,self.VSYS,self.PHI_BAR,self.R,self.Vrot,self.Vrad,self.Vtan,self.vlos_2D_mdl, self.kin_2D_mdls,self.bic_aic,self.errors_fit = rad()

		if self.vmode == "bisymmetric":
			#self.PA,self.INC,self.XC,self.YC,self.VSYS,self.PHI_BAR,self.R,self.Vrot,self.Vrad,self.Vtan,self.vlos_2D_mdl, self.kin_2D_mdls,self.PA_bar_mjr,self.PA_bar_mnr,self.bic_aic,self.errors_fit  = bisym_mod(galaxy, self.vel_map, self.evel_map, guess0, vary, n_it, rstart, rfinal, ring_space, frac_pixel, inner_interp, delta, pixel_scale,bar_min_max, errors, config, e_ISM, fit_method, use_mcmc,  stepsize)
			bis = Bisymmetric_model(galaxy, self.vel_map, self.evel_map, guess0, vary, n_it, rstart, rfinal, ring_space, frac_pixel, inner_interp, delta, pixel_scale,bar_min_max, errors, config, e_ISM, fit_method, use_mcmc,  stepsize)
			self.PA,self.INC,self.XC,self.YC,self.VSYS,self.PHI_BAR,self.R,self.Vrot,self.Vrad,self.Vtan,self.vlos_2D_mdl, self.kin_2D_mdls,self.PA_bar_mjr,self.PA_bar_mnr,self.bic_aic,self.errors_fit = bis()

		if "hrm" in self.vmode:
			#self.PA,self.INC,self.XC,self.YC,self.VSYS,self.R,self.Ck,self.Sk,self.vlos_2D_mdl, self.kin_2D_mdls,self.bic_aic,self.errors_fit  = m2_mod(galaxy, self.vel_map, self.evel_map, guess0, vary, n_it, rstart, rfinal, ring_space, frac_pixel, inner_interp, delta, pixel_scale, bar_min_max, errors, config, e_ISM, fit_method, use_mcmc, stepsize, self.m_hrm)
			hrm = Harmonic_model(galaxy, self.vel_map, self.evel_map, guess0, vary, n_it, rstart, rfinal, ring_space, frac_pixel, inner_interp, delta, pixel_scale, bar_min_max, errors, config, e_ISM, fit_method, use_mcmc, stepsize, self.m_hrm)
			self.PA,self.INC,self.XC,self.YC,self.VSYS,self.R,self.Ck,self.Sk,self.vlos_2D_mdl, self.kin_2D_mdls,self.bic_aic,self.errors_fit = hrm()

		self.redchi = self.bic_aic[-1] 
		#self.results()



class XS_out(Run_models):

	def results(self):

		if "hrm" not in self.vmode:
			e_Vrot,e_Vrad,e_Vtan, e_PA, e_INC, e_XC, e_YC, e_Vsys, e_theta = self.errors_fit	
		else:
			e_PA,e_INC,e_XC,e_YC,e_Vsys  = self.errors_fit[-5:]
			e_Ck,e_Sk = [self.errors_fit[k] for k in range(self.m_hrm)], [self.errors_fit[k] for k in range(self.m_hrm,2*self.m_hrm)]



		#
		## Write output into a table
		#

		if self.vmode == "circular" or self.vmode == "radial" or "hrm" in self.vmode:
			if self.survey not in self.osi :
				kin_params_table = "ana_kin_%s_model.%s.csv"%(self.vmode,self.survey)
			else:
				kin_params_table = "ana_kin_%s_model.%s.csv"%(self.vmode,self.galaxy)

			# write header of table
			if path.exists(kin_params_table) == True:
				pass
			else:
				hdr = ["object", "XC", "e_XC_l", "e_XC_u", "YC", "e_YC_l", "e_YC_u", "PA", "e_PA_l", "e_PA_u", "INC", "e_INC_l", "e_INC_u", "VSYS", "e_VSYS_l", "e_VSYS_u", "redchi" ]
				write(hdr,kin_params_table,column = False)


			kin_params = [self.galaxy,self.XC,e_XC,e_XC,self.YC,e_YC,e_YC,self.PA,e_PA,e_PA,self.INC,e_INC,e_INC,self.VSYS,e_Vsys,e_Vsys,self.redchi]

			write(kin_params,kin_params_table,column = False)

		if self.vmode == "bisymmetric":
			if self.survey not in self.osi:
				kin_params_table = "ana_kin_%s_model.%s.csv"%(self.vmode,self.survey)
			else:
				kin_params_table = "ana_kin_%s_model.%s.csv"%(self.vmode,self.galaxy)


			# write header of table
			if path.exists(kin_params_table) == True:
				pass
			else:
				hdr = ["object", "XC", "e_XC_l", "e_XC_u", "YC", "e_YC_l", "e_YC_u", "PA", "e_PA_l", "e_PA_u", "INC", "e_INC_l", "e_INC_u", "VSYS", "e_VSYS_l", "e_VSYS_u", "PHI_BAR", "e_PHI_BAR_l","e_PHI_BAR_u","PA_bar_mjr_sky","PA_bar_mnr_sky","redchi" ]
				write(hdr,kin_params_table,column = False)

			kin_params = [self.galaxy,self.XC,e_XC,e_XC,self.YC,e_YC,e_YC,self.PA,e_PA,e_PA,self.INC,e_INC,e_INC,self.VSYS,e_Vsys,e_Vsys,self.PHI_BAR,e_theta,e_theta,self.PA_bar_mjr,self.PA_bar_mnr,self.redchi]

			write(kin_params,kin_params_table,column = False)



		if "hrm" not in self.vmode:
			from plot_models import plot_kin_models
			plot_kin_models(self.galaxy, self.vmode,self.vel_map,self.R,self.Vrot,e_Vrot,self.Vrad,e_Vrad,self.Vtan,e_Vtan, self.VSYS, self.vlos_2D_mdl, self.ext)


			from save_fits_1D_model import save_model
			s = save_model(self.galaxy, self.vmode,self.R,self.Vrot,self.Vrad,self.Vtan,self.PA,self.INC,self.XC,self.YC,self.VSYS,self.PHI_BAR,self.PA_bar_mjr,self.PA_bar_mnr,self.errors_fit,self.bic_aic)



		if "hrm" in self.vmode:

			from plot_models_harmonic import plot_kin_models
			plot_kin_models(self.galaxy, self.vmode,self.vel_map,self.R,self.Ck,self.Sk,e_Ck,e_Sk,self.VSYS,self.INC,self.vlos_2D_mdl, self.ext,self.m_hrm,survey = self.survey)


			from save_fits_1D_model_harmonic import save_model
			s = save_model(self.galaxy, self.vmode,self.R,self.Ck,self.Sk,e_Ck,e_Sk,self.PA,self.INC,self.XC,self.YC,self.VSYS,self.m_hrm,self.errors_fit,self.bic_aic)
					

		from save_fits_2D_model import save_vlos_model
		if self.vmode == "bisymmetric":
			save_vlos_model(self.galaxy, self.vmode,self.vel_map,self.evel_map,self.vlos_2D_mdl,self.kin_2D_mdls,self.PA,self.INC,self.XC,self.YC,self.VSYS,theta = self.PHI_BAR, phi_bar_major = self.PA_bar_mjr, phi_bar_minor = self.PA_bar_mnr)
		else:
			save_vlos_model(self.galaxy, self.vmode,self.vel_map,self.evel_map,self.vlos_2D_mdl,self.kin_2D_mdls,self.PA,self.INC,self.XC,self.YC,self.VSYS, m_hrm = self.m_hrm)


		print("Done!")
		print("############################")


	def __call__(self):
		return self.results()





