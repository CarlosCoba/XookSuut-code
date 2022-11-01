import numpy as np
import matplotlib.pylab as plt
from astropy.io import fits
np.warnings.filterwarnings('ignore')
import os.path
from os import path



c = 3e5

from src.axis import AXIS 
from src.kinematic_centre_vsys import KC
from src.cbar import colorbar as cb
from src.isolated_pixels import filter_isolated_cells
from src.write_table import write
from src.filter_pixels import filter_SN
from src.circular_mode import Circular_model
from src.radial_mode import Radial_model
from src.bisymetric_mode import Bisymmetric_model
from src.harmonic_mode import Harmonic_model
from src.start_messenge import start


from src.save_fits_2D_model import save_vlos_model
from src.save_fits_1D_model_harmonic import save_model_h
from src.plot_models_harmonic import plot_kin_models_h
from src.save_fits_1D_model import save_model
from src.plot_models import plot_kin_models
from src.create_directories import direc_out

valid_strings_for_optional_inputs = ["", "-", ".", ",", "#","%", "&"]


def clean_vel_map(vel_map2D, evel_map, SN, osi, config ):


	vel_map = fits.getdata(vel_map2D)
	dims = np.size(vel_map.shape)
	if dims > 2 : 
		print("velocity-map or error-velmap has more than 2 dimensions, data dimensions = %s !"%dims)
		quit()
 
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


	config_general = config['general']
	smooth = config_general.getboolean("smooth", True)
	if smooth:
		print("Removing isolated pixels")
		# Maybe apply a filter to remove isolates pixels in the velocity map?	
		vel_ha = filter_isolated_cells(vel_ha,struct=np.ones((3,3)))

	return vel_ha, evel_map






def guess_vals(PA,INC,X0,Y0,VSYS,PHI_B ):
	# List of guess values
	guess = [PA,INC,X0,Y0,VSYS,PHI_B]
	return guess







class Run_models:

	def __init__(self, galaxy, vel_map, evel_map, SN, VSYS, PA, INC, X0, Y0, PHI_B, n_it, pixel_scale, vary_PA, vary_INC, vary_XC, vary_YC, vary_VSYS, vary_PHI, delta, rstart, rfinal, ring_space, frac_pixel, inner_interp, bar_min_max, vmode, survey, config, e_ISM, fit_method, prefix, osi):


		self.vmode = vmode
		self.galaxy = galaxy
		self.vel_map, self.evel_map = clean_vel_map(vel_map, evel_map, SN, osi, config)

		input_vel_map = self.vel_map
		input_evel_map = self.evel_map
		[ny,nx] = self.vel_map.shape
		self.e_ISM = e_ISM

		self.PA_bar_mjr,self.PA_bar_mnr,self.PHI_BAR = 0,0,0
		self.survey = survey
		self.m_hrm = 3
		
		if VSYS in osi :
			X0,Y0,VSYS,e_vsys = KC(self.vel_map,X0,Y0,pixel_scale)

		guess0 = guess_vals(PA,INC,X0,Y0,VSYS,PHI_B )
		vary = np.array( [vary_PA,vary_INC,vary_XC,vary_YC,vary_VSYS,vary_PHI] )
		sigma = []
		self.ext = np.dot([nx/2., -nx/2,-ny/2.,ny/2.], pixel_scale)
		self.osi = osi
		self.outdir = direc_out(config)

		if "hrm_" in vmode:
			try:
				self.m_hrm = int(vmode[4:])
				if self.m_hrm == 0 : 
					raise ValueError
			except(ValueError):
				print("XookSuut: provide a proper harmonic number different from zero, for example hrm_2")
				quit()
		# Print starting messenge
		start(galaxy,guess0,vmode,config)
		print("starting Least Squares analysis ..")
		if self.vmode == "circular": 
			circ = Circular_model(galaxy, self.vel_map, self.evel_map, guess0, vary, n_it, rstart, rfinal, ring_space, frac_pixel, inner_interp, delta, pixel_scale,bar_min_max,  config, self.e_ISM, fit_method,self.outdir)
			self.PA,self.INC,self.XC,self.YC,self.VSYS,self.PHI_BAR,self.R,self.Vrot,self.Vrad,self.Vtan,self.vlos_2D_mdl, self.kin_2D_mdls,self.bic_aic,self.errors_fit = circ()

		if self.vmode == "radial":
			rad = Radial_model(galaxy, self.vel_map, self.evel_map, guess0, vary, n_it, rstart, rfinal, ring_space, frac_pixel, inner_interp, delta, pixel_scale,bar_min_max,  config, self.e_ISM, fit_method,self.outdir)
			self.PA,self.INC,self.XC,self.YC,self.VSYS,self.PHI_BAR,self.R,self.Vrot,self.Vrad,self.Vtan,self.vlos_2D_mdl, self.kin_2D_mdls,self.bic_aic,self.errors_fit = rad()

		if self.vmode == "bisymmetric":
			bis = Bisymmetric_model(galaxy, self.vel_map, self.evel_map, guess0, vary, n_it, rstart, rfinal, ring_space, frac_pixel, inner_interp, delta, pixel_scale,bar_min_max,  config, self.e_ISM, fit_method,self.outdir)
			self.PA,self.INC,self.XC,self.YC,self.VSYS,self.PHI_BAR,self.R,self.Vrot,self.Vrad,self.Vtan,self.vlos_2D_mdl, self.kin_2D_mdls,self.PA_bar_mjr,self.PA_bar_mnr,self.bic_aic,self.errors_fit = bis()

		if "hrm" in self.vmode:
			hrm = Harmonic_model(galaxy, self.vel_map, self.evel_map, guess0, vary, n_it, rstart, rfinal, ring_space, frac_pixel, inner_interp, delta, pixel_scale, bar_min_max,  config, self.e_ISM, fit_method, self.m_hrm,self.outdir)
			self.PA,self.INC,self.XC,self.YC,self.VSYS,self.R,self.Ck,self.Sk,self.vlos_2D_mdl, self.kin_2D_mdls,self.bic_aic,self.errors_fit = hrm()

		self.redchi = self.bic_aic[-1] 
		#self.results()



class XS_out(Run_models):

	def results(self):

		e_centroid = self.e_ISM/np.sin(self.INC*np.pi/180)
		if "hrm" not in self.vmode:
			e_Vrot,e_Vrad,e_Vtan, e_PA, e_INC, e_XC, e_YC, e_Vsys, e_theta = self.errors_fit
			e_Vrot, e_Vrad,e_Vtan = np.sqrt(e_Vrot**2 + e_centroid**2 ), np.sqrt(e_Vrad**2 + e_centroid**2 ), np.sqrt(e_Vtan**2 + e_centroid**2 )	
		else:
			e_PA,e_INC,e_XC,e_YC,e_Vsys  = self.errors_fit[-5:]
			e_Ck,e_Sk = [np.sqrt( np.array(self.errors_fit[k])**2 + e_centroid**2) for k in range(self.m_hrm)], [np.sqrt(np.array(self.errors_fit[k])**2 + e_centroid**2) for k in range(self.m_hrm,2*self.m_hrm)]



		#
		## Write output into a table
		#

		if self.vmode == "circular" or self.vmode == "radial" or "hrm" in self.vmode:
			if self.survey not in self.osi :
				kin_params_table = "%sana_kin_%s_model.%s.csv"%(self.outdir,self.vmode,self.survey)
			else:
				kin_params_table = "%sana_kin_%s_model.%s.csv"%(self.outdir,self.vmode,self.galaxy)

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
				kin_params_table = "%sana_kin_%s_model.%s.csv"%(self.outdir,self.vmode,self.survey)
			else:
				kin_params_table = "%sana_kin_%s_model.%s.csv"%(self.outdir,self.vmode,self.galaxy)


			# write header of table
			if path.exists(kin_params_table) == True:
				pass
			else:
				hdr = ["object", "XC", "e_XC_l", "e_XC_u", "YC", "e_YC_l", "e_YC_u", "PA", "e_PA_l", "e_PA_u", "INC", "e_INC_l", "e_INC_u", "VSYS", "e_VSYS_l", "e_VSYS_u", "PHI_BAR", "e_PHI_BAR_l","e_PHI_BAR_u","PA_bar_mjr_sky","PA_bar_mnr_sky","redchi" ]
				write(hdr,kin_params_table,column = False)

			kin_params = [self.galaxy,self.XC,e_XC,e_XC,self.YC,e_YC,e_YC,self.PA,e_PA,e_PA,self.INC,e_INC,e_INC,self.VSYS,e_Vsys,e_Vsys,self.PHI_BAR,e_theta,e_theta,self.PA_bar_mjr,self.PA_bar_mnr,self.redchi]

			write(kin_params,kin_params_table,column = False)



		if "hrm" not in self.vmode:

			plot_kin_models(self.galaxy, self.vmode,self.vel_map,self.R,self.Vrot,e_Vrot,self.Vrad,e_Vrad,self.Vtan,e_Vtan, self.VSYS, self.vlos_2D_mdl, self.ext,out=self.outdir)


			s = save_model(self.galaxy, self.vmode,self.R,self.Vrot,self.Vrad,self.Vtan,self.PA,self.INC,self.XC,self.YC,self.VSYS,self.PHI_BAR,self.PA_bar_mjr,self.PA_bar_mnr,self.errors_fit,self.bic_aic, self.e_ISM,out=self.outdir)



		if "hrm" in self.vmode:

			plot_kin_models_h(self.galaxy, self.vmode,self.vel_map,self.R,self.Ck,self.Sk,e_Ck,e_Sk,self.VSYS,self.INC,self.vlos_2D_mdl, self.ext,self.m_hrm,survey = self.survey,out=self.outdir)


			s = save_model_h(self.galaxy, self.vmode,self.R,self.Ck,self.Sk,e_Ck,e_Sk,self.PA,self.INC,self.XC,self.YC,self.VSYS,self.m_hrm,self.errors_fit,self.bic_aic,self.e_ISM,out=self.outdir)
					


		if self.vmode == "bisymmetric":
			save_vlos_model(self.galaxy, self.vmode,self.vel_map,self.evel_map,self.vlos_2D_mdl,self.kin_2D_mdls,self.PA,self.INC,self.XC,self.YC,self.VSYS,theta = self.PHI_BAR, phi_bar_major = self.PA_bar_mjr, phi_bar_minor = self.PA_bar_mnr,out=self.outdir)
		else:
			save_vlos_model(self.galaxy, self.vmode,self.vel_map,self.evel_map,self.vlos_2D_mdl,self.kin_2D_mdls,self.PA,self.INC,self.XC,self.YC,self.VSYS, m_hrm = self.m_hrm,out=self.outdir)


		print("Done!")
		print("############################")


	def __call__(self):
		run = self.results()
		#return self.results()





