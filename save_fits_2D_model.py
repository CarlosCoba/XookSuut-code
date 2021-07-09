import numpy as np
from astropy.io import fits



def save_vlos_model(galaxy,vmode,vel_map,evel_map,vlos_2D_model, kin_2D_models,PA,INC,XC,YC,VSYS,save = 1, theta = False, phi_bar_major = False, phi_bar_minor = False):

	Vcirc_2D, Vrad_2D, Vtan_2D, R_n =  kin_2D_models

	if save == 1:

		# The best 2D model
		data = vlos_2D_model
		hdu = fits.PrimaryHDU(data)
		#hdu.data = vlos_2D_model

	
		hdu.header['NAME0'] = 'Two-dimensional vlos model'
    
		hdu.header['PA'] = PA
		hdu.header['INC'] = INC
		hdu.header['VSYS'] = VSYS
		hdu.header['XC'] = XC
		hdu.header['YC'] = YC
		
		hdu.writeto("./models/%s.%s.2D_vlos_model.fits"%(galaxy,vmode),overwrite=True)


		
		# Now the residual map
		data = vel_map- vlos_2D_model
		hdu = fits.PrimaryHDU(data)
		#hdu.data = vel_map- vlos_2D_model

	
		hdu.header['NAME0'] = 'residual map, data-model'
    
		hdu.header['PA'] = PA
		hdu.header['INC'] = INC
		hdu.header['VSYS'] = VSYS
		hdu.header['XC'] = XC
		hdu.header['YC'] = YC
		
		hdu.writeto("./models/%s.%s.residual.fits"%(galaxy,vmode),overwrite=True)


		
		# Now the chisquare map

		sigma = evel_map
		chisq = ( vel_map- vlos_2D_model )/sigma
		chisq_2 = chisq**2

		data = chisq_2
		hdu = fits.PrimaryHDU(data)
		#hdu.data = chisq_2

	
		hdu.header['NAME0'] = 'Chisquare map, (data-model)**2/sigma**2'
    
		hdu.header['PA'] = PA
		hdu.header['INC'] = INC
		hdu.header['VSYS'] = VSYS
		hdu.header['XC'] = XC
		hdu.header['YC'] = YC
		
		hdu.writeto("./models/%s.%s.chisq.fits"%(galaxy,vmode),overwrite=True)




		#
		# Save deprojected radius.
		#

		data = R_n
		hdu = fits.PrimaryHDU(data)
		#hdu.data = R_n

	
		hdu.header['NAME0'] = 'Two-dimensional deprojected radius'
		hdu.header['PA'] = PA
		hdu.header['INC'] = INC
		hdu.header['VSYS'] = VSYS
		hdu.header['XC'] = XC
		hdu.header['YC'] = YC
		
		hdu.writeto("./models/%s.%s.2D_R.fits"%(galaxy,vmode),overwrite=True)







		#
		# Save 2D kinematic models. Circular model is always saved.
		#
		
		# Circular velocity 2D model
		data = Vcirc_2D
		hdu = fits.PrimaryHDU(data)
		#hdu.data = Vcirc_2D

	
		hdu.header['NAME0'] = 'Two-dimensional circular model'
		hdu.header['PA'] = PA
		hdu.header['INC'] = INC
		hdu.header['VSYS'] = VSYS
		hdu.header['XC'] = XC
		hdu.header['YC'] = YC
		
		hdu.writeto("./models/%s.%s.2D_circ_model.fits"%(galaxy,vmode),overwrite=True)

		if vmode == "radial":
			# Radial velocity 2D model
			data = Vrad_2D
			hdu = fits.PrimaryHDU(data)
			#hdu.data = Vrad_2D

		
			hdu.header['NAME0'] = 'Two-dimensional radial model'
			hdu.header['PA'] = PA
			hdu.header['INC'] = INC
			hdu.header['VSYS'] = VSYS
			hdu.header['XC'] = XC
			hdu.header['YC'] = YC
			
			hdu.writeto("./models/%s.%s.2D_rad_model.fits"%(galaxy,vmode),overwrite=True)


		if vmode == "bisymmetric":
			# Radial velocity 2D model
			data = Vrad_2D
			hdu = fits.PrimaryHDU(data)
			#hdu.data = Vrad_2D

		
			hdu.header['NAME0'] = 'Two-dimensional V2r model'
			hdu.header['PA'] = PA
			hdu.header['INC'] = INC
			hdu.header['VSYS'] = VSYS
			hdu.header['XC'] = XC
			hdu.header['YC'] = YC
			hdu.header['THETA_BAR'] = theta
			hdu.header['PA_SKY_BAR_MAJOR'] = phi_bar_major
			hdu.header['PA_SKY_BAR_MINOR'] = phi_bar_minor

			
			hdu.writeto("./models/%s.%s.2D_rad_model.fits"%(galaxy,vmode),overwrite=True)

			# Tangential velocity 2D model
			data = Vtan_2D
			hdu = fits.PrimaryHDU(data)
			#hdu.data = Vtan_2D

		
			hdu.header['NAME0'] = 'Two-dimensional V2t model'
			hdu.header['PA'] = PA
			hdu.header['INC'] = INC
			hdu.header['VSYS'] = VSYS
			hdu.header['XC'] = XC
			hdu.header['YC'] = YC
			hdu.header['THETA_BAR'] = theta
			hdu.header['PA_SKY_BAR_MAJOR'] = phi_bar_major
			hdu.header['PA_SKY_BAR_MINOR'] = phi_bar_minor
			
			hdu.writeto("./models/%s.%s.2D_tan_model.fits"%(galaxy,vmode),overwrite=True)


	else: pass

