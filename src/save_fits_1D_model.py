import numpy as np
from astropy.io import fits
from src.phi_bar_sky import error_pa_bar_sky


def save_model(galaxy,vmode,R,Vrot,Vrad,Vtan,PA,INC,XC,YC,VSYS,THETA,PA_BAR_MAJOR,PA_BAR_MINOR,errors_fit,bic_aic, e_ISM, out):
	#m = len(MODELS)
	n = len(Vrot)
	e_PA,e_INC,e_XC,e_YC,e_Vsys,e_theta  = errors_fit[1]
	e_Vrot,e_Vrad,e_Vtan  = errors_fit[0]

	N_free, N_nvarys, N_data, bic, aic, redchi = bic_aic


        
	if vmode == "circular":
			#Vrot,e_Vrot = MODELS[0],eMODELS[0]
			data = np.zeros((3,n))
			data[0][:] = R
			data[1][:] = Vrot
			data[2][:] = e_Vrot

	if vmode == "radial":
			#Vrot,e_Vrot = MODELS[0],eMODELS[0]
			#Vrad,e_Vrad = MODELS[1],eMODELS[1]
			data = np.zeros((5,n))
			data[0][:] = R
			data[1][:] = Vrot
			data[2][:] = Vrad
			data[3][:] = e_Vrot
			data[4][:] = e_Vrad



	if vmode == "bisymmetric":
			#Vrot,e_Vrot = MODELS[0],eMODELS[0]
			#Vrad,e_Vrad = MODELS[1],eMODELS[1]
			#Vtan,e_Vtan = MODELS[2],eMODELS[2]
			data = np.zeros((7,n))
			data[0][:] = R
			data[1][:] = Vrot
			data[2][:] = Vrad
			data[3][:] = Vtan
			data[4][:] = e_Vrot
			data[5][:] = e_Vrad
			data[6][:] = e_Vtan


	if True:

		hdu = fits.PrimaryHDU(data)

		if vmode == "circular":
			hdu.header['NAME0'] = 'deprojected distance (arcsec)'
			hdu.header['NAME1'] = 'circular velocity (km/s)'
			hdu.header['NAME2'] = 'error circular velocity (km/s)'
		if vmode == "radial":
			hdu.header['NAME0'] = 'deprojected distance (arcsec)'
			hdu.header['NAME1'] = 'circular velocity (km/s)'
			hdu.header['NAME2'] = 'radial velocity (km/s)'
			hdu.header['NAME3'] = 'error circular velocity (km/s)'
			hdu.header['NAME4'] = 'error radial velocity (km/s)'
		if vmode == "bisymmetric":
			hdu.header['NAME0'] = 'deprojected distance (arcsec)'
			hdu.header['NAME1'] = 'circular velocity (km/s)'
			hdu.header['NAME2'] = 'radial velocity (km/s)'
			hdu.header['NAME3'] = 'tangencial velocity (km/s)'
			hdu.header['NAME4'] = 'error circular velocity (km/s)'
			hdu.header['NAME5'] = 'error radial velocity (km/s)'
			hdu.header['NAME6'] = 'error tangencial velocity (km/s)'

		hdu.header['redchisq'] = redchi
		hdu.header['Nfree'] = N_free
		hdu.header['Nvarys'] = N_nvarys
		hdu.header['Ndata'] = N_data
		hdu.header['BIC'] = bic
		hdu.header['AIC'] = aic
		hdu.header['PA'] = PA
		hdu.header['e_PA'] = e_PA
		hdu.header['INC'] = INC
		hdu.header['e_INC'] = e_INC
		hdu.header['VSYS'] = VSYS
		hdu.header['e_VSYS'] = e_Vsys
		hdu.header['XC'] = XC
		hdu.header['e_XC'] = e_XC
		hdu.header['YC'] = YC
		hdu.header['e_YC'] = e_YC

		if vmode == "bisymmetric":
			hdu.header['HIERARCH PHI_BAR'] = THETA
			hdu.header['HIERARCH e_PHI_BAR'] = e_theta
			hdu.header['HIERARCH PA_BAR_MAJOR'] = PA_BAR_MAJOR
			hdu.header['HIERARCH e_PA_BAR_MAJOR'] = error_pa_bar_sky(PA,INC,THETA,e_PA,e_INC,e_theta)
			hdu.header['HIERARCH PA_BAR_MINOR'] = PA_BAR_MINOR
			hdu.header['HIERARCH e_PA_BAR_MINOR'] = error_pa_bar_sky(PA,INC,THETA-90,e_PA,e_INC,e_theta)
		
		hdu.writeto("%smodels/%s.%s.1D_model.fits.gz"%(out,galaxy,vmode),overwrite=True)
