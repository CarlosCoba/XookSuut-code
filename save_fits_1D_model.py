import numpy as np
from astropy.io import fits



def save_model(galaxy,vmode,R,Vrot,e_Vrot,Vrad,e_Vrad,Vtan,e_Vtan,PA,INC,XC,YC,VSYS,THETA,PA_BAR_MAJOR,PA_BAR_MINOR,save = 1):
	#m = len(MODELS)
	n = len(Vrot)


	eVrot = e_Vrot
	eVrad = e_Vrad
	eVtan = e_Vtan


	if vmode == "circular":
			#Vrot,eVrot = MODELS[0],eMODELS[0]
			data = np.zeros((3,n))
			data[0][:] = R
			data[1][:] = Vrot
			data[2][:] = eVrot

	if vmode == "radial":
			#Vrot,eVrot = MODELS[0],eMODELS[0]
			#Vrad,eVrad = MODELS[1],eMODELS[1]
			data = np.zeros((5,n))
			data[0][:] = R
			data[1][:] = Vrot
			data[2][:] = Vrad
			data[3][:] = eVrot
			data[4][:] = eVrad



	if vmode == "bisymmetric":
			#Vrot,eVrot = MODELS[0],eMODELS[0]
			#Vrad,eVrad = MODELS[1],eMODELS[1]
			#Vtan,eVtan = MODELS[2],eMODELS[2]
			data = np.zeros((7,n))
			data[0][:] = R
			data[1][:] = Vrot
			data[2][:] = Vrad
			data[3][:] = Vtan
			data[4][:] = eVrot
			data[5][:] = eVrad
			data[6][:] = eVtan


	if save == 1:
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


		hdu.header['PA'] = PA
		hdu.header['INC'] = INC
		hdu.header['VSYS'] = VSYS
		hdu.header['XC'] = XC
		hdu.header['YC'] = YC

		if vmode == "bisymmetric":
			hdu.header['HIERARCH THETA-BAR'] = THETA
			hdu.header['HIERARCH PA-BAR-MAJOR'] = PA_BAR_MAJOR
			hdu.header['HIERARCH PA-BAR-MINOR'] = PA_BAR_MINOR
		
		hdu.writeto("./models/%s.%s.1D_model.fits"%(galaxy,vmode),overwrite=True)

