import numpy as np
from astropy.io import fits



def save_model(galaxy,vmode,R,Ck,Sk,e_Ck,e_Sk,PA,INC,XC,YC,VSYS,m_hrm,errors_fit,bic_aic):

	N_free, N_nvarys, N_data, bic, aic, redchi = bic_aic
	nx, ny = len(R), 4*m_hrm + 1 
	n = (Ck)

	data = np.zeros((ny,nx))
	data[0][:] = R
	for k in range(m_hrm):
		data[k+1][:] = Ck[k]
		data[m_hrm+1+k][:] = Sk[k]
		data[2*m_hrm+1+k][:] = e_Ck[k]
		data[3*m_hrm+1+k][:] = e_Sk[k]



	e_PA,e_INC,e_XC,e_YC,e_Vsys  = errors_fit[-5:]


	hdu = fits.PrimaryHDU(data)

	hdu.header['NAME0'] = 'deprojected distance (arcsec)'
	for k in range(1,m_hrm+1):
			hdu.header['NAME%s'%k] = 'C%s deprojected velocity (km/s)'%k
	for k in range(1,m_hrm+1):
			hdu.header['NAME%s'%(k+m_hrm)] = 'S%s deprojected velocity (km/s)'%k
	for k in range(1,m_hrm+1):
			hdu.header['NAME%s'%(k+2*m_hrm)] = 'error C%s (km/s)'%k
	for k in range(1,m_hrm+1):
			hdu.header['NAME%s'%(k+3*m_hrm)] = 'error S%s (km/s)'%k


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
	hdu.header['XC'] = XC
	hdu.header['e_XC'] = e_XC
	hdu.header['YC'] = YC
	hdu.header['e_YC'] = e_YC
	hdu.header['C0'] = VSYS
	hdu.header['e_C0'] = e_Vsys


	hdu.writeto("./models/%s.%s.1D_model.fits.gz"%(galaxy,vmode),overwrite=True)




