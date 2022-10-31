from astropy.io import fits

def save_mcmc_outs(galaxy,vmode,data, n_circ, n_noncirc, rings_pos, labels ):

	[ny,nx] = data.shape
	ndim = ny
	nlabels = len(labels)


	hdu = fits.PrimaryHDU(data)

	hdu.header['N_CIRC'] = n_circ
	hdu.header['N_NONCIRC'] = n_noncirc
	hdu.header['Rstart'] = rings_pos[0]
	hdu.header['Rstep'] = rings_pos[1]-rings_pos[0]

	for i in range(nlabels):
		hdu.header['NAME%s'%i] = "%s: value,1sigma_low,1sigma_up"%labels[i]


	if "hrm" in vmode:

		#for i in range(nlabels):
		#	hdu.header['NAME%s'%i] = "%s, value,1sigma_low,1sigma_up"%labels[i]

		hdu.header['NAME%s'%(nlabels+0)] = "PA: value,e_lower,e_upper"
		hdu.header['NAME%s'%(nlabels+1)] = "INC: value,1sigma_low,1sigma_up"
		hdu.header['NAME%s'%(nlabels+2)] = "XC: value,1sigma_low,1sigma_up"
		hdu.header['NAME%s'%(nlabels+3)] = "YC: value,1sigma_low,1sigma_up"
		hdu.header['NAME%s'%(nlabels+4)] = "c0: value,1sigma_low,1sigma_up"


	else:

		#for i in range(nlabels):
		#	hdu.header['NAME%s'%i] = "%s, value,1sigma_low,1sigma_up"%labels[i]



		hdu.header['NAME%s'%(nlabels+0)] = "PA: value,1sigma_low,1sigma_up"
		hdu.header['NAME%s'%(nlabels+1)] = "INC: value,1sigma_low,1sigma_up"
		hdu.header['NAME%s'%(nlabels+2)] = "XC: value,1sigma_low,1sigma_up"
		hdu.header['NAME%s'%(nlabels+3)] = "YC: value,1sigma_low,1sigma_up"
		hdu.header['NAME%s'%(nlabels+4)] = "VSYS: value,1sigma_low,1sigma_up"

		if vmode == "bisymmetric": hdu.header['NAME%s'%(nlabels+5)] = "PHI_BAR: value,1sigma_low,1sigma_up"

	hdu.writeto("./models/%s.%s.mcmc_errors.fits.gz"%(galaxy,vmode),overwrite=True)
