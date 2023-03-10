import numpy as np
from astropy.io import fits



def marginal_vals(galaxy,vmode,chain_res,n_circ,n_noncirc,out,nlabels,mcmc_outs):
	[acc_frac, steps, thin, burnin, Nwalkers, PropDist, ndim, act] = mcmc_outs
	data = chain_res
	[ny,nx]=data.shape
	hdu = fits.PrimaryHDU(data)

	hdu.header['PROPDIST'] = "Gaussian" if PropDist == "G" else "Cauchy"
	hdu.header['STEPS'] = steps	
	hdu.header['ACC_FRAC'] = round(acc_frac,2)	
	hdu.header['THIN'] = thin	
	hdu.header['NWALKERS'] = Nwalkers	
	hdu.header['NDIM'] = ndim	
	hdu.header['ACT'] = int(act)	
	hdu.header['NCIRC'] = n_circ 
	hdu.header['NNONCIRC'] = n_noncirc
	hdu.header['NCONSTNT'] = nlabels
	hdu.header['COLS'] = "[median,-1s,+1s,1s,-2s,+2s,2s]"



	if vmode == "circular":		
		hdu.header['NAME0'] = "VROT_k" 
	if vmode == "radial":
		hdu.header['NAME0'] = "VROT_k"
		hdu.header['NAME1'] = "VRAD_k"
	if vmode == "bisymmetric":
		hdu.header['NAME0'] = "VROT_k"
		hdu.header['NAME1'] = "VRAD_k"  		
		hdu.header['NAME2'] = "VTAN_k"
	if "hrm" in vmode:
		hdu.header['NAME0'] = "S_k"
		hdu.header['NAME1'] = "C_k"  		

	const0 = ["PA", "EPS", "X0", "Y0", "VSYS"]
	const1 = ["PA", "EPS", "X0", "Y0", "VSYS","PHI_BAR"]
	if "hrm" in vmode: const0[-1] = "C0"

	if vmode != "bisymmetric":
		for i,j in enumerate(const0):
			if nlabels == len(const0) :
				hdu.header[j] = data[ny-len(const0)+i][0]
			if nlabels == len(const0)+1 :
				hdu.header[j] = data[ny-len(const0)-1+i][0]
		if nlabels == 6 :
			x = "gamma" if PropDist=="C" else "lnsigma2"
			hdu.header[x] = data[-1][0]

	if vmode == "bisymmetric": 
		for i,j in enumerate(const1):
			if nlabels == len(const1) :
				hdu.header[j] = data[ny-len(const1)+i][0]
			if nlabels == len(const1)+1 :
				hdu.header[j] = data[ny-len(const1)-1+i][0]
		if nlabels == 7 :
			x = "gamma" if PropDist=="C" else "lnsigma2"
			hdu.header[x] = data[-1][0]		

	hdu.writeto("%smodels/%s.%s.marginal_dist.fits.gz"%(out,galaxy,vmode),overwrite=True)






	
