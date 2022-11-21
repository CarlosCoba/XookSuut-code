import numpy as np
import matplotlib.pylab as plt
import matplotlib
from matplotlib.ticker import ScalarFormatter
import corner
import itertools
from scipy.stats import multivariate_normal


from src.weights_interp import weigths_w
from src.kin_components import CIRC_MODEL
from src.kin_components import RADIAL_MODEL
from src.kin_components import BISYM_MODEL
from src.fig_params import fig_ambient
from src.axes_params import axes_ambient as axs
from src.create_2D_vlos_model import best_2d_model
from src.create_2D_kin_models import bidi_models

#params =   {'text.usetex' : True }
#plt.rcParams.update(params) 






def chain_res_mcmc(galaxy, vmode, theta, chain, mcmc_config, accept_rate, shape, rings_pos, ring_space, pixel_scale, inner_interp, phi_b = 0, outdir = False, config_psf=None ):

	chain_params = mcmc_config
	steps,thin,burnin,nwalkers,PropDist,save_plots = chain_params

	n_circ = len(theta[0])
	n_noncirc = len(theta[1])
	m = n_circ + 2*n_noncirc

	[nwalkers, steps,ndim]=chain.shape
	# These are the number of parameters from the kinematic models
	n_params_mdl = n_circ + 2*n_noncirc + 6
	
	labels_const = ["$\mathrm{\phi^{\prime}}~(^\circ)$", "$i~(^\circ)$", "$\mathrm{x_0}~(pix)$", "$\mathrm{y_0}~(pix)$", "$\mathrm{V_{sys}~(km/s)}$","$\\phi_{\mathrm{bar}}~(^\circ)$"]

	if PropDist == "G":
		if ndim == n_params_mdl:
			pa0,inc0,x0,y0,vsys0,phi0 = theta[-6:]
			truths_const = [pa0, inc0, x0, y0, vsys0,phi0]
		if ndim == n_params_mdl+1:
			pa0,inc0,x0,y0,vsys0,phi0,lnsigma2 = theta[-7:]
			truths_const = [pa0, inc0, x0, y0, vsys0, phi0,lnsigma2]
			labels_const.append("$\mathrm{\log~\sigma}$")
	if PropDist == "C":
		pa0,inc0,x0,y0,vsys0,phi0,lnsigma2 = theta[-7:]
		truths_const = [pa0, inc0, x0, y0, vsys0, phi0,lnsigma2]
		labels_const.append("$\mathrm{\gamma~(km/s)}$")


	theta_flat = np.hstack(theta.flatten())
	time = np.arange(steps)

	vel_labels = []
	if save_plots :
		fig, axes = fig_ambient(fig_size = (3.5,5), ncol = 1, nrow = n_circ, left = 0.2, right = 0.95, top = 0.99, hspace = 0, bottom = 0.1, wspace = 0.4 )
		labels = ["$\mathrm{ Vt_{%s} } $"%k for k in range(n_circ)]
		vel_labels = vel_labels +  ["Vt_%s"%k for k in range(n_circ)]
		for i in range(n_circ):
			ax = axes[i]
			#if nwalkers != 0 :
			#	for k in range(nwalkers):
			#		ax.plot(time,chain[k, :, i], alpha=0.5, lw = 0.1)
			ax.plot(chain[:,:, i], "k", alpha=0.5, lw = 0.1)
			ax.yaxis.set_label_coords(-0.15, 0.5)
			axs(ax, fontsize_yticklabel = 8, rotation = "horizontal")
			ax.set_ylabel(labels[i], fontsize = 8)
			if i != (n_circ-1) :ax.xaxis.set_ticklabels([])
		axes[-1].set_xlabel("$\mathrm{steps}$", fontsize = 10);
		plt.savefig("%sfigures/chain_progress.circ.%s_model.%s.png"%(outdir,vmode,galaxy), dpi=300)
		plt.clf()

		fig, axes = fig_ambient(fig_size = (3.5,5), ncol = 1, nrow = n_noncirc, left = 0.2, right = 0.95, top = 0.99, hspace = 0, bottom = 0.1, wspace = 0.4 )
		labels = ["$\mathrm{ V_{2r,%s} } $"%k for k in range(n_noncirc)]
		vel_labels = vel_labels +  ["V2r_%s"%k for k in range(n_noncirc)]
		for i in range(n_circ,n_circ+n_noncirc):
			ax = axes[i-n_circ]
			#if nwalkers != 0 :
			#	for k in range(nwalkers):
			#		ax.plot(time,chain[k, :, i], alpha=0.5, lw = 0.1)

			ax.plot(chain[:,:, i], "k", alpha=0.5, lw = 0.1)
			ax.yaxis.set_label_coords(-0.15, 0.5)
			axs(ax, fontsize_yticklabel = 8, rotation = "horizontal")
			ax.set_ylabel(labels[i-n_circ], fontsize = 8)
			if i != (n_circ+n_noncirc-1) :ax.xaxis.set_ticklabels([])
		axes[-1].set_xlabel("$\mathrm{steps}$", fontsize = 10);
		plt.savefig("%sfigures/chain_progress.rad.%s_model.%s.png"%(outdir,vmode,galaxy), dpi=300)
		plt.clf()

		fig, axes = fig_ambient(fig_size = (3.5,5), ncol = 1, nrow = n_noncirc, left = 0.2, right = 0.95, top = 0.99, hspace = 0, bottom = 0.1, wspace = 0.4 )
		labels = ["$\mathrm{ V_{2t,%s} } $"%k for k in range(n_noncirc)]
		vel_labels = vel_labels +  ["V2t_%s"%k for k in range(n_noncirc)]
		for i in range(n_circ+n_noncirc,m):
			ax = axes[i-(n_circ+n_noncirc)]
			#if nwalkers != 0 :
			#	for k in range(nwalkers):
			#		ax.plot(time,chain[k, :, i], alpha=0.5, lw = 0.1)
			ax.plot(chain[:,:, i], "k", alpha=0.5, lw = 0.1)
			ax.yaxis.set_label_coords(-0.15, 0.5)
			axs(ax, fontsize_yticklabel = 8, rotation = "horizontal")
			ax.set_ylabel(labels[i-(n_circ+n_noncirc)], fontsize = 8)
			if i != (m-1) :ax.xaxis.set_ticklabels([])
		axes[-1].set_xlabel("$\mathrm{steps}$", fontsize = 10);
		plt.savefig("%sfigures/chain_progress.tan.%s_model.%s.png"%(outdir,vmode,galaxy), dpi=300)
		plt.clf()

	# Apply thin to the chain
	good_samples = chain[::thin, :, :].reshape((-1, ndim))

	# Marginilize parameters
	chain_res = np.empty((ndim,4))
	for i in range(ndim):
		#
		# Show -+1 sigma
		#
		mcmc = np.percentile(good_samples[:, i], [15.865, 50, 84.135])
		q = np.diff(mcmc)
		median, lower,upper, std = mcmc[1], q[0], q[1], np.std(good_samples[:, i])
		chain_res[i,:] = [median,lower,upper,std]

	medians, errors, std = chain_res[:,0],chain_res[:,1:3],chain_res[:,-1]
	# +1 sigma
	sigma_l,sigma_u = errors[:,0],errors[:,1]

	if PropDist == "G":
		if ndim == n_params_mdl:
			pa,inc,x0,y0,Vsys,phi_b = medians[-6:]
			epa,einc,ex0,ey0,eVsys,ephi_b = std[-6:]
			V_k = medians[:-6]
			eV_k = std[:-6]
			theta_mdls = medians

		if ndim == n_params_mdl+1:
			pa,inc,x0,y0,Vsys,phi_b,lnsigma = medians[-7:]
			epa,einc,ex0,ey0,eVsys,ephi_b,elnsigma = std[-7:]
			V_k = medians[:-7]
			eV_k = std[:-7]
			theta_mdls = medians[:-1]

	if PropDist == "C":
		pa,inc,x0,y0,Vsys,phi_b,lnsigma = medians[-7:]
		epa,einc,ex0,ey0,eVsys,ephi_b,elnsigma = std[-7:]
		V_k = medians[:-7]
		eV_k = std[:-7]
		theta_mdls = medians[:-1]

	# TO DO ...
	#if save_plots:
	#	from mcmc_out import save_mcmc_outs
	#	save_mcmc_outs(galaxy,vmode,chain_res,n_circ,n_noncirc,rings_pos,vel_labels)






	#
	# Corner plot
	#

	nlabels = len(labels_const)
	#flat_samples = good_samples[:,m:m+nlabels]
	flat_samples = good_samples[:,m:m+nlabels]


	CORNER_KWARGS = dict(
	smooth=0.9,
	label_kwargs=dict(fontsize=10),
	plot_datapoints=False,
	labelpad = 0.3,
	max_n_ticks=3,
	contour_kwargs = dict(linewidths = 0.5))


	fig = corner.corner(flat_samples, labels=labels_const, truths=truths_const, truth_color = "#faa022", **CORNER_KWARGS);
	# Extract the axes
	axes_corner = np.array(fig.axes).reshape((nlabels, nlabels))

	di = np.diag_indices(nlabels)
	for i,j in zip(di[0],di[1]):
		ax = axes_corner[i, j]
		ax.set_title('{ $%s_{-%s}^{+%s}$ }'%(round(medians[m+i],2), abs(round(errors[m+i][0],2)), abs(round(errors[m+i][1],2))), fontsize = 10)


	for i in range(nlabels):
		for j in range(nlabels):

			if j == 0 :
				ax = axes_corner[i, j]

				if i ==  nlabels -1:
					axs(ax, fontsize_ticklabels = 10)
				else:
					axs(ax, fontsize_ticklabels = 10, remove_xticks = True)
				ax.tick_params(rotation=45)

			if i == nlabels -1 and j != 0:
				ax = axes_corner[i, j]
				axs(ax, fontsize_ticklabels = 10, remove_yticks = True)
				ax.tick_params(rotation=45)

			if j != 0 and i != nlabels -1 :
				ax = axes_corner[i, j]
				axs(ax, fontsize_ticklabels = 10, remove_yticks = True, remove_xticks = True)




	fig = matplotlib.pyplot.gcf()
	fig.set_size_inches(6, 6)
	plt.gcf().subplots_adjust(left = 0.13, bottom=0.13, top = 0.95)
	fig.savefig("%sfigures/corner.%s_model.%s.png"%(outdir,vmode,galaxy),dpi = 300)
	plt.clf()

	#import zeus
	#fig, axes = zeus.cornerplot(flat_samples, labels=labels_const);
	#fig.savefig("%sfigures/corner.zeus.%s_model.%s.png"%(outdir,vmode,galaxy),dpi = 300)
	#plt.clf()




	Vrot_flat,Vrad_flat, Vtan_flat= V_k[:n_circ],V_k[n_circ:n_circ+n_noncirc],V_k[n_circ+n_noncirc:n_circ+2*n_noncirc]
	eVrot_flat,eVrad_flat, eVtan_flat= eV_k[:n_circ],eV_k[n_circ:n_circ+n_noncirc],eV_k[n_circ+n_noncirc:n_circ+2*n_noncirc]

	Vrot = np.asarray(Vrot_flat)
	Vrad = np.asarray(Vrot)*0;Vrad[:n_noncirc]= Vrad_flat
	Vtan = np.asarray(Vrot)*0;Vtan[:n_noncirc]= Vtan_flat

	eVrot = np.asarray(eVrot_flat)
	eVrad = np.asarray(eVrot)*0;eVrad[:n_noncirc]= eVrad_flat
	eVtan = np.asarray(eVrot)*0;eVtan[:n_noncirc]= eVtan_flat

	V_k = [Vrot, Vrad, Vtan] 
	eV_k = [eVrot, eVrad, eVtan]
	e_constant_parms = [epa,einc,ex0,ey0,eVsys,ephi_b]

	errors = [[],[]]
	errors[0],errors[1] = eV_k,e_constant_parms

	from src.create_2D_vlos_model_mcmc import KinModel
	kinmodel = KinModel( 0, 0, theta_mdls, vmode, rings_pos, ring_space, pixel_scale, inner_interp, PropDist, 0, n_circ, n_noncirc, shape, config_psf,only_model = True)
	vlos_2D_model = kinmodel.interp_model(theta_mdls)

	interp_mdl = bidi_models(vmode, shape, V_k, pa, inc, x0, y0, Vsys, rings_pos, ring_space, pixel_scale, inner_interp) 
	kin_2D_models = interp_mdl.interp()

	return vlos_2D_model, kin_2D_models, V_k, pa, inc, x0, y0, Vsys, phi_b, errors

