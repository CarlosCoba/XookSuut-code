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
from src.create_2D_kin_models_mcmc import bidi_models
from src.pixel_params import eps_2_inc

#params =   {'text.usetex' : True }
#plt.rcParams.update(params) 






def chain_res_mcmc(galaxy, vmode, theta, mcmc_outs, shape, rings_pos, ring_space, pixel_scale, inner_interp, phi_b = 0, outdir = False, config_psf=None, plot_chain = False  ):

	[chain, acc_frac, steps, thin, burnin, Nwalkers, PropDist, ndim, act, theta] = mcmc_outs
	n_circ = len(theta[0])
	n_noncirc = len(theta[1])

	[nwalkers, steps,ndim]=chain.shape
	# These are the number of parameters from the kinematic models
	n_params_mdl = n_circ + n_noncirc + 5
	m = n_circ + n_noncirc

	# Labels of constant parameters
	labels_const = ["$\mathrm{\phi^{\prime}}~(^\circ)$", "$i~(^\circ)$", "$\mathrm{x_0}~(pix)$", "$\mathrm{y_0}~(pix)$", "$\mathrm{V_{sys}~(km/s)}$"]
	theta_flat = np.hstack(theta)
	pa0,eps0,x0,y0,vsys0 = theta_flat[m:m+5]
	inc0 = eps_2_inc(eps0)*180/np.pi
	truths_const = [pa0, inc0, x0, y0, vsys0]

	if PropDist == "G":
		if ndim == n_params_mdl+1:
			lnsigma2 = theta_flat[-1]
			truths_const.append(lnsigma2)
			labels_const.append("$\mathrm{\ln~\sigma_{int}^2}$")
	if PropDist == "C":
		lnsigma2 = theta_flat[-1]
		truths_const.append(lnsigma2)
		labels_const.append("$\mathrm{\gamma~(km/s)}$")

	nlabels = len(labels_const)
	#time = np.arange(steps)

	vel_labels = [] 
	if plot_chain :
		fig, axes = fig_ambient(fig_size = (3.5,5), ncol = 1, nrow = n_circ, left = 0.2, right = 0.95, top = 0.99, hspace = 0, bottom = 0.1, wspace = 0.4 )
		labels = ["$\mathrm{ V_{t}{%s} } $"%k for k in range(n_circ)]
		vel_labels = 	vel_labels + ["Vt_%s"%k for k in range(n_circ)]
		for i in range(n_circ):
			ax = axes[i]
			ax.plot(chain[:,:, i], alpha=0.5, lw = 0.1)
			ax.yaxis.set_label_coords(-0.15, 0.5)
			axs(ax, fontsize_yticklabel = 8, rotation = "horizontal")
			ax.set_ylabel(labels[i], fontsize = 8)
			if i != (n_circ-1) :ax.xaxis.set_ticklabels([])
		axes[-1].set_xlabel("$\mathrm{steps}$", fontsize = 10);
		plt.savefig("%sfigures/chain_progress.circ.%s_model.%s.png"%(outdir,vmode,galaxy), dpi=300)
		plt.clf()

		fig, axes = fig_ambient(fig_size = (3.5,5), ncol = 1, nrow = n_noncirc, left = 0.2, right = 0.95, top = 0.99, hspace = 0, bottom = 0.1, wspace = 0.4 )
		labels = ["$\mathrm{ V_{r,%s} } $"%k for k in range(n_noncirc)]
		vel_labels = 	vel_labels + ["Vr_%s"%k for k in range(n_noncirc)]
		for i in range(n_circ,n_circ+n_noncirc):
			ax = axes[i-n_circ]
			ax.plot(chain[:,:, i], alpha=0.5, lw = 0.1)
			ax.yaxis.set_label_coords(-0.15, 0.5)
			axs(ax, fontsize_yticklabel = 8, rotation = "horizontal")
			ax.set_ylabel(labels[i-n_circ], fontsize = 8)
			if i != (n_circ+n_noncirc-1) :ax.xaxis.set_ticklabels([])
		axes[-1].set_xlabel("$\mathrm{steps}$", fontsize = 10);
		plt.savefig("%sfigures/chain_progress.rad.%s_model.%s.png"%(outdir,vmode,galaxy), dpi=300)
		plt.clf()


	# Apply thin to the chain
	samples = chain[::thin, :, :].reshape((-1, ndim))
	# do a copy
	samples_copy = np.copy(samples)
	# change eps ---> inc and phi_bar ---> deg
	samples_copy[:,m+1] = eps_2_inc(samples_copy[:,m+1])*180/np.pi
	# Marginilize parameters
	chain_res = np.empty((ndim,7))
	for i in range(ndim):
		#
		# Show -+2 sigma
		#
		param = samples[:, i]
		mcmc = np.percentile(param, [2.275, 15.865, 50, 84.135,97.725])
		median = mcmc[2]
		[sigma2l,sigma1l,sigma1u,sigma2u] = median-mcmc[0],median-mcmc[1],mcmc[3]-median,mcmc[4]-median 

		std1= 0.5 * abs(sigma1u + sigma1l)
		std2= 0.5 * abs(sigma2u + sigma2l)
		chain_res[i,:] = [median,sigma1l,sigma1u,std1,sigma2l,sigma2u,std2]
	medians, errors, std = chain_res[:,0],chain_res[:,4:6],chain_res[:,-1]

	# Marginilize again but now including both inc and pa_bar in degrees
	chain_res_copy = np.empty((ndim,7))
	for i in range(ndim):
		param = samples_copy[:, i]
		mcmc = np.nanpercentile(param, [2.275, 15.865, 50, 84.135,97.225]); median = mcmc[2]
		[sigma2l,sigma1l,sigma1u,sigma2u] = median-mcmc[0],median-mcmc[1],mcmc[3]-median,mcmc[4]-median 

		std1= 0.5 * abs(sigma1u + sigma1l)
		std2= 0.5 * abs(sigma2u + sigma2l)
		chain_res_copy[i,:] = [median,sigma1l,sigma1u,std1,sigma2l,sigma2u,std2]
	medians_c, errors_c, std_c = chain_res_copy[:,0],chain_res_copy[:,4:6],chain_res_copy[:,-1]


	V_k,eV_k = medians[:m], std[:m]
	pa,eps,x0,y0,Vsys = medians[m:m+5]
	epa,eeps,ex0,ey0,eVsys, = std[m:m+5]


	if PropDist == "G":
		if ndim == n_params_mdl:
			theta_mdls = medians
		if ndim == n_params_mdl+1:
			lnsigma, elnsigma = medians[-1], std[-1]
			theta_mdls = medians[:-1]

	if PropDist == "C":
		lnsigma, elnsigma = medians[-1], std[-1]
		theta_mdls = medians[:-1]


	#########################################
	# Corner plot							#
	#########################################

	# Corner plot only includes constant parameters
	flat_samples = samples_copy[:,m:m+nlabels]
	# Just for visualization purposes I only include 5sigma interval in corner plot
	range0 = []
	for i in range(nlabels):
		param = flat_samples[:, i]
		low5sigma,up5sigma = np.nanpercentile(param, [2.8665157197904634e-05,99.9999713348428])
		range0.append((low5sigma,up5sigma))
	

	CORNER_KWARGS = dict(
	smooth=0.9,
	label_kwargs=dict(fontsize=10),
	plot_datapoints=False,
	labelpad = 0.3,
	max_n_ticks=3,
	levels=(1 - np.exp(-0.5),1 - np.exp(-4/2.)),
	range = range0,# include 5sigma range for each variable
	contour_kwargs = dict(linewidths = 0.5))


	fig = corner.corner(flat_samples, labels=labels_const, truths=truths_const, truth_color = "#faa022", **CORNER_KWARGS);
	# Extract the axes
	axes_corner = np.array(fig.axes).reshape((nlabels, nlabels))

	di = np.diag_indices(nlabels)
	for i,j in zip(di[0],di[1]):
		ax = axes_corner[i, j]
		ax.set_title('{ $%s_{-%s}^{+%s}$ }'%(round(medians_c[m+i],2), abs(round(errors_c[m+i][0],2)), abs(round(errors_c[m+i][1],2))), fontsize = 10)


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


	Vrot_flat,Vrad_flat= V_k[:n_circ],V_k[n_circ:]
	eVrot_flat,eVrad_flat= eV_k[:n_circ],eV_k[n_circ:]

	Vrot = np.asarray(Vrot_flat)
	Vrad = np.asarray(Vrot)*0;Vrad[:n_noncirc]= Vrad_flat

	eVrot = np.asarray(eVrot_flat)
	eVrad = np.asarray(eVrot)*0;eVrad[:n_noncirc]= eVrad_flat

	V_k = [Vrot, Vrad, Vrot*0] 
	eV_k = [eVrot, eVrad, eVrot*0]
	e_constant_parms = [epa,eeps,ex0,ey0,eVsys,0]

	errors = [[],[]]
	errors[0],errors[1] = eV_k,e_constant_parms
	from src.create_2D_vlos_model_mcmc import KinModel
	kinmodel = KinModel( 0, 0, theta_mdls, vmode, rings_pos, ring_space, pixel_scale, inner_interp, PropDist, 0, n_circ, n_noncirc, shape, config_psf,only_model = True)
	vlos_2D_model = kinmodel.interp_model(theta_mdls)

	interp_mdl = bidi_models(vmode, shape, V_k, pa, eps, x0, y0, Vsys, rings_pos, ring_space, pixel_scale, inner_interp) 
	kin_2D_models = interp_mdl.interp()

	# TO DO ...
	from src.save_mcmc_outs import marginal_vals
	marginal_vals(galaxy,vmode,chain_res,n_circ,n_noncirc,outdir,nlabels,mcmc_outs[1:-1])

	return vlos_2D_model, kin_2D_models, V_k, pa, eps, x0, y0, Vsys, 0, errors



