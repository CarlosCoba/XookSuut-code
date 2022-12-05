import numpy as np
import matplotlib
import matplotlib.pylab as plt
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



#chain dims = (nsteps, nwalkers, nparams)


def chain_res_mcmc(galaxy, vmode, theta, mcmc_outs, shape, rings_pos, ring_space, pixel_scale, inner_interp, phi_b = 0, outdir = False, config_psf=None, plot_chain = False ):

	[chain, acc_frac, steps, thin, burnin, Nwalkers, PropDist, ndim, act] = mcmc_outs
	n_circ,n_noncirc = len(theta[0]),0
	m = n_circ

	[nwalkers, steps,ndim]=chain.shape
	# These are the number of parameters from the kinematic models
	n_params_mdl = n_circ + 5

	# Labels of constant parameters
	labels_const = ["$\mathrm{\phi^{\prime}}~(^\circ)$", "$i~(^\circ)$", "$\mathrm{x_0}~(pix)$", "$\mathrm{y_0}~(pix)$", "$\mathrm{V_{sys}~(km/s)}$"]

	if PropDist == "G":
		if ndim == n_params_mdl:
			pa0,inc0,x0,y0,vsys0 = theta[-5:]
			truths_const = [pa0, inc0, x0, y0, vsys0]
		if ndim == n_params_mdl+1:
			pa0,inc0,x0,y0,vsys0,lnsigma2 = theta[-6:]
			truths_const = [pa0, inc0, x0, y0, vsys0, lnsigma2]
			labels_const.append("$\mathrm{\ln~\sigma_{int}^2}$")
	if PropDist == "C":
		pa0,inc0,x0,y0,vsys0,lnsigma2 = theta[-6:]
		truths_const = [pa0, inc0, x0, y0, vsys0, lnsigma2]
		labels_const.append("$\mathrm{\gamma~(km/s)}$")

	nlabels = len(labels_const)
	theta_flat = np.hstack(theta.flatten())
	time = np.arange(steps)

	vel_labels = []
	if plot_chain :
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


	# Apply thin to the chain
	good_samples = chain[::thin, :, :].reshape((-1, ndim))

	# Marginilize parameters
	chain_res = np.empty((ndim,7))
	for i in range(ndim):
		#
		# Show -+2 sigma
		#
		mcmc = np.percentile(good_samples[:, i], [2.275, 15.865, 50, 84.135,97.225])
		median = mcmc[2]
		[sigma2l,sigma1l,sigma1u,sigma2u] = median-mcmc[0],median-mcmc[1],mcmc[3]-median,mcmc[4]-median 

		std1= 0.5 * abs(sigma1u + sigma1l)
		std2= 0.5 * abs(sigma2u + sigma2l)
		chain_res[i,:] = [median,sigma1l,sigma1u,std1,sigma2l,sigma2u,std2]
	medians, errors, std = chain_res[:,0],chain_res[:,1:3],chain_res[:,-1]

	if PropDist == "G":
		if ndim == n_params_mdl:
			pa,inc,x0,y0,Vsys = medians[-5:]
			epa,einc,ex0,ey0,eVsys = std[-5:]
			V_k = medians[:-5]
			eV_k = std[:-5]
			theta_mdls = medians

		if ndim == n_params_mdl+1:
			pa,inc,x0,y0,Vsys,lnsigma = medians[-6:]
			epa,einc,ex0,ey0,eVsys,elnsigma = std[-6:]
			V_k = medians[:-6]
			eV_k = std[:-6]
			theta_mdls = medians[:-1]

	if PropDist == "C":
		pa,inc,x0,y0,Vsys,lnsigma = medians[-6:]
		epa,einc,ex0,ey0,eVsys,elnsigma = std[-6:]
		V_k = medians[:-6]
		eV_k = std[:-6]
		theta_mdls = medians[:-1]

	# TO DO ...
	from src.save_mcmc_outs import marginal_vals
	marginal_vals(galaxy,vmode,chain_res,n_circ,n_noncirc,outdir,nlabels,mcmc_outs[1:])

	#
	# Corner plot
	#

	# Corner plot only includes constant parameters
	flat_samples = good_samples[:,m:]


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
		ax.set_title('{ $%s_{-%s}^{+%s}$ }'%( round(medians[m+i],2), abs(round(errors[m+i][0],2)), abs(round(errors[m+i][1],2))), fontsize = 10)

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
		 


	Vrot = V_k
	eVrot = eV_k

	V_k = [Vrot, Vrot*0, Vrot*0] 
	eV_k = [eVrot, eVrot*0, eVrot*0]
	e_constant_parms = [epa,einc,ex0,ey0,eVsys,0]

	errors = [[],[]]
	errors[0],errors[1] = eV_k,e_constant_parms



	#V_k = [Vrot, Vrot*0, Vrot*0] 
	#create_2D = best_2d_model(vmode, shape,V_k, pa, inc, x0, y0, Vsys, rings_pos, ring_space, pixel_scale, inner_interp, config_psf) 
	#vlos_2D_model = create_2D.model2D()


	from src.create_2D_vlos_model_mcmc import KinModel
	kinmodel = KinModel( 0, 0, theta_mdls, vmode, rings_pos, ring_space, pixel_scale, inner_interp, PropDist, 0, n_circ, n_noncirc, shape, config_psf,only_model = True)
	vlos_2D_model = kinmodel.interp_model(theta_mdls)

	interp_mdl = bidi_models(vmode, shape, V_k, pa, inc, x0, y0, Vsys, rings_pos, ring_space, pixel_scale, inner_interp) 
	kin_2D_models = interp_mdl.interp()


	return vlos_2D_model, kin_2D_models, V_k, pa, inc, x0, y0, Vsys, 0, errors



