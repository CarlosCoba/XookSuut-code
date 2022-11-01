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






def chain_res_mcmc(galaxy, vmode, theta, chain, mcmc_config, accept_rate, shape, rings_pos, ring_space, pixel_scale, inner_interp, phi_b = 0, outdir = False ):

	chain_params = mcmc_config
	steps,thin,burnin,nwalkers,pdf,save_plots = chain_params

	n_circ = len(theta[0])
	n_noncirc = len(theta[1])
	m = n_circ + n_noncirc

	if pdf == "G":
		vrot0,vrad0,pa0,inc0,x0,y0,vsys0 = theta
		labels_const = ["$\mathrm{\phi^{\prime}}~(^\circ)$", "$i~(^\circ)$", "$\mathrm{x_0}~(pix)$", "$\mathrm{y_0}~(pix)$", "$\mathrm{Vsys~(km/s)}$"]
		truths_const = [pa0, inc0, x0, y0, vsys0]
	if pdf == "C":
		vrot0,vrad0,pa0,inc0,x0,y0,vsys0,lnsigma2 = theta
		labels_const = ["$\mathrm{\phi^{\prime}}~(^\circ)$", "$i~(^\circ)$", "$\mathrm{x_0}~(pix)$", "$\mathrm{y_0}~(pix)$", "$\mathrm{Vsys~(km/s)}$", "$\mathrm{\gamma~(km/s)}$"]
		truths_const = [pa0, inc0, x0, y0, vsys0,lnsigma2]

	theta_flat = np.hstack(theta.flatten())
	ndim = len(theta_flat)


	time = np.arange(steps)

	vel_labels = [] 
	if save_plots :
		fig, axes = fig_ambient(fig_size = (3.5,5), ncol = 1, nrow = n_circ, left = 0.2, right = 0.95, top = 0.99, hspace = 0, bottom = 0.1, wspace = 0.4 )
		labels = ["$\mathrm{ V_{t}{%s} } $"%k for k in range(n_circ)]
		vel_labels = 	vel_labels + ["Vt_%s"%k for k in range(n_circ)]
		for i in range(n_circ):
			ax = axes[i]
			if nwalkers != 0 :
				for k in range(nwalkers):
					ax.plot(time,chain[k, :, i], alpha=0.5, lw = 0.1)

			ax.set_xlim(0, len(time))
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
			if nwalkers != 0 :
				for k in range(nwalkers):
					ax.plot(time,chain[k, :, i], alpha=0.5, lw = 0.1)

			#ax.plot(chain[:, i], "k", alpha=0.5)
			ax.set_xlim(0, len(time))
			ax.yaxis.set_label_coords(-0.15, 0.5)
			axs(ax, fontsize_yticklabel = 8, rotation = "horizontal")
			ax.set_ylabel(labels[i-n_circ], fontsize = 8)
			if i != (n_circ+n_noncirc-1) :ax.xaxis.set_ticklabels([])
		axes[-1].set_xlabel("$\mathrm{steps}$", fontsize = 10);
		plt.savefig("%sfigures/chain_progress.rad.%s_model.%s.png"%(outdir,vmode,galaxy), dpi=300)
		plt.clf()




	medians = np.array([])
	errors = np.array([])
	sigma_errors = np.empty((ndim,2))

	# Apply thin to the chain
	good_samples = chain[:, ::thin, :].reshape((-1, ndim))

	chain_res = np.empty((ndim,3))
	for i in range(ndim):
		#
		# Show -+1 sigma
		#
		mcmc = np.percentile(good_samples[:, i], [15.865, 50, 84.135])
		q = np.diff(mcmc)
		median, lower,upper = mcmc[1], q[0], q[1]
		if pdf == "C":
			lower, upper = np.std(good_samples[:, i]), np.std(good_samples[:, i])
		chain_res[i,:] = [median, lower,upper]
		# Store median values of the distributions
		medians = np.append(medians,median)
		# I keep only the largest errors
		if lower > upper:
			errors = np.append(errors,lower)
		else:
			errors = np.append(errors,upper)

		# Store 1-sigma errors of the distributions
		sigma_errors[i] = [lower,upper]


	# TO DO ...
	#if save_plots :
	#	from mcmc_out import save_mcmc_outs
	#	save_mcmc_outs(galaxy,vmode,chain_res,n_circ,n_noncirc,rings_pos,vel_labels)


	#
	# Corner plot
	#

	nlabels = len(labels_const)
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
		ax.set_title('{ $%s_{-%s}^{+%s}$ }'%(round(medians[m+i],2), abs(round(sigma_errors[m+i][0],2)), abs(round(sigma_errors[m+i][1],2))), fontsize = 10)


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



	Vrot, Vrad, pa,inc,x0,y0,Vsys = medians[:n_circ],medians[n_circ:m],medians[m],medians[m+1],medians[m+2],medians[m+3],medians[m+4]
	eVrot, eVrad, epa,einc,ex0,ey0,eVsys = errors[:n_circ],errors[n_circ:m],errors[m],errors[m+1],errors[m+2],errors[m+3],errors[m+4]

	if n_circ != n_noncirc:
		Vrad = np.append(Vrad,np.zeros(n_circ-n_noncirc))
		eVrad = np.append(eVrad,np.zeros(n_circ-n_noncirc))

	V_k = [Vrot, Vrad, Vrad*0] 
	create_2D = best_2d_model(vmode, shape,V_k, pa, inc, x0, y0, Vsys, rings_pos, ring_space, pixel_scale, inner_interp) 
	vlos_2D_model = create_2D.model2D()

	interp_mdl = bidi_models(vmode, shape, V_k, pa, inc, x0, y0, Vsys, rings_pos, ring_space, pixel_scale, inner_interp) 
	kin_2D_models = interp_mdl.interp()
	
	errors = [eVrot,eVrad,eVrad,epa,einc,ex0,ey0,eVsys,0]

	return vlos_2D_model, kin_2D_models, V_k, pa, inc, x0, y0, Vsys, 0, errors



