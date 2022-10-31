import numpy as np
import matplotlib.pylab as plt
import matplotlib
from matplotlib.ticker import ScalarFormatter
import corner
import itertools
from scipy.stats import multivariate_normal


from weights_interp import weigths_w
from kin_components import CIRC_MODEL
from kin_components import RADIAL_MODEL
from kin_components import BISYM_MODEL

from fig_params import fig_ambient
from axes_params import axes_ambient as axs

#params =   {'text.usetex' : True }
#plt.rcParams.update(params) 






def chain_res_mcm(galaxy, vmode, sampler, theta, chain, step_size, steps, thin, burnin, accept_rate, shape, save_plots, rings_pos, ring_space, pixel_scale, inner_interp, lnsigma_int, m_hrm, n_circ, n_noncirc, phi_b = 0 ):


	vmode2 = vmode + "_%s"%m_hrm
	if lnsigma_int == False:
		pa0,inc0,x0,y0,vsys0 = theta[-5:]
		vk0 = theta[:-5]
		labels_const = ["$\mathrm{\phi^{\prime}}~(^\circ)$", "$i~(^\circ)$", "$\mathrm{x_0}~(pix)$", "$\mathrm{y_0}~(pix)$", "$\mathrm{c_0~(km/s)}$"]
		truths_const = [pa0, inc0, x0, y0, vsys0]
	else:
		pa0,inc0,x0,y0,vsys0,lnsigma2 = theta[-6:]
		vk0 = theta[:-6]

		labels_const = ["$\mathrm{\phi^{\prime}}$", "$i$", "$\mathrm{x_0}$", "$\mathrm{y_0}$", "$\mathrm{c_0}$", "$\mathrm{sigma}$"]
		truths_const = [pa0, inc0, x0, y0, vsys0,lnsigma2]


	#theta_flat = np.hstack(theta.flatten())
	ndim = len(theta)
	m = len(theta) - len(truths_const)



	fig,axes = fig_ambient(fig_size = (2,1.3), ncol = 1, nrow = 1, left = 0.11, right = 0.99, top = 0.99, hspace = 0, bottom = 0.15, wspace = 0.4 )
	ax = axes[0]
	XX = np.arange(steps)
	ax.plot(XX,accept_rate, c = "k", lw = 0.8)
	ax.set_xlabel("$\mathrm{steps}$",fontsize=6,labelpad=2)
	ax.set_ylabel("$\mathrm{acceptance~rate}$",fontsize=6,labelpad=2)
	axs(ax,fontsize_ticklabels = 4)
	plt.savefig("./plots/accept_rate.%s_model.%s.png"%(vmode2,galaxy),dpi = 300)
	plt.tight_layout()
	plt.clf()

	vel_labels = [] 

	if save_plots == True:

		# plot c1. c1 is always wheteber m_hrm is
		fig, axes = fig_ambient(fig_size = (3.5,5), ncol = 1, nrow = n_circ, left = 0.2, right = 0.95, top = 0.99, hspace = 0, bottom = 0.1, wspace = 0.4 )
		labels = ["$\mathrm{ c_{1,%s} } $"%k for k in range(n_circ)]
		vel_labels =  vel_labels + ["c1_%s"%k for k in range(n_circ)]
		for i in range(n_circ):
			ax = axes[i]
			ax.plot(chain[:, i], "k", alpha=0.5)
			ax.set_xlim(0, len(chain))
			ax.yaxis.set_label_coords(-0.15, 0.5)
			axs(ax, fontsize_yticklabel = 8, rotation = "horizontal")
			ax.set_ylabel(labels[i], fontsize = 8)
			if i != (n_circ-1) :ax.xaxis.set_ticklabels([])
		axes[-1].set_xlabel("$\mathrm{steps}$", fontsize = 10);
		plt.savefig("./plots/chain_progress.c1.%s_model.%s.png"%(vmode2,galaxy), dpi=300)
		plt.clf()


		# plots of c for m_hrm > 1
		if m_hrm >=2:

			for ii in range(2,m_hrm+1):
				fig, axes = fig_ambient(fig_size = (3.5,5), ncol = 1, nrow = n_noncirc, left = 0.2, right = 0.95, top = 0.99, hspace = 0, bottom = 0.1, wspace = 0.4 )
				labels = ["$\mathrm{ c_{%s,%s} } $"%(ii,k) for k in range(n_noncirc)]
				vel_labels =  vel_labels + ["c%s_%s "%(ii,k) for k in range(n_noncirc)]
				kk = 0
				n_c_k = n_circ + n_noncirc*(2*ii-2)
				nvels = n_circ + n_noncirc + n_noncirc*(2*ii-2)
				for i in range(n_c_k, nvels):
					ax = axes[kk]
					ax.plot(chain[:, i], "k", alpha=0.5)
					ax.set_xlim(0, len(chain))
					ax.yaxis.set_label_coords(-0.15, 0.5)
					axs(ax, fontsize_yticklabel = 8, rotation = "horizontal")
					ax.set_ylabel(labels[kk], fontsize = 8)
					if i != (n_noncirc-1) :ax.xaxis.set_ticklabels([])
					kk = kk + 1
				axes[-1].set_xlabel("$\mathrm{steps}$", fontsize = 10);
				plt.savefig("./plots/chain_progress.c%s.%s_model.%s.png"%(ii,vmode2,galaxy), dpi=300)
				plt.clf()

		# plots of s for m_hrm >= 1
		for ii in range(1,m_hrm+1):
			fig, axes = fig_ambient(fig_size = (3.5,5), ncol = 1, nrow = n_noncirc, left = 0.2, right = 0.95, top = 0.99, hspace = 0, bottom = 0.1, wspace = 0.4 )
			labels = ["$\mathrm{ s_{%s,%s} } $"%(ii,k) for k in range(n_noncirc)]
			vel_labels =  vel_labels + ["s%s_%s"%(ii,k) for k in range(n_noncirc)]
			kk = 0
			n_s_k = n_circ + n_noncirc*(2*ii-2)
			nvels = n_circ + n_noncirc + n_noncirc*(2*ii-2)
			for i in range(n_s_k, nvels):
				ax = axes[kk]
				ax.plot(chain[:, i], "k", alpha=0.5)
				ax.set_xlim(0, len(chain))
				ax.yaxis.set_label_coords(-0.15, 0.5)
				axs(ax, fontsize_yticklabel = 8, rotation = "horizontal")
				ax.set_ylabel(labels[kk], fontsize = 8)
				if i != (n_noncirc-1) :ax.xaxis.set_ticklabels([])
				kk = kk + 1
			axes[-1].set_xlabel("$\mathrm{steps}$", fontsize = 10);
			plt.savefig("./plots/chain_progress.s%s.%s_model.%s.png"%(ii,vmode2,galaxy), dpi=300)
			plt.clf()



	medians = np.array([])
	errors = np.array([])
	sigma_errors = np.empty((ndim,2))

	# Apply burnin and thin to the chain

	if sampler in [2]:
		good_samples = chain[burnin::thin]
	else:
		good_samples = chain


	chain_res = np.empty((ndim,3))
	for i in range(ndim):
		#
		# Show -+1 sigma
		#
		mcmc = np.percentile(good_samples[:, i], [15.865, 50, 84.135])
		q = np.diff(mcmc)
		median, lower,upper = mcmc[1], q[0], q[1]
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

	if save_plots == True:
		from mcmc_out import save_mcmc_outs
		save_mcmc_outs(galaxy,vmode2,chain_res,n_circ,n_noncirc,rings_pos,vel_labels)

	#
	# Corner plot
	#

	nlabels = len(labels_const)
	flat_samples = chain[burnin:,m:]


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
	fig.savefig("./plots/corner.%s_model.%s.png"%(vmode,galaxy),dpi = 300)
	plt.clf()

	if lnsigma_int == False:
		pa,inc,x0,y0,Vsys = medians[-5:]
		epa,einc,ex0,ey0,eVsys = errors[-5:]

	else:
		pa,inc,x0,y0,Vsys = medians[-6:-1]
		epa,einc,ex0,ey0,eVsys = errors[-6:-1]


	mask_c = [0] + [n_circ] + [n_circ + n_noncirc*(k+1) for k in range(m_hrm)]
	mask_s = [0] + [n_noncirc] + [n_noncirc + n_noncirc*(k+1) for k in range(m_hrm)]
	#C_k, S_k =  medians[:n_circ+int(0.5*m_hrm)*n_noncirc], medians[n_circ+int(0.5*m_hrm)*n_noncirc:-5]
	#eC_k, eS_k =  errors[:n_circ+int(0.5*m_hrm)*n_noncirc], errors[n_circ+int(0.5*m_hrm)*n_noncirc:-5]


	C_k, S_k =  medians[:n_circ+int(m_hrm-1)*n_noncirc], medians[n_circ+int(m_hrm-1)*n_noncirc:-5]
	eC_k, eS_k =  errors[:n_circ+int(m_hrm-1)*n_noncirc], errors[n_circ+int(m_hrm-1)*n_noncirc:-5]
	# We need to sort the velocities C1,C2...Cn, S1,S2...Sn
	C = [list(C_k[mask_c[i]:mask_c[i+1]]) for i in range(len(mask_c)-2)]
	S = [list(S_k[mask_s[i]:mask_s[i+1]]) for i in range(len(mask_s)-2)]
	eC = [list(eC_k[mask_c[i]:mask_c[i+1]]) for i in range(len(mask_c)-2)]
	eS = [list(eS_k[mask_s[i]:mask_s[i+1]]) for i in range(len(mask_s)-2)]	

	j = 1
	if n_circ != n_noncirc:
		zeros = [0]*(n_circ - n_noncirc)
		for k in range(m_hrm):
			if m_hrm == 1:
				S[k].extend(zeros)
				eS[k].extend(zeros)

			else:
				if j < m_hrm:
					C[j].extend(zeros)
					eC[j].extend(zeros)
					S[k].extend(zeros)
					eS[k].extend(zeros)

					j = j+1
				else:
					S[k].extend(zeros)
					eS[k].extend(zeros)



	V_k = C+S
	eV_k = eC + eS

	e_constant_parms = [epa,einc,ex0,ey0,eVsys]
	errors = eV_k + e_constant_parms

	from create_2D_vlos_model import best_2d_model
	create_2D = best_2d_model(vmode,shape,V_k, pa, inc, x0, y0, Vsys, rings_pos, ring_space, pixel_scale, inner_interp, m_hrm, phi_b = 0) 
	vlos_2D_model = create_2D.model2D()

	##########
	from create_2D_kin_models import bidi_models
	interp_mdl = bidi_models(vmode, shape, V_k, pa, inc, x0, y0, Vsys, rings_pos, ring_space, pixel_scale, inner_interp, m_hrm,0) 
	kin_2D_models = interp_mdl.interp()
	##########
	return vlos_2D_model, kin_2D_models, V_k, pa, inc , x0, y0, Vsys, errors



