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

#
# chain shape = (nwalkers, steps, ndim)
#


def chain_res_mcmc(galaxy, vmode, theta, chain, mcmc_config, accept_rate, shape, rings_pos, ring_space, pixel_scale, inner_interp, phi_b = 0, outdir = False, m_hrm=0, n_circ=0, n_noncirc=0 ):

	chain_params = mcmc_config
	steps,thin,burnin,nwalkers,pdf,save_plots = chain_params

	vmode2 = vmode + "_%s"%m_hrm
	if pdf == "G":
		pa0,inc0,x0,y0,vsys0 = theta[-5:]
		vk0 = theta[:-5]
		labels_const = ["$\mathrm{\phi^{\prime}}~(^\circ)$", "$i~(^\circ)$", "$\mathrm{x_0}~(pix)$", "$\mathrm{y_0}~(pix)$", "$\mathrm{c_0~(km/s)}$"]
		truths_const = [pa0, inc0, x0, y0, vsys0]
	if pdf == "C":
		pa0,inc0,x0,y0,vsys0,lnsigma2 = theta[-6:]
		vk0 = theta[:-6]

		labels_const = ["$\mathrm{\phi^{\prime}}$", "$i$", "$\mathrm{x_0}$", "$\mathrm{y_0}$", "$\mathrm{c_0}$", "$\mathrm{\gamma~(km/s)}$"]
		truths_const = [pa0, inc0, x0, y0, vsys0,lnsigma2]


	#theta_flat = np.hstack(theta.flatten())
	ndim = len(theta)
	m = len(theta) - len(truths_const)


	time = np.arange(steps)
	vel_labels = [] 

	if save_plots :

		# plot c1; 
		fig, axes = fig_ambient(fig_size = (3.5,5), ncol = 1, nrow = n_circ, left = 0.2, right = 0.95, top = 0.99, hspace = 0, bottom = 0.1, wspace = 0.4 )
		labels = ["$\mathrm{ c_{1,%s} } $"%k for k in range(n_circ)]
		vel_labels =  vel_labels + ["c1_%s"%k for k in range(n_circ)]
		for i in range(n_circ):
			ax = axes[i]
			if nwalkers != 0 :
				for k in range(nwalkers):
					ax.plot(time,chain[k, :, i], alpha=0.5, lw = 0.1)
			#ax.plot(chain[:, i], "k", alpha=0.5)
			ax.set_xlim(0, len(time))
			ax.yaxis.set_label_coords(-0.15, 0.5)
			axs(ax, fontsize_yticklabel = 8, rotation = "horizontal")
			ax.set_ylabel(labels[i], fontsize = 8)
			if i != (n_circ-1) :ax.xaxis.set_ticklabels([])
		axes[-1].set_xlabel("$\mathrm{steps}$", fontsize = 10);
		plt.savefig("%sfigures/chain_progress.c1.%s_model.%s.png"%(outdir,vmode2,galaxy), dpi=300)
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
					if nwalkers != 0 :
						for k in range(nwalkers):
							#ax.plot(time,chain[k*steps:(k+1)*steps, i], alpha=0.5, lw = 0.1)
							ax.plot(time,chain[k, :, i], alpha=0.5, lw = 0.1)

					ax.set_xlim(0, len(time))
					ax.yaxis.set_label_coords(-0.15, 0.5)
					axs(ax, fontsize_yticklabel = 8, rotation = "horizontal")
					ax.set_ylabel(labels[kk], fontsize = 8)
					if i != (nvels-1) :ax.xaxis.set_ticklabels([])
					kk = kk + 1
				axes[-1].set_xlabel("$\mathrm{steps}$", fontsize = 10);
				plt.savefig("%sfigures/chain_progress.c%s.%s_model.%s.png"%(outdir,ii,vmode2,galaxy), dpi=300)
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
				if nwalkers != 0 :
					for k in range(nwalkers):
						#ax.plot(time,chain[k*steps:(k+1)*steps, i], alpha=0.5, lw = 0.1)
						ax.plot(time,chain[k, :, i], alpha=0.5, lw = 0.1)

				ax.set_xlim(0, len(time))
				ax.yaxis.set_label_coords(-0.15, 0.5)
				axs(ax, fontsize_yticklabel = 8, rotation = "horizontal")
				ax.set_ylabel(labels[kk], fontsize = 8)
				if i != (nvels-1) :ax.xaxis.set_ticklabels([])
				kk = kk + 1
			axes[-1].set_xlabel("$\mathrm{steps}$", fontsize = 10);
			plt.savefig("%sfigures/chain_progress.s%s.%s_model.%s.png"%(outdir,ii,vmode2,galaxy), dpi=300)
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

	# TO DO ....
	#if save_plots :
	#	from mcmc_out import save_mcmc_outs
	#	save_mcmc_outs(galaxy,vmode2,chain_res,n_circ,n_noncirc,rings_pos,vel_labels)

	#
	# Corner plot
	#

	nlabels = len(labels_const)
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
	fig.savefig("%sfigures/corner.%s_model.%s.png"%(outdir,vmode2,galaxy),dpi = 300)
	plt.clf()

	if pdf == "G":
		pa,inc,x0,y0,Vsys = medians[-5:]
		epa,einc,ex0,ey0,eVsys = errors[-5:]
		V_k = theta[:-5]
		eV_k = errors[:-5]

	if pdf == "C":
		pa,inc,x0,y0,Vsys = medians[-6:-1]
		epa,einc,ex0,ey0,eVsys = errors[-6:-1]
		V_k = theta[:-6]
		eV_k = errors[:-6]

	"""
	mask_c = [0] + [n_circ] + [n_circ + n_noncirc*(k+1) for k in range(m_hrm)]
	mask_s = [0] + [n_noncirc] + [n_noncirc + n_noncirc*(k+1) for k in range(m_hrm)]


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
	"""



	V_k_noncirc = V_k[n_circ:]
	eV_k_noncirc = eV_k[n_circ:]
	# First fill C_k and S_k with zeros 
	C_k, S_k = [[0 for k in range(n_circ)] for k in range(m_hrm)], [[0 for k in range(n_circ)] for k in range(m_hrm)]
	eC_k, eS_k = [[0 for k in range(n_circ)] for k in range(m_hrm)], [[0 for k in range(n_circ)] for k in range(m_hrm)]
	#C_k[0] will always exist
	C_k[0] = V_k[:n_circ]
	eC_k[0] = eV_k[:n_circ]
	if m_hrm !=1:
		for k in range(1,m_hrm):
				(C_k[k])[:n_noncirc] = V_k_noncirc[(k-1)*n_noncirc:k*n_noncirc]
				(eC_k[k])[:n_noncirc] = eV_k_noncirc[(k-1)*n_noncirc:k*n_noncirc]
		for k in range(m_hrm,2*m_hrm):
				(S_k[k-m_hrm])[:n_noncirc] = V_k_noncirc[(k-1)*n_noncirc:k*n_noncirc]
				(eS_k[k-m_hrm])[:n_noncirc] = eV_k_noncirc[(k-1)*n_noncirc:k*n_noncirc]
	else:
		S_k[0][:n_noncirc] = V_k_noncirc
		eS_k[0][:n_noncirc] = eV_k_noncirc



	#V_k = C+S
	#eV_k = eC + eS

	V_k = C_k+S_k
	eV_k = eC_k + eS_k

	e_constant_parms = [epa,einc,ex0,ey0,eVsys]
	errors = eV_k + e_constant_parms

	create_2D = best_2d_model(vmode,shape,V_k, pa, inc, x0, y0, Vsys, rings_pos, ring_space, pixel_scale, inner_interp, m_hrm, phi_b = 0) 
	vlos_2D_model = create_2D.model2D()

	##########
	interp_mdl = bidi_models(vmode, shape, V_k, pa, inc, x0, y0, Vsys, rings_pos, ring_space, pixel_scale, inner_interp, m_hrm,0) 
	kin_2D_models = interp_mdl.interp()
	##########
	return vlos_2D_model, kin_2D_models, V_k, pa, inc , x0, y0, Vsys, 0, errors



