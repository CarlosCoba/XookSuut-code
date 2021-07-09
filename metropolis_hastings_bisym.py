import numpy as np
import matplotlib.pylab as plt
import matplotlib.pylab as plt
from matplotlib.ticker import ScalarFormatter
import corner
import itertools
import warnings



from weights_interp import weigths_w
from kin_components import CIRC_MODEL
from kin_components import RADIAL_MODEL
from kin_components import BISYM_MODEL

from figure import fig_ambient
from axis import AXIS
 
def Rings(xy_mesh,pa,inc,x0,y0,pixel_scale):
	(x,y) = xy_mesh

	X = (- (x-x0)*np.sin(pa) + (y-y0)*np.cos(pa))
	Y = (- (x-x0)*np.cos(pa) - (y-y0)*np.sin(pa))

	R= np.sqrt(X**2+(Y/np.cos(inc))**2)

	return R*pixel_scale


m = 0
def bayesian_mcmc(galaxy, shape, vel_map, e_vel_map, guess, vmode, config, rings_pos, ring_space, steps, e_ISM = 5, pixel_scale = 1, r_bar_min = None, r_bar_max = None):

	vrot0,vrad0,pa0,inc0,X0,Y0,vsys0,vtan0,phi0 = guess

	n_circ = len(vrot0)
	non_zero = vrad0 != 0
	vrad0 = vrad0[non_zero]
	vtan0 = vtan0[non_zero]

	#n_noncirc = len(vtan0)
	#mask_r = rings_pos < 10
	#vrad0 = vrad0[mask_r]
	#vtan0 = vtan0[mask_r]
	n_noncirc = len(vtan0)


	global m
	m = n_circ+2*n_noncirc

	
	[ny,nx] = shape

	nrings = len(rings_pos)
	n_annulus = nrings - 1  


	interp_model = np.zeros((ny,nx))

	X = np.arange(0, nx, 1)
	Y = np.arange(0, ny, 1)
	XY_mesh = np.meshgrid(X,Y)
	X0,Y0 = X0+1e-3,Y0+1e-3 
	r_n = Rings(XY_mesh,pa0*np.pi/180,inc0*np.pi/180,X0,Y0,pixel_scale)

	#e_vel_map[e_vel_map <= 0 ] = 1
	#e_vel_map[~np.isfinite(e_vel_map)] = 1


	def model(theta):

		global m
		m = n_circ+2*n_noncirc


		Vrot, Vrad, Vtan, pa,inc,x0,y0,Vsys,phi_b = theta[:n_circ],theta[n_circ:n_circ+n_noncirc],theta[n_circ+n_noncirc:m],theta[m],theta[m+1],theta[m+2],theta[m+3],theta[m+4],theta[m+5]

		

		Vrot_dic = {}
		Vrad_dic = {}
		Vtan_dic = {}
		for j in range(n_circ):
		    Vrot_dic["Vrot{0}".format(j)] = Vrot[j]

		for j in range(n_noncirc):
		    Vrad_dic["Vrad{0}".format(j)] = Vrad[j]
		    Vtan_dic["Vtan{0}".format(j)] = Vtan[j]


		model_array = np.array([])
		obs_array = np.array([])
		error_array = np.array([])
		Xpos,Ypos = np.array([]),np.array([])

		interp_model = np.zeros((ny,nx))
		def eval_mdl(xy_mesh,pa,inc,x0,y0,Vsys,phi_b, r_0 = None, r_space = ring_space,ii=0):

			Vrot_ = Vrot_dic["Vrot%s"%ii]
			if ii + 1 <= n_noncirc:
				Vrad_ = Vrad_dic["Vrad%s"%ii]
				Vtan_ = Vtan_dic["Vtan%s"%ii]
				bisymm = BISYM_MODEL(xy_mesh,Vrot_,Vrad_,pa,inc,x0,y0,Vtan_,phi_b)
			else:
				bisymm = BISYM_MODEL(xy_mesh,Vrot_,0,pa,inc,x0,y0,0,0)

			W = weigths_w(xy_mesh,shape,pa,inc,x0,y0,r_0,r_space,pixel_scale=pixel_scale)
			w0,w1 = weigths_w(xy_mesh,shape,pa,inc,x0,y0,r_0,r_space,pixel_scale=pixel_scale) 
			modl0 = bisymm*w0
			modl1 = bisymm*w1

			mdl = bisymm*W
			return mdl,Vsys


		for N in range(n_annulus):

			mdl_ev = 0
			r_space_k = rings_pos[N+1] - rings_pos[N] 
			mask = np.where( (r_n >= rings_pos[N] ) & (r_n < rings_pos[N+1]) )
			x,y = XY_mesh[0][mask], XY_mesh[1][mask] 
			XY = (x,y)

			for kk in range(2):
				Vxy,Vsys = eval_mdl(XY, pa,inc,x0,y0,Vsys,phi_b, r_0 = rings_pos[N], r_space = r_space_k, ii = N+kk)
				mdl_ev = mdl_ev + Vxy[kk]


				if N == 0 and kk == 0:
							
					mask1 = np.where( (r_n < rings_pos[0] ) )
					x1,y1 = XY_mesh[0][mask1], XY_mesh[1][mask1] 
					XY1 = (x1,y1)


					#
					#
					# inner interpolation
					#
					#
						
					#(a) velocity rise linearly from zero

					r_space_0 =  rings_pos[0]
					Vxy,Vsys  = eval_mdl(XY1, pa,inc,x0,y0,Vsys,phi_b, r_0 = 0, r_space = r_space_0, ii = 0)


					model_array = np.append(model_array,Vxy[1])
					obs_array = np.append(obs_array,vel_map[mask1])
					error_array = np.append(error_array,e_vel_map[mask1])
					interp_model[mask1] = Vxy[1]


				m_ = mdl_ev
				o_ = vel_map[mask]
				e_ = e_vel_map[mask]

			model_array = np.append(model_array,m_)
			obs_array = np.append(obs_array,o_)
			error_array = np.append(error_array,e_)
			#interp_model[mask] = mdl_ev + Vsys
			Xpos,Ypos = np.append(Xpos, x), np.append(Ypos, y)

		model_array = model_array + Vsys
		res =  model_array - obs_array

		return model_array, obs_array, error_array, Xpos,Ypos





	def ln_likelihood(theta):
		 # we will pass the parameters (theta) to the model function

		model_array, obs_array, error_array, Xpos,Ypos = model(theta)
		OBS, MODEL,ERROR = model_array, obs_array, error_array
		N = len(OBS)

		#log_L = -0.5 * np.nansum( np.log(2*np.pi*ERROR** 2) + ( (OBS - MODEL) / ERROR) ** 2)
		#log_L = -0.5 * np.nansum( ( (OBS - MODEL) / ERROR) ** 2 )
		log_L = -1 * np.nansum( ( (OBS - MODEL)) ** 2 )/N
		#log_L = -0.5 * np.nansum( ( (OBS - MODEL) / ERROR) ** 2 ) - (N/2.)*np.log(2*np.pi) - np.nansum( np.log(ERROR) )
		#log_L = -0.5 * np.nansum( ( (OBS - MODEL) / ERROR) ** 2 + np.log(2*np.pi*ERROR**2) )

		return log_L 


	def ln_prior(theta):
		Vrot, Vrad, Vtan, pa,inc,x0,y0,Vsys,phi_b =  theta[:n_circ],theta[n_circ:n_circ+n_noncirc],theta[n_circ+n_noncirc:m],theta[m],theta[m+1],theta[m+2],theta[m+3],theta[m+4],theta[m+5]
		# flat prior: log(1) = 0

		Vrot_arr = np.asarray(Vrot)
		Vrad_arr = np.asarray(Vrad)
		Vtan_arr = np.asarray(Vtan)

		#apply conditions on priors here
		if  0 <= pa <= 360 and 0 < inc < 90 and 0 < x0 < nx and 0 < y0 < ny and 0 < Vsys < 3e5 and 0 < phi_b < 180:
			mask_1 = (Vrot_arr > -400) & (Vrot_arr < 400) 
			mask_2 = (Vrad_arr > -300) & (Vrad_arr < 300) 
			mask_3 = (Vtan_arr > -300) & (Vtan_arr < 300)
			mask_product = mask_2*mask_3
			if False in mask_1 or False in mask_product:
				return -np.inf
			else:  
				return 0.0
		else:
			return -np.inf



	def ln_posterior(theta):
		return ln_prior(theta) + ln_likelihood(theta)


	def tune(pars, sigpars, all_pars, prng):

		"""
		Adaptive move

		"""
		nu = np.unique(all_pars[:, 0]).size
		npars = pars.size

		# Accumulate at least as many *accepted* moves as the
		# elements of the covariance matrix before computing it.
		if nu > npars*(npars + 1.)/2.:

			eps = 0.01
			diag = np.diag((sigpars*eps)**2)
			cov = 2.38**2/npars*(np.cov(all_pars.T) + diag)
			pars = prng.multivariate_normal(pars, cov)

		else:

			pars = prng.normal(pars, sigpars)

		return pars


	prng = np.random.RandomState(456)  # Random stream independent of global one
	def run_mcmc(ln_posterior, nsteps, ndim, theta0, stepsize, args=()):
		# Create the array of size (nsteps, ndims) to hold the chain
		chain = np.zeros((nsteps, ndim))
		rejected = []
		accepted = []

		# Create the array of size nsteps to hold the log-likelihoods for each point
		# Initialize the first entry of this with the log likelihood at theta0

		log_likes = np.zeros(nsteps)
		log_likes[0] = ln_posterior(theta0, *args)

		accept_rate = np.zeros(nsteps)
		naccept = 0
		sigma = stepsize

		theta = theta0
		for i in range(1, nsteps):

			# Randomly draw a new theta from the proposal distribution.
			theta_new =tune(theta, sigma, chain[:i-1], prng)
		
			# Calculate the probability for the new state
			log_like_new = ln_posterior(theta_new, *args)

			# Compare it to the probability of the old state
			log_p_accept = log_like_new - log_likes[i - 1]
		
			# Chose a random number r between 0 and 1 to compare with p_accept
			r = np.random.rand()
			r = prng.uniform()
		

			if (log_p_accept > 1) or (log_p_accept > np.log(r)):

				theta = theta_new
				#Store only accepted moves
				accepted.append(theta)
				log_likes[i] = log_like_new
				naccept += 1
				#accept_rate[i] = naccept / i


			else:
				#Store here rejected moves
				rejected.append(theta_new)
				log_likes[i] = log_likes[i - 1]
				


			accept_rate[i] = naccept / i
			chain[i-1] = theta_new	


		accepted = np.asarray(accepted)
		n_accepted = len(accepted)
		accepted_chain = accepted.reshape((n_accepted,ndim))

		rejected = np.asarray(rejected)
		n_rejected = len(rejected)
		rejected_chain = rejected.reshape((n_rejected,ndim))

		return accepted_chain,rejected_chain,accept_rate



	theta = vrot0,vrad0,vtan0,pa0,inc0,X0,Y0,vsys0,phi0
	theta0 = np.concatenate([x.ravel() for x in theta])
	ndim = len(theta0)

	step_size = np.array([np.ones(n_circ)*1e1,np.ones(n_noncirc)*1e1,np.ones(n_noncirc)*1e1,1,1,1e-1,1e-1,1,1])*1e-1
	stepsize = 	np.hstack(step_size)

	print("Running chain ...")

	chain,reject,accept_rate = run_mcmc(ln_posterior, steps, ndim, theta0, stepsize)
	threshold_accept_rat = 0.23


	axes = fig_ambient(fig_size = (1.5,1.5), ncol = 1, nrow = 1, left = 0.12, right = 0.99, top = 0.99, hspace = 0, bottom = 0.15, wspace = 0.4 )
	ax = axes[0]
	XX = np.arange(steps)
	ax.plot(XX,accept_rate)
	ax.set_xlabel("$\mathrm{steps}$",fontsize=6,labelpad=2)
	ax.set_ylabel("$\mathrm{acceptance~rate}$",fontsize=6,labelpad=2)
	AXIS(ax,fontsize_ticklabel = 4)
	plt.savefig("./plots/accept_rate.%s_model.%s.png"%(vmode,galaxy),dpi = 300)
	plt.tight_layout()
	plt.clf()


	ncols = n_circ
	nrows = 5
	if n_circ >= 6:
		nrows = 4
	if n_circ <= 3:
		ncols = 3
		nrows = 5	
	if ncols > 3 and ncols < n_circ:
		nrows = 5

	axes = fig_ambient(fig_size = (6,1.5), ncol = ncols, nrow = nrows, left = 0.02, right = 0.99, top = 0.99, hspace = 0, bottom = 0.1, wspace = 0.4 )
	for i in range(n_circ):
		globals()['ax_circ%s' % i] = axes[i]
		globals()['ax_circ%s' % i].plot(chain[:, i], linewidth = 1)
		AXIS(globals()['ax_circ%s' % i],fontsize_ticklabel = 4)
		globals()['ax_circ%s' % i].set_ylabel("$\mathrm{Vcirc,%s}$"%i,fontsize=4,labelpad=0)
		ax = globals()['ax_circ%s' % i]
		ax.set_aspect(np.diff(ax.get_xlim())/np.diff(ax.get_ylim()))



	for i in range(n_noncirc):
		globals()['ax_rad%s' % i] = axes[i+n_circ]
		globals()['ax_rad%s' % i].plot(chain[:, i+n_noncirc], linewidth = 1)
		AXIS(globals()['ax_rad%s' % i],fontsize_ticklabel = 4)
		globals()['ax_rad%s' % i].set_ylabel("$\mathrm{Vrad,%s}$"%i,fontsize=4,labelpad=0)
		ax = globals()['ax_rad%s' % i]
		ax.set_aspect(np.diff(ax.get_xlim())/np.diff(ax.get_ylim()))


	for i in range(n_noncirc):
		globals()['ax_tan%s' % i] = axes[i+(2*n_circ)]
		globals()['ax_tan%s' % i].plot(chain[:, i+2*n_noncirc], linewidth = 1)
		AXIS(globals()['ax_tan%s' % i],fontsize_ticklabel = 4)
		globals()['ax_tan%s' % i].set_ylabel("$\mathrm{Vtan,%s}$"%i,fontsize=4,labelpad=0)
		ax = globals()['ax_tan%s' % i]
		ax.set_aspect(np.diff(ax.get_xlim())/np.diff(ax.get_ylim()))


	ax_pa = axes[3*n_circ]
	ax_pa.plot(chain[:, m], linewidth = 1)
	AXIS(ax_pa,fontsize_ticklabel = 4)
	ax_pa.set_ylabel("$\mathrm{\phi^{\prime}}$",fontsize=4,labelpad=0)
	ax_pa.set_aspect(np.diff(ax_pa.get_xlim())/np.diff(ax_pa.get_ylim()))

	ax_inc = axes[3*n_circ+1]
	ax_inc.plot(chain[:, m+1], linewidth = 1)
	AXIS(ax_inc,fontsize_ticklabel = 4)
	ax_inc.set_ylabel("$i$",fontsize=4,labelpad=0)
	ax_inc.set_aspect(np.diff(ax_inc.get_xlim())/np.diff(ax_inc.get_ylim()))


	ax_x0 = axes[3*n_circ+2]
	ax_x0.plot(chain[:, m+2], linewidth = 1)
	AXIS(ax_x0,fontsize_ticklabel = 4)
	ax_x0.set_ylabel("$\mathrm{XC}$",fontsize=4,labelpad=0)
	ax_x0.set_aspect(np.diff(ax_x0.get_xlim())/np.diff(ax_x0.get_ylim()))

	ax_y0 = axes[3*n_circ+3]
	ax_y0.plot(chain[:, m+3])
	AXIS(ax_y0,fontsize_ticklabel = 4)
	ax_y0.set_ylabel("$\mathrm{YC}$",fontsize=4,labelpad=0)
	ax_y0.set_aspect(np.diff(ax_y0.get_xlim())/np.diff(ax_y0.get_ylim()))


	ax_vsys = axes[3*n_circ+4]
	ax_vsys.plot(chain[:, m+4], linewidth = 1)
	AXIS(ax_vsys,fontsize_ticklabel = 4)
	ax_vsys.set_ylabel("$\mathrm{Vsys}$",fontsize=4,labelpad=0)
	ax_vsys.set_aspect(np.diff(ax_vsys.get_xlim())/np.diff(ax_vsys.get_ylim()))


	ax_theta = axes[3*n_circ+5]
	ax_theta.plot(chain[:, m+5], linewidth = 1)
	AXIS(ax_theta,fontsize_ticklabel = 4)
	ax_theta.set_ylabel("$\\theta_{\mathrm{bar}}$",fontsize=4,labelpad=0)
	ax_theta.set_aspect(np.diff(ax_theta.get_xlim())/np.diff(ax_theta.get_ylim()))

	ax_pa.text(n_circ/2.,0.05,"$\mathrm{step number}$",fontsize = 4, transform = ax_pa.transAxes, zorder = 1e2)


	for i in range(n_circ - n_noncirc):
		globals()['ax%s' % i] = axes[n_circ+n_noncirc+i]
		globals()['ax%s' % i].remove()

		globals()['ax1%s' % i] = axes[2*n_circ+n_noncirc+i]
		globals()['ax1%s' % i].remove()


	used_axis = nrows*ncols - (3*n_circ + 6)	
	if used_axis < nrows*ncols:	
		for i in range(used_axis):
			globals()['ax_remove%s' % i] = axes[3*n_circ+6+i]
			globals()['ax_remove%s' % i].remove()


	plt.savefig("./plots/mcmc_steps.%s_model.%s.png"%(vmode,galaxy),dpi = 300)
	plt.tight_layout()
	plt.clf()



	theta_best = chain.mean(0)
	theta_best = np.nanmedian(chain, axis = 0)
	theta_std = chain.std(0)



	#
	# Show only the corner plot for the constant parameters
	#
	labels = ["$\mathrm{\phi^{\prime}}$", "$i$", "$\mathrm{XC}$", "$\mathrm{YC}$", "$\mathrm{Vsys}$", "$\\phi_{\mathrm{bar}}$"]
	nlabels = len(labels)
	flat_samples = chain[:,m:m+6]


	fig = corner.corner(
	    flat_samples, labels=labels, truths=[pa0, inc0, X0, Y0, vsys0, phi0]
	);

	plt.savefig("./plots/corner.%s_model.%s.png"%(vmode,galaxy),dpi = 300)
	plt.tight_layout()
	plt.clf()


	#
	# Show only the corner plot for the kinematic components
	#

	if n_circ < 20:
		nlabels = len(labels)

		labels = ["$\mathrm{Vcirc,%s}$"%i for i in range(n_circ)]
		flat_samples = chain[:,0:n_circ]
		fig = corner.corner(flat_samples, labels=labels, truths=[vrot0[i] for i in range(n_circ)]);
		#plt.tight_layout()
		warnings.simplefilter("ignore")
		plt.savefig("./plots/corner.vcirc.%s_model.%s.pdf"%(vmode,galaxy), dpi=300)
		plt.clf()

		labels = ["$\mathrm{Vrad,%s}$"%i for i in range(n_noncirc)]
		flat_samples = chain[:,n_circ:n_circ+n_noncirc]
		fig = corner.corner(flat_samples, labels=labels, truths=[vrad0[i] for i in range(n_noncirc)]);
		warnings.simplefilter("ignore")
		plt.savefig("./plots/corner.vrad.%s_model.%s.pdf"%(vmode,galaxy), dpi=300)
		#plt.tight_layout()
		plt.clf()


		labels = ["$\mathrm{Vtan,%s}$"%i for i in range(n_noncirc)]
		flat_samples = chain[:,n_circ+n_noncirc:m]
		fig = corner.corner(flat_samples, labels=labels, truths=[vtan0[i] for i in range(n_noncirc)]);
		warnings.simplefilter("ignore")
		plt.savefig("./plots/corner.vtan.%s_model.%s.pdf"%(vmode,galaxy), dpi=300)
		#plt.tight_layout()
		plt.clf()
	else:
		print("I wont save the corner plot, there are too many variables !! ")

	medians = np.array([])
	errors = np.array([])


	for i in range(ndim):
		#
		# Show -+1 sigma
		#
		mcmc = np.percentile(chain[:, i], [15.865, 50, 84.135])
		q = np.diff(mcmc)
		median, lower,upper = mcmc[1], q[0], q[1]
		# Store median values of the distributions
		medians = np.append(medians,median)
		# I only keep the largest error
		if lower > upper:
			errors = np.append(errors,lower)
		else:
			errors = np.append(errors,upper)
		 
		#print(mcmc[1], q[0], q[1], labels[i])


	Vrot, Vrad, Vtan, pa,inc,x0,y0,Vsys,phi_b = medians[:n_circ],medians[n_circ:n_circ+n_noncirc],medians[n_circ+n_noncirc:m],medians[m],medians[m+1],medians[m+2],medians[m+3],medians[m+4],medians[m+5]
	eVrot, eVrad, eVtan, epa,einc,ex0,ey0,eVsys,ephi_b = errors[:n_circ],errors[n_circ:n_circ+n_noncirc],errors[n_circ+n_noncirc:m],errors[m],errors[m+1],errors[m+2],errors[m+3],errors[m+4],errors[m+5]

	#Vrot, Vrad, Vtan, pa,inc,x0,y0,Vsys,phi_b = theta_best[:n_circ],theta_best[n_circ:n_circ+n_noncirc],theta_best[n_circ+n_noncirc:m],theta_best[m],theta_best[m+1],theta_best[m+2],theta_best[m+3],theta_best[m+4],theta_best[m+5]

	#eVrot, eVrad, eVtan, epa,einc,ex0,ey0,eVsys,ephi_b = theta_std[:n_circ],theta_std[n_circ:n_circ+n_noncirc],theta_std[n_circ+n_noncirc:m],theta_std[m],theta_std[m+1],theta_std[m+2],theta_std[m+3],theta_std[m+4],theta_std[m+5]



	if n_circ != n_noncirc:
		Vrad = np.append(Vrad,np.zeros(n_circ-n_noncirc))
		Vtan = np.append(Vtan,np.zeros(n_circ-n_noncirc))
		eVrad = np.append(eVrad,np.zeros(n_circ-n_noncirc))
		eVtan = np.append(eVtan,np.zeros(n_circ-n_noncirc))



	from create_2D_vlos_model import best_2d_model
	V_k = [Vrot, Vrad, Vtan] 
	vlos_2D_model = best_2d_model(vmode, shape,V_k, pa, inc, x0, y0, Vsys, rings_pos, ring_space = ring_space, phi_b = phi_b, pixel_scale = pixel_scale) 

	##########
	from create_2D_kin_models import bidi_models
	kin_2D_models = bidi_models(vmode, shape, V_k, pa, inc, x0, y0, Vsys, rings_pos, ring_space = ring_space, pixel_scale = pixel_scale) 
	##########
	
	Std_errors = eVrot,eVrad,epa,einc,ex0,ey0,eVsys,ephi_b,eVtan

	return chain,vlos_2D_model, kin_2D_models, Vrot, Vrad, Vsys,  pa, inc , x0, y0, Vtan, phi_b, Std_errors




















