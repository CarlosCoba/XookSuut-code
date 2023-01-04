import matplotlib.pylab as plt
from dynesty import plotting as dyplot
from src.axes_params import axes_ambient as AX
from matplotlib.gridspec import GridSpec
from matplotlib import gridspec

height, width = 18.0, 14 # width [cm]
cm_to_inch = 0.393701 # [inch/cm]
figWidth = width * cm_to_inch # width [inch]
figHeight = height * cm_to_inch # width [inch]



	

def dplots(res,truths,vmode, galaxy, PropDist, int_scatter, n_circ, n_noncirc):
	ndim=len(truths)
	nrow, ncol = ndim, 2
	labels = ["$v_{%s}$"%k for k in range(ndim)]
	labels_const = ["$\mathrm{\phi^{\prime}}$", "$\epsilon$", "$\mathrm{x_0}$", "$\mathrm{y_0}$", "$\mathrm{V_{sys}}$"]
	if "hrm" in vmode:
		labels_const[-1] = "$c_0$"

	if vmode == "bisymmetric":
		labels_const.append("$\\phi_{\mathrm{bar}}$")


	if PropDist == "G":
		if int_scatter :
			labels_const.append("$\mathrm{\ln~\sigma_{int}^2}$")
	if PropDist == "C":
		labels_const.append("$\mathrm{\gamma~(km/s)}$")
	nconst = len(labels_const)
	labels[-nconst:] = labels_const[:]	

	fig, axs = plt.subplots(nrow, ncol, figsize = (figWidth, figHeight),  \
	gridspec_kw={'height_ratios': [1.5 for k in range(nrow)], 'width_ratios': [4.5 for k in range(ncol)]})

	# plotting the original run
	dyplot.traceplot(res, truths=truths, truth_color="#faa022",
		                         show_titles=True, title_kwargs={'fontsize': 12, 'y': 1.05},
		                         trace_cmap='plasma', kde=False, max_n_ticks = 4, labels = labels,
		                         fig=(fig, axs))

	plt.gcf().subplots_adjust(bottom=0.1, top = 0.95)
	plt.savefig("./XS/figures/%s.%s.dyplot.trace.png"%(galaxy,vmode),dpi=300)
	plt.close()
	plt.clf()


	nrow, ncol = 4, 1
	fig, axs = plt.subplots(nrow, ncol, figsize = (	4, 6))
	dyplot.runplot(res, color='dodgerblue', fig = (fig, axs))
	plt.gcf().subplots_adjust(bottom=0.1, top = 0.95, left = 0.2, right = 0.98)
	#fig.tight_layout()
	plt.savefig("./XS/figures/%s.%s.dyplot.runplot.png"%(galaxy,vmode))
	plt.close()
	plt.clf()
