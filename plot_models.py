import numpy as np
import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
from axis import AXIS 
from CBAR import colorbar as cb
from colormaps_CLC import vel_map
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from error_bars import error_bar

cmap = vel_map()

def plot_kin_models(galaxy,vmode,vel_ha,R,Vrot,eVrot,Vrad,eVrad,Vtan,eVtan,VSYS, MODEL, ext,plot = 0, save = 1):
	e1,e2,e3 = len(eVrot),len(eVrad),len(eVtan)

	mask_MODEL = np.divide(MODEL,MODEL)
	MODEL_copy = np.copy(MODEL-VSYS)
	MODEL = MODEL*np.divide(vel_ha, vel_ha)
	MODEL = MODEL - VSYS
	vel_ha = vel_ha - VSYS
	fig=plt.figure(figsize=(6,2))
	gs2 = GridSpec(1, 3)
	gs2.update(left=0.045, right=0.62,top=0.815,hspace=0.01,bottom=0.135,wspace=0.)


	ax=plt.subplot(gs2[0,0])
	ax1=plt.subplot(gs2[0,1])
	ax2=plt.subplot(gs2[0,2])



	vmin = abs(np.nanmin(MODEL))
	vmax = abs(np.nanmax(MODEL))
	max_vel = np.nanmax([vmin,vmax])
	
	vmin = -(max_vel//50 + 1)*50
	vmax = (max_vel//50 + 1)*50


	im0 = ax.imshow(vel_ha,cmap = cmap, origin = "lower",vmin = vmin,vmax = vmax, aspect = "auto", extent = ext, interpolation = "nearest")
	im1 = ax1.imshow(MODEL_copy,cmap = cmap, origin = "lower", aspect = "auto", vmin = vmin,vmax = vmax, extent = ext, interpolation = "nearest")

	residual = (vel_ha- MODEL)
	im2 = ax2.imshow(residual,cmap = cmap, origin = "lower", aspect = "auto",vmin = -50,vmax = 50, extent = ext, interpolation = "nearest")

	"""
	from astropy.convolution import Gaussian2DKernel
	from astropy.convolution import convolve
	import matplotlib
	matplotlib.rcParams['contour.negative_linestyle']= 'solid'

	kernel = Gaussian2DKernel(x_stddev=4)
	vloss = convolve(vel_ha, kernel)
	z = np.ma.masked_array(vloss.astype(int),mask= np.isnan(vel_ha) )
	N = vmax // 50
	ax.contour(z, levels = [i*50 for i in np.arange(-N,N+1)], colors = "k", alpha = 1, zorder = 1e3, extent = ext, linewidths = 0.6)
	ax1.contour(z, levels = [i*50 for i in np.arange(-N,N+1)], colors = "k", alpha = 1, zorder = 1e3, extent = ext, linewidths = 0.6)
	ax2.contour(z, levels = [i*50 for i in np.arange(-N,N+1)], colors = "k", alpha = 1, zorder = 1e3, extent = ext, linewidths = 0.6)
	if np.nanmean(Vtan) < 0 :
		Vtan = -Vtan
		Vrad = -Vrad
	"""

	AXIS(ax,tickscolor = "k")
	AXIS(ax1,tickscolor = "k",remove_yticks= True)
	AXIS(ax2,tickscolor = "k",remove_yticks= True)



	ax.set_ylabel('$\mathrm{ \Delta Dec~(arc)}$',fontsize=8,labelpad=0)
	ax.set_xlabel('$\mathrm{ \Delta RA~(arc)}$',fontsize=8,labelpad=0)
	ax1.set_xlabel('$\mathrm{ \Delta RA~(arc)}$',fontsize=8,labelpad=0)
	ax2.set_xlabel('$\mathrm{ \Delta RA~(arc)}$',fontsize=8,labelpad=0)

	ax.text(0.05,1.01, "$\mathrm{vlos}$", fontsize = 7, transform = ax.transAxes)
	ax1.text(0.05,1.01,"$\mathrm{model}$", fontsize = 7, transform = ax1.transAxes)
	ax2.text(0.05,1.01,"$\mathrm{residual}$",fontsize = 7, transform = ax2.transAxes)


	gs2 = GridSpec(1, 1)
	gs2.update(left=0.68, right=0.995,top=0.815,bottom=0.135)
	ax3=plt.subplot(gs2[0,0])


	if vmode == "circular":
		ax3.plot(R,Vrot, color = "#362a1b",linestyle='-', alpha = 0.6, linewidth=0.8, label = "$\mathrm{V_{t}}$")

		if e1 > 0:
			ax3.fill_between(R, Vrot-eVrot, Vrot+eVrot, color = "#362a1b", alpha = 0.3, linewidth = 0)

	if vmode == "radial":

		ax3.plot(R,Vrot, color = "#362a1b",linestyle='-', alpha = 0.6, linewidth=0.8, label = "$\mathrm{V_{t}}$")

		if e1 > 0:
			ax3.fill_between(R, Vrot-eVrot, Vrot+eVrot, color = "#362a1b", alpha = 0.3, linewidth = 0)

		ax3.plot(R,Vrad, color = "#c73412",linestyle='-', alpha = 0.6, linewidth=0.8, label = "$\mathrm{V_{r}}$")
		if e2 > 0:
			ax3.fill_between(R, Vrad-eVrad, Vrad+eVrad, color = "#c73412", alpha = 0.3, linewidth = 0)




	if vmode == "bisymmetric":
		ax3.plot(R,Vrot, color = "#362a1b",linestyle='-', alpha = 1, linewidth=0.8, label = "$\mathrm{V_{t}}$")

		if e1 > 0:
			ax3.fill_between(R, Vrot-eVrot, Vrot+eVrot, color = "#362a1b", alpha = 0.3, linewidth = 0)


		ax3.plot(R,Vrad, color = "#c73412",linestyle='-', alpha = 1, linewidth=0.8, label = "$\mathrm{V_{2,r}}$")

		if e2 > 0:
			ax3.fill_between(R, Vrad-eVrad, Vrad+eVrad, color = "#c73412", alpha = 0.3, linewidth = 0)


		ax3.plot(R,Vtan, color = "#2fa7ce",linestyle='-', alpha = 1, linewidth=0.8, label = "$\mathrm{V_{2,t}}$")

		if e3 > 0:
			ax3.fill_between(R, Vtan-eVtan, Vtan+eVtan, color = "#2fa7ce", alpha = 0.3, linewidth = 0)





	ax3.legend(loc = "center", fontsize = 6.5, bbox_to_anchor = (0, 1, 1, 0.1), ncol = 3, frameon = False)

	vels = [Vrot, Vrad, Vtan]
	max_vel,min_vel = int(np.nanmax(vels)),int(np.nanmin(vels)) 
	min_vel = abs(min_vel)

	#ax3.set_ylim(min_vel-50,max_vel+50)
	ax3.set_ylim(-50*(min_vel//50)-50,50*(max_vel//50)+80)
	ax3.plot([0,np.nanmax(R)],[0,0],color = "k",linestyle='-', alpha = 0.6,linewidth = 0.3)
	ax3.set_xlabel('$\mathrm{r~(arcsec)}$',fontsize=8,labelpad = 0)
	ax3.set_ylabel('$\mathrm{V_{rot}~(km~s^{-1})}$',fontsize=8,labelpad = 0)

	
	if np.nanmax(R) // 10 > 1:
		ax3.xaxis.set_major_locator(MultipleLocator(10))
		# For the minor ticks, use no labels; default NullFormatter.
		ax3.xaxis.set_minor_locator(MultipleLocator(2))
	else:
		ax3.xaxis.set_major_locator(MultipleLocator(2))
		# For the minor ticks, use no labels; default NullFormatter.
		ax3.xaxis.set_minor_locator(MultipleLocator(1))



	if np.nanmax(R) // 100 > 1:
		ax3.xaxis.set_major_locator(MultipleLocator(200))
		# For the minor ticks, use no labels; default NullFormatter.
		ax3.xaxis.set_minor_locator(MultipleLocator(100))



	if np.nanmax(Vrot) // 100 > 1:
		ax3.yaxis.set_major_locator(MultipleLocator(100))
		# For the minor ticks, use no labels; default NullFormatter.
		ax3.yaxis.set_minor_locator(MultipleLocator(20))
	else:
		ax3.yaxis.set_major_locator(MultipleLocator(50))
		# For the minor ticks, use no labels; default NullFormatter.
		ax3.yaxis.set_minor_locator(MultipleLocator(25))



	AXIS(ax3,tickscolor = "k")
	cb(im1,ax,orientation = "horizontal", colormap = cmap, bbox= (0.5,1.135,1,1),width = "100%", height = "5%",label_pad = -17, label = "$\mathrm{(km~s^{-1})}$",font_size=6)
	cb(im2,ax2,orientation = "horizontal", colormap = cmap, bbox= (0,1.135,1,1),width = "100%", height = "5%",label_pad = -17, label = "$\mathrm{(km~s^{-1})}$",font_size=6)


	if save == 1 and plot == 1:
		plt.savefig("./plots/kin_%s_model_%s.png"%(vmode,galaxy),dpi = 300)
		plt.show()
		plt.clf()
	else:

		if plot == 1:
			plt.show()
			plt.clf()
		if save == 1:
			plt.savefig("./plots/kin_%s_model_%s.png"%(vmode,galaxy),dpi = 300)
			plt.clf()




