import numpy as np
import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
from matplotlib import gridspec
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)


from src.axes_params import axes_ambient as axs 
from src.cbar import colorbar as cb
from src.colormaps_CLC import vel_map


#params =   {'text.usetex' : True }
#plt.rcParams.update(params) 

cmap = vel_map()

def plot_kin_models(galaxy,vmode,vel_ha,R,Vrot,eVrot,Vrad,eVrad,Vtan,eVtan,VSYS, model, ext, out):

	mask_model = np.divide(model,model)
	model_copy = np.copy(model-VSYS)
	model = model*np.divide(vel_ha, vel_ha)
	model = model - VSYS
	vel_ha = vel_ha - VSYS
	Vrad[Vrad == 0] = np.nan
	Vtan[Vtan == 0] = np.nan

	width, height = 18.0, 6 # width [cm]
	#width, height = 3, 15 # width [cm]
	cm_to_inch = 0.393701 # [inch/cm]
	figWidth = width * cm_to_inch # width [inch]
	figHeight = height * cm_to_inch # width [inch]
  
	fig = plt.figure(figsize=(figWidth, figHeight), dpi = 300)
	#nrows x ncols
	widths = [1.2, 1.2, 1.2, 0.4, 2]
	heights = [1]
	gs2 = gridspec.GridSpec(1, 5,  width_ratios=widths, height_ratios=heights)
	gs2.update(left=0.04, right=0.99,top=0.8,bottom=0.12, hspace = 0.03, wspace = 0)



	ax=plt.subplot(gs2[0,0])
	ax1=plt.subplot(gs2[0,1])
	ax2=plt.subplot(gs2[0,2])
	ax3=plt.subplot(gs2[0,4])

	vmin = abs(np.nanmin(model))
	vmax = abs(np.nanmax(model))
	max_vel = np.nanmax([vmin,vmax])
	
	vmin = -(max_vel//50 + 1)*50
	vmax = (max_vel//50 + 1)*50


	im0 = ax.imshow(vel_ha,cmap = cmap, origin = "lower",vmin = vmin,vmax = vmax, aspect = "auto", extent = ext, interpolation = "nearest", zorder = 2)
	im1 = ax1.imshow(model_copy,cmap = cmap, origin = "lower", aspect = "auto", vmin = vmin,vmax = vmax, extent = ext, interpolation = "nearest", zorder = 2)

	residual = (vel_ha- model)
	im2 = ax2.imshow(residual,cmap = cmap, origin = "lower", aspect = "auto",vmin = -50,vmax = 50, extent = ext, interpolation = "nearest", zorder = 2)


	# If you dont want to add these lines ontop of the 2D plots, then comment the following.
	# it seems that  calling astropy.convolution causes that some package is downloaded each time this is run, why ?
	#"""
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
	#ax2.contour(z, levels = [i*50 for i in np.arange(-N,N+1)], colors = "k", alpha = 1, zorder = 1e3, extent = ext, linewidths = 0.6)
	#if np.nanmean(Vtan) < 0 :
	#	Vtan = -Vtan
	#	Vrad = -Vrad
	#"""

	axs(ax,tickscolor = "k")
	axs(ax1,tickscolor = "k",remove_yticks= True)
	axs(ax2,tickscolor = "k",remove_yticks= True)



	ax.set_ylabel('$\mathrm{ \Delta Dec~(arcsec)}$',fontsize=10,labelpad=0)
	ax.set_xlabel('$\mathrm{ \Delta RA~(arcsec)}$',fontsize=10,labelpad=0)
	ax1.set_xlabel('$\mathrm{ \Delta RA~(arcsec)}$',fontsize=10,labelpad=0)
	ax2.set_xlabel('$\mathrm{ \Delta RA~(arcsec)}$',fontsize=10,labelpad=0)

	ax.text(0.05,1.01, "$\mathrm{vlos}$", fontsize = 10, transform = ax.transAxes)
	ax1.text(0.05,1.01,"$\mathrm{model}$", fontsize = 10, transform = ax1.transAxes)
	ax2.text(0.05,1.01,"$\mathrm{residual}$",fontsize = 10, transform = ax2.transAxes)


	if vmode == "circular":
		ax3.plot(R,Vrot, color = "#362a1b",linestyle='-', alpha = 1, linewidth=0.8, label = "$\mathrm{V_{t}}$")
		ax3.fill_between(R, Vrot-eVrot, Vrot+eVrot, color = "#362a1b", alpha = 0.3, linewidth = 0)

	if vmode == "radial":

		ax3.plot(R,Vrot, color = "#362a1b",linestyle='-', alpha = 1, linewidth=0.8, label = "$\mathrm{V_{t}}$")
		ax3.fill_between(R, Vrot-eVrot, Vrot+eVrot, color = "#362a1b", alpha = 0.3, linewidth = 0)

		ax3.plot(R,Vrad, color = "#c73412",linestyle='-', alpha = 0.6, linewidth=0.8, label = "$\mathrm{V_{r}}$")
		ax3.fill_between(R, Vrad-eVrad, Vrad+eVrad, color = "#c73412", alpha = 0.3, linewidth = 0)




	if vmode == "bisymmetric":
		ax3.plot(R,Vrot, color = "#362a1b",linestyle='-', alpha = 1, linewidth=0.8, label = "$\mathrm{V_{t}}$")
		ax3.fill_between(R, Vrot-eVrot, Vrot+eVrot, color = "#362a1b", alpha = 0.3, linewidth = 0)


		ax3.plot(R,Vrad, color = "#c73412",linestyle='-', alpha = 1, linewidth=0.8, label = "$\mathrm{V_{2,r}}$")
		ax3.fill_between(R, Vrad-eVrad, Vrad+eVrad, color = "#c73412", alpha = 0.3, linewidth = 0)


		ax3.plot(R,Vtan, color = "#2fa7ce",linestyle='-', alpha = 1, linewidth=0.8, label = "$\mathrm{V_{2,t}}$")
		ax3.fill_between(R, Vtan-eVtan, Vtan+eVtan, color = "#2fa7ce", alpha = 0.3, linewidth = 0)




	#bbox_to_anchor =(x0, y0, width, height)
	ax3.legend(loc = "center", fontsize = 10, bbox_to_anchor = (0, 1, 1, 0.2), ncol = 3, frameon = False)

	vels = [0*Vrot, Vrot, Vrad, Vtan]
	max_vel,min_vel = int(np.nanmax(vels)),int(np.nanmin(vels)) 
	min_vel = abs(min_vel)

	ax3.set_ylim(-50*(min_vel//50)-50,50*(max_vel//50)+80)
	ax3.plot([0,np.nanmax(R)],[0,0],color = "k",linestyle='-', alpha = 0.6,linewidth = 0.3)
	ax3.set_xlabel('$\mathrm{r~(arcsec)}$',fontsize=10,labelpad = 0)
	ax3.set_ylabel('$\mathrm{V_{rot}~(km~s^{-1})}$',fontsize=10,labelpad = 0.5)


	axs(ax3,tickscolor = "k")
	cb(im1,ax,orientation = "horizontal", colormap = cmap, bbox= (0.5,1.14,1,1),width = "100%", height = "5%",label_pad = -24, label = "$\mathrm{(km~s^{-1})}$",font_size=10, labelsize = 8)
	cb(im2,ax2,orientation = "horizontal", colormap = cmap, bbox= (0,1.14,1,1),width = "100%", height = "5%",label_pad = -24, label = "$\mathrm{(km~s^{-1})}$",font_size=10, labelsize = 8)

	ax.grid(visible = True, which = "major", axis = "both", color='w', linestyle='-', linewidth=0.5, zorder =  1, alpha = 0.5)
	ax1.grid(visible = True, which = "major", axis = "both", color='w', linestyle='-', linewidth=0.5, zorder = 1, alpha = 0.5)
	ax2.grid(visible = True, which = "major", axis = "both", color='w', linestyle='-', linewidth=0.5, zorder = 1, alpha = 0.5)
	ax3.grid(visible = True, which = "major", axis = "both", color='w', linestyle='-', linewidth=0.5, zorder = 1, alpha = 0.5)

	plt.savefig("%sfigures/kin_%s_model_%s.png"%(out,vmode,galaxy))
	plt.clf()




