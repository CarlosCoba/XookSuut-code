import numpy as np
import matplotlib.pylab as plt
from astropy.io import fits
from scipy.optimize import curve_fit


from write_table import write
import errno
import os


from figure import  fig_ambient
from axis import AXIS

from os import path

"""
Rotation Curve Models
"""



# From Bouche + 2015
def exp(r_sky,v_max,R_turn):
	return v_max*(1-np.exp(-r_sky/R_turn))

#Courteau + 1997
def arctan(r_sky,v_max,R_turn):
	return (2/np.pi)*v_max*np.arctan(r_sky/R_turn)

def tanh(r_sky,v_max,R_turn):
	return v_max*np.tanh(r_sky/R_turn)

#From Haeun Chung, to model the rising RC
def tanh_linear(r_sky,v_max,R1,R2):
		R_turn = R1
		# R2 can be negative in case of descending RCs
		v=v_max*(np.tanh(r_sky/R_turn) + r_sky*R2 )
		return v

#Courteau + 1997
def multi_param(r_sky,v_max,R_turn,gamma):
		beta = 0
		x=R_turn/r_sky
		A = (1+x)**beta
		B = (1+x**gamma)**(1./gamma)
		v=v_max*A/B
		return v
# Bertola 1991
def bertola(r_sky,v_max,k,gamma):
	v = v_max*r_sky/(r_sky**2 + k**2)**(gamma/2.)
	return v


vmax_rot = 400.
def fit_RC(r_array,v_array,method):
	max_R = np.nanmax(r_array)
	R = np.linspace(0.1,max_R,100)
	if method == "exp":
		try:
			popt, pcov = curve_fit(exp, r_array, v_array, bounds=([0,0], [vmax_rot,60]))
			vmax,r_turn = popt
			best_fit = exp(r_array,vmax,r_turn)
			best_fit_interp = exp(R,vmax,r_turn)
		except(RuntimeError):
			vmax,r_turn,best_fit,best_fit_interp = 0,0,r_array*0,r_array*0
		return vmax,r_turn,best_fit,best_fit_interp

	if method == "arctan":
		try:
			popt, pcov = curve_fit(arctan, r_array, v_array, bounds=([0,0], [vmax_rot,60]))
			vmax,r_turn = popt
			best_fit = arctan(r_array,vmax,r_turn)
			best_fit_interp = arctan(R,vmax,r_turn)
		except(RuntimeError):
			vmax,r_turn,best_fit,best_fit_interp = 0,0,r_array*0,r_array*0
		return vmax,r_turn,best_fit,best_fit_interp

	if method == "tanh":
		try:
			popt, pcov = curve_fit(tanh, r_array, v_array, bounds=([0,0], [vmax_rot,60]))
			vmax,r_turn = popt
			best_fit = tanh(r_array,vmax,r_turn)
			best_fit_interp = tanh(R,vmax,r_turn)
		except(RuntimeError):
			vmax,r_turn,best_fit,best_fit_interp = 0,0,r_array*0,r_array*0
		return vmax,r_turn,best_fit,best_fit_interp

	if method == "tanh-linear":
		try:
			# R2 can be negative, common values 0.010 < R2 < 0.070
			popt, pcov = curve_fit(tanh_linear, r_array, v_array, bounds=([0,0,0], [vmax_rot,60,0.1]))
			vmax,r_turn,R2 = popt
			best_fit = tanh_linear(r_array,vmax,r_turn,R2)
			best_fit_interp = tanh_linear(R,vmax,r_turn,R2)
		except(RuntimeError):
			vmax,r_turn,R2,best_fit,best_fit_interp = 0,0,0,r_array*0,r_array*0
		return vmax,r_turn,R2,best_fit,best_fit_interp


	if method == "multi-param":
		try:
			popt, pcov = curve_fit(multi_param, r_array, v_array, bounds=([0,0,-1], [vmax_rot,60,12]))
			vmax, r_turn,gamma = popt
			best_fit = multi_param(r_array,vmax,r_turn,gamma)
			best_fit_interp = multi_param(R,vmax,r_turn,gamma)
		except(RuntimeError):
			vmax,r_turn,gamma,best_fit,best_fit_interp = 0,0,0,r_array*0,r_array*0
		return vmax,r_turn,gamma,best_fit,best_fit_interp


	if method == "Bertola+1991":
		try:
			popt, pcov = curve_fit(bertola, r_array, v_array, bounds=([0,0,1.], [vmax_rot,60,3/2.]))
			vmax, k,gamma = popt
			best_fit = bertola(r_array,vmax,k,gamma)
			best_fit_interp = bertola(R,vmax,k,gamma)
		except(RuntimeError):
			vmax, k,gamma,best_fit = 0,0,0,r_array*0,r_array*0
		return vmax, k,gamma,best_fit,best_fit_interp





model = ["exp", "arctan", "tanh","tanh-linear","multi-param","Bertola+1991"]
model = ["exp", "arctan", "tanh","multi-param","Bertola+1991"]


axes = fig_ambient(fig_size = (2,1.7), ncol = 1, nrow = 1, left = 0.15, right = 0.99, top = 0.99, hspace = 0, bottom = 0.15, wspace = 0 )
ax = axes[0]

def fit_rotcur(galaxy,vmode,e_Vrot,survey):
	try:

		data = fits.getdata("./models/%s.%s.1D_model.fits"%(galaxy,vmode))

		R = data[0]
		V = data[1]
		
		R_plot = np.linspace(0.1,np.nanmax(R),100)

		min_vel,max_vel = int(np.nanmin(V)),int(np.nanmax(V))
		temp = []
		v_max_temp = []
		R_turn_temp = []
		
		for_table = [galaxy]

		for i in model:
			best_params = fit_RC(R,V,i)
			data = best_params[:-1]
			Vmax,Rturn = data[0],data[1]
			if Vmax >= vmax_rot: Vmax = 0 
			if Vmax == 0: Vmax = np.nan 
			temp.append(Vmax)
			temp.append(Rturn)

			for_table.append(Vmax)
			for_table.append(Rturn)
			v_max_temp.append(Vmax)
			R_turn_temp.append(Rturn)

			ax.plot(R,V,"k", linestyle='-', alpha = 0.6, linewidth=0.8,zorder = 100)#, label = "$\mathrm{V_{t}}$")
			ax.errorbar(R,V, yerr=[e_Vrot,e_Vrot], fmt='o', color = "k",markersize = 1, elinewidth = 0.6,capsize  = 1.5)
			ax.fill_between(R, V-e_Vrot, V+e_Vrot, color = "darkgray", alpha = 0.4)
			if len(R_plot) == len(best_params[-1]):
				ax.plot(R_plot,best_params[-1],label = i,linewidth=0.5)


		#The average values for Vmax and Rturn
		mean_Vmax,median_Vmax,std_Vmax = np.nanmedian(v_max_temp),np.nanmean(v_max_temp),np.nanstd(v_max_temp)
		mean_Rt,median_Rt,std_Rt = np.nanmedian(R_turn_temp),np.nanmean(R_turn_temp),np.nanstd(R_turn_temp)

		for_table.extend([mean_Vmax,median_Vmax,std_Vmax,mean_Rt,median_Rt,std_Rt])
		table = for_table



		if survey != "":
			table_name = "vmax_out_params.%s_model.%s.csv"%(vmode,survey) 
			#write(table,table_name,column = False)
		else:
			table_name = "vmax_out_params.%s_model.%s.csv"%(vmode,galaxy) 




		# write header of table
		if path.exists(table_name) == True:
			pass
		else:
			hdr = ["object"]
			for i in model: hdr.extend(["Vmax_%s_model"%(i), "Rturn_%s_model"%(i)])
			hdr.extend(["mean_Vmax","median_Vmax","std_Vmax","mean_Rt","median_Rt","std_Rt"])
			write(hdr,table_name,column = False)


		write(table,table_name,column = False)

		plt.legend( fontsize = "xx-small")
		ax.set_xlabel("$\mathrm{r~(arcsec)}$", fontsize = 8,labelpad = 0)
		ax.set_ylabel("$\mathrm{V_{rot}~(km~s^{-1})}$",fontsize=8,labelpad = 0)
		ax.set_ylim(-50,max_vel+50)
		#ax.legend(loc = "center", fontsize = 6.5, bbox_to_anchor = (0, 1, 1, 0.1), ncol = 3, frameon = False)
		AXIS(ax)
		plt.savefig("./vmax_rturn/%s.fit_rotcur.%s.png"%(galaxy,vmode),dpi = 300)
		#plt.show()
		plt.cla()
		return median_Vmax,median_Rt

	except(IOError, OSError):
		table = [galaxy, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
		if survey != "":
			write(table,"vmax_out_params.%s_model.%s.csv"%(vmode,survey),column = False)
		else:
			write(table,"vmax_out_params.%s_model.%s.csv"%(vmode,galaxy),column = False)

		return 1,1	
		pass
