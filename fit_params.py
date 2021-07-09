import numpy as np
import numpy as np
import matplotlib.pylab as plt
import scipy
import sys
import lmfit
from lmfit import Model
from lmfit import Parameters, fit_report, minimize
from matplotlib.gridspec import GridSpec
#import sys

from recompute_chi import result
from read_config import config_file
from weights_interp import weigths_w


import astropy.units as u
from astropy.convolution import convolve,convolve_fft




 
def Rings(xy_mesh,pa,inc,x0,y0,pixel_scale):
	(x,y) = xy_mesh

	X = (- (x-x0)*np.sin(pa) + (y-y0)*np.cos(pa))
	Y = (- (x-x0)*np.cos(pa) - (y-y0)*np.sin(pa))

	R= np.sqrt(X**2+(Y/np.cos(inc))**2)

	return R*pixel_scale





def RADIAL_MODEL2(w1,Vsys,w2,Vrot,w3,Vrad):
	vlos = w1*Vsys + w2*Vrot + w3*Vrad
	return np.ravel(vlos)





def polynomial(x,a0,a1,a2,a3,a4,a5):
	x = np.asarray(x)
	y = a0 + a1*x +a2*x**2 +a3*x**3 + a4*x**4 + a5*x**5
	return y

def linear(x,a0,a1):
	x = np.asarray(x)
	y = a0 + a1*x 
	return y

def fit_polynomial(x,dato):
	x = np.asarray(x)
	dato = np.asarray(dato)


	def residual_line(pars,x,data=None):
		parvals = pars.valuesdict()
		a0 = parvals['a0']
		a1 = parvals['a1']
		a2 = parvals['a2']
		a3 = parvals['a3']
		a4 = parvals['a4']
		a5 = parvals['a5']
		model = polynomial(x,a0,a1,a2,a3,a4,a5)
		objective = model - data
		return objective**2


	fit_param = Parameters()
	fit_param.add('a0', value=0)
	fit_param.add('a1', value=0)
	fit_param.add('a2', value=0)
	fit_param.add('a3', value=0)
	fit_param.add('a4', value=0)
	fit_param.add('a5', value=0)

	out = minimize(residual_line, fit_param, args=(x,), kws={'data': dato},method='Powell', nan_policy = "omit")
	best = out.params
	a0, a1, a2, a3, a4, a5 = best["a0"].value,best["a1"].value, best["a2"].value, best["a3"].value, best["a4"].value, best["a5"].value
	return a0, a1, a2, a3, a4, a5


def fit_linear(x,dato):
	x = np.asarray(x)
	dato = np.asarray(dato)


	def residual_line(pars,x,data=None):
		parvals = pars.valuesdict()
		a0 = parvals['a0']
		a1 = parvals['a1']
		model = linear(x,a0,a1)
		objective = model - data
		return objective**2


	fit_param = Parameters()
	fit_param.add('a0', value=1e3)
	fit_param.add('a1', value=0)

	out = minimize(residual_line, fit_param, args=(x,), kws={'data': dato},method='Powell')
	best = out.params
	a0, a1 = best["a0"].value,best["a1"].value
	return a0, a1


from kin_components import CIRC_MODEL
from kin_components import RADIAL_MODEL
from kin_components import BISYM_MODEL

def fit(shape, vel_map, e_vel_map, guess,vary,vmode,config, rings_pos, ring_space, fit_method = "leastsq", e_ISM = 5, pixel_scale = 1):
	"""
	vary = [Vrot,Vrad,PA,INC,XC,YC,VSYS,theta,Vtan]
	"""
	vrot0,vr20,pa0,inc0,X0,Y0,vsys0,vt20,theta0 = guess
	constant_params = [pa0,inc0,X0,Y0,vsys0,theta0]
	vary_vrot,vary_vrad,vary_pa,vary_inc,vary_xc,vary_yc,vary_vsys,vary_theta,vary_vtan = vary

	[ny,nx] = shape

	nrings = len(rings_pos)
	n_annulus = nrings - 1  


	interp_model = np.zeros((ny,nx))
	mask_ring = np.ones((ny,nx), dtype = bool)



	X = np.arange(0, nx, 1)
	Y = np.arange(0, ny, 1)
	XY_mesh = np.meshgrid(X,Y)
	X0,Y0 = X0+1e-3,Y0+1e-3 
	r_n = Rings(XY_mesh,pa0*np.pi/180,inc0*np.pi/180,X0,Y0,pixel_scale)



	if vmode == "circular":


		def vmodel_dataset(pars, i,  xy_mesh,  r_0 = None, r_space = ring_space):


			parvals = pars.valuesdict()
			pa = parvals['pa']
			inc = parvals['inc']
			Vsys = parvals['Vsys']
			Vrot = parvals['Vrot_%i'% i]
			x0,y0 = parvals['x0'],parvals['y0']

			modl = (CIRC_MODEL(xy_mesh,Vrot,pa,inc,x0,y0))*weigths_w(xy_mesh,shape,pa,inc,x0,y0,r_0,r_space,pixel_scale=pixel_scale)


			return modl,Vsys

		def residual(pars, data= None):
			"""Calculate total residual for fits of VMODELS to several data sets."""

			resid = np.array([])


			# make residual per data set

			for N in range(n_annulus):

				mdl_ev = 0
				r_space_k = rings_pos[N+1] - rings_pos[N] 
				mask = np.where( (r_n >= rings_pos[N] ) & (r_n < rings_pos[N+1]) )
				x,y = XY_mesh[0][mask], XY_mesh[1][mask] 
				XY = (x,y)




				for kk in range(2):
					Vxy,Vsys = vmodel_dataset(pars, N+kk,  XY, r_0 = rings_pos[N], r_space = r_space_k )

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
						Vxy,Vsys = vmodel_dataset(pars, 0,  XY1, r_0 = 0, r_space = r_space_0)
					
						interp_model[mask1] = Vxy[1] + Vsys

				interp_model[mask] = mdl_ev + Vsys




			
			sigma = np.sqrt(e_vel_map**2 + e_ISM**2)
			#interp_model[interp_model == 0]  = np.nan


			convolved = 0

			if convolved == True:
				from radio_beam import Beam
				my_beam = Beam(2.5*u.arcsec, 2.5*u.arcsec, 0*u.deg)
				pix_scale = pixel_scale * u.arcsec
				gauss_kern = my_beam.as_kernel(pix_scale, x_size = nx, y_size = ny)


				extend = np.zeros((3*ny,3*nx))
				extend[ny:2*ny,nx:2*nx] = interp_model
				convolve_extend = convolve_fft(extend, gauss_kern, mask = extend == 0 )
				interp_model_conv = convolve_extend[ny:2*ny,nx:2*nx]
				interp_model_conv[interp_model == 0]  = 0




			else:
				interp_model_conv = interp_model

			interp_model_conv[interp_model_conv == 0] = np.nan
			res = vel_map - interp_model_conv
			Resid = res/sigma


			return Resid.flatten()

			

    
#####



		fit_params = Parameters()


		for iy  in range(nrings):
				fit_params.add('Vrot_%i' % (iy),value=vrot0[iy], vary = vary[0], min = -400, max = 400)

		if config == "":

			fit_params.add('Vsys', value=vsys0, vary = vary[6])
			fit_params.add('pa', value=pa0, vary = vary[2], min = 0, max = 360)
			fit_params.add('inc', value=inc0, vary = vary[3], min = 0, max = 90)
			fit_params.add('x0', value=X0, vary = vary[4],  min = 0, max = nx)
			fit_params.add('y0', value=Y0, vary = vary[5], min = 0, max = ny)


		else:
			k = 0
			for res,const in zip(config_file(config),constant_params):
				param, fit, val, vmin, vmax = str(res["param"]), bool(float(res["fit"])), eval(res["val"]), eval(res["min"]), eval(res["max"])
				if param != "phi_b":  
					if fit == False:
						fit_params.add(param, value = constant_params[k], vary = fit)
					else:
						fit_params.add(param, value = constant_params[k], vary = fit, min = vmin, max = vmax)
				k = k+1


		#out = minimize(residual, fit_params, args=(XY_MESH,), kws={'data': vel_val},method = fit_method, nan_policy = "omit")
		#out = minimize(residual, fit_params,method = fit_method, nan_policy = "omit")
		out = minimize(residual, fit_params,method = "leastsq", nan_policy = "omit")

		best = out.params

		Vtan, Vrad, Vrot = [],[],[]
		std_Vtan, std_Vrad, std_Vrot = [],[],[]

		for iy  in range(nrings):
				Vrot.append(best["Vrot_%s"%iy].value)

				std_Vrot.append(best["Vrot_%s"%iy].stderr)


		pa, inc, x0, y0, Vsys = best["pa"].value, best["inc"].value, best["x0"].value,best["y0"].value, best["Vsys"].value

		std_pa, std_inc, std_x0, std_y0, std_Vsys = best["pa"].stderr, best["inc"].stderr, best["x0"].stderr, best["y0"].stderr, best["Vsys"].stderr



		N_free = out.nfree
		red_chi = out.redchi




		from create_2D_vlos_model import best_2d_model
		V_k = [Vrot, 0, 0] 
		vlos_2D_model = best_2d_model(vmode, shape, V_k, pa, inc, x0, y0, Vsys, rings_pos, ring_space = ring_space, pixel_scale = pixel_scale) 


		##########
		from create_2D_kin_models import bidi_models
		kin_2D_models = bidi_models(vmode, shape, V_k, pa, inc, x0, y0, Vsys, rings_pos, ring_space = ring_space, pixel_scale = pixel_scale) 
		##########


		std_Vrad,std_Vtan = std_Vrot*0, std_Vrot*0
		std_theta = 0		
		Std_errors = [std_Vrot,std_Vrad,std_pa, std_inc, std_x0, std_y0, std_Vsys, std_theta, std_Vtan]


		return vlos_2D_model, kin_2D_models, Vrot, Vsys, pa, inc , abs(x0), abs(y0) ,red_chi, N_free, Std_errors




	if vmode == "radial":
		def vmodel_dataset(pars, i,  xy_mesh,  r_0 = None, r_space = ring_space):


			parvals = pars.valuesdict()
			pa = parvals['pa']
			inc = parvals['inc']
			Vsys = parvals['Vsys']
			Vrot = parvals['Vrot_%i'% i]
			Vrad = parvals['Vrad_%i'% i]
			x0,y0 = parvals['x0'],parvals['y0']

			modl = (RADIAL_MODEL(xy_mesh,Vrot,Vrad,pa,inc,x0,y0))*weigths_w(xy_mesh,shape,pa,inc,x0,y0,r_0,r_space,pixel_scale=pixel_scale)


			return modl,Vsys

		def residual(pars, data= None):
			"""Calculate total residual for fits of VMODELS to several data sets."""

			resid = np.array([])


			# make residual per data set

			for N in range(n_annulus):

				mdl_ev = 0
				r_space_k = rings_pos[N+1] - rings_pos[N] 
				mask = np.where( (r_n >= rings_pos[N] ) & (r_n < rings_pos[N+1]) )
				x,y = XY_mesh[0][mask], XY_mesh[1][mask] 
				XY = (x,y)




				for kk in range(2):
					Vxy,Vsys = vmodel_dataset(pars, N+kk,  XY, r_0 = rings_pos[N], r_space = r_space_k)

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
						Vxy,Vsys = vmodel_dataset(pars, 0,  XY1, r_0 = 0, r_space = r_space_0)
					
						interp_model[mask1] = Vxy[1] + Vsys

				interp_model[mask] = mdl_ev + Vsys




			
			sigma = np.sqrt(e_vel_map**2 + e_ISM**2)



			convolved = 0

			if convolved == True:
				from radio_beam import Beam
				my_beam = Beam(2.5*u.arcsec, 2.5*u.arcsec, 0*u.deg)
				pix_scale = pixel_scale * u.arcsec
				gauss_kern = my_beam.as_kernel(pix_scale, x_size = nx, y_size = ny)


				extend = np.zeros((3*ny,3*nx))
				extend[ny:2*ny,nx:2*nx] = interp_model
				convolve_extend = convolve_fft(extend, gauss_kern, mask = extend == 0 )
				interp_model_conv = convolve_extend[ny:2*ny,nx:2*nx]
				interp_model_conv[interp_model == 0]  = 0


			else:
				interp_model_conv = interp_model

			interp_model[interp_model == 0]  = np.nan
			res = vel_map - interp_model_conv
			Resid = res/sigma



			return Resid.flatten()





		fit_params = Parameters()

		for iy  in range(nrings):

				fit_params.add('Vrot_%i' % (iy),value=vrot0[iy], vary = vary[0], min = -400, max = 400)
				fit_params.add('Vrad_%i' % (iy), value=vr20[iy], vary = vary[1], min = -200, max = 200)

		if config == "":

		
			fit_params.add('Vsys', value=vsys0, vary = vary[6])
			fit_params.add('pa', value=pa0, vary = vary[2], min = 0, max = 360)
			fit_params.add('inc', value=inc0, vary = vary[3], min = 0, max = 90)
			fit_params.add('x0', value=X0, vary = vary[4], min = 0, max = nx)
			fit_params.add('y0', value=Y0, vary = vary[5], min = 0, max = ny)

		else:
			k = 0
			for res,const in zip(config_file(config),constant_params):
				param, fit, val, vmin, vmax = str(res["param"]), bool(float(res["fit"])), eval(res["val"]), eval(res["min"]), eval(res["max"])
				if param != "phi_b":  
					if fit == False:
						fit_params.add(param, value = constant_params[k], vary = fit)
					else:
						fit_params.add(param, value = constant_params[k], vary = fit, min = vmin, max = vmax)
				k = k+1



		#out = minimize(residual, fit_params,method = fit_method, nan_policy = "omit")
		out = minimize(residual, fit_params,method = "leastsq", nan_policy = "omit")
		best = out.params


		Vtan, Vrad, Vrot = [],[],[]
		std_Vtan, std_Vrad, std_Vrot = [],[],[]

		for iy  in range(nrings):
				Vrot.append(best["Vrot_%s"%iy].value)
				Vrad.append(best["Vrad_%s"%iy].value)

				std_Vrot.append(best["Vrot_%s"%iy].stderr)
				std_Vrad.append(best["Vrad_%s"%iy].stderr)

		pa, inc, x0, y0, Vsys = best["pa"].value, best["inc"].value, best["x0"].value,best["y0"].value, best["Vsys"].value

		std_pa, std_inc, std_x0, std_y0, std_Vsys = best["pa"].stderr, best["inc"].stderr, best["x0"].stderr, best["y0"].stderr, best["Vsys"].stderr


		N_free = out.nfree
		red_chi = out.redchi


		from create_2D_vlos_model import best_2d_model
		V_k = [Vrot, Vrad, Vrot] 
		vlos_2D_model = best_2d_model(vmode, shape,V_k, pa, inc, x0, y0, Vsys, rings_pos, ring_space = ring_space, pixel_scale = pixel_scale) 

		##########
		from create_2D_kin_models import bidi_models
		kin_2D_models = bidi_models(vmode, shape, V_k, pa, inc, x0, y0, Vsys, rings_pos, ring_space = ring_space, pixel_scale = pixel_scale) 
		##########


		std_Vtan =  std_Vrot*0
		std_theta = 0		
		Std_errors = [std_Vrot,std_Vrad,std_pa, std_inc, std_x0, std_y0, std_Vsys, std_theta, std_Vtan]



		return vlos_2D_model, kin_2D_models, Vrot, Vrad, Vsys, pa, inc , abs(x0), abs(y0) ,red_chi, N_free, Std_errors


	if vmode == "bisymmetric":

		def vmodel_dataset(pars, i,  xy_mesh,  r_0 = None, r_space = ring_space):


			parvals = pars.valuesdict()
			pa = parvals['pa']
			inc = parvals['inc']
			Vsys = parvals['Vsys']
			Vrot = parvals['Vrot_%i'% i]
			Vrad = parvals['Vrad_%i'% i]
			Vtan = parvals['Vtan_%i'% i]
			x0,y0 = parvals['x0'],parvals['y0']


			if Vrad != 0 and Vtan != 0:
				phi_b = parvals['phi_b']

				modl = (BISYM_MODEL(xy_mesh,Vrot,Vrad,pa,inc,x0,y0,Vtan,phi_b))*weigths_w(xy_mesh,shape,pa,inc,x0,y0,r_0,r_space,pixel_scale=pixel_scale)
			else:

				modl = (BISYM_MODEL(xy_mesh,Vrot,0,pa,inc,x0,y0,0,0))*weigths_w(xy_mesh,shape,pa,inc,x0,y0,r_0,r_space,pixel_scale=pixel_scale)

			return modl,Vsys

		def residual(pars, data= None):
			"""Calculate total residual for fits of VMODELS to several data sets."""

			resid = np.array([])


			# make residual per data set

			for N in range(n_annulus):

				mdl_ev = 0
				r_space_k = rings_pos[N+1] - rings_pos[N] 
				mask = np.where( (r_n >= rings_pos[N] ) & (r_n < rings_pos[N+1]) )
				x,y = XY_mesh[0][mask], XY_mesh[1][mask] 
				XY = (x,y)




				for kk in range(2):
					Vxy,Vsys = vmodel_dataset(pars, N+kk,  XY, r_0 = rings_pos[N], r_space = r_space_k)

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
						Vxy,Vsys = vmodel_dataset(pars, 0,  XY1, r_0 = 0, r_space = r_space_0)
					
						interp_model[mask1] = Vxy[1] + Vsys

				interp_model[mask] = mdl_ev + Vsys




			
			sigma = np.sqrt(e_vel_map**2 + e_ISM**2)



			convolved = 0

			if convolved == True:
				from radio_beam import Beam
				my_beam = Beam(2.5*u.arcsec, 2.5*u.arcsec, 0*u.deg)
				pix_scale = pixel_scale * u.arcsec
				gauss_kern = my_beam.as_kernel(pix_scale, x_size = nx, y_size = ny)


				extend = np.zeros((3*ny,3*nx))
				extend[ny:2*ny,nx:2*nx] = interp_model
				convolve_extend = convolve_fft(extend, gauss_kern, mask = extend == 0 )
				interp_model_conv = convolve_extend[ny:2*ny,nx:2*nx]
				interp_model_conv[interp_model == 0]  = 0


			else:
				interp_model_conv = interp_model


			interp_model_conv[interp_model_conv == 0]  = np.nan
			res = vel_map - interp_model_conv
			Resid = res/sigma


			return Resid.flatten()



		fit_params = Parameters()

		for iy in range(nrings):
				if vr20[iy] == 0 and vt20[iy] ==0:
					vary_vrad = False
					vary_vtan = False


				else:

					vary_vrad = vary[1]
					vary_vtan = vary[7]
					fit_params.add('phi_b', value=theta0, vary = vary_theta, min = 0 , max = 180)



				fit_params.add('Vrot_%i' % (iy),value=vrot0[iy], vary = vary[0], min = -400, max = 400)
				fit_params.add('Vrad_%i' % (iy), value=vr20[iy], vary = vary_vrad,  min = -300, max = 300)
				fit_params.add('Vtan_%i' % (iy), value=vt20[iy], vary = vary_vtan, min = -300, max = 300)




		if config == "":
			fit_params.add('Vsys', value=vsys0, vary = vary[6])
			fit_params.add('pa', value=pa0, vary = vary[2], min = 0, max = 360)
			fit_params.add('inc', value=inc0, vary = vary[3], min = 0, max = 90)
			fit_params.add('x0', value=X0, vary = vary[4], min = 0, max = nx)
			fit_params.add('y0', value=Y0, vary = vary[5], min = 0, max = nx)

	
		else:
			k = 0
			for res,const in zip(config_file(config),constant_params):
				param, fit, val, vmin, vmax = str(res["param"]), bool(float(res["fit"])), eval(res["val"]), eval(res["min"]), eval(res["max"]) 
				if fit == False:
					fit_params.add(param, value = constant_params[k], vary = fit)
				else:
					fit_params.add(param, value = constant_params[k], vary = fit, min = vmin, max = vmax)

				k = k+1




		out = minimize(residual, fit_params,method = fit_method, nan_policy = "omit")
		#if fit_method == "emcee":
			#fit_params.params.add('__lnsigma', value=np.log(0.1), min=np.log(0.001), max=np.log(2))
		#	out = minimize(residual, fit_params,method = "emcee", burn = 300, steps = 1000,thin = 100, progress = False, nan_policy = "omit", is_weighted=False)


		#if fit_method == "ampgo":
		#	out = minimize(residual, fit_params,method = "ampgo", nan_policy = "omit")

		best = out.params



		Vtan, Vrad, Vrot = [],[],[]
		std_Vtan, std_Vrad, std_Vrot = [],[],[]

		for iy in range(nrings):
				Vtan.append(best["Vtan_%s"%iy].value)
				Vrot.append(best["Vrot_%s"%iy].value)
				Vrad.append(best["Vrad_%s"%iy].value)

				std_Vtan.append(best["Vtan_%s"%iy].stderr)
				std_Vrot.append(best["Vrot_%s"%iy].stderr)
				std_Vrad.append(best["Vrad_%s"%iy].stderr)


		pa, inc, x0, y0, Vsys, theta = best["pa"].value, best["inc"].value, best["x0"].value,best["y0"].value, best["Vsys"].value, best["phi_b"].value

		std_pa, std_inc, std_x0, std_y0, std_Vsys, std_theta = best["pa"].stderr, best["inc"].stderr, best["x0"].stderr, best["y0"].stderr, best["Vsys"].stderr, best["phi_b"].stderr



		N_free = out.nfree
		red_chi = out.redchi



		from create_2D_vlos_model import best_2d_model
		V_k = [Vrot, Vrad, Vtan] 
		vlos_2D_model = best_2d_model(vmode, shape,V_k, pa, inc, x0, y0, Vsys, rings_pos, ring_space = ring_space, phi_b = theta, pixel_scale = pixel_scale) 


		##########
		from create_2D_kin_models import bidi_models
		kin_2D_models = bidi_models(vmode, shape, V_k, pa, inc, x0, y0, Vsys, rings_pos, ring_space = ring_space, pixel_scale = pixel_scale) 
		##########
	
		Std_errors = [std_Vrot,std_Vrad,std_pa, std_inc, std_x0, std_y0, std_Vsys, std_theta, std_Vtan]


		return vlos_2D_model, kin_2D_models, Vrot, Vrad, Vsys,  pa, inc , x0, y0, Vtan, theta, red_chi, N_free, Std_errors







