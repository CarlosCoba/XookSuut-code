import numpy as np
import matplotlib.pylab as plt

from lmfit import Model
from lmfit import Parameters, fit_report, minimize
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from lmfit import Model

import scipy.optimize as optimization

"""
def vrot(r_sky,v_max,R_turn,alpha=2):
#def vrot(r_sky,v_max,R_turn):
	#v=v_max*r_sky/(R_turn**alpha+r_sky**alpha)**(1./alpha)

	R=r_sky/R_turn
	v=(2/np.pi)*v_max*np.arctan(R)

	return v
"""



def vrot(r_sky,v_max,R_turn,beta,gamma):

		x=R_turn/r_sky
		A = (1+x)**beta
		B = (1+x**gamma)**(1./gamma)
		v=v_max*A/B

		return v


def vrot2(r_sky,v_max,R_turn,gamma):
		beta = 0
		x=R_turn/r_sky
		A = (1+x)**beta
		B = (1+x**gamma)**(1./gamma)
		v=v_max*A/B
		return v




def Rturn_Vmax(r_array,v_array,negative = False):
			r_array,v_array = abs(r_array), abs(v_array)
			r_array = np.append(r_array,0.1)
			v_array = np.append(v_array,5)
			v0 = np.nanmedian(v_array)

			try:
				#x0=[250,5,0,1]
				#vmax ,r_turn,beta,gamma = optimization.curve_fit(vrot, r_array, v_array, p0=x0, bounds = ([0,0,0,0],[500,60,1,50]))[0]

				mod = Model(vrot)
				mod.set_param_hint('v_max', value=v0, vmin = 0, vmax = 360)
				mod.set_param_hint('beta', value=0, vary = False, min=0, max=1.0)
				mod.set_param_hint('R_turn', value=5, min=0, max=35)
				mod.set_param_hint('gamma', value=1, min=-2, max=12)
				pars = mod.make_params()

				result = mod.fit(v_array, r_sky = r_array)#, fit_method = "Powell")
				best = result.best_values
				vmax ,r_turn,gamma,beta = best["v_max"],best["R_turn"],best["gamma"],best["beta"]


			except (RuntimeError,TypeError,ValueError):
					vmax,r_turn,gamma,beta = 500,0,1,1

			return vmax, r_turn,beta,gamma




