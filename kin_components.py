import numpy as np
import numpy as np
import matplotlib.pylab as plt
import scipy
import sys
import lmfit
from lmfit import Model
from lmfit import Parameters, fit_report, minimize
from matplotlib.gridspec import GridSpec
from weights_interp import weigths_w
 
def Rings(xy_mesh,pa,inc,x0,y0):
	(x,y) = xy_mesh

	X = (- (x-x0)*np.sin(pa) + (y-y0)*np.cos(pa))
	Y = (- (x-x0)*np.cos(pa) - (y-y0)*np.sin(pa))

	R= np.sqrt(X**2+(Y/np.cos(inc))**2)

	return R




def CIRC_MODEL(xy_mesh,Vrot,pa,inc,x0,y0):
	(x,y) = xy_mesh
	pa,inc=(pa)*np.pi/180,inc*np.pi/180
	R  = Rings(xy_mesh,pa,inc,x0,y0)
	cos_tetha = (- (x-x0)*np.sin(pa) + (y-y0)*np.cos(pa))/R
	sin_tetha = (- (x-x0)*np.cos(pa) - (y-y0)*np.sin(pa))/(np.cos(inc)*R)
	vlos = np.sin(inc)*(Vrot*cos_tetha)
	return np.ravel(vlos)


def RADIAL_MODEL(xy_mesh,Vrot,Vr2,pa,inc,x0,y0):
	(x,y) = xy_mesh
	pa,inc=(pa)*np.pi/180,inc*np.pi/180
	R  = Rings(xy_mesh,pa,inc,x0,y0)
	cos_tetha = (- (x-x0)*np.sin(pa) + (y-y0)*np.cos(pa))/R
	sin_tetha = (- (x-x0)*np.cos(pa) - (y-y0)*np.sin(pa))/(np.cos(inc)*R)
	vlos = np.sin(inc)*(Vrot*cos_tetha + Vr2*sin_tetha)
	return np.ravel(vlos)


def BISYM_MODEL(xy_mesh,Vrot,Vrad,pa,inc,x0,y0,Vtan,phi_b):
	(x,y) = xy_mesh
	pa,inc,phi_b=(pa)*np.pi/180,inc*np.pi/180,phi_b*np.pi/180
	R  = Rings(xy_mesh,pa,inc,x0,y0)
	cos_tetha = (- (x-x0)*np.sin(pa) + (y-y0)*np.cos(pa))/R
	sin_tetha = (- (x-x0)*np.cos(pa) - (y-y0)*np.sin(pa))/(np.cos(inc)*R)

	m = 2
	theta = np.arctan(sin_tetha/cos_tetha)
	#phi_b = np.arctan(np.tan(phi_b-pa)/np.cos(inc))

	phi_b = phi_b
	theta_b = theta - phi_b
	

	vlos = np.sin(inc)*((Vrot*cos_tetha) - Vtan*np.cos(m*theta_b)*cos_tetha - Vrad*np.sin(m*theta_b)*sin_tetha)

	return np.ravel(vlos)


