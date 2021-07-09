import numpy as np
import matplotlib.pylab as plt
 
def Rings(xy_mesh,pa,inc,x0,y0,pixel_scale=1):
	(x,y) = xy_mesh


	X = (- (x-x0)*np.sin(pa) + (y-y0)*np.cos(pa))
	Y = (- (x-x0)*np.cos(pa) - (y-y0)*np.sin(pa))

	R= np.sqrt(X**2+(Y/np.cos(inc))**2)
	R = R*pixel_scale


	return R




def weigths_w(xy_mesh,shape,pa,inc,x0,y0,r_0,ring_space,pixel_scale):
	pa, inc  = pa*np.pi/180, inc*np.pi/180  
	r_n = Rings(xy_mesh,pa,inc,x0,y0,pixel_scale)
	a_k = r_0

	a_k_plus_1 = a_k + ring_space


	w_k_n = (1 - (r_n -a_k)/ring_space)
	w_k_plus_1_n = (r_n -a_k)/ring_space


	return np.ravel(w_k_n),np.ravel(w_k_plus_1_n) 

