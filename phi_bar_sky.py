import numpy as np


def pa_bar_sky(pa_disk,inc,phi_b):
	pa_disk,inc,phi_b = pa_disk*np.pi/180,inc*np.pi/180,phi_b*np.pi/180
	phi_bar = pa_disk + np.arctan(np.tan(phi_b)*np.cos(inc))
	return phi_bar*180/np.pi
