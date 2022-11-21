import numpy as np

def pa_bar_sky(pa_disk,inc,phi_b):
	pa_disk,inc,phi_b = pa_disk*np.pi/180,inc*np.pi/180,phi_b*np.pi/180
	phi_bar = pa_disk + np.arctan(np.tan(phi_b)*np.cos(inc))
	return phi_bar*180/np.pi




#
# Propagate errors
#
def error_pa_bar_sky(pa_disk,inc,phi_b, e_pa_disk,e_inc,e_phi_b):

	if e_pa_disk*e_inc*e_phi_b != 0:
		pa_disk,inc,phi_b = pa_disk*np.pi/180,inc*np.pi/180,phi_b*np.pi/180
		e_pa_disk,e_inc,e_phi_b = e_pa_disk*np.pi/180,e_inc*np.pi/180,e_phi_b*np.pi/180

		#Partial derivatives
		# phi_bar = pa_disk + np.arctan(np.tan(phi_b)*np.cos(inc))

		d_pa_disk = 1.
		d_phi_b = (1./np.cos(phi_b))**2*np.cos(inc) / (1 + np.tan(phi_b)**2*np.cos(inc)**2 )
		d_inc = -np.tan(phi_b)*np.cos(inc) / ( 1 + np.tan(phi_b)**2*np.cos(inc)**2  )

		e_pa_bar_sky = (e_pa_disk*d_pa_disk)**2 + (e_phi_b*d_phi_b)**2 + (e_inc*d_inc)**2

		return np.sqrt(e_pa_bar_sky)*180/np.pi
	else:
		return 0
