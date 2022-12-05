import numpy as np
from src.pixel_params import Rings
import random


def Rings_r_4(R,vel,ring,delta=0):
	[ny,nx]= vel.shape
	M=np.ones((ny,nx))

	if ring == 0:
		mask = (R>=ring) & (R <= ring+delta)
	else:
		mask = (R>=ring-0.5*delta) & (R <= ring+0.5*delta)		
	
	s = M*mask
	return s




def ring_pixels(r_n,ring,delta):

	a_k = ring
	mask1 = np.where( (r_n >= a_k - delta) & (r_n < a_k + delta) ) 
	mask2 = np.where( (r_n <= a_k) ) 


	return mask1, mask2





def resampling(velmap,error,rings,delta,pa,inc,xc,yc,pixel_scale,seed=10):		

	pa,inc=pa*np.pi/180,inc*np.pi/180
	[ny,nx] = velmap.shape
	X = np.arange(0, nx, 1)
	Y = np.arange(0, ny, 1)
	XY_mesh = np.meshgrid(X,Y)

	r_n = Rings(XY_mesh,pa,inc,xc,yc,pixel_scale)
	velmap_c = np.copy(velmap)

	pnrg = np.random.RandomState(seed)
	k = 0
	for ring in rings:
		masks_pix = ring_pixels(r_n,ring,delta)
		mask_pix = masks_pix[0]
		mask_pix1 = masks_pix[1]
		if k == 0:
			mask_pix = mask_pix1
			k = k + 1

		error_pix = error[mask_pix]
		n_errors = len(error_pix)
		not_nan_errors = error_pix[np.isfinite(error_pix)]
		n_data = len(not_nan_errors)
		try:
			random_errors = pnrg.choice(not_nan_errors, size = n_errors, replace = True)
		except(ValueError):
			random_errors = 0
		velmap_c[mask_pix] = velmap_c[mask_pix] + random_errors 
		#M[mask_pix] = new_vals

	return velmap_c








