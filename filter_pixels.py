import numpy as np
from scipy.signal import convolve2d
import matplotlib.pylab as plt
from astropy.io import fits





def filter_SN(vel_map,evel_map, SN):

	evel_map_copy = np.copy(evel_map)
	evel_map[evel_map < SN ] = 0
	evel_map[evel_map > SN ] = 1

	evel_map = np.asarray(evel_map, dtype = bool)


	kernel = np.ones((3,3))
	kernel[1,1] = 0
	mask = convolve2d(evel_map, kernel, mode='same', fillvalue=1)

	result = evel_map.copy()
	result[np.logical_and(mask>3, evel_map==0)] = 1


	res = ~result
	#res[res == 0] = np.nan

	#plt.imshow(res*vel_map, cmap = "jet")
	#plt.show()

	return res*vel_map



