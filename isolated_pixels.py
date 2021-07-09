import numpy as np
from scipy import ndimage
import matplotlib.pylab as plt

def filter_isolated_cells(image, struct):
    """ Return array with completely isolated single cells removed
    :param array: Array with completely isolated single cells
    :param struct: Structure array for generating unique regions
    :return: Array with minimum region size > 1
    """

    array= np.divide(image,image)
    array[~np.isfinite(array)] = 0
    #struc = np.array([[0,1,0],[1,1,1],[0,1,0]])

    filtered_array = np.copy(array)
    id_regions, num_ids = ndimage.label(filtered_array, structure=struct)
    id_sizes = np.array(ndimage.sum(array, id_regions, range(num_ids + 1)))
    area_mask = (id_sizes == 1)
    filtered_array[area_mask[id_regions]] = 0

    filter_image=image*filtered_array
    filter_image[filter_image==0] = np.nan


    return filter_image

# Run function on sample array
#filtered_array = filter_isolated_cells(square, struct=np.ones((3,3)))
