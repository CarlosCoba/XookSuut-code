import numpy as np
#import matplotlib.pylab as plt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def colorbar(im,axis,orientation="vertical",font_size=8,colormap="rainbow",ticks='None',label=None,bbox=(1, 0.0, 1, 1),width="5%",height="100%",label_pad=0):
	cax1 = inset_axes(axis,
		width=width, # width = 5% of parent_bbox width
		#height="33%", # height : 100%
		loc=3,
		#bbox_to_anchor=(0.75, 0.05, 1, 1),
		height=height,
		bbox_to_anchor=bbox,
		bbox_transform=axis.transAxes,
		borderpad=0
		)
	if ticks == 'None':
		cbar1=plt.colorbar(im,cax=cax1,orientation=orientation,cmap=colormap)
	else:
		cbar1=plt.colorbar(im,cax=cax1,orientation=orientation,cmap=colormap,ticks=ticks)

	if label != None:
		if orientation == "vertical":
			rot = 90
		else:
			rot = 0
		cbar1.set_label(fontsize=font_size,label=label,rotation=rot,labelpad=label_pad)
		#cbar1.ax.get_yaxis().labelpad = label_pad

	cbar1.ax.tick_params(labelsize=5)
	cbar1.ax.tick_params(axis='y', direction='in',rotation=90 )
	cax1.tick_params(direction='in', pad = 1, width = 0.5)

	cbar1.outline.set_edgecolor("k")
	cbar1.outline.set_linewidth(0.2)



	return cax1

