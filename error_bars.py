import matplotlib.pylab as plt
import numpy as np

def error_bar(axis,x,y, e_y_l, e_y_u, color):
	x,y, e_y_l, e_y_u = np.asarray(x),np.asarray(y),np.asarray(e_y_l),np.asarray(e_y_u)
	e_y_l = abs(e_y_l)
	e_y_u = abs(e_y_u)
	#min_error = 

	axis.bar(x,2 + 2, bottom=y-2, width=.15, color= color, zorder=0, align='center', edgecolor  = "k", linewidth  = 0.3)
#ax.errorbar(x, y, yerr=2 * (top_err + bot_err), marker='o', linestyle='none')
