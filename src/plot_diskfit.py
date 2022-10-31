import numpy as np
import matplotlib.pylab as plt
from astropy.io import fits
from clc_modules.fig_params import fig_ambient
from clc_modules.axes_params import axes_ambient as axss


rd = np.genfromtxt("diskfit.out", usecols = 0, dtype = float, delimiter = ",")
vtd = np.genfromtxt("diskfit.out", usecols = 2, dtype = float, delimiter = ",")
v2td = np.genfromtxt("diskfit.out", usecols = 6, dtype = float, delimiter = ",")
v2rd = np.genfromtxt("diskfit.out", usecols = 8, dtype = float, delimiter = ",")


data = fits.getdata("./models/test-test.bisymmetric.1D_model.fits.gz")
r = data[0]
vt = data[1]
vrad = data[2]
vtan = data[3]


rd = rd*0.2


fig,axs = fig_ambient(fig_size = (2,1.5), ncol = 1, nrow = 1, top = 0.9 )
ax = axs[0]
axss(ax)

ax.scatter(rd, vtd, s = 5, zorder = 10, facecolors='none', edgecolors='k', lw = 0.5, label = "$\mathrm{DiskFit}$")
ax.scatter(rd, v2td, s = 5, zorder = 10, facecolors='none', edgecolors='k', lw = 0.5)
ax.scatter(rd, v2rd, s = 5, zorder = 10, facecolors='none', edgecolors='k', lw = 0.5)


#ax3.plot(R,Vrot, color = "#362a1b",linestyle='-', alpha = 1, linewidth=0.8, label = "$\mathrm{V_{t}}$")


#ax.scatter(r, vt, s = 2, c="k")
ax.plot(r, vt, color = "#362a1b",linestyle='-', alpha = 1, linewidth=0.8, label = "$\mathrm{V_{t}}$")
#ax.scatter(r, vrad, s = 2, c="k")
ax.plot(r,vrad, color = "#c73412",linestyle='-', alpha = 0.6, linewidth=0.8, label = "$\mathrm{V_{2,r}}$")
#ax.scatter(r, vtan, s = 2, c="k")
ax.plot(r,vtan, color = "#2fa7ce",linestyle='-', alpha = 1, linewidth=0.8, label = "$\mathrm{V_{2,t}}$")




max_vel,min_vel = int(np.nanmax(vt)),int(np.nanmin(vt)) 
min_vel = abs(min_vel)

ax.legend(loc = "center", fontsize = 4.5, bbox_to_anchor = (0, 0.95, 1, 0.2), ncol = 4, frameon = False)
ax.set_ylim(-5,50*(max_vel//50)+80)
ax.plot([0,np.nanmax(rd)],[0,0],color = "k",linestyle='-', alpha = 0.6,linewidth = 0.3)
ax.set_xlabel('$\mathrm{r~(arcsec)}$',fontsize=8,labelpad = 0)
ax.set_ylabel('$\mathrm{V_{rot}~(km~s^{-1})}$',fontsize=8,labelpad = 0)
plt.savefig("DiskFit", dpi = 300)

plt.show()
