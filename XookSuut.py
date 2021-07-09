#! /usr/bin/python3
import sys
nargs=len(sys.argv)
from initialize_XS import rotcur

"""
#################################################################################
# 				The XookSuut-code.				#
#				version 21.04.11				#
# 				C. Lopez-Coba					#
#################################################################################


INPUT:
object = 	Object name.
vel_map = 	2D velocity map in km/s.
evel_map = 	2D error map in km/s. If no error map is passed then set this to "".
SN = 		S/N cut applied on the velocity map. 
		Better results are obtained when SN has low values, 15-20 km/s.
		if evel_map is set to 0, then the whole velocity map will be used for estimating the rotation curve.
VSYS	=	Guess value for Vsys. If VSYS is set to "", then it will be used the central value as guess (i.e., VSYS =  vel_map[Y0][X0]).
PA	= 	Guess value for the position angle of the major axis.		
INC	=	Guess value for the inclination.
X0,Y0	=	Guess values for the kinematic center. These are in pixel coordinates. 
N_it 	=	Number of iteratios to find the best parameters. 
pixel_scale	Size of the pixel in arcsecs.
vary_PA =	Boolean.
vary_INC =	Boolean.
vary_XC =	Boolean.
vary_YC =	Boolean.
vary_VSYS =	Boolean.
vary_PHI_BAR =	Boolean. Position angle of the bisimetric distortion on the galaxy plane.
delta =		Width of the rings (2*delta) in the tabulated model. If delta is set to "", then delta = 0.5*ring_space
ring_space =	Spacing between rings used to create the interpolated model. 
rstart	=	Initial radius in the interpolated model. 
rfinal	=	Final radius in the interpolated model.
frac_pixel = 	Fraction of total pixels within a ring required to compute the tabulated model. If frac_pixel = 1, 
		then 100% of the pixels within a ring are required to be filled with data to compute the tabulated model.

r_bar_min,max= 	Maximum and minimum length of the bisymmetric perturbation. If only a value is passed then it'll be considered as r_bar_max.
model	= 	The different kinematic models. You must choose between: circular motion (circular), radial flows (radial), or barlike flows (bisymmetric).
errors	=	Boolean. If you want to compute errors for the derived parameters using a Metropolis-Hastings analysis.
survey =	String. If the object belongs to a specific galaxy survey.
config =	configure file to pass initial guess and constrains for the constant params (x0, y0, Vsys, pa, inc, phi_bar). This file is optional.
		If this file is passed to XookSuut, then it will ommit the previos guess values (VSYS, PA, INC, X0, Y0) as well as the VARY_ entrance.
save_plots =	boolean
e_ISM =		Error in the emission line centroid.
steps =		Number of steps for Metropolis-Hastings analysis
method =	Fitting method for minimization. Choose between <Powell> or <leastsq>
use_mh = 	Boolean. If 1 then the outputs of the Metropolis-Hastings analysis are used as best values. If 0 the outputs from the chi square minimization using the 
		specified fitting method are used as best values.
prefix	=	Prefix to pass to the object's name.

A configuration file has the follow entrance

#
#
# XS config file
# Example of configuration file
# param col. are the constant parameters to fit. Do not touch this column.
# val col. corresponds to the initial values for the considered parameters
# fit col. is a boolen. If set to 1 means the parameter is fitted, other wise set 0.
# min col. is the minimum value for the considered parameter. If fit is seto to 0, the min value is not taken into account. 
# max col. is the maximum value for the considered parameter. If fit is seto to 0, the max value is not taken into account. 
#
#
#
param	val	fit	min	max
pa	35	1	0	360
inc	35	1	0	90
x0	25	1	0	50
y0	25	1	0	50
Vsys	11168.266579019075	0	0	3e6
phi_b	45	1	0	180





					OUTPUT:


-Tables:

*Table containing the best fitted values for the constant parameters (x0, y0, Vsys, pa, inc, phi_bar)
ana_kin_model.csv

*Table containing different estimations of the maximum circular velocity.
vmax_out_params.bisymmetric_model.csv


* fits files
2D array of the LoV of the best kinematic model
1D array of the different kinematic models as function of the deprojected radius


* plots
plot of the best 2D kinematic models
plot of the asymptotic velocity estimated with Vt 



************************************************
************    ADDITIONAL *********************
************************************************

The following directories must be created in the working directory.
./plots
./models
./vmax_rturn
"""


if __name__ == "__main__":


	if (nargs == 34 ):
		galaxy = sys.argv[1]
		vel_map = sys.argv[2]
		evel_map = sys.argv[3]
		SN = float(sys.argv[4])
		VSYS = sys.argv[5]
		PA = float(sys.argv[6])
		INC = float(sys.argv[7])
		X0 = float(sys.argv[8])
		Y0 = float(sys.argv[9])
		n_it = sys.argv[10]
		pixel_scale = float(sys.argv[11])
		vary_PA,vary_INC,vary_XC,vary_YC,vary_VSYS, vary_PHI = bool(float(sys.argv[12])), bool(float(sys.argv[13])), bool(float(sys.argv[14])), bool(float(sys.argv[15])), 			bool(float(sys.argv[16])), bool(float(sys.argv[17]))
		ring_space = float(sys.argv[18])
		delta = sys.argv[19]


		rstart = float(sys.argv[20])
		rfinal = float(sys.argv[21])

		frac_pixel = eval(sys.argv[22])
		r_bar_min_max = eval(sys.argv[23])
		vmode = sys.argv[24]

		save_plots = bool(float(sys.argv[25]))
		errors = bool(float(sys.argv[26]))
		survey = sys.argv[27]
		config = sys.argv[28]
		e_ISM = sys.argv[29]
		steps = sys.argv[30]

		method = sys.argv[31]
		use_metropolis = bool(float(sys.argv[32]))
		prefix = sys.argv[33]

		if VSYS != "": VSYS = eval(sys.argv[5])
		if n_it == "": n_it = 5
		if n_it != "": n_it = int(float(n_it))
		if delta == "":
			delta = ring_space/2. 
		else:
			delta = float(delta)



		if pixel_scale == "": pixe_scale = 1
		if rstart == "": rstart = 2.5
		if rfinal == "": rfinal = 40
		if frac_pixel == "": frac_pixel = 2/3.
		if errors == "": errors = bool(1)
		if survey == "": survey = ""
		if config == "": config = ""
		if save_plots == "": save_plots = 1
		if e_ISM == "": e_ISM = 5
		if e_ISM != "": e_ISM = float(e_ISM)
		if use_metropolis == "" : use_metropolis = 0 

		if steps == "": steps = 1e4
		if steps != "": steps = int(float(steps))

		if type(r_bar_min_max)  == tuple:
			bar_min_max = [r_bar_min_max[0], r_bar_min_max[1] ]
		else:

			bar_min_max = [rstart, r_bar_min_max ]

		if prefix != "": galaxy = "%s-%s"%(galaxy,prefix)			


		#if delta != "": delta = float(delta)
		#if ring_space != "": ring_space = float(ring_space)

		rotcur(galaxy, vel_map, evel_map, SN, VSYS, PA, INC, X0, Y0, n_it, pixel_scale, vary_PA, vary_INC, vary_XC, vary_YC, vary_VSYS, vary_PHI, delta, rstart, rfinal, ring_space, frac_pixel, 		bar_min_max,vmode, save_plots, errors, survey,config,e_ISM, steps, method , use_metropolis)


	else:
		print ("USE: XookSuut.py object vel_map [evel_map] SN [VSYS] PA INC X0 Y0 [N_it=5] [pixel_scale=1] vary_PA[0,1] vary_INC[0,1] vary_XC[0,1] vary_YC[0,1] vary_VSYS[0,1] vary_PHI_bar[0,1] ring_space [Delta] [Rstart=2.5] [Rfinal=40] [frac_pixel=2/3.] [R_bar_min,R_bar_max] [model=cicular,radial,bisymmetric] [save_plots=1] [errors=0] [survey] [config_file] [e_ISM=5] [steps = 1e4] [method = Powell, leastsq] [use_mh = 0] [prefix]" )

		exit()



"""


#
# RUNNING THE CODE:
#
#
# EXAMPLE
#

XookSuut.py test_galaxy test.fits ""  20 "" 200 50 27 27 5 1 1 1 1 1 1 1 2.5 "" 3 46 2/3. 3,20 bisymmetric 1 0 test "" 5 1e4 leastsq 1 ""



"""





