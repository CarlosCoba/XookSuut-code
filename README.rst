

*******************************************
XookSuut (XS)
*******************************************

|logo|



====

:Authors: Carlos Lopez Coba, LiHwai Lin, Sebastian F. Sanchez
:Contact: carlos.lopezcoba@gmail.com




Description
===========
XookSuut or XS for short, is a python tool developed to model non-circular motions on 2D velocity maps,
such as those obtained from Integral Field Spectroscopy data, i.e., stellar and ionized-gas velocity maps; even though  XS can run
on any velocity map whose representation is in 2D maps.
In order to derive the best kinematic model XS performs a Least-Squares analysis together with Bayesian inference methods to sample
the propability space given by the posterior distribution (~likelihood*priors), i.e. MCMC, or by sampling the volume space occupied
by the priors, i.e., nested sampling.
XS adopts the most popular Monte Carlo Chain Marcov (MCMC) packages emcee & zeus, and dynesty for dynamic nested sampling, to infer the best fit parameters from the
considered kinematic model.
XS is able to model circular rotation models, axisymmetric radial flows, bisymmetric flows, and a general harmonic decomposition of the LOSV.
To derive the best set of parameters on each kinematic model XS uses most of the pixels from the input velocity
map; thus, large dimension images could take time large CPU time for deriving the best model. 
Note: In general Monte Carlo sampling can be computational more expensive than simple Least-Square analyses; therefore it is
recomended to perform this analysis using multiple CPUs (adecuated within XS).
The novelty of XS resides in its ability to compute kinematic models by sampling the posterior distribution, or by intrating the so-called evidence.

Dependencies
===========

            * ::
            
                Python >= 3.6


Installation
===========

1. Go to the XookSuut-code directory
cd /XookSuut-code/

2.  pip install -e .
-e stands for editable, meaning that you will be able to work on the script and invoke the latest version without need to reinstall.

3. Try it. Go to any folder and type XookSuut

you must get the following ::

USE: XookSuut name vel_map.fits [error_map.fits,SN] pixel_scale PA INC X0 Y0 [VSYS] vary_PA vary_INC vary_X0 vary_Y0 vary_VSYS ring_space [delta] Rstart,Rfinal cover kin_model fit_method N_it [R_bar_min,R_bar_max] [config_file] [prefix]


Uninstall
===========

pip uninstall XookSuut


Use
===========

XS is designed to run in command line, although you can easely set-up a python script .py so you can run it as a script.
This last could be very usefull in case you want to analyse several objects in parallel !
Please read the run_example.txt file to see how to run XS.
XS requires a 2D velocity map in (km/s) and optionally a map containing the uncertainties on the velocities.
In addition, XS requires guess values for the disk geometry, and parameters describing the rings position on the galaxy plane.
Another input is the desired kinematic ic model, and in the case of non-circular models, the maximum radial extention of the non-circular flows.


Examples
===========
Following are some of the outputs you will obtain from running XS on a velocity map. These examples correpond to a simulated velocity map with an oval distortion.

Results for circular rotation model:
|circ|

Radial flow model:
|rad|

Bisymmetric model:
|bis|

Following are shown the corner plots, or the individual marginalization distributons for the different parameters describing the 
previous bisymmetric model. This particular model contains 40 independent variables !.

Corner plot showing the constant parameters (i.e, PA, INC, XC, YC, VSYS, PA_BAR, LNSIGMA2):
|corner_const|

Corner plot for the circular velocities vt (km/s):
|corner_vt|

Corner plot for the bisymmetric component v2r (km/s):
|corner_v2r|

Corner plot for the bisymmetric component v2t (km/s):
|corner_v2t|

Harmonic expasion with harmonic number m  = 2
|hrm|



Referencing XookSuut
=================
 
If you are using XS in your work, please cite the XS release paper ().
In addition to that, XS is influenced by the DiskFit (Spekkens & Sellwood 2007) and RESWRI (Schoenmakers et al. 1997) packages 
since it includes kinematic models from these two codes, so don't forget to mention them in your work.
XS also relies on the following MCMC packages, emcee from (Foreman-Mackey+2013) and Zeus (Karamanis+2021,2022); and Dynesty (Speagle 2020, Koposov +2022) for nested sampling.
Also, if you use the XS colormap (red-black-blue) in a different context, I would appreciate it, if you include XS in the acknowledgment section.


.. |logo| image:: logo.png
    :scale: 20 %
    :target: https://github.com/CarlosCoba/XookSuut-code


.. |circ| image:: kin_circular_model_example.png
    :scale: 20 %
    :target: https://github.com/CarlosCoba/XookSuut-code


.. |rad| image:: kin_radial_model_example.png
    :scale: 20 %
    :target: https://github.com/CarlosCoba/XookSuut-code


.. |bis| image:: kin_bisymmetric_model_example.png
    :scale: 20 %
    :target: https://github.com/CarlosCoba/XookSuut-code

.. |corner_const| image:: multicorner.png
    :scale: 10 %
    :target: https://github.com/CarlosCoba/XookSuut-code

.. |corner_vt| image:: multicorner_vt.png
    :scale: 20 %
    :target: https://github.com/CarlosCoba/XookSuut-code

.. |corner_v2r| image:: multicorner_v2r.png
    :scale: 20 %
    :target: https://github.com/CarlosCoba/XookSuut-code

.. |corner_v2t| image:: multicorner_v2t.png
    :scale: 20 %
    :target: https://github.com/CarlosCoba/XookSuut-code

.. |hrm| image:: kin_hrm_2_model_example.png
    :scale: 20 %
    :target: https://github.com/CarlosCoba/XookSuut-code

