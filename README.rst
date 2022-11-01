@c


*******************************************
XookSuut version 2.0.0
*******************************************

|logo|



====

:Authors: Carlos Lopez Coba, LiHwai Lin, Sebastian F. Sanchez
:Contact: carlos.lopezcoba@gmail.com




Description
===========
XookSuut or XS, is a python tool developed to model non-circular motions on 2D velocity maps,
such as those obtained from Integral Field Spectroscopy data, i.e., stellar and ionized-gas velocity maps. However XS can run
in any velocity map whose representation is in 2D. 
XS is able to model circular rotation models, axissymietric radial flows, bisymmetric flows, and a general harmonic decomposition of the LOSV.
In order to derive the best set of parameters for each kinematic model XS performs a Least-Square analysis, using all spaxels from the input velocity
map; thus, large images could take time larger times for deriving the best model. 
In addition, XS is able to derive the best set of parameters by sampling their posterior distribution by implementing Markov-Chain Monte Carlo methods.
In this way each parameter is obtained from marginalizing their individual posterior distribution.
Note that these are different approaches and could give different results ! 




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

XS is designed to run in command line, but you can easely set-up a python script .py for you can run it as a .py,
this could be very usefull in case you want to analyse several objects at the same time !
Please read the run_example.txt file to see how to run XS.


Examples
===========
Following are some of the outputs you can obtain from running XS on a velocity map. These examples correpond to a simulated velocity map with an oval distortion.

Results for circular rotation model:
|circ|


Radial flow model:
|rad|

Bisymmetric model:
|bis|

Corner plot of constant parameters
|corner|

V2r chain
|chain_rad|

V2t chain
|chain_tan|

Harmonic expasion with m = 2
|hrm|



Referencing XookSuut
=================

If you are using XS in your work, please cite the XS release paper (), and dont forget citing DiskFit (Spekkens & Sellwood 2007) and RESWRI (Schoenmakers et al. 1997)
where the models are inspired.
Also, if you use the XS colormap in a different context, I would appreciate it, if you include XS in the acknowledgment section.



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

.. |corner| image:: corner.bisymmetric_model.example.png
    :scale: 20 %
    :target: https://github.com/CarlosCoba/XookSuut-code

.. |chain_rad| image:: chain_progress.rad.bisymmetric_model.example.png
    :scale: 20 %
    :target: https://github.com/CarlosCoba/XookSuut-code

.. |chain_tan| image:: chain_progress.tan.bisymmetric_model.example.png
    :scale: 20 %
    :target: https://github.com/CarlosCoba/XookSuut-code

.. |hrm| image:: kin_hrm_2_model_example.png
    :scale: 20 %
    :target: https://github.com/CarlosCoba/XookSuut-code

