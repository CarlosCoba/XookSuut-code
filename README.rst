


*******************************************
XookSuut version 1.0.0
*******************************************

|logo|



====

:Authors: Carlos Lopez Coba, LiHwai Lin, Sebastian F. Sanchez
:Contact: carlos.lopezcoba@gmail.com




Description
===========
XookSuut is a python tool developed to model non-circular motions over 2D velocity maps. 




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

USE: XookSuut name vel_map.fits [error_map.fits,SN] pixel_scale PA INC X0 Y0 [VSYS] vary_PA vary_INC vary_X0 vary_Y0 vary_VSYS ring_space [delta] Rstart,Rfinal frac_pixel inner_interp  kin_model fit_method N_it [R_bar_min,R_bar_max] [errors] [stepsize_mcmc] [use_mcmc_res=0,1] [plot_chain_mcmc=0,1] [e_ISM] [survey] [config_file] [prefix]



Uninstall
===========

pip uninstall XookSuut




Referencing XookSuut
=================

If you are using this software in a work, please cite this paper:



.. |logo| image:: logo.png
    :scale: 25 %
    :target: https://github.com/CarlosCoba/XookSuut-code



