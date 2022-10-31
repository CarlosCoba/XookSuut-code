import os
from os import path

current_directory = "."
path_models = "./models/"
path_plots = "./plots/"


if path.exists(path_plots) == False:
	os.mkdir(path_plots)

if path.exists(path_models) == False:
	os.mkdir(path_models)



