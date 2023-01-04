#!/usr/bin/env python3
import os
from setuptools import setup, find_packages


def readme():
   with open("README.rst") as f:
       return f.read()



with open("requirements.txt", "r") as r:
   filtered_lines = filter(lambda line: not line.startswith("#"), r.readlines())
   requirements = list(map(lambda s: s.replace("\n", ""), filtered_lines))
   print(requirements)


all_packages = find_packages()
setup(
    name="XookSuut",
    version="3.2.0",
    description="A Python tool for modeling non-ciruclar motions on 2D velocity maps",
    long_description=readme(),
    keywords="kinematics",
    url="https://github.com/CarlosCoba/XookSuut-code.git",
    author="C. Lopez-Coba",
    author_email="calopez@asiaa.sinica.edu.tw",
	include_package_data=True,
    license="MIT",
    packages=all_packages,
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'XookSuut=src.XookSuut_inputs:input_params'
        ]
    }
)
