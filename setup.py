#!/usr/bin/env python
from setuptools import setup

project_name = "phox"

setup(
    name=project_name,
    version="0.1",
    packages=[project_name],
    install_requires=[
        'numpy>=1.19',
        'scipy>=1.7.1',
        'dphox',
        'simphox',
        'xarray'
    ],
)
