#!/usr/bin/env python
from setuptools import setup, find_packages

project_name = "phox"

setup(
    name=project_name,
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.19',
        'scipy>=1.7.1',
        'dphox>=0.0.1a4',
        'simphox>=0.0.1a4',
        'holoviews'
    ],
)
