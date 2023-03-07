#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="jupyter_image_diff",
    version="0.1.0",
    description="Fast comparison of Numpy array images in Jupyter notebooks",
    author="Magnus Dalen Kvalevåg",
    author_email="magnus.kvalevag@ntnu.no",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "ipywidgets==7.7.2",  # Fixes a bug that crashes VS Code notebooks
        "ipycanvas",
    ],
)
