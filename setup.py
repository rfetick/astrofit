# -*- coding: utf-8 -*-
"""
Created on Tue Jun 05 14:13:40 2018

@author: rfetick
"""

from setuptools import setup, find_packages
 
setup(name='astrofit',
      version='1.1',
      url='unknown',
      license='GNU-GPL v3.0',
      author='Romain JL Fetick',
      description='Functions and tools for optics and fitting.',
      packages=find_packages(exclude=['tests']),
      zip_safe=False,
      )
