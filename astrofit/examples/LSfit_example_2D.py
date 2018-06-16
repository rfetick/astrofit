# -*- coding: utf-8 -*-
"""
Created on Thu Jan 04 17:51:47 2018

Purpose: Learn how to use LSfit2D

@author: rfetick
"""

# IMPORT
from numpy import mgrid, zeros
from numpy.random import rand
from pylab import title, pcolor, axis, clf
from matplotlib.pyplot import figure

from astrofit import gauss2D, LSfit2D

###############################################################################
###         Print documentation
###############################################################################

print(LSfit2D.__doc__)


###############################################################################
###         Fitting of a 2D gaussian function
###############################################################################


# Initialize
Npoints = 50
[X,Y] = mgrid[0:Npoints,0:Npoints]

paramTrue = zeros(7)
paramTrue[1] = 1 #amplitude
paramTrue[2] = 10 #sigma_X
paramTrue[3] = 10 #sigma_Y
paramTrue[4] = Npoints/2 #center_X
paramTrue[5] = Npoints/2 #center_Y

# Create true gauss2D
Gtrue = gauss2D(X,Y,paramTrue)

# Generate noisy data
noiseMag = 1.0
Gnoisy = Gtrue + noiseMag*(rand(Npoints,Npoints)-.5)

# Fitting with LSfit2D
paramINIT = [.3,.8,7,12.,40,20,0]

paramOUT = LSfit2D(gauss2D,Gnoisy,X,Y,paramINIT)
Gfit = gauss2D(X,Y,paramOUT)

# Show results
Ginit = gauss2D(X,Y,paramINIT)

Gplot = zeros([Npoints,Npoints*4])
Gplot[:,0:Npoints] = Gtrue
Gplot[:,Npoints:2*Npoints] = Gnoisy
Gplot[:,2*Npoints:3*Npoints] = Ginit
Gplot[:,3*Npoints:4*Npoints] = Gfit

figure(0)
clf()
pcolor(Gplot)
axis('equal')
title("2D Gauss (true, noisy, init, fit)")
