# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 23:10:48 2018

Test usual functions

@author: rfetick
"""

## IMPORT

from numpy import linspace, zeros, mgrid
from pylab import plot, legend, show, title, pcolor, axis, clf, grid
from matplotlib.pyplot import figure

from astrofit import moffat, gauss, gauss2D, moffat2D

## Test Gauss 1D and Moffat 1D
X = linspace(0,10,num=100)

bck = 1.
amp = 1.
sigma = 1.
center = 5.
beta = 2.

G = gauss(X,[bck,amp,sigma,center])
M = moffat(X,[bck,amp,sigma,center,beta])

figure(0)
plot(X,G,linewidth=2)
plot(X,M,linewidth=2)
grid()
legend(("Gauss","Moffat"))
title("1D Gauss and 1D Moffat")
show()

## Test Gauss 2D
xmax = 100
[X,Y] = mgrid[0:xmax,0:xmax]
A = zeros(7)
A[1] = 1
A[2] = 10
A[3] = 10
A[4] = xmax/2
A[5] = xmax/2
g = gauss2D(X,Y,A)
figure(1)
pcolor(g)
axis('equal')
title("2D Gauss")
show()

## Test Moffat 2D
xmax = 100
[X,Y] = mgrid[0:xmax,0:xmax]
A = zeros(8)
A[1] = 1
A[2] = 10
A[3] = 10
A[4] = xmax/2
A[5] = xmax/2
A[7] = 2
m = moffat2D(X,Y,A)
figure(2)
pcolor(m)
axis('equal')
title("2D Moffat")
show()