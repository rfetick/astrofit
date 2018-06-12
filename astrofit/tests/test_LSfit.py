# -*- coding: utf-8 -*-
"""
Created on Thu Jan 04 17:51:47 2018
Updated on Mon Jun 04 19:30:00 2018
    Added test for LSfit2D

Test the LSfit method

@author: rfetick
"""

# IMPORT
from numpy import linspace, array, floor, mgrid, zeros
from numpy.random import rand
from pylab import plot, legend, show, title, pcolor, axis, clf, grid
from matplotlib.pyplot import figure

from astrofit import gauss, LSfit, LSparam, gauss2D, LSfit2D

###############################################################################
###         Gaussian
###############################################################################

print("-------------- LSfit --------------")

Npoints = 50

paramTrue = [1.0,5.0,1.0,5.0] #[bck,amp,sigma,center]
paramInit = [1.5,4.0,2.0,6.0]

noiseAmp = 2

# DEFINE NOISY DATA
X = linspace(0,10,num=Npoints)
paramTrue = array(paramTrue)
Ytrue = gauss(X,paramTrue)
Ynoisy = Ytrue + noiseAmp*(rand(Npoints)-.5)

# MINIMIZATION
param0 = array(paramInit)
Ystart = gauss(X,param0)
param = LSfit(gauss,Ynoisy,X,param0)
Yfit = gauss(X,param)

# SHOW RESULTS
print("Param      : [bck,amp,sig,mid]")
print("Param true : "+str(paramTrue))
print("Param start: "+str(param0))
print("Param fit  : "+str(floor(param*100)/100.0))

figure(0)
clf()
plot(X,Ynoisy,'orange',linewidth=2)
plot(X,Ytrue,'b',linewidth=3)
plot(X,Ystart,'g',linewidth=2, linestyle='--')
plot(X,Yfit,'r',linewidth=2)
grid()
legend(("Noisy data","True curve","Init fitting","LSfit solution"))
title("Gaussian fit")
show()

###############################################################################
###                  GAUSS using class LSparam
###     Check class LSparam from LSfit.py for more information
###############################################################################

# I introduce on purpose bad constraints on the parameters to show you the effects
# However this may lead to errors of convergence

print("-------------- LSfit constrained --------------")

# MINIMIZATION
param0 = LSparam(paramInit)
param0.fixed = [False,False,True,False]
#param0.set_bound_down(2)
param = LSfit(gauss,Ynoisy,X,param0)
YfitCS = gauss(X,param)

# SHOW RESULTS
print("Param      : [bck,amp,sig,mid]")
print("Param true : "+str(paramTrue))
print("Param start: "+str(param0.value))
print("Param fixed: "+str(param0.fixed))
print("Param fit  : "+str(floor(param*100)/100.0))

figure(1)
clf()
plot(X,Ynoisy,'orange',linewidth=2)
plot(X,Ytrue,'b',linewidth=3)
plot(X,Ystart,'g',linewidth=2, linestyle='--')
plot(X,YfitCS,'r',linewidth=2)
grid()
legend(("Noisy data","True curve","Init fitting","LSfit solution"))
title('Gaussian fit with constrained param')
show()

###############################################################################
###         Gaussian 2D
###############################################################################

print("-------------- Gaussian 2D --------------")

[X,Y] = mgrid[0:Npoints,0:Npoints]
A = zeros(7)
A[1] = 1
A[2] = 10
A[3] = 10
A[4] = Npoints/2
A[5] = Npoints/2
Gtrue = gauss2D(X,Y,A)

Gnoisy = Gtrue + 1.*(rand(Npoints,Npoints)-.5)

#Ainit = A.copy()
#Ainit[1] = 0.8
#Ainit[2] = 7.0
#Ainit[4] += 10.

Ainit = [.3,.8,7,12.,40,20,0]

Ginit = gauss2D(X,Y,Ainit)

param = LSfit2D(gauss2D,Gnoisy,X,Y,Ainit)
Gfit = gauss2D(X,Y,param)


Gplot = zeros([Npoints,Npoints*4])
Gplot[:,0:Npoints] = Gtrue
Gplot[:,Npoints:2*Npoints] = Gnoisy
Gplot[:,2*Npoints:3*Npoints] = Ginit
Gplot[:,3*Npoints:4*Npoints] = Gfit

figure(2)
clf()
pcolor(Gplot)
axis('equal')
title("2D Gauss (true, noisy, init, fit)")
show()
