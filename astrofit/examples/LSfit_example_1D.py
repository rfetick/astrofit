# -*- coding: utf-8 -*-
"""
Created on Thu Jan 04 17:51:47 2018

Purpose: Learn how to use LSfit in 1D
         Learn how to add eventual constraints on parameters

@author: rfetick
"""

# IMPORT
from numpy import linspace, array, floor
from numpy.random import rand
from pylab import plot, legend, title, clf, grid
from matplotlib.pyplot import figure

from astrofit import gauss, LSfit, LSparam

###############################################################################
###         Print documentation
###############################################################################

print(LSfit.__doc__)


###############################################################################
###         Fitting of a gaussian function
###############################################################################

print("-------------- LSfit (unconstrained) --------------")

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

###############################################################################
###                  GAUSS using class LSparam
###     Check class LSparam from LSfit.py for more information
###############################################################################

# I introduce on purpose bad constraints on the parameters
# It will show the effects on the results
# The solution will be on purpose of bad quality

print("-------------- LSfit (constrained) --------------")

# MINIMIZATION
param0 = LSparam(paramInit)

# Algorithm is not allowed to modify the sigma
param0.fixed = [False,False,True,False]

# You can also define a lower/upper bound:
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