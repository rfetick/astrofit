# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 19:40:35 2018

Purpose: learn how to use the `status` output of LSfit and LSfit2D

@author: rfetick
"""

# IMPORT
from numpy import linspace, array, floor
from numpy.random import rand
from pylab import plot, legend, show, title, clf, grid, xlabel, ylabel, semilogy
from matplotlib.pyplot import figure

from astrofit import gauss, LSfit, LSparam

###############################################################################
###         Print documentation
###############################################################################

print(LSfit.__doc__)
print("------------------------------------------------")

###############################################################################
###         Fitting of a gaussian function
###############################################################################

Npoints = 50

paramTrue = [2.0,5.0,1.0,4.0] #[bck,amp,sigma,center]
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
# we retrieve the "status"
param, status = LSfit(gauss,Ynoisy,X,param0)
Yfit = gauss(X,param)

# SHOW FITTING
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

# SHOW STATUS
print("Keys of `status` are: "+str(status.keys()))

# evaluation of data fidelity
figure(1)
clf()
plot(status["xhi2"])
grid()
title("$\chi^2$")
xlabel("Iteration")

# evolution of parameters toward their final value
figure(2)
clf()
plot((status["param"])[:,0])
plot((status["param"])[:,1])
plot((status["param"])[:,2])
plot((status["param"])[:,3])
grid()
title("Evolution of parameters")
xlabel("Iteration")
ylabel("Value")
legend(["background","amplitude","sigma","center"])
