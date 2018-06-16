# -*- coding: utf-8 -*-
"""
Created on Tue Jun 05 21:55:50 2018

@author: rfetick
"""

from matplotlib.pyplot import figure
from pylab import clf
from numpy import floor

from astrofit import Zernike

Npix = 300

###############################################################################
###         INSTANCE METHODS
###############################################################################

## Define one Zernike with J

pist = Zernike(Npix,J=0)
TTX = Zernike(Npix,J=1)

figure(0)
clf()
pist.look()

figure(1)
clf()
TTX.look()

## Define one Zernike with [M,N]

defoc = Zernike(Npix,N=2,M=0)

figure(2)
clf()
defoc.look()

## Define multiple Zernike, with coefficients

phase = Zernike(Npix,J=[1,2,3])
print('Initial Zernike coeffs are'+str(phase.coeffs))
phase.coeffs = [100,-80,80]
print('New Zernike coeffs are'+str(phase.coeffs))

figure(3)
clf()
phase.look()

print(phase) # print some information about your Zernike

phase2 = phase.getMode(2) # get the mode number 2, multiplied by its coeff
phase2unit = phase.getUnitMode(2) # get the mode number 2, with unit coeff

phaseMap = phase.getSum() # get table of values associated to the Zernike

## Fit a phase onto a Zernike instance

zernikeFit = Zernike(Npix,J=[0,1,2,3,4,5])
zernikeFit.fit(phaseMap)
print('Fitted coeffs: '+str(floor(zernikeFit.coeffs)))
print('(to be compared with phase.coeffs)')

## Transform a Zernike to a complex wavefront

WF = phase.toWF() # complex wavefront exp(i*phase)

###############################################################################
###         STATIC METHODS
###############################################################################

## Fit a phase onto some Zernike functions

coeff = Zernike.fitMode(phaseMap,J=[0,1,2,3,4,5])
print('Fitted coeffs: '+str(floor(coeff)))
print('(to be compared with phase.coeffs)')

## Convert J to [M,N] and vice-versa

J=4
MN = Zernike.JtoMN(J)
print('J='+str(J)+' is equivalent to [M,N]='+str(MN))
print('[M,N]='+str(MN)+' is equivalent to J='+str(Zernike.MNtoJ(MN[0],MN[1])))

## Plot pyramid

Zernike.pyramid(Npix,5)

J=4
print('J='+str(J)+' is '+str(Zernike.JtoSTR(J)))