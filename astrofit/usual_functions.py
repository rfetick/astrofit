# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 23:17:57 2018
Updated on Tue Feb 20 20:00:00 2018
    Parameters are converted to float to prevent integer processing

@author: rfetick
"""

from numpy import exp, cos, sin, array, pi
from scipy.special import jv

from scipy.signal import convolve2d

###############################################################################
###############################################################################

def airy2D(Npix,samp,OCC=0):
    """
    ### USAGE ###
    Create a 2D diffraction pattern
    ### INPUTS ###
    Npix - Size of array
    samp - Sampling = (LAMBDA*FOC)/(PIX*D)
    ### KEYWORDS ###
    OCC    eventual occultation ratio = D_secondary / D_primary
    """
    omega = pi/samp
    r = circarr(Npix,Npix)
    r[Npix/2,Npix/2] = 1e-5
    res = 4.0/((1-OCC**2)**2)*((jv(1,omega*r)-OCC*jv(1,OCC*omega*r))/(omega*r))**2
    res[Npix/2,Npix/2] = 1.0
    
    return res/res.sum()

###############################################################################
###############################################################################

def gauss(X,A):
    """
    ### USAGE ###
    Create a 1D Gaussian function
    ### INPUTS ###
    X are the coordinates where to evaluate the function
    A is the voctor of parameter defined as following
    A[0]   Constant baseline level
    A[1]   Peak value
    A[2]   Sigma
    A[3]   Peak centroid
    """
    A = array(A).astype(float)
    return A[1]*exp(-((X-A[3])**2)/(2*A[2]**2))+A[0]

###############################################################################
###############################################################################

def gauss2D(X,Y,A):
    """
    ### USAGE ###
    Create a 2D Gaussian function
    ### INPUTS ###
    [X,Y] is the meshgrid where to evaluate the function
    A is the voctor of parameter defined as following
    A[0]   Constant baseline level
    A[1]   Peak value
    A[2]   Sigma X
    A[3]   Sigma Y
    A[4]   Peak centroid (x)
    A[5]   Peak centroid (y)
    A[6]   Tilt angle (clockwise)
    """
    A = array(A).astype(float)
    
    alpha_X = A[2]
    alpha_Y = A[3]
    
    # Rotational angles
    xNum = (cos(A[6])/alpha_X)**2 + (sin(A[6])/alpha_Y)**2
    yNum = (cos(A[6])/alpha_Y)**2 + (sin(A[6])/alpha_X)**2
    xyNum = sin(2*A[6])/alpha_Y**2 - sin(2*A[6])/alpha_X**2

    # Compute Moffat
    u  = xNum*(X-A[4])**2 + xyNum*(X-A[4])*(Y-A[5]) + yNum*(Y-A[5])**2
    
    return A[1]*exp(-.5*u)+A[0]

###############################################################################
###############################################################################

def moffat(X,A):
    """
    ### USAGE ###
    Create a 1D Moffat function
    ### INPUTS ###
    X are the coordinates where to evaluate the function
    A is the voctor of parameter defined as following
    A[0]   Constant baseline level
    A[1]   Peak value
    A[2]   Peak half-width
    A[3]   Peak centroid
    A[4]   Moffat power law
    ### OUTPUT ###
    Y the Moffat function evaluated on the X coordinates
    """
    
    A = array(A).astype(float)
    
    # Compute Moffat
    u  = ((X-A[3])/A[2])**2
    moff = A[1]/(u + 1.)**A[4] + A[0]
    
    return moff

###############################################################################
###############################################################################

def moffat2D(X,Y,A,circular=False,energy=False):
    """
    ### USAGE ###
    Create a 2D Moffat function
    ### INPUTS ###
    [X,Y] is the meshgrid where to evaluate the function
    A is the voctor of parameter defined as following
    A[0]   Constant baseline level
    A[1]   Peak value
    A[2]   Peak half-width (ALPHA x)
    A[3]   Peak half-width (ALPHA y)
    A[4]   Peak centroid (x)
    A[5]   Peak centroid (y)
    A[6]   Tilt angle (clockwise)
    A[7]   Moffat power law
    ### KEYWORDS ###
    circular - Ignore A[3] and use A[2] instead
    energy   - A[1] is not the amplitude anymore, but the energy of Moffat
                before adding A[0]. Energy is defined only if A[7]>1
    ### OUTPUT ###
    The 2D-Moffat evaluated on the meshgrid
    """
    
    A = array(A).astype(float)
    
    alpha_X = A[2]
    alpha_Y = A[3]
    
    if circular:
        alpha_Y = alpha_X
    
    # Rotational angles
    xNum = (cos(A[6])/alpha_X)**2 + (sin(A[6])/alpha_Y)**2
    yNum = (cos(A[6])/alpha_Y)**2 + (sin(A[6])/alpha_X)**2
    xyNum = sin(2*A[6])/alpha_Y**2 - sin(2*A[6])/alpha_X**2

    # Compute Moffat
    u  = xNum*(X-A[4])**2 + xyNum*(X-A[4])*(Y-A[5]) + yNum*(Y-A[5])**2
    moff = A[1]/(u + 1.)**A[7]
    
    if energy:
        if A[7]<=1:
            raise ValueError("A[7] > 1 is required for energy normalization")
        moff *= (A[7]-1.0)/(pi*alpha_X*alpha_Y)
    
    return moff + A[0]

###############################################################################
###############################################################################

def moffatAiry(X,Y,A,**kwargs):
    """
    Convolve a Moffat with Airy
    A[0] to A[7] are the Moffat parameters
    A[8] is the Airy sampling
    """
    return convolve2d(moffat2D(X,Y,A[0:8],**kwargs),airy2D(X.shape[0],A[8]))
