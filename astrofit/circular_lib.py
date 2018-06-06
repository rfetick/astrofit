# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 16:05:38 2018

@author: rfetick
"""

import numpy as np

def circarr(nx,ny,center=[0,0],sigma=[1,1]):
    """
    Create a 2D array with values defined as radius to center.
    """
    y,x = np.ogrid[-nx/2:nx/2,-ny/2:ny/2]
    return np.sqrt(((x-center[0])**2)/sigma[0]**2 + ((y-center[1])**2)/sigma[1]**2)

def circavg(tab,center=[0,0]):
    """
    Compute the circular average of an array.
    Returns a vector with average for each radius
    """
    nx,ny = tab.shape
    r = circarr(nx,ny,center=[0,0])
    cAVG = np.zeros(int(r.max()))
    for i in np.arange(int(r.max())):
        index = np.where((r>=i) * (r<(i+1)))
        cAVG[i] = tab[index[0],index[1]].sum()/index[0].size
    return cAVG
        
def circvar(tab,center=[0,0]):
    """
    Compute the circular variance of an array.
    Returns a vector with variance for each radius
    """
    nx,ny = tab.shape
    r = circarr(nx,ny,center=[0,0])
    cVAR = np.zeros(int(r.max()))
    cAVG = circavg(tab)
    for i in np.arange(int(r.max())):
        index = np.where((r>=i) * (r<(i+1)))
        temp = (tab[index[0],index[1]] - cAVG[i])**2
        cVAR[i] = temp.sum()/index[0].size
    return cVAR


def imcenter(im,size=-1,maxi=True,GC=False,center=None):
    """
    Center a tabular on its maximum, or on its center of gravity
    im     : the 2D image to center
    size   : the 2 elements list of the size for the new image
    maxi   : boolean to choose for maximum centering (default = True)
    GC     : boolean to choose for gravity centering (default = False)
    center : user defined centering on a chosen pixel (default = None)
    """
    
    ### SET to False the default choice, since user wants another option
    if (GC == True) or center != None:
        maxi = False 
    
    sX = len(im)
    sY = len(im[0])
    
    if len(size) != 2:
        raise ValueError("size keyword should contain 2 elements")
        
    if (size[0] < 0) or (size[1] < 0):
        raise ValueError("size keyword should contain only positive numbers")
    
    ### CENTER ON USER DEFINED CENTER
    if center != None:
        cX = center[0]
        cY = center[1]
    
    ### CENTER ON MAX, default method
    if maxi==True:
        index = np.where(im==im.max())
        if len(index[0])>1:
            print "Imcenter warning: more than one maximum found"
        cX = index[0][0]
        cY = index[1][0]
          
    ### CENTER ON GRAVITY CENTER
    if GC==True:
        x = np.arange(sX)
        y = np.arange(sY)
        cX = int(np.round(((x*im.sum(axis=1)).sum())/(im.sum())))
        cY = int(np.round(((y*im.sum(axis=0)).sum())/(im.sum())))
        
    ### COMMON BLOCK FOR CENTERING!!!
    if (cX-size[0]/2 < 0) or (cY-size[1]/2 < 0) or (cX+size[0]/2 + ((size[0]/2)%2) >= sX) or (cY+size[1]/2 + ((size[1]/2)%2) >= sY):
        raise ValueError("Defined size is too big, cannot center on max. Output image would expand out of input image.")
    vX = np.arange(cX-size[0]/2,cX+size[0]/2 + (size[0]/2)%2)
    vY = np.arange(cY-size[1]/2,cY+size[1]/2 + (size[1]/2)%2)
    newim = im[vX][:,vY]
        
    return newim
    
    
    
    

    