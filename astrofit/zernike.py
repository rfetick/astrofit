# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 18:28:23 2018
Version: 1.0

Generation of the Zernike polynomials

@author: rfetick
"""

from numpy import arange, meshgrid, exp, sqrt, zeros, where, arctan, pi, cos, sin, size, fft
from math import factorial
import matplotlib.pyplot as plt

class Zernike(object):
    """
    Zernike class
    
    ###   ATTRIBUTES   ###
    modes : all the zernike modes (Npix,Npix,nbZer)
    coeffs : all the coeffs associated with modes (RMS unit)
    mask : mask of 0 and 1 on which the Zernike is computed
    
    ###    METHODS     ###
    getMode
    getUnitMode
    getSum
    fit
    look
    
    ### STATIC METHODS ###
    JtoMN
    MNtoJ
    field
    JtoSTR
    fitMode
    pyramid
    
    """
    
    @staticmethod
    def _checkJMN(J=None,M=None,N=None):
        """
        Check input values of J, M and N
        """
        if (J==None)*(M==None)*(N==None):
            raise ValueError("You should define J or (M,N)")
        if J!=None:
            if M!=None or N!=None:
                raise ValueError("You cannot enter J at the same time as M or N")
            if J < 0:
                raise ValueError("J should be positive or null")
        if M!=None:
            if N==None:
                raise ValueError("You should define M and N at the same time")
        if N!=None:
            if N < 0:
                raise ValueError("N should be positive or null")
            if M==None:
                raise ValueError("You should define M and N at the same time")
            if abs(M)>abs(N):
                raise ValueError("abs(M) should be lower or equal to N")
            if (N-M)%2 != 0:
                raise ValueError("N minus M should be an even number")

    
    
    
    def __init__(self,Npix,J=None,M=None,N=None,unit=None,ampWF=1.):
        """
        Constructor of a Zernike
        Npix : Number of pixels for the output Zernike
        J    : J number of the Zernike (J=0 for piston)
        M    : Azimutal number of Zernike
        N    : Radial number of Zernike (N=0 for piston)
        unit : Radius (in pixels) of the unit circle in which to plot the Zernike
                Default value is unit=Npix/2, so Zernike is incribed exactly in the square array
                Setting unit<Npix/2 allows to set some spacing around Zernike
        """
        self.Npix = Npix
        self.ampWF = ampWF #the amplitude of an eventual complex wavefront
        
        Zernike._checkJMN(J=J,M=M,N=N)
        
        if J!=None:
            if isinstance(J, (int, long, float)):
                self._J = [J]
            else:
                self._J = J
            self._M = []
            self._N = []
            for jj in self._J:
                MN = Zernike.JtoMN(jj)
                self._M.append(MN[0])
                self._N.append(MN[1])

        if M!=None:
            if isinstance(M, (int, long, float)):
                self._M = [M]
            else:
                self._M = M
        if N!=None:
            if isinstance(N, (int, long, float)):
                self._N = [N]
            else:
                self._N = N
                
            self._J = []
            for ii in arange(len(self._N)):
                self._J.append(Zernike.MNtoJ(self._M[ii],self._N[ii]))
                
                
        self._nbZer = len(self._J)
        self.coeffs = zeros(self._nbZer)+1.
        
        self.modes = zeros((Npix,Npix,self._nbZer))
        for ii in arange(self._nbZer):
            self.modes[:,:,ii] = Zernike._field(Npix,J=self._J[ii],unit=unit)
            
        if unit==None:
            unit = Npix/2.
        y,x = meshgrid(arange(Npix)-Npix/2,arange(Npix)-Npix/2)
        r = sqrt((x**2) + (y**2))/unit
        self.mask = r <= 1
    
      
    def __repr__(self):
        disp  = "Zernike instance ["+str(self.Npix)+"x"+str(self.Npix)+" pix]"
        disp += ", with "+str(self._nbZer)+" modes"
        disp += "\nJ      = "+self._J.__repr__()
        disp += "\ncoeffs = "+self.coeffs.__repr__()
        return disp
    
    def getUnitMode(self,number):
        """
        Get a unique mode of your Zernike, with a coeff=1
        """
        return self.modes[:,:,number]
         
    def getMode(self,number):
        """
        Get a unique mode of your Zernike
        Mode is weighted with its associated coefficient
        """
        return self.modes[:,:,number]*self.coeffs[number]
    
    def getSum(self):
        """
        Get the sum of all your Zernike modes
        Modes are weighted with their associated coefficients
        """
        tot = zeros((self.Npix,self.Npix))
        for ii in arange(self._nbZer):
            tot += self.getMode(ii)
        return tot
    
    @staticmethod
    def JtoMN(J):
        """
        Transform a J value to a [M,N] value
        """
        Zernike._checkJMN(J=J,M=None,N=None)
        N=0
        while J >= N*(N+1)/2:
            N += 1
        N -= 1
        M = J - N*(N+1)/2
        M = 2*M - N
        return [M,N]
    
    @staticmethod
    def MNtoJ(M,N):
        """
        Transform a [M,N] value to a J value
        """
        Zernike._checkJMN(J=None,M=M,N=N)
        return N*(N+1)/2+(M+N)/2
    
    @staticmethod
    def field(Npix,M=None,N=None,J=None,circ=1,unit=None):
        """
        Main method to compute one or multiple Zernike
        Output is 3D array (Npix,Npix,nbZer)
        
        Npix : Number of pixels for your array
        circ : If the Zernike is inscribed a circle
        unit : Radius of the unit circle in pixels 
        """
        Zernike._checkJMN(J=J,M=M,N=N)
        if J!=None:
            modes = zeros((Npix,Npix,size(J)))
            if isinstance(J, (int, long, float)):
                modes[:,:,0] = Zernike._field(Npix,J=J)
            else:
                for ii in arange(size(J)):
                    modes[:,:,ii] = Zernike._field(Npix,J=J[ii])
        else:
            modes = zeros((Npix,Npix,size(M)))
            if isinstance(M, (int, long, float)):
                modes[:,:,0] = Zernike._field(Npix,M=M,N=N)
            else:
                for ii in arange(size(M)):
                    modes[:,:,ii] = Zernike._field(Npix,M=M[ii],N=N[ii])
        return modes
    
    @staticmethod
    def _field(Npix,M=None,N=None,J=None,circ=1,unit=None):
        """
        Main method to compute a unique Zernike
        Npix : Number of pixels for your array
        circ : If the Zernike is inscribed a circle
        unit : Radius of the unit circle in pixels 
        """
        Zernike._checkJMN(J=J,M=M,N=N)
        
        if J!=None:
            MN = Zernike.JtoMN(J)
            M = MN[0]
            N = MN[1]
            
        if unit==None:
            unit = Npix/2.
        
        # COMPUTE R and THETA
        y,x = meshgrid(arange(Npix)-Npix/2,arange(Npix)-Npix/2)
        r = sqrt((x**2) + (y**2))/unit
        theta = zeros((Npix,Npix))
        indices = where(x != 0)
        theta[indices] = arctan((y[indices]+0.)/(x[indices]+0.))
        
        quadrant1 = where((x <= 0) * (y > 0))
        quadrant2 = where((x < 0) * (y <= 0))
        quadrant3 = where((x >= 0) * (y < 0))
        xNull = where((x == 0) * (y >= 0))
        xyNull = where((x == 0) * (y <= 0))
    
        theta[quadrant1] = theta[quadrant1]+pi
        theta[quadrant2] = theta[quadrant2]+pi
        theta[quadrant3] = theta[quadrant3]+2*pi
        theta[xNull] = pi/2
        theta[xyNull] = 3*pi/2
        
        #theta = (2*pi-theta + pi/2)%(2*pi)
        #plt.pcolormesh(theta.T)
        
        # COMPUTE ZERNIKE
        rmn = zeros((Npix, Npix))

        for k in arange((N-abs(M))/2 + 1):
            a = ((-1.)**k)*factorial(N-k)+0.
            b = factorial(k)*factorial((N+abs(M))/2-k)*factorial((N-abs(M))/2-k)+0.
            rmn = rmn + a/b*r**(N-2*k)
            
        if M >= 0:
            z = rmn*cos(M*theta)
        else:
            z = rmn*sin(abs(M)*theta)
            
        if circ == 1:
            z *= (r <= 1)
            z = z/sqrt((z**2).sum())
        
        return z
    
    @staticmethod
    def fitMode(phase,J=None,M=None,N=None,unit=None):
        """
        Fit a phase with some Zernike modes
        J, M, N: The Zernike modes to fit on
        Returns coeffs of fitting on these modes
        """
        Zernike._checkJMN(J=J,M=M,N=N)
        s = phase.shape
        if len(s)!=2:
            raise ValueError("phase should be a 2D array")
        if s[0]!=s[1]:
            raise ValueError("phase should be a square array")
        modes = Zernike.field(s[0],M=M,N=N,J=J,unit=unit)
        nbZer = modes.shape[2]
        coeffs = zeros(nbZer)
        for ii in arange(nbZer):
            coeffs[ii] = (phase*modes[:,:,ii]).sum()
        return coeffs
    
    
    def fit(self,phase):
        """
        Fit a phase screen on the Zernike modes
        self.coeffs are set accordingly
        """
        s = phase.shape
        if len(s)!=2:
            raise ValueError("phase should be a 2D array")
        if s[0]!=s[1]:
            raise ValueError("phase should be a square array")
        if s[0]!=self.Npix:
            raise ValueError("phase should be size of self.Npix="+str(self.Npix))
        for ii in arange(self._nbZer):
            self.coeffs[ii] = (phase*self.modes[:,:,ii]).sum()
        
        
    def getPSF(self):
        """
        Return the associated PSF
        PSF = |Fourier[exp(1j*Zernike_phase)]^2|
        """
        PSF = abs(fft.fftshift(fft.fft2(self.toWF())))**2
        return PSF

    def toWF(self):
        """
        Transform the Zernike into a complex wavefront
        """
        return self.ampWF*self.mask*exp(1j*self.getSum())

    def look(self,func=None,cmap='inferno'):
        """
        Give a quick look on your Zernike
        'func' defines the color scaling
        """
        z = self.getSum()
        if func==None:
            cax = plt.pcolormesh(z.T)
        else:
            cax = plt.pcolormesh(func(z.T-z.min()+1.))
        
        plt.colorbar(cax)
        plt.set_cmap(cmap)
        plt.title('Zernike J='+str(self._J))
        plt.axis('square')
        plt.axis([0, self.Npix, 0, self.Npix])
        plt.xlabel('X pixels',fontsize=15)
        plt.ylabel('Y pixels',fontsize=15)
    
    @staticmethod
    def pyramid(Npix,N):
        """
        Plot a pyramid of Zernike for N radial orders
        Npix is the size of each Zernike in pixels
        """
        M = N
        J = Zernike.MNtoJ(M,N)
        pyr = zeros((2*Npix*(N+1)-Npix,Npix*(N+1)))
        yy, xx = meshgrid(arange(Npix),arange(Npix)+Npix*(N+1))
        for jj in arange(J+1):
            mn = Zernike.JtoMN(jj)
            z = Zernike(Npix,J=jj)
            pyr[xx+Npix*(mn[0]-1),yy+Npix*(N+1-mn[1]-1)] = z.getSum()
        fig, ax = plt.subplots()
        cax = plt.pcolormesh(pyr.T)
        plt.colorbar(cax)
        plt.axis('image')
        plt.title('Zernike pyramid from J=0 to J='+str(J-1))
        plt.xlabel('M azimutal')
        plt.ylabel('N radial')
        ax.set_xticks(Npix*(arange(2*N+1) + .5), minor=False)
        ax.set_yticks(Npix*(arange(N+1) + .5), minor=False)
        yticks = [] 
        xticks = []
        for mm in arange(2*N + 1):
            xticks.append(str(mm-N))
        for nn in arange(N+1):
            yticks.append(str(N-nn))
        ax.set_xticklabels(xticks)
        ax.set_yticklabels(yticks)
    
    @staticmethod
    def JtoSTR(J):
        """
        Get desination of some Zernike
        """
        if J==0:
            return "piston"
        elif J==1:
            return "tip-tilt X"
        elif J==2:
            return "tip-tilt Y"
        elif J==3:
            return "astigmatism"
        elif J==4:
            return "defocus"
        elif J==5:
            return "astigmatism"
        elif J==6:
            return "trefoil"
        elif J==7:
            return "coma"
        elif J==8:
            return "coma"
        elif J==9:
            return "trefoil"
        else:
            return "Warning (unknown Zernike)"
        
        
        