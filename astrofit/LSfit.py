# -*- coding: utf-8 -*-
"""
=================
LSfit
=================
Least-square fitting library
See LSfit, LSfit2D and LSparam documentation

Version 1.2

REVISIONS:
Created on Tue Jan 02 17:57:41 2018
Updated on Sat Jan 13 13:01:00 2018
    Stabilized algo with matrix conditioning
Updated on Sat Jun 09 11:30:00 2018
    Added **kwargs to be able to pass keywords to the user function
    Changed dp_min from 1e-8 to 1e-7, to reduce computation time
    Changed version from V.1.0 to V.1.1
Updated on Sun Jun 10 23:00:00 2018
    Simplified code (less keywords)
    LSfit checks number of outputs and eventually returns "status" as second output
    Changed version from V.1.1 to V.1.2

FUTURE IMPROVEMENTS:
- Check the WEIGHTED least-square convergence
- Enable parameter tied to another one
    

Licence GNU-GPL v3.0

@author: rfetick
"""

from numpy import size, zeros, arange, array, dot, diag, eye, floor, log10, inf, NaN
from numpy.linalg import cond, eig
from scipy.linalg import inv, LinAlgError

import inspect
import dis

###############################################################################
#                         LSfit2D main function                               #
###############################################################################

def LSfit2D(funct,data,X,Y,param0, weights=-1, quiet=True, **kwargs):
    """ 
    #####  USAGE  #####
     Least-square minimization algorithm.
     Parametric least-square fitting.
     
     Minimises Xhi2 = sum(weights*(f(X,Y,param) - data)^2)
     where the sum(.) runs over the coordinates X and Y
     minimization is performed over the unknown "param"
     "data" is the noisy observation
     "f(X,Y,param)" is your model, set by the vector of parameters "param" and evaluated on coordinates (X,Y)
    
    Algorithm is Levenberg-Marquardt, with adaptive damping factor
    
    #####  SYNTAX  #####
    Two syntaxes are possible
    
    param        = LSfit(funct,data,X,Y,param0)
    param,status = LSfit(funct,data,X,Y,param0)
    
    "param"  is the returned solution
    "status" contains extra information about convergence
             status = {"xhi2","mu","param"}
    
    #####  INPUTS  #####
     funct   - [Function] The function to call
     data    - [2D array] Noisy data
     X       - [2D array] Coordinates where to evaluate the function
     Y       - [2D array] Coordinates where to evaluate the function
     param0  - [Vector,LSparam] First estimation of parameters
                 param0 can be simply a vector
                 if param0 is a LSparam, you can define bounds, fixed values...
                 see LSparam documentation
    
    ##### OPTIONAL INPUTS #####
     weights - [2D array] Weights (1/noise^2) on each coordinate (X,Y)
                  Default : no weighting (weights=1)
                  weights = 1/sigma^2 for a gaussian noise
                  weights = 1/data for a Poisson-like noise
     quiet   - [Boolean] Don't print status of algorithm
                  Default : quiet = True
    **kwargs - [Dictionary] Use a dictionary to pass some keywords to your function
    
    #####  OUTPUT  #####
     param   - [Vector] Best parameters for least-square fitting
     status  - [dictionary] Optional output, contains some information about convergence
    """
    global LSfit2DfunctUSER
    LSfit2DfunctUSER = funct
    
    global LSfit2DxUSER
    LSfit2DxUSER = X
    
    global LSfit2DyUSER
    LSfit2DyUSER = Y
    
    if size(weights) > 1:
        weights = weights.flatten()
    
    nout = _n_out(offset=3)
    if nout > 2:
        raise ValueError("Number of outpt argument for LSfit2D can be only 1 or 2")
    
    if nout==2:
       param, status = LSfit(_LSfitFUNCT,data.flatten(),X.flatten(),param0, weights=weights, quiet=quiet, **kwargs)
       return param, status
    
    param = LSfit(_LSfitFUNCT,data.flatten(),X.flatten(),param0, weights=weights, quiet=quiet, **kwargs)
    
    return param


def _LSfitFUNCT(X,param, **kwargs):
    """
    Function to be called in internal as interface with user function
    """
    return (LSfit2DfunctUSER(LSfit2DxUSER,LSfit2DyUSER,param, **kwargs)).flatten()

###############################################################################
#                         LSfit main function                                 #
###############################################################################
    

def LSfit(funct,data,X,param0, weights=-1, quiet=True, **kwargs):
    """ 
    #####  USAGE  #####
     Least-square minimization algorithm.
     Parametric least-square fitting.
     
     Minimises Xhi2 = sum(weights*(f(X,param) - data)^2)
     where the sum(.) runs over the coordinates X
     minimization is performed over the unknown "param"
     "data" is the noisy observation
     "f(X,param)" is your model, set by the vector of parameters "param" and evaluated on coordinates X
    
    Algorithm is Levenberg-Marquardt, with adaptive damping factor
    
    #####  SYNTAX  #####
    Two syntaxes are possible
    
    param        = LSfit(funct,data,X,param0)
    param,status = LSfit(funct,data,X,param0)
    
    "param"  is the returned solution
    "status" contains extra information about convergence
             status = {"xhi2","mu","param"}
    
    #####  INPUTS  #####
     funct   - [Function] The function to call
     data    - [Vector] Noisy data
     X       - [Vector] Coordinates where to evaluate the function
     param0  - [Vector,LSparam] First estimation of parameters
                 param0 can be simply a vector
                 if param0 is a LSparam, you can define bounds, fixed values...
                 see LSparam documentation
    
    ##### OPTIONAL INPUTS #####
     weights - [Vector] Weights (1/noise^2) on each coordinate X
                  Default : no weighting (weights=1)
                  weights = 1/sigma^2 for a gaussian noise
                  weights = 1/data for a Poisson-like noise
     quiet   - [Boolean] Don't print status of algorithm
                  Default : quiet = True
    **kwargs - [Dictionary] Use a dictionary to pass some keywords to your function
    
    #####  OUTPUT  #####
     param   - [Vector] Best parameters for least-square fitting
     status  - [dictionary] Optional output, contains some information about convergence
    """
    # INITIALIZATION
    Xarr = array(X)
    sX = size(Xarr)
    param = LSparam(param0)
    param.initParam()
    mu_cond = 0
    Id = eye(param.nb_valid_param) # Identity matrix for conditioning

    J = zeros((sX,param.nb_param)) # Jacobian
    h = 1e-9 #step to compute Jacobian
    bad_cond = 1e10 #conditioning higher than this value is bad
    
    status = {"xhi2":[],"mu":[],"param":[]}
    nout = _n_out(offset=3)
    if nout > 2:
        raise ValueError("Number of outpt argument for LSfit can be only 1 or 2")
    
    # STOPPING CONDITIONS
    J_min = 1e-5*param.nb_param**2 # stopping criteria based on small Jacobian
    dp_min = 1e-7 #(1e-8) stopping criteria based on small variation of parameters
    max_iter = 1e4 # stopping criteria based on max number of iteration
    stop_loop = False # boolean for stopping the loop
    stop_trace = "LSfit started but not finished correctly"
    
    # WEIGHTS
    if size(weights)==1 and weights<=0:
        weights = zeros(sX)+1
    elif size(weights)>1 and size(weights)!=sX:
        raise ValueError("WEIGHTS should be a scalar or same size as X")
    
    # LEVENBERG-MARQUARDT
    mu = 1
    
    # Print some information
    f = funct(Xarr,param.value,**kwargs)    
    Xhi2 = sum(weights*(f-data)**2)
    status["xhi2"].append(Xhi2)
    if not quiet:
        print("[Iter=0] Xhi2 = "+str(Xhi2))
    
    
    # LOOP
    
    iteration = 0
    f = funct(Xarr,param.value,**kwargs)
    
    while (iteration < max_iter) and not stop_loop:
        
        mu_cond = 0
        
        ## Iterate over each parameter to compute derivative
        for ip in param.validValue:          
            J[:,ip] = weights*(funct(Xarr,param.step(ip,h),**kwargs)-f)/h
        
        ## Compute dvalue = -{(transpose(J)*J)^(-1)}*transpose(J)*(weights*(func-data))
        ## Reduce the matrix J only on the moving parameters
        JTJ = dot(J[:,param.validValue].T,J[:,param.validValue])
        JTJ += mu*diag(JTJ.diagonal())
        
        ## Try to improve conditioning
        c = cond(JTJ)
        if c > bad_cond:
            mu_cond = _improve_cond(JTJ,bad_cond,Id)

         
        ## Try inversion, here we might encounter numerical issues
        try:
            param.dvalue[param.validValue] = - dot(dot(inv(JTJ+mu_cond*Id),J[:,param.validValue].T),weights*(f-data))
        except LinAlgError as exception_message:
            _print_info_on_error(JTJ,iteration,mu,param.value)
            raise LinAlgError(exception_message)
        except ValueError as exception_message:
            _print_info_on_error(JTJ,iteration,mu,param.value)
            raise ValueError(exception_message)
        
        ## Xhi square old
        Xhi2 = sum(weights*(f-data)**2)
        
        if nout==2:
            status["xhi2"].append(Xhi2)
            status["mu"].append(mu)
            status["param"].append(param.value)
            
        # Step forward with dvalue
        param.forward()
        # New value of f(.)
        f = funct(Xarr,param.value,**kwargs)
                
        ## Levenberg-Marquardt update for mu
        Xhi2_new = sum(weights*(f-data)**2)
        if Xhi2_new > Xhi2:
            mu = min(10*mu,1e10)
        else:
            mu = max(0.1*mu,1e-10)
        
        ## Stop loop based on small variation of parameter
        if param.dvalueNorm() < dp_min*param.valueNorm():
            stop_loop = True
            stop_trace = "Parameter not evolving enough at iter="+str(iteration)
        
        ## Stop loop based on small Jacobian
        if abs(J[:,param.validValue]).sum() < J_min:
            stop_loop = True
            stop_trace = "Jacobian small enough at iter="+str(iteration)
        
        ## Increment Loop
        iteration += 1  
    
    
    ## END of LOOP and SHOW RESULTS
    
    if iteration >= max_iter:
        stop_trace = "Maximum iteration reached (iter="+str(iteration)+")"
    
    if not quiet:
        print("[Iter="+str(iteration)+"] Xhi2 = "+str(Xhi2))
        print(" ")
        print("Stopping condition: "+stop_trace)
        print(" ")
    
    if nout==2:
        status["param"] = array(status["param"])
        return param.value, status
    
    return param.value


###############################################################################
#         Print matrix information when an error occurs                       #
###############################################################################

def _print_info_matrix(M):
    
    print("Matrix values:")
    for j in arange(len(M)):
        line_str = "["
        for i in arange(len(M[0])):
            line_str = line_str + " " + _num2str(M[j][i])
        print(line_str + " ]")

    print("Conditioning: "+_num2str(cond(M)))
    
    line_str = "["
    for i in arange(len(M)):
        line_str = line_str + " " + _num2str(eig(M)[0][i])
    print("Eigenvalues: " + line_str + " ]")

def _print_info_on_error(M,iteration,mu,values):
    print("LSFit encountered an error at iter = "+str(iteration))
    print("mu = "+str(mu))
    print("##### Parameter #####")
    print(str(values))
    print("##### ########## #####")


def _num2str(x):
    if x==0:
        num_str="0.00"
    elif x is inf or x == inf:
        num_str = "inf"
    elif x is NaN:
        num_str = "NaN"
    else:
        power = floor(log10(abs(x)))
        str_power = str(int(power))
        if len(str_power)==1:
            str_power = "0" + str_power
        if power>=0:
            str_power = "+" + str_power
        if power<0 and len(str_power)==2:
            str_power = "-0"+str_power[1]
        num_str = ('%.2f' % (x/10**power)) + "e" + str_power

    return num_str


###############################################################################
#                   Improve conditioning                                      #
###############################################################################
def _improve_cond(M,bad_cond,Id):
    mu = 1e-10
    while cond(M+mu*Id) > bad_cond and mu<1e10:
        mu = 10*mu
    return mu
    

###############################################################################
#                      Expected outputs                                       #
###############################################################################

def _n_out(offset=0):
    """
    Return how many values the caller is expecting
    Function from Stackoverflow
    https://stackoverflow.com/questions/16481156/find-out-into-how-many-values-a-return-value-will-be-unpacked
    """
    f = inspect.currentframe().f_back.f_back
    i = f.f_lasti + offset
    bytecode = f.f_code.co_code
    instruction = ord(bytecode[i])
    if instruction == dis.opmap['UNPACK_SEQUENCE']:
        return ord(bytecode[i + 1])
    elif instruction == dis.opmap['POP_TOP']:
        return 0
    else:
        return 1


###############################################################################
#                   Define a class to precise parameters                      #
###############################################################################
    
class LSparam(object):
    """
    Class LS param, to be used is LSfit
    Define more precisely your parameters
    Only "value" (see hereunder) is mandatory
    
    ATTRIBUTES
    value         - [List] Initial guess for your parameters
    fixed         - [List] Set to 'True' if parameters are fixed
    bound_up      - [List] Set the value of eventual up-bounds
    bound_down    - [List] Set the value of eventual down-bounds
    is_bound_up   - [List] Set to 'True' to activate bounds
    is_bound_down - [List] Set to 'True' to activate bounds
    
    """
    
    def __init__(self,value):
        if isinstance(value,LSparam):
            self.copyLSparam(value)
        else:
            if isinstance(value, (int, long, float, complex)):
                valueList = [value]
            else:
                valueList = value
            L = len(valueList)
            # User can influence these attributes
            self.value = array(valueList,dtype=float)
            self.fixed = [False for i in arange(L)]
            self.bound_up = array([0 for i in arange(L)],dtype=float)
            self.bound_down = array([0 for i in arange(L)],dtype=float)
            self.is_bound_up = [False for i in arange(L)]
            self.is_bound_down = [False for i in arange(L)]
            # User shouldn't access these attributes
            self.nb_param = L
            self.valueInit = self.value
            self.valueOld = self.value
            self.dvalue = array([0 for i in arange(L)],dtype=float)
            self.validValue = [i for i in arange(L)] #indices of not fixed parameters
            self.nb_valid_param = L
            
    def __repr__(self):
        return "LSparam with "+str(self.nb_param)+" parameters"
    
    def copyLSparam(self,objToCopy):
        """
        Copy old LSparam into current one
        """
        all_attr = objToCopy.__dict__
        for key in all_attr:
            setattr(self,key,getattr(objToCopy,key))
    
    def show(self):
        """
        Display information
        """
        print("########## LSparam ##########")
        print("Values         : " + str(self.value))
        print("Fixed          : " + str(self.fixed) )      
        print("Bounds up      : " + str(self.bound_up))
        print("Is bounded up  : " + str(self.is_bound_up) )      
        print("Bounds down    : " + str(self.bound_down))
        print("Is bounded down: " + str(self.is_bound_down))
        print("#############################")
        
    def one(self,i):
        """
        Returns a vector of size nb_param
        This vector is nul, excepted the i-th component equals 1
        Allows to compute partial derivatives for example
        """
        a = zeros(self.nb_param)
        a[i] = 1.0
        return a
    
    def step(self,i,h):
        """
        Return the parameter stepped at the i-th position
        (p+h*[0,,0...0,1,0...0])
        
        If some tied parameters, include them in the step
        """
        
        P  = self.value + h*self.one(i)
                
        return P
    
    def forward(self):
        """
        Steps forward the param.value with param.dvalue
        """
        self.valueOld = self.value
        # Before stepping, check if we are inside the bounds
        conditionUP = ((self.value + self.dvalue) < self.bound_up) | (1- array(self.is_bound_up))
        conditionDOWN = ((self.value + self.dvalue) > self.bound_down) | (1- array(self.is_bound_down))
        conditionFIXED = 1 - array(self.fixed)
        self.value = self.value + self.dvalue * (conditionUP & conditionDOWN & conditionFIXED)
        
    def initParam(self):
        """
        Be sure that everyhing is allright before going on
        """
        self.check()
        self.nb_param = len(self.value)
        self.valueInit = self.value
        self.valueOld = self.value
        self.dvalue = array([0 for i in arange(self.nb_param)],dtype=float)
        self.validValue = []
        for i in arange(self.nb_param):
            if (not self.fixed[i]):
                self.validValue.append(i)
        self.nb_valid_param = len(self.validValue)
    
    def valueNorm(self):
        """
        Return norm_1 of value
        """
        nb_varying = sum(1 - array(self.fixed))
        return sum(abs( self.value * (1 - array(self.fixed)) ))/nb_varying
    
    def dvalueNorm(self):
        """
        Return norm_1 of dvalue
        """
        nb_varying = sum(1 - array(self.fixed))
        return sum(abs( self.dvalue * (1 - array(self.fixed)) ))/nb_varying
    
    def check(self):
        """
        Check values consistency
        """
        a=array([len(self.bound_down),len(self.bound_up),len(self.is_bound_down),len(self.is_bound_up)])
        if sum(abs(a-len(self.value))):
            raise ValueError("Number of elements mismatch")
        for i in arange(len(self.value)):
            if not self.fixed[i] and self.value[i] in [-inf,inf]:
                raise ValueError("Parameter number "+str(i)+" is infinite")
            if self.is_bound_down[i]:
                if self.is_bound_up[i] and self.bound_down[i]>self.bound_up[i]:
                    raise ValueError("Bound_down number "+str(i)+" is higher than its bound_up")
                if self.value[i] < self.bound_down[i]:
                    raise ValueError("Paramter number "+str(i)+" is initialized lower than its bound_down")
            if self.is_bound_up[i]:
                if self.value[i] > self.bound_up[i]:
                    raise ValueError("Paramter number "+str(i)+" is initialized higher than its bound_up")
        
    def set_bound_down(self,val):
        """
        Set all down bounds with same value
        """
        self.bound_down = val*(zeros(self.nb_param)+1)
        self.is_bound_down = [True for i in arange(self.nb_param)]
        
    def set_bound_up(self,val):
        """
        Set all up bounds with same value
        """
        self.bound_up = val*(zeros(self.nb_param)+1)
        self.is_bound_up = [True for i in arange(self.nb_param)]
