import numpy as np

def modfunc(x, u_ant, v_ant):
    """
    Returns a vector containing the values of a 3x3 model exponential matrix with source parameters A0,l0,m0.
    
    INPUTS:
    x               Vector of source parameters.  Amp,l,m= x[0],x[1],x[2]  
    u_ant, v_ant    Value of (u,v)baseline measurements at the sampled (timeslot) points.
    """
    
    n_ant= u_ant.shape[0] #3
    
    l= x[0]
    m= x[1]
    Amp= x[2]
    
    model= Amp*np.exp(-2j*np.pi*(u_ant*l+v_ant*m))
    model[range(n_ant), range(n_ant), :]= 0
    
    model_stream= model[np.where(model!=0)]
    ## with t timeslots each
    return model_stream

def modfunc_jac(x,u_ant,v_ant):
    """
    Returns the Hessian and Jacobian corresponding to the function defined by model equation.
    
    INPUTS:
    x               Vector of source parameters.  A,l,m= x[0],x[1],x[2] 
    u_ant, v_ant    Value of (u,v)baseline measurements at the sampled (timeslot) points. use this as a loop
   
    """
    t= u_ant.shape[2]
    n_ant= u_ant.shape[0]
    
    l= x[0]
    m= x[1]
    Amp= x[2]
    
    ## Calculating JHJ explicitly from J
    Psi= -2j*np.pi*(u_ant*l+v_ant*m)
    expPsi= lambda sign: np.exp(sign*Psi)
    
    coeU = -2j*np.pi*Amp*u_ant
    coeV = -2j*np.pi*Amp*v_ant
    ## remember the -(minus) for negative exponents

    Jkl= coeU*expPsi(1) 
    Jkl[range(n_ant), range(n_ant), :]=0
    
    Jkm= coeV*expPsi(1) 
    Jkm[range(n_ant), range(n_ant), :]=0
      
    JkA= expPsi(1)
    JkA[range(n_ant), range(n_ant), :]=0
    
    
    Jk= np.vstack((Jkl[np.where(Jkl!=0)],Jkm[np.where(Jkm!=0)],JkA[np.where(Jkl!=0)])).T 
    
    Jh= Jk.conj().T ## Hermitian of J
    Complete= Jh.dot(Jk)
    
    Jh2= Jh[:-1,:]
    Complete2= Complete[:-1,:-1] ## 2 parameter case_only position
    
    return Jh,Complete

def modfunc_res(x, u_ant, v_ant, data):
    """
    Returns a vector containing the residual values over timeslots.
    
    INPUTS:
    data            Vector of measured values.
    u_ant, v_ant    Value of (u,v)baseline measurements at the sampled (timeslot) points.
    x               Vector of source parameters.  A,l,m= x[0],x[1],x[2]    
    """
    
    residual= data - modfunc(x, u_ant, v_ant)
    
    return residual
