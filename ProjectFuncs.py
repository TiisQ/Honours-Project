import numpy as np

def modfunc(x, u_ant, v_ant):
    """
    Returns a vector containing the values of a (3*2t,1) model exponential matrix flattened with source     parameters A0,l0,m0.
    
    INPUTS:
    x               Vector of source parameters.  l,m,Amp= x[0],x[1],x[2]  
    u_ant, v_ant    Value of (u,v)baseline measurements at the sampled (timeslot) points.
    """
    
    n_ant= u_ant.shape[0] #3 for test case
    
    l= x[:,0] # for multiple sources x must be Num_of_Sources long,for now x[0] is len 1 because there is 1 source
    m= x[:,1]
    Amp= x[:,2]
    
    model_stream= 0  ## change 14/09
    for i in range(len(x)):  ## change 14/09
        model= Amp[i]*np.exp(-2j*np.pi*(u_ant*l[i]+v_ant*m[i]))
        model[range(n_ant), range(n_ant), :]= 0

        model_stream += model[np.where(model!=0)]  ## change 14/09
        ## with t timeslots each
    return model_stream

def modfunc_jac(x,u_ant,v_ant):
    """
    Returns the Hessian and Jacobian corresponding to the function defined by model equation.
    
    INPUTS:
    x               Vector of source parameters.  l,m,Amp= x[0],x[1],x[2] 
    u_ant, v_ant    Value of (u,v)baseline measurements at the sampled (timeslot) points. use this as a loop
   
    """
    t= u_ant.shape[2]
    n_ant= u_ant.shape[0]
    
    l= x[:,0]
    m= x[:,1]
    Amp= x[:,2]
    
    J_mult=[]
    for i in range(len(x)): ## change 14/09
    
        ## Calculating JHJ explicitly from J
        Psi= -2j*np.pi*(u_ant*l[i]+v_ant*m[i])
        expPsi= lambda sign: np.exp(sign*Psi)

        coeU = -2j*np.pi*Amp[i]*u_ant
        coeV = -2j*np.pi*Amp[i]*v_ant
        ## remember the -(minus) for negative exponents

        Jkl= coeU*expPsi(1) 
        Jkl[range(n_ant), range(n_ant), :]=0

        Jkm= coeV*expPsi(1) 
        Jkm[range(n_ant), range(n_ant), :]=0

        JkA= expPsi(1)
        JkA[range(n_ant), range(n_ant), :]=0

        Jk_it= np.vstack((Jkl[np.where(Jkl!=0)],Jkm[np.where(Jkm!=0)],JkA[np.where(Jkl!=0)]))  ##to vstack again, remove .T
        J_mult.append(Jk_it) ## change 14/09
        #print("Shape of J_mult: ",np.shape(J_mult))
    
    #print("J_mult", np.array(J_mult).shape)
    
    Jk= np.concatenate(J_mult,axis=0) ## change 14/09
    #print("Shape of Jk: ",np.shape(Jk))
    
    Jh= Jk.conj() ## Hermitian of J
    #print("Shape of Jh: ",np.shape(Jh))
    Complete= Jh.dot(Jk.T)
    
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
