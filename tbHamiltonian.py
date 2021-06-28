import numpy as np
from scipy.linalg import toeplitz

def tbH(length,onsiteV,spinOpt='spinless'):
    '''This function aims to give matrix element of c_{i+1}^\dagger c_i, its H.c., and diagonal elements in
    single-particle Hilbert space'''
    if spinOpt!='spinless' and spinOpt!='spinful':
        assert (0)
    R=np.zeros((1,length))
    C=np.zeros((length,1))
    R[0,1]=1
    C[1,0]=1
    H_1=toeplitz(c=C,r=np.zeros((1,length)))#c_{i+1}^\dagger c_{i}
    H_2=toeplitz(c=np.zeros((length,1)),r=R)#c_{i}^\dagger c_{i+1}
    H_diag=np.diag(onsiteV.flatten())
    if spinOpt=='spinful':
        H_1=np.block([[H_1,np.zeros((length,length))],[np.zeros((length,length)),H_1]])
        H_2=np.block([[H_2,np.zeros((length,length))],[np.zeros((length,length)),H_2]])
        H_diag=np.block([[H_diag,np.zeros((length,length))],[np.zeros((length,length)),H_diag]])
    return (H_diag,H_1,H_2)