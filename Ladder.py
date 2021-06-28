import numpy as np
from scipy.linalg import toeplitz
###############################################################################
#  Convention:
#
#  A-A-A-A-A-A          <A,i|H|A,i>=E_i^A        <B,i|H|B,i>=E_i^B
#     /|\               <A,i|H|A,i+j>=-t_A[j-1]       <B,i|B|A,i+j>=-t_B[j-1]
#  B-B-B-B-B-B    <A,i|H|B,i>=-t_0   <A,i|H|B,i+1>=-t_1   <A,i|H|B,i-1>=-t_{-1} 
#
###############################################################################
def ladder(length,tA,tB,EA,EB,t_AB_R,t_AB_L):
    if t_AB_R[0,0]!=t_AB_L[0,0]:
        assert(0)
    HA=toeplitz(c=np.c_[EA,-tA.conj()])
    HB=toeplitz(c=np.c_[EB,-tB.conj()])
    HAB=toeplitz(c=t_AB_L,r=t_AB_R)
    HBA=HAB.conj().T
    H=np.block([[HA,HAB],[HBA,HB]])
    return H