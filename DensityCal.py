import numpy as np
from math import pi
### Define sc as the simple cubic lattice with structure like
#   A-B-A-B-A-B
#   | | | | | |
#   A-B-A-B-A-B
### Define fcc as the face center cubic lattice with structure like
#   A-B-A-B-A-B
#   | | | | | |
#   B-A-B-A-B-A

#ALWAYS CHOOSE A TO BE THE SITE WITH HIGHER ONSITE ENERGY!
def energy_1d(kz,tz,gap):  #for 1-D, sc and fcc are the same
    E=np.sqrt((2*tz*np.cos(kz))**2+gap**2)
    return E

def density_1d_plus(kz,tz,gap): #for 1-D, sc and fcc are the same
    phiAplus=2*tz*np.cos(kz)
    phiBplus=gap-np.sqrt(phiAplus**2+gap**2)
    nAplus=phiAplus**2/(phiAplus**2+phiBplus**2)
    nBplus=phiBplus**2/(phiAplus**2+phiBplus**2)
    return np.asarray([nAplus,nBplus])

def density_1d_minus(kz,tz,gap):  #for 1-D, sc and fcc are the same   
    phiAminus=2*tz*np.cos(kz)
    phiBminus=gap+np.sqrt(phiAminus**2+gap**2)
    nAminus=phiAminus**2/(phiAminus**2+phiBminus**2)
    nBminus=phiBminus**2/(phiAminus**2+phiBminus**2)
    return np.asarray([nAminus,nBminus])
        
def sc_integrand_A(kx,ky,kz,txy,tz,gap,mu,T): #integrand for sc case 
    y=(1/(np.exp((-2*txy*(np.cos(kx)+np.cos(ky))+energy_1d(kz,tz,gap)-mu)/T)+1)*density_1d_plus(kz,tz,gap)[0]+
       1/(np.exp((-2*txy*(np.cos(kx)+np.cos(ky))-energy_1d(kz,tz,gap)-mu)/T)+1)*density_1d_minus(kz,tz,gap)[0])/(2*pi)**3
    return y

def sc_integrand_B(kx,ky,kz,txy,tz,gap,mu,T): #integrand for sc case 
    y=(1/(np.exp((-2*txy*(np.cos(kx)+np.cos(ky))+energy_1d(kz,tz,gap)-mu)/T)+1)*density_1d_plus(kz,tz,gap)[1]+
       1/(np.exp((-2*txy*(np.cos(kx)+np.cos(ky))-energy_1d(kz,tz,gap)-mu)/T)+1)*density_1d_minus(kz,tz,gap)[1])/(2*pi)**3
    return y

def energy_3d_fcc(kx,ky,kz,txy,tz,gap): #3-D fcc
    E=np.sqrt((2*tz*np.cos(kz)+2*txy*(np.cos(kx)+np.cos(ky)))**2+gap**2)
    return E

def density_3d_fcc_plus(kx,ky,kz,txy,tz,gap): #3-D fcc
    phiAplus=2*tz*np.cos(kz)+2*txy*(np.cos(kx)+np.cos(ky))
    phiBplus=gap-np.sqrt(phiAplus**2+gap**2)
    nAplus=phiAplus**2/(phiAplus**2+phiBplus**2)
    nBplus=phiBplus**2/(phiAplus**2+phiBplus**2)
    return np.asarray([nAplus,nBplus])

def density_3d_fcc_minus(kx,ky,kz,txy,tz,gap):  #for 1-D, sc and fcc are the same   
    phiAminus=2*tz*np.cos(kz)+2*txy*(np.cos(kx)+np.cos(ky))
    phiBminus=gap+np.sqrt(phiAminus**2+gap**2)
    nAminus=phiAminus**2/(phiAminus**2+phiBminus**2)
    nBminus=phiBminus**2/(phiAminus**2+phiBminus**2)
    return np.asarray([nAminus,nBminus])

def fcc_integrand_A(kx,ky,kz,txy,tz,gap,mu,T): #integrand for sc case 
    y=(1/(np.exp((energy_3d_fcc(kx,ky,kz,txy,tz,gap)-mu)/T)+1)*density_3d_fcc_plus(kx,ky,kz,txy,tz,gap)[0]+
       1/(np.exp((-energy_3d_fcc(kx,ky,kz,txy,tz,gap)-mu)/T)+1)*density_3d_fcc_minus(kx,ky,kz,txy,tz,gap)[0])/(2*pi)**3
    return y

def fcc_integrand_B(kx,ky,kz,txy,tz,gap,mu,T): #integrand for sc case 
    y=(1/(np.exp((energy_3d_fcc(kx,ky,kz,txy,tz,gap)-mu)/T)+1)*density_3d_fcc_plus(kx,ky,kz,txy,tz,gap)[1]+
       1/(np.exp((-energy_3d_fcc(kx,ky,kz,txy,tz,gap)-mu)/T)+1)*density_3d_fcc_minus(kx,ky,kz,txy,tz,gap)[1])/(2*pi)**3
    return y

def sc_density_calculator(txy,tz,gap,mu,T):#calculate the density given chemical potential for all case
    (nA,_)=nquad(sc_integrand_A,[[-pi,pi],[-pi,pi],[-pi,pi]],args=(txy,tz,gap,mu,T))
    (nB,_)=nquad(sc_integrand_B,[[-pi,pi],[-pi,pi],[-pi,pi]],args=(txy,tz,gap,mu,T))
    return np.asarray([nA,nB])

def fcc_density_calculator(txy,tz,gap,mu,T):#calculate the density given chemical potential for all case
    (nA,_)=nquad(fcc_integrand_A,[[-pi,pi],[-pi,pi],[-pi,pi]],args=(txy,tz,gap,mu,T))
    (nB,_)=nquad(fcc_integrand_B,[[-pi,pi],[-pi,pi],[-pi,pi]],args=(txy,tz,gap,mu,T))
    return np.asarray([nA,nB])

