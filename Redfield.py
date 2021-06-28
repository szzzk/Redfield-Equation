import numpy as np
from tbHamiltonian import tbH
from RK4 import rk4
import matplotlib.pyplot as plt
from scipy.linalg import solve_sylvester  
from scipy.linalg import block_diag
import numpy.linalg as LA
from math import pi
from scipy.linalg import toeplitz
from DensityCal import sc_density_calculator,fcc_density_calculator
from Ladder import ladder

global PARA
PARA={
      'hoppingz':1,
      'hoppingxy':0,
      'intercoupling':1, #coupling between junction
      'onsiteU':0,
      'DIRV':-0.0*0.01*(4*pi),# direct interaction magnitude
      'length':100,
      'juncL':4, #the region at ends where the charges' contribution to the potential is neglected
      'xdim':1,
      'ydim':1,
      'dopemu':0, #onsite potential [-dopemu,dopemu]
      'muL':2,#local chemical potential at left with respect to center of the band(gap)
      'muR':2,#local chemical potential at right with respect to center of the band(gap)
      'gap':0,#gap of the insulator, if there is (SHOULD BE POSITIVE!)
      'Efield':0, # (set it to be zero forever to get correct ground state)
      'frequency':0,
      'lattice':'sc', #'sc' for simple cubic, 'fcc' for face-centered cubic
      'ACmagnitude':0.05, #the oscillating magnitude of electrochemical potential
      'DCmagnitude':0, #the biased magnitude of electrochemical potential on right side
      'spinoption':'spinless', #spinless assumes SU(2) symmetry of state at all time, spinful treats complete problem
      'T':0.1, #temperature
      'currentMode':'DC',      
      }

DISSPARA={
        'SBcoupling':0.1, #=pi*J
        }

TIME_PARAMETERS={
                'timestep':0.005,
                'iteration':0,
                }

if PARA['lattice']=='sc' and PARA['DIRV']!=0:
    nL=sc_density_calculator(txy=PARA['hoppingxy'],tz=PARA['hoppingz'],gap=PARA['gap'],mu=PARA['muL'],T=PARA['T'])
    nR=sc_density_calculator(txy=PARA['hoppingxy'],tz=PARA['hoppingz'],gap=PARA['gap'],mu=PARA['muR'],T=PARA['T'])
    PARA['PosCharge']=np.c_[np.matlib.repmat(nL,1,int(PARA['length']/4)),np.matlib.repmat(nR,1,int(PARA['length']/4))] #postive charge background per spin
elif PARA['lattice']=='fcc' and PARA['DIRV']!=0:
    nL=fcc_density_calculator(txy=PARA['hoppingxy'],tz=PARA['hoppingz'],gap=PARA['gap'],mu=PARA['muL'],T=PARA['T'])
    nR=fcc_density_calculator(txy=PARA['hoppingxy'],tz=PARA['hoppingz'],gap=PARA['gap'],mu=PARA['muR'],T=PARA['T'])    
    PARA['PosCharge']=np.c_[np.ones((1,int(PARA['length']/4)))*(nL[0]+nL[1])/2,np.ones((1,int(PARA['length']/4)))*(nR[0]+nR[1])/2]
elif PARA['lattice']=='sc' and PARA['DIRV']==0:
    PARA['PosCharge']=np.ones((1,PARA['length']))
elif PARA['lattice']=='fcc' and PARA['DIRV']==0:
    PARA['PosCharge']=np.ones((1,int(PARA['length']/2)))
    
def smooth(length,window): #gives matrix that is used to calculate smoothed density
    if length<=window or window % 2 ==0:
        assert(0)
    mat=toeplitz(np.c_[np.ones((1,int((window+1)/2)))*1/window,np.zeros((1,length-int((window+1)/2)))])
    mat[:int((window+1)/2),:window]=1/window
    mat[-int((window+1)/2):,-window:]=1/window
    return mat
   
def tevolveplot(density,jDensity,pumpedCharge):
    trange=np.arange(TIME_PARAMETERS['iteration']+1)*TIME_PARAMETERS['timestep']
    
    plt.figure(1)
    plt.plot(trange,jDensity[int((jDensity.shape[0]-1)/2),:].T)
    plt.xlabel('time')
    plt.ylabel('current')
    
    plt.figure(2)
    plt.plot(trange,pumpedCharge)
    plt.xlabel('time')
    plt.ylabel('chargeTransferred')
   # plt.ylim(-0.0003,0.0003)
   
def sort(d,v):
    sortind=np.argsort(d)  
    dd=d[sortind]
    vv=v[:,sortind]
    return [dd,vv]

def function(t,rho):
    HFint=interaction(rho)
    (P,rhoP)=jumpOp(t,HFint,kx,ky) #output M and N matrix defined in the note
    return -1j * ((Hamiltonians(kx,ky)+HFint) @ rho-rho @ (Hamiltonians(kx,ky)+HFint))-(P @ rho + rho @ P)+rhoP

def mult2oneD(multDrho):
    '''This function transform a multi-dimensional density matrix to one-D and give occupancy and current.'''
    if len(multDrho.shape)==2:
        multDrho=np.reshape(multDrho,(multDrho.shape[0],multDrho.shape[1],1,1))
    oneDrho=np.sum(np.sum(multDrho,axis=3),axis=2)/(PARA['xdim']*PARA['ydim'])#the density matrix that give quantities along z-direction (sum over kxky)
    occ=np.diag(oneDrho).real
    if PARA['lattice']=='sc':
        current=-1j*PARA['hoppingz']*(np.diag(oneDrho,k=1)-np.diag(oneDrho,k=-1)) #include negative charge
        current[int(PARA['length']/2)-1]=-1j*PARA['intercoupling']*(oneDrho[int(PARA['length']/2)-1,int(PARA['length']/2)]-oneDrho[int(PARA['length']/2),int(PARA['length']/2)-1])
    elif PARA['lattice']=='fcc':
        jab=-1j*PARA['hoppingz']*(np.diag(oneDrho,k=int(PARA['length']/2)+1)-np.diag(oneDrho,k=-int(PARA['length']/2)-1)) #current from A atom to B atom
        jba=-1j*PARA['hoppingz']*(np.diag(oneDrho,k=-int(PARA['length']/2)+1)-np.diag(oneDrho,k=int(PARA['length']/2)-1))[1:-1] #current from B atom to A atom
        current=(jab+jba)/2
    return [oneDrho,occ,current.real]

def time_evolve(initialrho,iteration,timestep):
    global kx,ky
    rho=initialrho
    occ=np.zeros((PARA['length'],iteration+1))
    if PARA['lattice']=='sc':    
        current=np.zeros((PARA['length']-1,iteration+1)) 
    elif PARA['lattice']=='fcc':
        current=np.zeros((int(PARA['length']/2)-1,iteration+1)) 
    [oneDrho,occ[:,0],current[:,0]]=mult2oneD(initialrho)    
    pumpedCharge=np.zeros(iteration+1)
    pumpedCharge[0]=current[int((current.shape[0]-1)/2),0]*timestep
    for ii in range(iteration):
        for kxcount in np.arange(PARA['xdim']):
            for kycount in np.arange(PARA['ydim']):
                kx=2*pi/PARA['xdim']*kxcount
                ky=2*pi/PARA['ydim']*kycount
                rho[:,:,kxcount,kycount]=rk4(function,ii*timestep,rho[:,:,kxcount,kycount],timestep)
        [oneDrho,occ[:,ii+1],current[:,ii+1]]=mult2oneD(rho)    
        pumpedCharge[ii+1]=pumpedCharge[ii]+current[int((current.shape[0]-1)/2),ii+1]*timestep
        #print('Time-evolution process:','{:.1%}'.format((ii+1)/iteration))
    finalrho=rho
    return [occ,current,pumpedCharge,finalrho]

def Hamiltonians(kx,ky,lat=PARA['lattice']):#define the Hamiltonian for sc lattice in 3-D (same as 1-D)
    if lat=='sc':
        staggerMu=np.empty((int(PARA['length']/2),1)) #staggered chemical potential
        staggerMu[::2]=1
        staggerMu[1::2]=-1    
        Lelements=tbH(int(PARA['length']/2),PARA['gap']*staggerMu-PARA['dopemu']*np.ones((int(PARA['length']/2),1)),PARA['spinoption'])      
        Relements=tbH(int(PARA['length']/2),PARA['gap']*staggerMu+PARA['dopemu']*np.ones((int(PARA['length']/2),1)),PARA['spinoption'])          
        LH=Lelements[0]+(-PARA['hoppingz'])*Lelements[1]+(-PARA['hoppingz'])*Lelements[2]
        RH=Relements[0]+(-PARA['hoppingz'])*Relements[1]+(-PARA['hoppingz'])*Relements[2]
        totalH=block_diag(LH,RH)
        totalH[int(PARA['length']/2),int(PARA['length']/2)-1]=-PARA['intercoupling']
        totalH[int(PARA['length']/2)-1,int(PARA['length']/2)]=-PARA['intercoupling']
        totalH=totalH+np.eye(PARA['length'])*(-2*PARA['hoppingxy']*(np.cos(kx)+np.cos(ky)))
    elif lat=='fcc':
        tab=np.c_[np.asarray([[-2*PARA['hoppingxy']*(np.cos(kx)+np.cos(ky)),-PARA['hoppingz']]]),np.zeros((1,int(PARA['length']/2)-2))]
        totalH=ladder(int(PARA['length']/2),tA=np.zeros((1,int(PARA['length']/2)-1)),tB=np.zeros((1,int(PARA['length']/2)-1)),
                      EA=PARA['gap'],EB=-PARA['gap'],t_AB_R=tab,t_AB_L=tab)
        totalH=totalH+np.diag(np.r_[-PARA['dopemu']*np.ones(int(PARA['length']/4)),PARA['dopemu']*np.ones(int(PARA['length']/4)),
                                    -PARA['dopemu']*np.ones(int(PARA['length']/4)),PARA['dopemu']*np.ones(int(PARA['length']/4))])
    return totalH

def jumpOp(time,intU,kx,ky):
    global leftmu,rightmu,staticmuL,staticmuR
    if PARA['currentMode']=='AC': 
        if time!=0:
            Er=(leftmu+rightmu)/2+PARA['ACmagnitude']*np.sin(PARA['frequency']*time) #fermi level of left and right baths
            El=Er-2*PARA['ACmagnitude']*np.sin(PARA['frequency']*time)
        else:
            Er=(leftmu+rightmu)/2+PARA['onsiteU']/2
            El=Er
    elif PARA['currentMode']=='DC':
        if time!=0:
            Er=(leftmu+rightmu)/2+PARA['DCmagnitude']
            El=Er-2*PARA['DCmagnitude']
        else:
            Er=(leftmu+rightmu+PARA['muL']+PARA['muR'])/2+PARA['onsiteU']/2+PARA['DCmagnitude']
            El=Er-2*PARA['DCmagnitude']
            
    [D,V]=LA.eigh(Hamiltonians(kx,ky)+intU)
    PL=np.zeros((PARA['length'],PARA['length']))
    PR=np.zeros((PARA['length'],PARA['length']))    
    if PARA['lattice']=='sc':    
        PL[0,0]=1
        PR[-1,-1]=1
    elif PARA['lattice']=='fcc':
        PL[0,0]=1
        PL[int(PARA['length']/2),int(PARA['length']/2)]=1
        PR[-1,-1]=1
        PR[int(PARA['length']/2)-1,int(PARA['length']/2)-1]=1
        
    fl=1/(np.exp((D-El)/PARA['T'])+1)
    fr=1/(np.exp((D-Er)/PARA['T'])+1)

    rhoL=DISSPARA['SBcoupling']*(V @ np.diag(fl) @ V.conj().T) #n matrix for left side
    rhoR=DISSPARA['SBcoupling']*(V @ np.diag(fr) @ V.conj().T) #n matrix for right side
    
    rhoPL=rhoL @ PL + PL @ rhoL #{rho,P}
    rhoPR=rhoR @ PR + PR @ rhoR

    return ((PL+PR)*DISSPARA['SBcoupling'],rhoPL+rhoPR)


def staticstate_solver(trialInt=np.zeros((PARA['length'],PARA['length'])),iteration=2500,thres=10**(-7),w=0.01): #trialInt inputs the trial interaction
    '''try to solve Arho+rhoB+N=0--solving static state for given Hamiltonian, with the dissipator at t=0'''
    global leftmu,rightmu
    rhoRecord=np.zeros((PARA['length'],PARA['length'],PARA['xdim'],PARA['ydim']),dtype=complex)
    oldint=np.empty((PARA['length'],PARA['length']),dtype=complex)
    
    if (PARA['onsiteU']==0 and PARA['DIRV']==0): 
        for kxcount in np.arange(PARA['xdim']):
            for kycount in np.arange(PARA['ydim']):                
                rightmu=0
                leftmu=0
                kx=2*pi/PARA['xdim']*kxcount
                ky=2*pi/PARA['ydim']*kycount    
                (P,rhoP)=jumpOp(0,np.zeros((PARA['length'],PARA['length'])),kx,ky) #output M and N matrix at t=0
                A0=-1j*Hamiltonians(kx,ky)-P
                B0=1j*Hamiltonians(kx,ky)-P
                rhoRecord[:,:,kxcount,kycount]=solve_sylvester(A0,B0,-rhoP) #solve non-interacting case as initial guess
                oldint=interaction(rhoRecord)
    
    if PARA['onsiteU']!=0 or PARA['DIRV']!=0: 
        
        if np.all(trialInt==np.zeros((PARA['length'],PARA['length']))):            
            depL=6 #depletion size ansatz
            if PARA['lattice']=='sc':
                ansatzpotential=np.tanh((np.arange(PARA['length'])-(PARA['length']-1)/2)/depL)*(-(PARA['muR']-PARA['muL'])/2+PARA['DCmagnitude']) #estimate the potential on each sites                 
                ansatzU=PARA['PosCharge']*PARA['onsiteU']
            else:
                ansatzpotential=np.tanh((np.arange(int(PARA['length']/2))-(int(PARA['length']/2)-1)/2)/depL)*(-(PARA['muR']-PARA['muL'])/2+PARA['DCmagnitude']) #estimate the potential on each sites                 
                ansatzpotential=np.r_[ansatzpotential,ansatzpotential]
                ansatzU=np.zeros(PARA['length'])      
            
            if PARA['gap']==0:
                ansatzpotential=np.zeros(PARA['length'])
            
            if PARA['juncL']!=0:
                leftmu=(ansatzpotential.flatten())[PARA['juncL']-1]
                rightmu=(ansatzpotential.flatten())[-PARA['juncL']]
            else:
                leftmu=(ansatzpotential.flatten())[0]
                rightmu=(ansatzpotential.flatten())[-1]
                
            oldint=np.diag((ansatzpotential+ansatzU).flatten())
        else:
            oldint=trialInt.copy()        
        
        for ii in range(iteration): #Use iteration to find self-consistent solution
            for kxcount in np.arange(PARA['xdim']):
                for kycount in np.arange(PARA['ydim']):
                    kx=2*pi/PARA['xdim']*kxcount
                    ky=2*pi/PARA['ydim']*kycount
                    (P,rhoP)=jumpOp(0,oldint,kx,ky)
                    A=-1j*(Hamiltonians(kx,ky)+oldint)-P
                    B=1j*(Hamiltonians(kx,ky)+oldint)-P
                    [eigA,_]=LA.eig(A)
                    #print('smallest eigenvalue of A:',ii,np.min(np.abs(np.real(eigA))))
                    if ii==0:
                        rhoRecord[:,:,kxcount,kycount]=solve_sylvester(A,B,-rhoP)
                        [_,occ,_]=mult2oneD(rhoRecord)
#                        netcharge=occ-PARA['PosCharge']
                    else:
                        newrho=solve_sylvester(A,B,-rhoP)
                        rhoRecord[:,:,kxcount,kycount]=w*newrho+(1-w)*rhoRecord[:,:,kxcount,kycount]
                        [_,occ,_]=mult2oneD(newrho)
#                        netcharge=occ-PARA['PosCharge']
            if np.any(occ<0):
                rhoRecord[:,:,:,:]=np.nan
                print('Positivity is broken, forced to quit')
                break
                
            newint=interaction(rhoRecord)
            maxdiff=np.max(abs(newint-oldint).flatten())
            print(ii,maxdiff)
            if np.all(abs(newint-oldint)<= thres):
                if np.any(np.abs(eigA.real)<=10**(-15)):
                    print('the solution may not be reliable')
                break
            if ii==iteration-1:
                rhoRecord[:,:,:,:]=np.nan
                print('Iteration does not converge, the difference of newint and oldint is',maxdiff)
            if ii==0:    
                oldint=(newint+newint.conj().T)/2*w+(1-w)*oldint
            else:    
                oldint=(newint+newint.conj().T)/2
    return rhoRecord

def interaction(rho): #output matrix related to interaction and virtual hopping and external electric field
    global DirectV,leftmu,rightmu
    [oneDrho,occ,_]=mult2oneD(rho)
    intU=PARA['onsiteU']*np.diag(np.abs(occ)) #on-site interaction term 
    if PARA['lattice']=='sc':
        intV = np.diag((DirectV @ (occ-PARA['PosCharge']).reshape(-1,1)).flatten()) #the off-site interaction                       
    elif PARA['lattice']=='fcc':
        occ=(occ[:int(PARA['length']/2)]+occ[int(PARA['length']/2):])/2
        intV = np.diag((np.matlib.repmat(DirectV @ (occ-PARA['PosCharge']).reshape(-1,1),2,1)).flatten()) #the off-site interaction 
    if PARA['juncL']!=0:
        leftmu=intV[PARA['juncL']-1,PARA['juncL']-1]
        rightmu=intV[-PARA['juncL'],-PARA['juncL']]
    else:
        leftmu=intV[0,0]
        rightmu=intV[-1,-1] 
    return intU+intV

           
def IVcurve(Vrange,optrange,opt): #optrange is U when opt='U', mu when opt='mu', SBcoupling when opt='J'
    global DirectV 
    LRI()
    I=np.zeros((Vrange.shape[0],optrange.shape[0]))
    potential=np.empty((Vrange.shape[0],PARA['length']))*np.nan
    density=np.empty((Vrange.shape[0],PARA['length']))*np.nan
    Zeroind=np.argmin(abs(Vrange))
    lastint=np.zeros((PARA['length'],PARA['length']))
    
    for jj in range(optrange.shape[0]):
        if opt=='U':
            PARA['onsiteU']=optrange[jj]
        elif opt=='mu':
            PARA['dopemu']=optrange[jj]
        elif opt=='J':
            DISSPARA['SBcoupling']=optrange[jj]
        elif opt=='intert':
            PARA['intercoupling']=optrange[jj]
        
        PARA['DCmagnitude']=Vrange[Zeroind]#First solve V=0 case to get trial interaction
        zerorho=staticstate_solver() #find rho when V=0 for different U as a trial density matrix
        [_,occ,tmp]=mult2oneD(zerorho)
        I[Zeroind,jj]=tmp[int((tmp.shape[0]-1)/2)]
        print('V:',Vrange[Zeroind],'I:',I[Zeroind,jj],'opt:',optrange[jj])
        zeroint=interaction(zerorho) #gives the interaction matrix for V=0
        lastint[:,:]=zeroint[:,:]
        potential[Zeroind,:]=np.diag(zeroint)
        density[Zeroind,:]=occ
        
        for ii in range(Zeroind+1,Vrange.shape[0]):                        
            PARA['DCmagnitude']=Vrange[ii]
            staticrho=staticstate_solver() #find steady solution trialInt=lastint
            [_,occ,tmp]=mult2oneD(staticrho)
            I[ii,jj]=tmp[int((tmp.shape[0]-1)/2)]
            print('V:',Vrange[ii],'I:',I[ii,jj],'opt:',optrange[jj])
            if np.isnan(I[ii,jj])==False:
                lastint=interaction(staticrho)
                potential[ii,:]=np.diag(lastint)
                density[ii,:]=occ
        
        lastint[:,:]=zeroint[:,:]
        for ii in range(Zeroind-1,-1,-1):            
            PARA['DCmagnitude']=Vrange[ii]
            staticrho=staticstate_solver() #find steady solution trialInt=lastint
            [_,occ,tmp]=mult2oneD(staticrho)
            I[ii,jj]=tmp[int((tmp.shape[0]-1)/2)]
            print('V:',Vrange[ii],'I:',I[ii,jj],'opt:',optrange[jj])
            if np.isnan(I[ii,jj])==False:
                lastint=interaction(staticrho)
                potential[ii,:]=np.diag(lastint)
                density[ii,:]=occ
                
        plt.figure(1)
        plt.plot(Vrange,I[:,jj],'.',label='%.1f'%float(optrange[jj]))
        plt.legend(loc='upper left')
    return [I,potential,density]

def LRI():#generates long range interaction matrix elements
    global DirectV
    if PARA['gap']!=0:
        if PARA['length'] % 4 !=0:
            assert(0)            
    if PARA['lattice']=='sc':        
        DirectV_V=PARA['DIRV']*np.arange(PARA['length'])
    elif PARA['lattice']=='fcc':
        DirectV_V=PARA['DIRV']*np.arange(int(PARA['length']/2))
#    DirectV_V[0]=DirectV_V[1]
    DirectV=toeplitz(np.r_[DirectV_V]) #Long range direct interaction matrix
    if PARA['juncL']!=0:
        DirectV[:,:PARA['juncL']]=0 #force that the interaction contributed from the ends are zero 
        DirectV[:,-PARA['juncL']:]=0 
        
def Single_run():
    global kx,ky,DirectV,staticmuL,staticmuR
    LRI()
    staticrho=staticstate_solver(thres=10**(-9)) #find steady solution
    staticmuR=rightmu
    staticmuL=leftmu
    [density,jDensity,pumpedCharge,frho]=time_evolve(staticrho,TIME_PARAMETERS['iteration'],TIME_PARAMETERS['timestep'])
    #tevolveplot(density,jDensity,pumpedCharge)
    return [density,jDensity,pumpedCharge,frho,staticrho]     

[density,jdensity,_,frho,staticrho]=Single_run()
[oneDrho,_,_]=mult2oneD(staticrho)
potential=np.diag(interaction(staticrho))

if PARA['lattice'] == 'sc':
    print('net charge is',-np.sum(density.T-PARA['PosCharge']))
    netcharge=density[:,0]-PARA['PosCharge'].T[:,0]
    plt.figure(1)
    plt.plot(np.arange(int(PARA['length'])),netcharge[:int(PARA['length'])],'b')
    plt.figure(2)
    plt.plot(np.arange(PARA['length']),potential,'y')
else:
    netcharge=(density[:int(PARA['length']/2),0]+density[int(PARA['length']/2):,0])/2-PARA['PosCharge'].T[:,0]
    plt.plot(np.arange(int(PARA['length']/2)),netcharge[:int(PARA['length']/2)],'y')