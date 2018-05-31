#import libraries
import numpy as np
import scipy.linalg as LA
from scipy import integrate
import matplotlib.pylab as pl
import tensorlib as T

#intitialize constants
d=2
dt=0.01
g=1.0
glist=np.arange(0,2,0.1)
tsteps=1000
Sz=[[1,0],[0,-1]]
Sx=[[0,1],[1,0]]
chi_max=30



def tebd(G,s,U,chi_max ) :
    #apply to alternate bonds
    for index in [0,1] :
        i1=index
        i2=(index+1)%2
        
        tensors=[G[i1],G[i2],np.diag(s[i1]),np.diag(s[i2]),np.diag(s[i2]),U]
        contraction=[[-5,-1,-3],[-4,-2,-6],[-3,-4],[0,-5],[-6,3],[1,2,-1,-2]]
        order=[-3,-4,-1,-2,-5,-6]
        theta=T.network( tensors, contraction,order)
        #svd and truncate
        X,s[i1],Z=T.tensor_svd(theta,[0,1],[2,3],chi_max)
        #update tensors
        G[i1]=T.network([np.diag(s[i2]**-1),X],[[0,-1],[-1,1,2]],[-1])
        G[i2]=T.network([Z,np.diag(s[i2]**-1)],[[0,1,-1],[-1,2]],[-1])


def twosite(G,s,O) :
    E=0
    for index in range (0,2) :
        i1=index
        i2=(index+1)%2
        tensors=[G[i1],G[i2],np.diag(s[i1]),np.diag(s[i2]),np.diag(s[i2])]
        contraction=[[-1,1,-2],[-3,2,-4],[-2,-3],[0,-1],[-4,3]]
        order=[-2,-3,-1,-4]
        state=T.network(tensors,contraction,order)
        statec=np.conj(state)
        tensors=[state,O,statec]
        contraction=[[-1,-2,-3,-4],[-2,-3,-5,-6],[-1,-5,-6,-4]]
        order=[-2,-3,-5,-6,-1,-4]
        E+=T.network(tensors,contraction,order)
    return E/2.0



E=[];Ee=[];diff=[];S=[];mag=[];
M=np.kron(Sz,Sz)
for g in glist :
    f = lambda k,g : -2*np.sqrt(1+g**2-2*g*np.cos(k))/np.pi/2.
    E0_exact = integrate.quad(f, 0, np.pi, args=(g,))[0]

    H=-np.kron(Sz,Sz)+g*np.kron(Sx,np.eye(d))
    U=np.reshape(LA.expm(-H*dt),[d,d,d,d])
    
    #make starting tensors
    G=[]
    s=[]
    for i in range (0,2) :
        G.append(np.zeros([1,2,1]));
        s.append(np.ones([1]));

    G[0][0,0,0]=1.0
    G[1][0,0,0]=1.0
    
    #run itebd
    for i in range (0,500):
        tebd(G,s,U,chi_max)
        
    #calculate observables
    En=twosite(G,s,np.reshape(H,[d,d,d,d]))
    entr=-np.dot(s[0],np.log(s[0]))
    mag.append(twosite(G,s,np.reshape(M,[d,d,d,d])))    
    Ee.append(E0_exact)
    E.append(En)
    S.append(entr)
    diff.append(En-E0_exact)


pl.plot(glist,S)
pl.show()




