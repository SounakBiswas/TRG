import math
import numpy as np
import scipy.linalg as LA
import optimal as opt
import master
import slave
import tensorlib_new as T
import time
import sys
chig=6;
chiv=6;
chiw=6;
chiy=6;
chiu=4;
chis=4
beta=1.0/(2.26)
optSteps=400
Lx=256;
nsites=2*(Lx**2)
curr_step=1;
#intitialization for Ising
A=np.zeros([2,2,2,2]);
#make A
logZ=0;
for i in range(0,2):
  for j in range(0,2) :
    for k in  range(0,2) :
      for l in  range(0,2) :
        i1=2*i-1
        i2=2*j-1
        i3=2*k-1
        i4=2*l-1
        A[i,j,k,l]=math.exp(beta*(i1*i2+i2*i3+i3*i4+i1*i4))
#A,free=T.normalize(A,[0,1],[2,3])
print A-np.transpose(A,[0,3,2,1])
sys.stdin.read(1)
#print A
Ac=np.conjugate(A)
tensors=[A,Ac]
contraction=[[1,-7,4,6],[2,-7,5,3]]
T.network(tensors,contraction,[-7])
A=tensors[0]
print A-np.transpose(A,[0,1,5,3,4,2])
sys.stdin.read(1)

