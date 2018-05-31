import math
import numpy as np
import scipy.linalg as LA
import optimal as opt
import master
import slave
import tensorlib_new as T
import time
import sys
beta=1.0/(2.2691)
Lx=512;
nsites=2*(Lx**2)
ntensors=Lx**2;
chi_max=40;
#intitialization of tensors
A=np.zeros([2,2,2,2]);
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
    
fname="TRGIsing_%f_%d.dat"%(1/beta,chi_max)
outfile=open(fname,"w")
dim=np.shape(A)[0]
norm=np.trace(np.reshape(A,[dim**2,dim**2]))
A=A/norm
logZ=0;
logZ+= (ntensors)*math.log(norm)
##Exact Coarse Graining
tensors=[A,A,A,A] 
contraction=[[1,-9,-12,7],[2,-9,-10,3],[-10,-11,6,4],[-12,-11,5,8]]
T.network(tensors,contraction,[-9,-11,-10,-12])
A=tensors[0].reshape([dim**2,dim**2,dim**2,dim**2]) 
ntensors=ntensors/4;
dim=np.shape(A)[0]
norm=np.trace(np.reshape(A,[dim**2,dim**2]))
G=norm;
G_old=G
A=A/norm
logZ+= (ntensors)*math.log(norm)
print -(1/beta)*logZ/nsites


S1=S2=S3=S4=np.zeros([chi_max,chi_max,chi_max])
A_tensors=[S1,S2,S3,S4]
A_shapes=[np.shape(tensor) for tensor in A_tensors]
A_contr=[[-3,-4,7],[9,-1,-2],[-2,-4,8],[6,-3,-1]]
contr_tmp=[A_contr[j][:] for j in range(0,len(A_contr))]
shapes_tmp=[A_shapes[j][:] for j in range(0,len(A_shapes))]
instr_A,transp_A=master.instr_gen(shapes_tmp,contr_tmp);



# RG transformation
rg_step=1
while(ntensors>4) :
    print "ntensors=",ntensors
    S1,S2=T.svdD(A,[0,3],[1,2],chi_max)
    #test=np.tensordot(S1,S2,[])
    S3,S4=T.svdD(A,[0,1],[2,3],chi_max)
    A_tensors=[S1,S2,S3,S4]
    slave.contract(A_tensors,instr_A,transp_A)
    A=A_tensors[0];



    dim=np.shape(A)
    norm=np.trace(np.reshape(A,[dim[0]*dim[1],dim[2]*dim[3]]))
    A=A/norm;

    if(rg_step>0) : 
      f=norm;
      print "f=",f
      L=f
      T_inv=A/L
      tmp=[T_inv]
      T.network(tmp,[[-1,2,-1,3]],[-1])
      e,v=LA.eigh(tmp[0])
      #print e


      if(e[-1]>0) :
        charge=(6.0/math.pi)*math.log(e[-1]);
        print "Central charge =", (6.0/math.pi)*math.log(e[-1]);
        print "1st scaling dimension  =",((charge/(12.0))-math.log(e[-2])/(2*math.pi))
        print "2st scaling dimension  =",((charge/(12.0))-math.log(e[-3])/(2*math.pi))
        print "2st scaling dimension  =",((charge/(12.0))-math.log(e[-4])/(2*math.pi))
        print "2st scaling dimension  =",((charge/(12.0))-math.log(e[-5])/(2*math.pi))
        print "2st scaling dimension  =",((charge/(12.0))-math.log(e[-6])/(2*math.pi))
        print "2st scaling dimension  =",((charge/(12.0))-math.log(e[-7])/(2*math.pi))
        print "2st scaling dimension  =",((charge/(12.0))-math.log(e[-8])/(2*math.pi))
      print "f =", f
      charge= (6.0/math.pi)*math.log(e[-1]);
      l1=((charge/(12.0))-math.log(e[-2])/(2*math.pi))
      l2=((charge/(12.0))-math.log(e[-3])/(2*math.pi))
      l3=((charge/(12.0))-math.log(e[-4])/(2*math.pi))
      l4=((charge/(12.0))-math.log(e[-5])/(2*math.pi))
      l5=((charge/(12.0))-math.log(e[-6])/(2*math.pi))
      l6=((charge/(12.0))-math.log(e[-7])/(2*math.pi))
      l7=((charge/(12.0))-math.log(e[-8])/(2*math.pi))
      #sys.stdin.read(1)
    
    

    ntensors=ntensors/2
    logZ+= (ntensors)*math.log(norm)
    print -(1/beta)*logZ/nsites
    print>>outfile,"%d %f %f %f %f %f %f %f %f "%(rg_step,charge,l1,l2,l3,l4,l5,l6,l7)
    rg_step=rg_step+1
    


f1=open('dataTRG.dat',"a")
print>>f1,"%.16f %.16f"%(1/beta,-(1/beta)*logZ/nsites)


