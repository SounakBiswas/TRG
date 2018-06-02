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
chi_max=30;
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
#tensors=[A,A,A,A] 
#contraction=[[1,-9,-12,7],[2,-9,-10,3],[-10,-11,6,4],[-12,-11,5,8]]
#T.network(tensors,contraction,[-9,-11,-10,-12])
#A=tensors[0].reshape([dim**2,dim**2,dim**2,dim**2]) 
#ntensors=ntensors/4;
#dim=np.shape(A)[0]
#norm=np.trace(np.reshape(A,[dim**2,dim**2]))
#G=norm;
#G_old=G
#A=A/norm
#logZ+= (ntensors)*math.log(norm)

M_tensors=[A,A]
M_shapes=[np.shape(tensor) for tensor in M_tensors]
M_contr=[[1,-7,4,6],[2,3,5,-7]]
contr_tmp=[M_contr[j][:] for j in range(0,len(M_contr))]
shapes_tmp=[M_shapes[j][:] for j in range(0,len(M_shapes))]
instr_V,transp_V=master.instr_gen(shapes_tmp,contr_tmp);

M_tensors=[A,A]
M_shapes=[np.shape(tensor) for tensor in M_tensors]
M_contr=[[1,2,-7,5],[-7,3,4,6]]
contr_tmp=[M_contr[j][:] for j in range(0,len(M_contr))]
shapes_tmp=[M_shapes[j][:] for j in range(0,len(M_shapes))]
instr_H,transp_H=master.instr_gen(shapes_tmp,contr_tmp);

U=np.zeros([dim,dim])
M_tensors=[A,U,U]
M_shapes=[np.shape(tensor) for tensor in M_tensors]
M_contr=[[-5,2,-6,4],[-5,1],[-6,3]]
contr_tmp=[M_contr[j][:] for j in range(0,len(M_contr))]
shapes_tmp=[M_shapes[j][:] for j in range(0,len(M_shapes))]
instr_tH,transp_tH=master.instr_gen(shapes_tmp,contr_tmp);

M_tensors=[A,U,U]
M_shapes=[np.shape(tensor) for tensor in M_tensors]
M_contr=[[1,-5,3,-6],[-5,2],[-6,4]]
contr_tmp=[M_contr[j][:] for j in range(0,len(M_contr))]
shapes_tmp=[M_shapes[j][:] for j in range(0,len(M_shapes))]
instr_tV,transp_tV=master.instr_gen(shapes_tmp,contr_tmp);


# RG transformation
rg_step=1
while(ntensors>4) :
    print "ntensors=",ntensors
    M_tensors=[A,A]
    slave.contract(M_tensors,instr_V,transp_V)
    print "contracted"
    M=M_tensors[0]
    dim=np.shape(M)
    M=np.reshape(M,[dim[0]*dim[1],dim[2],dim[3]*dim[4],dim[5]])
    MMD=np.tensordot(M,np.conjugate(M),axes=([1,2,3],[1,2,3]))
    #print np.shape(M),"MMD",np.shape(MMD)
    SL2,UL=LA.eigh(MMD)
    full=np.shape(SL2)[0]
    chi=min(chi_max,full)
    eps1=np.sum(SL2[0:full-chi])
       

    dim=np.shape(M)
    MMD=np.tensordot(M,np.conjugate(M),axes=([0,1,3],[0,1,3]))
    SR2,UR=LA.eigh(MMD)
    full=np.shape(SR2)[0]
    chi=min(chi_max,full)
    eps2=np.sum(SR2[0:full-chi])
    #print "eps",eps1,eps2
    if(eps1<eps2) :
        U=UL[:,full-chi:full]
    else :
        U=UR[:,full-chi:full]
    M_tensors=[M,U,U]
    slave.contract(M_tensors,instr_tH,transp_tH)
    A=M_tensors[0]
    print "shapes1",np.shape(A)


    


    M_tensors=[A,A]
    slave.contract(M_tensors,instr_H,transp_H)
    M=M_tensors[0]
    dim=np.shape(M)
    M=np.reshape(M,[dim[0],dim[1]*dim[2],dim[3],dim[4]*dim[5]])
    MMD=np.tensordot(M,np.conjugate(M),axes=([0,2,3],[0,2,3]))
    SL2,UL=LA.eigh(MMD)
    full=np.shape(SL2)[0]
    chi=min(chi_max,full)
    eps1=np.sum(SL2[0:full-chi])
       

    dim=np.shape(M)
    #M=np.reshape(M,[dim[0],dim[1]*dim[2],dim[3],dim[4]*dim[5]])
    MMD=np.tensordot(M,np.conjugate(M),axes=([0,1,2],[0,1,2]))
    SR2,UR=LA.eigh(MMD)
    full=np.shape(SR2)[0]
    chi=min(chi_max,full)
    eps2=np.sum(SR2[0:full-chi])
    print "eps",eps1,eps2
    if(eps1<eps2) :
        U=UL[:,full-chi:full]
    else :
        U=UR[:,full-chi:full]
    #print "shapes2",np.shape(M),np.shape(U)
    M_tensors=[M,U,U]
    slave.contract(M_tensors,instr_tV,transp_tV)
    A=M_tensors[0]
    print "shapes2",np.shape(A)
    dim=np.shape(A)
    #print "shapef",dim
    norm=np.trace(np.reshape(A,[dim[0]*dim[1],dim[2]*dim[3]]))
    #print "norm",norm
    #sys.stdin.read(1)
    A=A/norm;

    #if(rg_step>4) : 
    #  f=norm;
    #  print "f=",f
    #  L=f
    #  T_inv=A/L
    #  tmp=[T_inv]
    #  T.network(tmp,[[-1,2,-1,3]],[-1])
    #  e,v=LA.eigh(tmp[0])
    #  #print e


    #  if(e[-1]>0) :
    #    charge=(6.0/math.pi)*math.log(e[-1]);
    #    print "Central charge =", (6.0/math.pi)*math.log(e[-1]);
    #    print "1st scaling dimension  =",((charge/(12.0))-math.log(e[-2])/(2*math.pi))
    #    print "2st scaling dimension  =",((charge/(12.0))-math.log(e[-3])/(2*math.pi))
    #    print "2st scaling dimension  =",((charge/(12.0))-math.log(e[-4])/(2*math.pi))
    #    print "2st scaling dimension  =",((charge/(12.0))-math.log(e[-5])/(2*math.pi))
    #    print "2st scaling dimension  =",((charge/(12.0))-math.log(e[-6])/(2*math.pi))
    #    print "2st scaling dimension  =",((charge/(12.0))-math.log(e[-7])/(2*math.pi))
    #    print "2st scaling dimension  =",((charge/(12.0))-math.log(e[-8])/(2*math.pi))
    #  print "f =", f
    #  charge= (6.0/math.pi)*math.log(e[-1]);
    #  l1=((charge/(12.0))-math.log(e[-2])/(2*math.pi))
    #  l2=((charge/(12.0))-math.log(e[-3])/(2*math.pi))
    #  l3=((charge/(12.0))-math.log(e[-4])/(2*math.pi))
    #  l4=((charge/(12.0))-math.log(e[-5])/(2*math.pi))
    #  l5=((charge/(12.0))-math.log(e[-6])/(2*math.pi))
    #  l6=((charge/(12.0))-math.log(e[-7])/(2*math.pi))
    #  l7=((charge/(12.0))-math.log(e[-8])/(2*math.pi))
    #  #sys.stdin.read(1)
    
    

    ntensors=ntensors/4
    logZ+= (ntensors)*math.log(norm)
    print "free en",-(1/beta)*logZ/nsites
    #print>>outfile,"%d %f %f %f %f %f %f %f %f "%(rg_step,charge,l1,l2,l3,l4,l5,l6,l7)
    rg_step=rg_step+1
    


f1=open('dataTRG.dat',"a")
print>>f1,"%.16f %.16f"%(1/beta,-(1/beta)*logZ/nsites)


