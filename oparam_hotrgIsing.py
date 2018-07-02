import math
import numpy as np
import scipy.linalg as LA
import optimal as opt
import master
import slave
import tensorlib_new as T
#from memory_profiler import profile
import time
import sys
beta=1.0/(2.2691)
Lx=1024;
nsites=2*(Lx**2)
ntensors=Lx**2;
chi_max=70;
#intitialization of tensors
A=np.zeros([2,2,2,2]);
I=np.zeros([2,2,2,2]);
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
        I[i,j,k,l]=(i1+i2+i3+i4)*math.exp(beta*(i1*i2+i2*i3+i3*i4+i1*i4))
    
fname="TRGIsing_%f_%d.dat"%(1/beta,chi_max)
outfile=open(fname,"w")
dim=np.shape(A)[0]
norm=np.trace(np.reshape(A,[dim**2,dim**2]))
A=A/norm
logZ=0;
logZ+= (ntensors)*math.log(norm)

M_shapes=[[chi_max,chi_max,chi_max,chi_max],[chi_max,chi_max,chi_max,chi_max], [chi_max,chi_max,chi_max,chi_max], [chi_max,chi_max,chi_max,chi_max]]
M_contr=[[-1,-2,-3,-7],[-3,-4,-1,-8], [-5,-8,-6,-4],[-6,-7,-5,-2]]
contr_tmp=[M_contr[j][:] for j in range(0,len(M_contr))]
shapes_tmp=[M_shapes[j][:] for j in range(0,len(M_shapes))]
instr_final,transp_final=master.instr_gen(shapes_tmp,contr_tmp);

M_shapes=[[chi_max,chi_max,chi_max,chi_max],[chi_max,chi_max,chi_max,chi_max], [chi_max,chi_max,chi_max,chi_max], [chi_max,chi_max,chi_max,chi_max]]
M_contr=[[2,-5,-6,-9],[4,-5,-6,-10], [3,-10,-7,-8],[1,-9,-7,-8]]
contr_tmp=[M_contr[j][:] for j in range(0,len(M_contr))]
shapes_tmp=[M_shapes[j][:] for j in range(0,len(M_shapes))]
instr_V1,transp_V1=master.instr_gen(shapes_tmp,contr_tmp);

M_shapes=[[chi_max,chi_max,chi_max,chi_max],[chi_max,chi_max,chi_max,chi_max], [chi_max,chi_max,chi_max,chi_max], [chi_max,chi_max,chi_max,chi_max]]
M_contr=[[-6,-5,2,-9],[-6,-5,4,-10], [-7,-10,3,-8],[-7,-9,1,-8]]
contr_tmp=[M_contr[j][:] for j in range(0,len(M_contr))]
shapes_tmp=[M_shapes[j][:] for j in range(0,len(M_shapes))]
instr_V2,transp_V2=master.instr_gen(shapes_tmp,contr_tmp);

M_shapes=[[chi_max,chi_max,chi_max,chi_max],[chi_max,chi_max,chi_max,chi_max], [chi_max,chi_max,chi_max,chi_max], [chi_max,chi_max,chi_max,chi_max]]
M_contr=[[-8,1,-3,-4],[-3,2,-7,-5], [-6,4,-7,-5],[-8,3,-6,-4]]
contr_tmp=[M_contr[j][:] for j in range(0,len(M_contr))]
shapes_tmp=[M_shapes[j][:] for j in range(0,len(M_shapes))]
instr_H1,transp_H1=master.instr_gen(shapes_tmp,contr_tmp);

M_shapes=[[chi_max,chi_max,chi_max,chi_max],[chi_max,chi_max,chi_max,chi_max], [chi_max,chi_max,chi_max,chi_max], [chi_max,chi_max,chi_max,chi_max]]
M_contr=[[-8,-4,-3,1],[-3,-5,-7,2], [-6,-5,-7,4],[-8,-4,-6,3]]
contr_tmp=[M_contr[j][:] for j in range(0,len(M_contr))]
shapes_tmp=[M_shapes[j][:] for j in range(0,len(M_shapes))]
instr_H2,transp_H2=master.instr_gen(shapes_tmp,contr_tmp);

M_shapes=[[chi_max,chi_max,chi_max,chi_max],[chi_max,chi_max,chi_max,chi_max], [chi_max,chi_max,chi_max],[chi_max, chi_max, chi_max]]
M_contr=[[-8,2,-9,-6],[-5,-6,-7,4],[-5,-8,1],[-7,-9,3]]
contr_tmp=[M_contr[j][:] for j in range(0,len(M_contr))]
shapes_tmp=[M_shapes[j][:] for j in range(0,len(M_shapes))]
instr_tH,transp_tH=master.instr_gen(shapes_tmp,contr_tmp);

M_shapes=[[chi_max,chi_max,chi_max,chi_max],[chi_max,chi_max,chi_max,chi_max], [chi_max,chi_max,chi_max],[chi_max, chi_max, chi_max]]
M_contr=[[1,-5,-7,-8],[-7,-6,3,-9],[-5,-6,2],[-8,-9,4]]
contr_tmp=[M_contr[j][:] for j in range(0,len(M_contr))]
shapes_tmp=[M_shapes[j][:] for j in range(0,len(M_shapes))]
instr_tV,transp_tV=master.instr_gen(shapes_tmp,contr_tmp);


# RG transformation
rg_step=1
while(ntensors>4) :
    print "ntensors=",ntensors
    #Pure tensors contraction.
    M_tensors=[A,A,A,A]
    slave.contract(M_tensors,instr_V1,transp_V1)
    MMD=M_tensors[0]
    dim=np.shape(MMD)
    SL2,UL=LA.eigh(MMD.reshape([dim[0]*dim[1],dim[0]*dim[1]]))
    full=np.shape(SL2)[0]
    chi=min(chi_max,full)
    eps1=np.sum(SL2[0:full-chi])
       

    M_tensors=[A,A,A,A]
    slave.contract(M_tensors,instr_V2,transp_V2)
    MMD=M_tensors[0]
    SR2,UR=LA.eigh(MMD.reshape([dim[0]*dim[1],dim[0]*dim[1]]))
    full=np.shape(SR2)[0]
    chi=min(chi_max,full)
    eps2=np.sum(SR2[0:full-chi])
    #print "eps",eps1,eps2
    if(eps1<eps2) :
        U=UL[:,full-chi:full]
    else :
        U=UR[:,full-chi:full]
    U=U.reshape([dim[0],dim[1],chi])
    M_tensors=[A,A,U,U]
    slave.contract(M_tensors,instr_tH,transp_tH)
    A=M_tensors[0]
    #Impurity tensors contraction.
    M_tensors=[I,I,A,A]
    slave.contract(M_tensors,instr_V1,transp_V1)
    MMD=M_tensors[0]
    dim=np.shape(MMD)
    SL2,UL=LA.eigh(MMD.reshape([dim[0]*dim[1],dim[0]*dim[1]]))
    full=np.shape(SL2)[0]
    chi=min(chi_max,full)
    eps1=np.sum(SL2[0:full-chi])
    M_tensors=[I,I,A,A]
    slave.contract(M_tensors,instr_V2,transp_V2)
    MMD=M_tensors[0]
    SR2,UR=LA.eigh(MMD.reshape([dim[0]*dim[1],dim[0]*dim[1]]))
    full=np.shape(SR2)[0]
    chi=min(chi_max,full)
    eps2=np.sum(SR2[0:full-chi])
    #print "eps",eps1,eps2
    if(eps1<eps2) :
        U=UL[:,full-chi:full]
    else :
        U=UR[:,full-chi:full]
    M_tensors=[I,A,U,U]
    slave.contract(M_tensors,instr_tH,transp_tH)
    I=M_tensors[0]

   


    print "shapes1",np.shape(A)


    


    M_tensors=[A,A,A,A]
    slave.contract(M_tensors,instr_H1,transp_H1)
    MMD=M_tensors[0]
    dim=np.shape(MMD)
    SL2,UL=LA.eigh(MMD.reshape([dim[1]*dim[2],dim[1]*dim[2]]))
    full=np.shape(SL2)[0]
    chi=min(chi_max,full)
    eps1=np.sum(SL2[0:full-chi])
       

    M_tensors=[A,A,A,A]
    slave.contract(M_tensors,instr_H2,transp_H2)
    MMD=M_tensors[0]
    dim=np.shape(MMD)
    SR2,UR=LA.eigh(MMD.reshape([dim[1]*dim[2],dim[1]*dim[2]]))
    full=np.shape(SR2)[0]
    chi=min(chi_max,full)
    eps2=np.sum(SR2[0:full-chi])
    print "eps",eps1,eps2
    if(eps1<eps2) :
        U=UL[:,full-chi:full]
    else :
        U=UR[:,full-chi:full]
    #print "shapes2",np.shape(M),np.shape(U)
    U=U.reshape([dim[1],dim[2],chi])
    U=U.reshape([dim[0],dim[1],chi])
    M_tensors=[A,A,U,U]
    slave.contract(M_tensors,instr_tV,transp_tV)
    A=M_tensors[0]

    #IMpurity tensors renormalization.
    M_tensors=[I,A,A,I]
    slave.contract(M_tensors,instr_H1,transp_H1)
    MMD=M_tensors[0]
    dim=np.shape(MMD)
    SL2,UL=LA.eigh(MMD.reshape([dim[1]*dim[2],dim[1]*dim[2]]))
    full=np.shape(SL2)[0]
    chi=min(chi_max,full)
    eps1=np.sum(SL2[0:full-chi])
       

    M_tensors=[I,A,I,A]
    slave.contract(M_tensors,instr_H2,transp_H2)
    MMD=M_tensors[0]
    dim=np.shape(MMD)
    SR2,UR=LA.eigh(MMD.reshape([dim[1]*dim[2],dim[1]*dim[2]]))
    full=np.shape(SR2)[0]
    chi=min(chi_max,full)
    eps2=np.sum(SR2[0:full-chi])
    print "eps",eps1,eps2
    if(eps1<eps2) :
        U=UL[:,full-chi:full]
    else :
        U=UR[:,full-chi:full]
    #print "shapes2",np.shape(M),np.shape(U)
    U=U.reshape([dim[1],dim[2],chi])
    U=U.reshape([dim[0],dim[1],chi])
    M_tensors=[I,A,U,U]
    slave.contract(M_tensors,instr_tV,transp_tV)
    I=M_tensors[0]

    print "shapes2",np.shape(A)
    dim=np.shape(A)
    #print "shapef",dim
    norm=np.trace(np.reshape(A,[dim[0]*dim[1],dim[2]*dim[3]]))
    #print "norm",norm
    #sys.stdin.read(1)
    A=A/norm;
    I=I/norm;

    if(rg_step>4) : 
      f=norm;
      print "f=",f
      L=f
      T_inv=A/(L**(1/3.0))
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
      print>>outfile,"%d %f %f %f %f %f %f %f %f "%(rg_step,charge,l1,l2,l3,l4,l5,l6,l7)
    
    

    ntensors=ntensors/4
    logZ+= (ntensors)*math.log(norm)
    print "free en",-(1/beta)*logZ/nsites
    rg_step=rg_step+1
    
M_tensors=[I,A,A,A]
slave.contract(M_tensors,instr_final,transp_final)
logZp= logZ+math.log(M_tensors[0])

M_tensors=[A,A,A,A]
slave.contract(M_tensors,instr_final,transp_final)
logZ+= math.log(M_tensors[0])


f1=open('dataTRG.dat',"a")
print>>f1,"%.16f %.16f %.16f"%(1/beta,-(1/beta)*logZ/nsites,math.exp(logZp/logZ))


