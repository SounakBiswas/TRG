import numpy as np
import math
import scipy.linalg as LA
import optimal
import sys

#Python version containingp functionality of Vidal/Pfiefer MATLAB code.

# Done : Basic contractions, scalar output support by default in numpy, Traces.

#To be done :Disjoint networks,outer product, optimal sequences, instruction set

#Not to be done : Stupid Error checks. Test for optimality,

def network(tensors,contraction,order) :
  n=len(tensors);
  while (len(order)>0) :
    #look for outer products
    nzeros=0;
    while ((len(order)>0) and (order[0]==0)) :
      order=order[1:]
      nzeros+=1;
    if(nzeros!=0) : 
      if (nzeros+1==len(contraction)) :
        outerprod=range(0,len(tensors));
      else :
        outerprod=[]
        for k in range(0,len(tensors)) :
          ctr=0
          for m in range(0,len(contraction[k])) :
            for i in range (0,nzeros+1) :
              j=order[i]
              if (j==contraction[k][m]) :
                ctr+=1;
          if(ctr<=nzeros):
            outerprod.append(k)
      contract_new=[];	  
      for i in range (0,len(outerprod)) :
        contract_new=contract_new+contraction[outerprod[i]];
      contraction[0]=contract_new;
      for i in range (1,len(outerprod)) :
        tensor1=tensors[outerprod[0]]
        tensor2=tensors[outerprod[i]]
        tensor1=tensor1.reshape(list(np.shape(tensor1))+[1]);
        tensor2=tensor2.reshape(list(np.shape(tensor2))+[1]);
        tensors[outerprod[0]]=np.tensordot(tensor1,tensor2,axes=(-1,-1))
      for i in range(len(outerprod)-1,0,-1) :
        tensors.pop(outerprod[i]);
        contraction.pop(outerprod[i]);
      if(len(order)==0) :
        if(len(contraction)>0) :
            neworder=np.argsort(contraction[0])
            tensors[0]=np.transpose(tensors[0],neworder)
            contraction[0]=[contraction[0][i] for i in neworder]
        return tensors[0]

    j=order[0]
    pair=[-1,-1]
    c=0
    legs=[[],[]]
    for k in range (0,len(tensors)) :
      for m in range (0,len(contraction[k])) :
	if(j==contraction[k][m]) :
	  pair[c]=k
	  c+=1
	  break
    if (pair[1]==-1) :
	trace1=[]
	trace2=[]
	trace3=[i for i in range (0,len(contraction[pair[0]]))]
	dim2=list(np.shape(tensors[pair[0]]))
	dim=1
	n_traced_ind=0;
	newlist= contraction[pair[0]][:]
	for m in range(0,len(contraction[pair[0]])) :
	  for p in range (m+1,len(contraction[pair[0]])) :
	    m1=contraction[pair[0]][m]
	    p1=contraction[pair[0]][p]
	    if m1==p1 :
	      n_traced_ind=n_traced_ind+1;
	      trace1.append(m)
	      trace2.append(p)
	      dim*=np.shape(tensors[pair[0]])[m]
	      newlist.remove(m1)
	      newlist.remove(p1)
	      trace3.remove(m)
	      trace3.remove(p)
	      dim2.remove(np.shape(tensors[pair[1]])[m])
	      dim2.remove(np.shape(tensors[pair[0]])[m])
	      break
	if (n_traced_ind==1) :
           tensors[pair[0]]=np.trace(tensors[pair[0]],axis1=trace1[0],axis2=trace2[0])
	else :
	   newindices=trace1+trace2+trace3
	   tensors[pair[0]]=np.transpose(tensors[pair[0]],newindices)
	   dim2=[dim,dim]+dim2
	   tensors[pair[0]]=tensors[pair[0]].reshape(dim2)
           tensors[pair[0]]=np.trace(tensors[pair[0]])
	contraction[pair[0]]=newlist
        order=order[len(trace1):]
    else :
	newlist=contraction[pair[0]]+contraction[pair[1]];
	for m in range(0,len(contraction[pair[0]])) :
	  for p  in range(0,len(contraction[pair[1]])) :
	    m1=contraction[pair[0]][m]
	    p1=contraction[pair[1]][p]
	    if (m1==p1) :
	      legs[0].append(m)
	      legs[1].append(p)
	      newlist.remove(m1)
	      newlist.remove(m1)
        tensors[pair[0]]=np.tensordot(tensors[pair[0]],tensors[pair[1]],axes=(legs[0],legs[1]))
        #print "legs", legs[0],legs[1];
        #sys.stdin.read(1)
        contraction[pair[0]]=newlist;
        tensors.pop(pair[1])
        contraction.pop(pair[1])
        order=order[len(legs[0]):]
  if(len(contraction)>0) :
         neworder=np.argsort(contraction[0])
         #print contraction[0]
         #print neworder
         tensors[0]=np.transpose(tensors[0],neworder)
         contraction[0]=[contraction[0][i] for i in neworder]
  return tensors[0]


def tensor_svd(A,list1, list2) :
    dims=list1+list2
    #print dims, np.shape(A)
    A=np.transpose(A,dims)
    shape1=list(np.shape(A)[0:len(list1)])
    shape2=list(np.shape(A)[len(list1):len(dims)])
    dim1=np.prod(np.shape(A)[0: len(list1)])
    dim2=np.prod(np.shape(A)[len(list1): len(dims)])
    A=np.reshape(A,[dim1,dim2])
    dim3=min(dim1,dim2)
    #print np.shape(A)
    X,Y,Z=LA.svd(A,full_matrices=False)
    #print Y
    #sys.stdin.read(1)
    #chi=min(np.sum(Y>10**(-15)),chi_max)
    X=np.reshape(X,shape1+[dim3])
    Z=np.reshape(Z,[dim3]+shape2)
    #print np.shape(X),np.shape(Y),np.shape(Z)
    return X,Y,Z

def svdD(A,list1, list2,chi_max) :
    dims=list1+list2
    #print dims, np.shape(A)
    A=np.transpose(A,dims)
    shape1=list(np.shape(A)[0:len(list1)])
    shape2=list(np.shape(A)[len(list1):len(dims)])
    dim1=np.prod(np.shape(A)[0: len(list1)])
    dim2=np.prod(np.shape(A)[len(list1): len(dims)])
    A=np.reshape(A,[dim1,dim2])
    dim3=min(dim1,dim2)
    #print np.shape(A)
    X,Y,Z=LA.svd(A,full_matrices=False)
    #print Y
    #sys.stdin.read(1)
    chi=min(np.sum(Y>10**(-15)),chi_max)
    X=X[:,0:chi]
    Z=Z[0:chi,:]
    Y=Y[0:chi]
    Y=np.diag(np.sqrt(Y))
    X=np.reshape(X,shape1+[chi])
    Z=np.reshape(Z,[chi]+shape2)
    X=np.tensordot(X,Y,axes=([2,0]))
    Z=np.tensordot(Y,Z,axes=([1,0]))
    #print np.shape(X),np.shape(Y),np.shape(Z)
    return X,Z
def normalize(A,list1, list2) :
    dims=list1+list2
    #print dims, np.shape(A)
    A=np.transpose(A,dims)
    shape1=list(np.shape(A)[0:len(list1)])
    shape2=list(np.shape(A)[len(list1):len(dims)])
    dim1=np.prod(np.shape(A)[0: len(list1)])
    dim2=np.prod(np.shape(A)[len(list1): len(dims)])
    A=np.reshape(A,[dim1,dim2])
    #print np.shape(A)
    X,Y,Z=LA.svd(A)
    #print Y
    free=np.sqrt(np.sum(Y**2))
    Y=Y/np.sqrt(np.sum(Y**2))
    Y=np.diag(Y)
    temp=np.tensordot(X,Y,axes=(1,0))
    X=np.tensordot(temp,Z,axes=(1,0))
    X=np.reshape(X,shape1+shape2)
    return X,free

    #print np.shape(X),np.shape(Y),np.shape(Z)
def make_env(tensors,contraction,pos,order) :
  contraction_temp=contraction[:]
  tensors_temp=tensors[:]
  print contraction_temp
  B=network(tensors_temp,contraction_temp,order)
  print contraction_temp
  Bc=np.conjugate(B)
  free=[]
  
  for i in range(0,len(contraction[pos])) :
    if contraction[pos][i]>0 :
      free.append(contraction[pos][i]);
  for j in range(0,len(contraction)):
    for k in range(0,len(contraction[j])) :
      if contraction[j][k]>0 :
        contraction[j][k]*=-1
      for i in range(0,len(contraction[pos])) :
         if (contraction[pos][i]<0 and j!=pos) :
           if contraction[j][k]==contraction[pos][i] :
             contraction[j][k]*=-1
  #print contraction_temp[0]
  #print free

  tensors.pop(pos);
  contraction.pop(pos);
  for i in range(0,len(contraction_temp[0])) :
    if contraction_temp[0][i] not in free :
      contraction_temp[0][i]*=-1
  tensors.append(Bc);
  contraction.append(contraction_temp[0])
  temp=contraction[:]
  order=optimal.optimal_w(temp)
  #print order
  network(tensors,contraction,order)

