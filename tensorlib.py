import numpy as np
import math
import scipy.linalg as LA

#Python version containingp functionality of Vidal/Pfiefer MATLAB code.

# Done : Basic contractions, scalar output support by default in numpy, Traces.

#To be done :Disjoint networks,outer product, optimal sequences, instruction set

#Not to be done : Stupid Error checks. Test for optimality,

def network(tensors,contraction,order) :
  n=len(tensors);
  while (len(order)>0) :
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

	newlist= contraction[pair[0]][:]
	for m in range(0,len(contraction[pair[0]])) :
	  for p in range (m+1,len(contraction[pair[0]])) :
	    m1=contraction[pair[0]][m]
	    p1=contraction[pair[0]][p]
	    if m1==p1 :
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
        contraction[pair[0]]=newlist;
        tensors.pop(pair[1])
        contraction.pop(pair[1])
        order=order[len(legs[0]):]
  if(len(contraction)>0) :
      tensors[0]=np.transpose(tensors[0],np.argsort(contraction[0]))
  return tensors[0]


def tensor_svd(A,list1, list2,chi_max) :
    dims=list1+list2
    A=np.transpose(A,dims)
    shape1=list(np.shape(A)[0:len(list1)])
    shape2=list(np.shape(A)[len(list1):len(dims)])
    dim1=np.prod(np.shape(A)[0: len(list1)])
    dim2=np.prod(np.shape(A)[len(list1): len(dims)])
    A=np.reshape(A,[dim1,dim2])
    X,Y,Z=LA.svd(A)
    chi=min(np.sum(Y>10**(-10)),chi_max)
    Y=Y[0:chi]
    Y=Y/np.sqrt(np.sum(Y**2))
    X=X[:,0:chi]
    Z=Z[0:chi,:]
    X=np.reshape(X,shape1+[chi])
    Z=np.reshape(Z,[chi]+shape2)
    return X,Y,Z



















