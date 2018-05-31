#Generaes contraction information for tensor  networks
#instructions are returned in form of a list with elements instruction[i]
#ins[i][0] is [tensor1,tensor2] for contraction and [tensor1] for trace
#If instruction is a simple trace ins[i][1] is a list of traced indices from both tensors
#If instruction is group trace :ins[i][1] transposes to collect traced indices
#                        :ins[i][2] reshapes to make the traced subspace a 2D
#if instruction is contract : ins[i][1][0] and ins[i][0][1] contains contraction axes
#if instruction is a group of outerproducts : ins[i][0] is a list of tensors to form outerprod
#                                             ins[i][1] is an empty array
import sys
import optimal
import numpy as np
def instr_gen(shapes,contraction) :
  #shapes=[list(np.shape(tensor)) for tensor in tensors]
  instr=[]
  neworder=[]
  contraction_temp=contraction[:]
  order=optimal.optimal_w(contraction_temp)
#  print "order ",order
#  print "contr ",contraction
#  print "shapes",len(shapes)
  while (len(order)>0) :
  #  print "instr",instr
    nzeros=0;
    while ((len(order)>0) and (order[0]==0)) :
      order=order[1:]
    #  print order
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
            shapes[0]=[shapes[0][i] for i in neworder]
        return instr,neworder
    j=order[0]
    pair=[-1,-1]
    c=0
    legs=[[],[]]
    
    for k in range (0,len(shapes)) :
      for m in range (0,len(contraction[k])) :
	if(j==contraction[k][m]) :
	  pair[c]=k
	  c+=1
	  break
    if (pair[1]==-1) :  #trace
	trace1=[]
	trace2=[]
	trace3=[i for i in range (0,len(contraction[pair[0]]))]
	dim2=shapes[pair[0]][:]
	dim=1
        n_traced_ind=0
	newlist= contraction[pair[0]][:]
	for m in range(0,len(contraction[pair[0]])) :
	  for p in range (m+1,len(contraction[pair[0]])) :
	    m1=contraction[pair[0]][m]
	    p1=contraction[pair[0]][p]
	    if m1==p1 :
              n_traced_ind+=1;
	      trace1.append(m)
	      trace2.append(p)
	      dim*=shapes[pair[0]][m]
	      newlist.remove(m1)
	      newlist.remove(p1)
	      trace3.remove(m)
	      trace3.remove(p)
	      dim2.remove(shapes[pair[0]][m])
	      dim2.remove(shapes[pair[0]][m])
	      break
	if (n_traced_ind==1) :
	   instr.append([[pair[0]],[trace1[0],trace2[0]]])
           shapes[pair[0]]=[shapes[i] for i in range(0,trace3)]
        else :
	   newindices=trace1+trace2+trace3
	   shapes[pair[0]]=dim2
	   dim2=[dim,dim]+list(dim2)
           shapes[pair[0]]=[shapes[i] for i in range(0,trace3)]
	   instr.append([[pair[0]],newindices,dim2])
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
              
        #shapes[pair[0]]=np.tensordot(shapes[pair[0]],shapes[pair[1]],axes=(legs[0],legs[1]))
        newshapes0=[shapes[pair[0]][i] for i in range(0,len(shapes[pair[0]])) if i not in legs[0]]
        newshapes1=[shapes[pair[1]][i] for i in range(0,len(shapes[pair[1]])) if i not in legs[1]]

        contraction[pair[0]]=newlist;
        shapes[pair[0]]=newshapes0+newshapes1
        shapes.pop(pair[1])
        #sys.stdin.read(1)
        contraction.pop(pair[1])
        order=order[len(legs[0]):]
	instr.append([[pair[0],pair[1]],legs])
  if len(contraction)>0 :
      neworder=np.argsort(contraction[0])
      contraction[0]=[contraction[0][i] for i in neworder]
      shapes[0]=[shapes[0][i] for i in neworder]
  return instr,neworder

def make_env(shapes,contraction,pos) :
  contraction_temp=contraction[:]
  shapes_temp=shapes[:]
  instruction1,transpose1=instr_gen(shapes_temp,contraction_temp)
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
  shapes.pop(pos);
  contraction.pop(pos);
  for i in range(0,len(contraction_temp[0])) :
    if contraction_temp[0][i] not in free :
      contraction_temp[0][i]*=-1
  contraction.append(contraction_temp[0])
  shapes.append(shapes_temp[0])
  instruction2,transpose2=instr_gen(shapes,contraction)
  env_instr= [instruction1,transpose1,instruction2,transpose2]
  return env_instr;
