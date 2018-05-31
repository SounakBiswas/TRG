import numpy as np
def contract(tensors,instruction,transp) :
  n=len(tensors);
  for i in range(0,len(instruction)) :
    #trace
    ins=instruction[i]
    if (len(ins[0])==1) :
      tensors[ins[0][0]]=np.transpose(tensors[ins[0][0]],ins[1])
      tensors[ins[0][0]]=tensors[ins[0][0]].reshape(ins[2])
      tensors[ins[0][0]]=np.trace(tensors[ins[0][0]])
    else :
      if (len(ins[1])==0) :
        for j in range (1,len(ins[0])) :
          tensor1=tensors[ins[0][0]]
          tensor2=tensors[ins[0][j]]
          tensor1=tensor1.reshape(list(np.shape(tensor1))+[1]);
          tensor2=tensor2.reshape(list(np.shape(tensor2))+[1]);
          tensors[ins[0][0]]=np.tensordot(tensor1,tensor2,axes=(-1,-1))
    

        for j in range(len(ins[0])-1,0,-1) :
          tensors.pop(ins[0][j]);


      else :
        tensors[ins[0][0]]=np.tensordot(tensors[ins[0][0]],tensors[ins[0][1]],axes=(ins[1][0],ins[1][1]))
        tensors.pop(ins[0][1])
  if (len(transp)!=0) :
     tensors[0]=np.transpose(tensors[0],transp)

def make_env(tensors,pos,instruction) :
  instruction1=instruction[0]
  transpose1=instruction[1]
  instruction2=instruction[2]
  transpose2=instruction[3]
  tensors_temp=tensors[:]
  contract(tensors_temp,instruction1,transpose1)
  Bc=np.conjugate(tensors_temp[0])
  tensors.pop(pos)
  tensors.append(Bc)
  contract(tensors,instruction2,transpose2)
  return tensors[0]


      


      
      
      
