import numpy as np
import scipy.linalg as LA
maxc=30
chi=4
def findcost(indices1,indices2) :
    #a=np.zeros(maxc+1);
    contracted=0;
    
    for i in range (0,len(indices1)) :
        for j in range (0,len(indices2)) :
            if(indices1[i]==indices2[j]) :
                contracted+=1;

    #contracted=len(np.intersect1d(indices1,indices2))

    complexity=len(indices1)+len(indices2)-contracted;
    #a[complexity]=1;
    return chi**complexity
def contract_indices(indices1,indices2) :
    contracted=[];
    legs1=indices1[:]
    legs2=indices2[:]
    
    for i in range (0,len(indices1)) :
        for j in range (0,len(indices2)) :
            if(indices1[i]==indices2[j]) :
                legs1.remove(indices1[i])
                legs2.remove(indices2[j])
    indices3= legs1+legs2
    legs3=indices3[:]
    #takeout traces
    for i in range (0,len(indices3)) :
        for j in range (i+1,len(indices3)) :
            if (indices3[i]==indices3[j]) :
                legs3.remove(indices3[i])
                legs3.remove(indices3[i])
    return legs3
def less(cost1,cost2) :
    p=maxc;
    flag=0
    while(p>=0) :
        if ((cost2[p]!=cost1[p])) :
            if cost1[p]<cost2[p] :
                flag=1
            break;
        p-=1
    if flag==1 :
        return True
    else :
        return False

def sumcost(cost1,cost2,cost3) :
    cost=np.add(cost1,cost2)
    return np.add(cost,cost3)








def optimal(contraction) :
    ntensors=len(contraction);
    n=2**ntensors
    cost=np.zeros([n],dtype=int);
    touched=np.zeros([n],dtype=bool);
    root=np.zeros([n],dtype=int);
    path=[];
    S=[];
    indices=[];
    S1=[]
    for i in range(0,n) :
        path.append([]);
    for i in range(0,ntensors) :
        S1.append(2**i)
        root[2**i]=i
        path[2**i]=[]
        #for j in range (0,maxc+1) :
        #  cost[2**i][j]=0
    S.append(S1)
    indices.append(contraction)
    for c in range (2,ntensors+1) :
        ctr=0;
        i3=c-1
        S1=[]
        newindices=[]
        for d in range(1,c/2+1) :
            cmd=c-d;
            i1=d-1
            i2=cmd-1
            for i in range(0,len(S[i1])) :
                for j in range(0,len(S[i2])) :
                    ta=S[i1][i]
                    tb=S[i2][j]
                    if( ta & tb ==0 ) :
                        final=ta+tb;
                        if(touched[final]==False) :
                            touched[final]=True
                            S1.append(final);
                            newindices.append(contract_indices(indices[i1][i],indices[i2][j]))
                        cost_temp=findcost(indices[i1][i],indices[i2][j])
                        cost_tot=cost[ta]+cost[tb]+cost_temp
                        if (cost_tot<cost[final]) or cost[final]==0 :
                            cost[final]=cost_tot
                            root[final]=root[ta]
                            path[final]=path[ta]+path[tb]+[[root[ta],root[tb]]]
        S.append(S1)
        indices.append(newindices)
        
    #print cost[n-1]
    #print path[n-1]
    return path[n-1]



#contraction=[[3,-3,-2,-1],[4,-12,-11,-8],[-3,-8,-9,-4],[-2,-4,-5,-6],[-5,-9,-10,-7],[-1,-6,-7,1],[-10,-11,-12,2]]
#contraction=[[1,-1,-2],[2,-3,-4],[3,-5,-6],[-2,-3,-7,-8],[-4,-5,-9,-10],[-7,-8,-11,-12,-9,-13],[-11,-12,-14,-15],[-13,-10,-16,-17],[-1,4,-14],[-15,5,-16],[-17,6,-6]]
#optimal(contraction)




def optimal_w(contraction) :
  order=[];
  path=optimal(contraction)
  for i in range (0,len(path)) : 
    pair0=path[i][0];
    pair1=path[i][1];
    #look for trace-ables
    traced=[];
    for j in range(0,len(contraction[pair0])) :
      tracedleg=contraction[pair0][j]
      traceflag=0
      for k in range(j+1,len(contraction[pair0])) : 
	if(contraction[pair0][j]==contraction[pair0][k]) :
	   traced.append(contraction[pair0][k])
           traceflag=1
      #if(traceflag==1) :
        #contraction[pair0].remove[tracedleg];
        #contraction[pair0].remove[tracedleg];
    #print "trace1",traced
    #order=order+traced;
    traced=[];
    for j in range(0,len(contraction[pair1])) :
      for k in range(j+1,len(contraction[pair1])) : 
	if(contraction[pair1][j]==contraction[pair1][k]) :
	   traced.append(contraction[pair1][k])
           traceflag=1
      #if(traceflag==1) :
        #contraction[pair1].remove[tracedleg];
        #contraction[pair1].remove[tracedleg];
    #print "trace2",traced
    #order=order+traced;

    legs=[]
    newlist=contraction[pair0]+contraction[pair1]
    for j in range(0,len(contraction[pair0])) :
      for k in range(0,len(contraction[pair1])) : 
	if(contraction[pair0][j]==contraction[pair1][k]) :
	  order.append(contraction[pair0][j]);
	  legs.append(contraction[pair0][j])
	  newlist.remove(contraction[pair0][j])
	  newlist.remove(contraction[pair0][j])
	  #print "order_append", contraction[pair0][j]
    contraction[pair0]=newlist
    #print legs ,"huh"
    if(len(legs)==0) : 
      order.append(0)
    #print "order", order

  return order

       

#contraction=[[3,-3,-2,-1],[4,-12,-11,-8],[-3,-8,-9,-4],[-2,-4,-5,-6],[-5,-9,-10,-7],[-1,-6,-7,1],[-10,-11,-12,2]]
#new=contraction[:]
#print optimal(new)


    
