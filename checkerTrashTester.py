#Russell Hornbuckle
#2018
#checkers
import pandas as pd
import numpy as np   
import math as mt  
import checkersGame as cg 
import pickle as pk
from checkersFrank import checkersFrank as cf
#sideCOE=1
#import checkersFrank as cF
testState=np.array((1,1,1,1,1,1,1,1,1,1,1,1,0,-1,-1,0,0,0,0,0,0,-1,-1,0,0,0,2,0,0,-1,-1,-1))
#testState=testState.reshape((8,4))
#print(cG.defaultState)
#out=pd.DataFrame(cg.defaultState)
#gameRes=pd.Series(np.array([.78345]))
#gameRes=gameRes.append(pd.Series(cg.defaultState),ignore_index=True)
#open("testOut.txt",mode='x')
#open("testOut.txt",mode='r')
#out.to_pickle("testOut")
frank=cf()
frank.setSide(0)
#frank.initWeights(experiment here)
#testState=testState[:]*-1
#frank.printMoves(testState)
#frank.stateEvaluatorMaster(testState)
#bio=open("/home/russell/Documents/Frank/bio-Two.npz",mode="xb+")
#wOne=np.array(np.random.standard_normal((32,16)))
#wTwo=np.array(np.random.standard_normal((16,1)))
#game=np.array(np.zeros((1,1)))
#bio=np.load("Frank/bio-One.npz")
#wOne=bio['arr_0']
#wTwo=bio['arr_1']
#game=bio['arr_2']
##np.savez("/home/russell/Documents/Frank/bio-Two.npz",wOne,wTwo,game)



























































































































"""def rowRule(iNum,jNum):
        q=iNum%2
        if(q==1):
            return jNum,jNum+1
        else:
            return jNum,jNum-1
def contains( small, big):
    for i in range(0,len(big)-len(small)+1):
        for j in range(0,len(small)):
            if(big[i+j] != small[j]):
                break
        else:
            return True #i, i+len(small)
    return False
def constainsTwo( small, big):
    for i in range(0,len(big)-len(small)+1):
        for j in range(0,len(small)):
            if(big[i+j] != small[j]):
                break
        else:
            return i, i+len(small)
    return False
def stateFarmer(iNum,jNum,currentState):
    #Empty Move Checker
    farmSet=[]
    cur=currentState[iNum,jNum]
    jOne,jTwo=rowRule(iNum,jNum)
    if(iNum+1*sideCOE in range(0,8)):    
        if(currentState[iNum+1*sideCOE,jOne]==0):
            newState=np.copy(currentState)
            newState[iNum,jNum]=0
            newState[iNum+1*sideCOE,jOne]=cur
            newState=newState.reshape((1,-1)).tolist()
            farmSet.extend(newState)
        if(jTwo in range(0,4)):
            if(currentState[iNum+1*sideCOE,jTwo]==0):
                newState=np.copy(currentState)
                newState[iNum,jNum]=0
                newState[iNum+1*sideCOE,jTwo]=cur
                newState=newState.reshape((1,-1)).tolist()
                farmSet.extend(newState)
    if(cur==2):
        if(iNum-1*sideCOE in range(0,8)):    
            if(currentState[iNum-1*sideCOE,jOne]==0):
                newState=np.copy(currentState)
                newState[iNum,jNum]=0
                newState[iNum-1*sideCOE,jOne]=cur
                newState=newState.reshape((1,-1)).tolist()
                farmSet.extend(newState)
            if(jTwo in range(0,4)):
                if(currentState[iNum-1*sideCOE,jTwo]==0):
                    newState=np.copy(currentState)
                    newState[iNum,jNum]=0
                    newState[iNum-1*sideCOE,jTwo]=cur
                    newState=newState.reshape((1,-1)).tolist()
                    farmSet.extend(newState)    
    return farmSet
def priorityStateFarmer(iNum,jNum,currentState,prioritySet):
    #Priority Move Checker
    if(((iNum==7)and(sideCOE==1))or((iNum==0)and(sideCOE==-1))):
        currentState[iNum,jNum]=2
    curList=currentState.reshape((1,-1)).tolist()
    jOne,jTwo=rowRule(iNum,jNum)
    cur=currentState[iNum,jNum]

    if(cur>=1):
        if(iNum+2*sideCOE in range(0,8)):   
            if(currentState[iNum+1*sideCOE,jOne]<=-1):
                #jOneCases
                if(jOne>jTwo):
                    if(jOne+1 in range(0,4)):
                        if(currentState[iNum+2*sideCOE,jOne+1]==0):
                            newState=np.copy(currentState)
                            newState[iNum,jNum]=0
                            newState[iNum+1*sideCOE,jOne]=0
                            newState[iNum+2*sideCOE,jOne+1]=cur
                            newStateSet=[newState.reshape((1,-1)).tolist()]
                            
                            if(curList  in prioritySet):
                                #print("here")
                                del prioritySet[prioritySet.index(curList)]
                                #removeCurrentStateFromPrioritySet
                            prioritySet.extend(priorityStateFarmer(iNum+2*sideCOE,jOne+1,newState,newStateSet))
                            #newStateFuncwithRecursion
                else:
                    if(jOne-1 in range(0,4)):
                        if(currentState[iNum+2*sideCOE,jOne-1]==0):
                            newState=np.copy(currentState)
                            newState[iNum,jNum]=0
                            newState[iNum+1*sideCOE,jOne]=0
                            newState[iNum+2*sideCOE,jOne-1]=cur
                            newStateSet=[newState.reshape((1,-1)).tolist()]
                            
                            if(curList in prioritySet):
                                #print("here")
                                del prioritySet[prioritySet.index(curList)]
                                #removeCurrentStateFromPrioritySet
                            prioritySet.extend(priorityStateFarmer(iNum+2*sideCOE,jOne-1,newState,newStateSet))
                            #newStateFuncwithRecursion
            if(jTwo in range(0,4)):
                #jTwoCases
                if(currentState[iNum+1*sideCOE,jTwo]<=-1):
                    if(jOne>jTwo):
                        if(currentState[iNum+2*sideCOE,jTwo]==0):
                            newState=np.copy(currentState)
                            newState[iNum,jNum]=0
                            newState[iNum+1*sideCOE,jTwo]=0
                            newState[iNum+2*sideCOE,jTwo]=cur
                            newStateSet=[newState.reshape((1,-1)).tolist()]
                            
                            if(curList  in prioritySet):
                                #print("here")
                                del prioritySet[prioritySet.index(curList)]
                                #removeCurrentStateFromPrioritySet
                            prioritySet.extend(priorityStateFarmer(iNum+2*sideCOE,jTwo,newState,newStateSet))
                            #newStateFuncwithRecursion
                    else:
                        #if(jTwo in range(0,4)):
                        if(currentState[iNum+2*sideCOE,jTwo]==0):
                            newState=np.copy(currentState)
                            newState[iNum,jNum]=0
                            newState[iNum+1*sideCOE,jTwo]=0
                            newState[iNum+2*sideCOE,jTwo]=cur
                            newStateSet=[newState.reshape((1,-1)).tolist()]
                            
                            if(curList  in prioritySet):
                                #print("here")
                                del prioritySet[prioritySet.index(curList)]
                            #removeCurrentStateFromPrioritySet
                            prioritySet.extend(priorityStateFarmer(iNum+2*sideCOE,jTwo,newState,newStateSet))
                            #newStateFuncwithRecursion
        if(cur==2):
            if(iNum-2*sideCOE in range(0,8)):
                if(currentState[iNum-1*sideCOE,jOne]<=-1):
                    #jOneCases
                    if(jOne>jTwo):
                        if(jOne+1 in range(0,4)):
                            if(currentState[iNum-2*sideCOE,jOne+1]==0):
                                newState=np.copy(currentState)
                                newState[iNum,jNum]=0
                                newState[iNum-1*sideCOE,jOne]=0
                                newState[iNum-2*sideCOE,jOne+1]=cur
                                newStateSet=[newState.reshape((1,-1)).tolist()]
                                
                                if(curList  in prioritySet):
                                    del prioritySet[prioritySet.index(curList)]
                                #removeCurrentStateFromPrioritySet
                                prioritySet.extend(priorityStateFarmer(iNum-2*sideCOE,jOne+1,newState,newStateSet))
                                #newStateFuncwithRecursion
                    else:
                        if(jOne-1 in range(0,4)):
                            if(currentState[iNum-2*sideCOE,jOne-1]==0):
                                newState=np.copy(currentState)
                                newState[iNum,jNum]=0
                                newState[iNum-1*sideCOE,jOne]=0
                                newState[iNum-2*sideCOE,jOne-1]=cur
                                newStateSet=[newState.reshape((1,-1)).tolist()]
                                
                                if(curList  in prioritySet):
                                    del prioritySet[prioritySet.index(curList)]
                                    #removeCurrentStateFromPrioritySet
                                prioritySet.extend(priorityStateFarmer(iNum-2*sideCOE,jOne-1,newState,newStateSet))
                                #newStateFuncwithRecursion
                if(jTwo in range(0,4)):
                    #jTwoCases
                    if(currentState[iNum-1*sideCOE,jTwo]<=-1):
                        if(jOne>jTwo):
                            if(currentState[iNum-2*sideCOE,jTwo]==0):
                                #newStateFuncwithRecursion
                                newState=np.copy(currentState)
                                newState[iNum,jNum]=0
                                newState[iNum-1*sideCOE,jTwo]=0
                                newState[iNum-2*sideCOE,jTwo]=cur
                                newStateSet=[newState.reshape((1,-1)).tolist()]
                                
                                if(curList  in prioritySet):
                                    del prioritySet[prioritySet.index(curList)]
                                #removeCurrentStateFromPrioritySet
                                prioritySet.extend(priorityStateFarmer(iNum-2*sideCOE,jTwo,newState,newStateSet))
                        else:
                            if(jTwo+1 in range(0,4)):
                                if(currentState[iNum-2*sideCOE,jTwo]==0):
                                    newState=np.copy(currentState)
                                    newState[iNum,jNum]=0
                                    newState[iNum-1*sideCOE,jTwo]=0
                                    newState[iNum-2*sideCOE,jTwo]=cur
                                    newStateSet=[newState.reshape((1,-1)).tolist()]
                                    
                                    if(curList in prioritySet):
                                        #print("here")
                                        del prioritySet[prioritySet.index(curList)]
                                #removeCurrentStateFromPrioritySet
                                    prioritySet.extend(priorityStateFarmer(iNum-2*sideCOE,jTwo,newState,newStateSet))
                                    #newStateFuncwithRecursion
        return prioritySet
def stateProvider(currentState):
    curList=currentState.tolist()
    currentState=currentState.reshape((8,4))
    workingSet=[]
    prioritySet=[]
    #prioritySet.extend(curList)
    for i in range(0,8):
        for j in range(0,4):
            if(currentState[i,j]>0):
                stateF=stateFarmer(i,j,currentState)
                if(not(stateF)):
                    pass
                else:
                    workingSet.extend(stateF)
                laughingSet=[]
                prioritySet.extend(priorityStateFarmer(i,j,currentState,laughingSet))
                if(curList  in prioritySet):
                    del prioritySet[prioritySet.index(curList)]
    if(len(prioritySet)>0):
        return prioritySet
    else:
        return workingSet
moveset=stateProvider(testState)
print(len(moveset))
for i in range(0,len(moveset)):
    movey=np.array(moveset[i])
    #print(moveset)
    movey=movey.reshape((8,4))
    
    print(i,end='')
    print(":")
    for j in range(7,-1,-1):
        print(movey[j,:])
#^Good code for checkersTuring
#On that note, I'll have to reverse the order that 
#moveDO=np.asarray(moveset)
#print(moveDO.shape)
"""