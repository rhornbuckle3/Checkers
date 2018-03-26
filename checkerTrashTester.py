import pandas as pd
import numpy as np  
import math as mt  
import checkersGame as cG 
import pickle as pk
sideCOE=1
#import checkersFrank as cF
testState=np.array((1,1,1,1,1,1,1,1,1,1,1,1,0,-1,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1))
testState=testState.reshape((8,4))
#print(cG.defaultState)
out=pd.DataFrame(cG.defaultState)
gameRes=pd.Series(np.array([.78345]))
gameRes=gameRes.append(pd.Series(cG.defaultState),ignore_index=True)
#open("testOut.txt",mode='x')
#open("testOut.txt",mode='r')
#out.to_pickle("testOut")
def rowRule(iNum,jNum):
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
    jOne,jTwo=rowRule(iNum,jNum)
    cur=currentState[iNum,jNum]
    jOne,jTwo=rowRule(iNum,jNum)
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
                            if(contains(currentState.reshape((1,-1)).tolist(),prioritySet)):
                                start,end=constainsTwo(currentState.reshape((1,-1)).tolist(),prioritySet)
                                del prioritySet[start:end]
                                #removeCurrentStateFromPrioritySet
                            prioritySet.extend(priorityStateFarmer(iNum+2*sideCOE,jOne+1,newState,newStateSet))
                            #newStateFuncwithRecursion
                else:
                    if(currentState[iNum+2*sideCOE,jOne-1]==0):
                        newState=np.copy(currentState)
                        newState[iNum,jNum]=0
                        newState[iNum+1*sideCOE,jOne]=0
                        newState[iNum+2*sideCOE,jOne-1]=cur
                        newStateSet=[newState.reshape((1,-1)).tolist()]
                        if(contains(currentState.reshape((1,-1)).tolist(),prioritySet)):
                            start,end=constainsTwo(currentState.reshape((1,-1)).tolist(),prioritySet)
                            del prioritySet[start:end]
                            #removeCurrentStateFromPrioritySet
                        prioritySet.extend(priorityStateFarmer(iNum+2*sideCOE,jOne-1,newState,newStateSet))
                        #newStateFuncwithRecursion
            if(jTwo in range(0,4)):
                #jTwoCases
                if(currentState[iNum+1*sideCOE,jTwo]<=-1):
                    if(jOne>jTwo):
                        if(currentState[iNum+2*sideCOE,jTwo-1]==0):
                            newState=np.copy(currentState)
                            newState[iNum,jNum]=0
                            newState[iNum+1*sideCOE,jTwo]=0
                            newState[iNum+2*sideCOE,jTwo]=cur
                            newStateSet=[newState.reshape((1,-1)).tolist()]
                            if(contains(currentState.reshape((1,-1)).tolist(),prioritySet)):
                                start,end=constainsTwo(currentState.reshape((1,-1)).tolist(),prioritySet)
                                del prioritySet[start:end]
                                #removeCurrentStateFromPrioritySet
                            prioritySet.extend(priorityStateFarmer(iNum+2*sideCOE,jTwo,newState,newStateSet))
                            #newStateFuncwithRecursion
                    else:
                        if(jTwo+1 in range(0,4)):
                            if(currentState[iNum+2*sideCOE,jTwo]==0):
                                newState=np.copy(currentState)
                                newState[iNum,jNum]=0
                                newState[iNum+1*sideCOE,jTwo]=0
                                newState[iNum+2*sideCOE,jTwo]=cur
                                newStateSet=[newState.reshape((1,-1)).tolist()]
                                if(contains(currentState.reshape((1,-1)).tolist(),prioritySet)):
                                    start,end=constainsTwo(currentState.reshape((1,-1)).tolist(),prioritySet)
                                    del prioritySet[start:end]
                                    #removeCurrentStateFromPrioritySet
                                prioritySet.append(priorityStateFarmer(iNum+2*sideCOE,jTwo,newState,newStateSet))
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
                                if(contains(currentState.reshape((1,-1)).tolist(),prioritySet)):
                                    start,end=constainsTwo(currentState.reshape((1,-1)).tolist(),prioritySet)
                                    del prioritySet[start:end]
                                    #removeCurrentStateFromPrioritySet
                                prioritySet.extend(priorityStateFarmer(iNum-2*sideCOE,jOne+1,newState,newStateSet))
                                #newStateFuncwithRecursion
                    else:
                        if(currentState[iNum-2*sideCOE,jOne-1]==0):
                            newState=np.copy(currentState)
                            newState[iNum,jNum]=0
                            newState[iNum-1*sideCOE,jOne]=0
                            newState[iNum-2*sideCOE,jOne-1]=cur
                            newStateSet=[newState.reshape((1,-1)).tolist()]
                            if(contains(currentState.reshape((1,-1)).tolist(),prioritySet)):
                                start,end=constainsTwo(currentState.reshape((1,-1)).tolist(),prioritySet)
                                del prioritySet[start:end]
                                #removeCurrentStateFromPrioritySet
                            prioritySet.extend(priorityStateFarmer(iNum-2*sideCOE,jOne-1,newState,newStateSet))
                            #newStateFuncwithRecursion
                if(jTwo in range(0,4)):
                    #jTwoCases
                    if(currentState[iNum-1*sideCOE,jTwo]<=-1):
                        if(jOne>jTwo):
                            if(currentState[iNum-2*sideCOE,jTwo-1]==0):
                                #newStateFuncwithRecursion
                                newState=np.copy(currentState)
                                newState[iNum,jNum]=0
                                newState[iNum-1*sideCOE,jTwo]=0
                                newState[iNum-2*sideCOE,jTwo-1]=cur
                                newStateSet=[newState.reshape((1,-1)).tolist()]
                                if(contains(currentState.reshape((1,-1)).tolist(),prioritySet)):
                                    start,end=constainsTwo(currentState.reshape((1,-1)).tolist(),prioritySet)
                                    del prioritySet[start:end]
                                    #removeCurrentStateFromPrioritySet
                                prioritySet.extend(priorityStateFarmer(iNum-2*sideCOE,jTwo-1,newState,newStateSet))
                        else:
                            if(jTwo+1 in range(0,4)):
                                if(currentState[iNum-2*sideCOE,jTwo+1]==0):
                                    newState=np.copy(currentState)
                                    newState[iNum,jNum]=0
                                    newState[iNum-1*sideCOE,jTwo]=0
                                    newState[iNum-2*sideCOE,jTwo+1]=cur
                                    newStateSet=[newState.reshape((1,-1)).tolist()]
                                    if(contains(currentState.reshape((1,-1)).tolist(),prioritySet)):
                                        start,end=constainsTwo(currentState.reshape((1,-1)).tolist(),prioritySet)
                                        del prioritySet[start:end]
                                        #removeCurrentStateFromPrioritySet
                                    prioritySet.extend(priorityStateFarmer(iNum-2*sideCOE,jTwo+1,newState,newStateSet))
                                    #newStateFuncwithRecursion
        return prioritySet
def stateProvider(currentState):
    currentState.reshape((8,4))
    workingSet=[]
    prioritySet=[]
    prioritySet.extend(currentState.reshape((1,-1)).tolist())
    for i in range(0,8):
        for j in range(0,4):
            if(currentState[i,j]>0):
                stateF=stateFarmer(i,j,currentState)
                if(not(stateF)):
                    pass
                else:
                    workingSet.extend(stateF)
                currentList=currentState.reshape((1,-1)).tolist()
                prioritySet.extend(priorityStateFarmer(i,j,currentState,currentList))
                #print(len(prioritySet))
                #if(prioritySet[(len(prioritySet)-33):(len(prioritySet)-1)]==currentState.reshape((1,-1)).tolist()):
                if(contains(currentState.reshape((1,-1)).tolist(),prioritySet)):
                    start,end=constainsTwo(currentState.reshape((1,-1)).tolist(),prioritySet)
                    del prioritySet[start:end]
    if(prioritySet[(len(prioritySet)-33):(len(prioritySet)-1)]==currentState.reshape((1,-1)).tolist()):
        prioritySet.remove(prioritySet[(len(prioritySet)-33):(len(prioritySet)-1)])
    if(not(currentState.reshape((1,-1)).tolist()==prioritySet[0:31])):
        print('priority set')
        print(len(workingSet))
        return prioritySet
    else:
        print('farm set')
        return workingSet
moveset=stateProvider(testState)
print(len(moveset))
for i in range(0,len(moveset)):
    movey=np.array(moveset[i])
    movey=movey.reshape((8,4))
    
    print(i,end='')
    print(":")
    for j in range(7,-1,-1):
        print(movey[j,:])
#^Good code for checkersTuring
#On that note, I'll have to reverse the order that 
#moveDO=np.asarray(moveset)
#print(moveDO.shape)
