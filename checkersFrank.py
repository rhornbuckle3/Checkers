import numpy as np
import pandas as pd 
import math as mt 
import checkersGame as cG
#checkers-Act
class checkersFrank:
    def __init__(self):
        self.currentState=cG.defaultState
        self.wOne=np.array(np.zeros((32,16)))
        self.wTwo=np.array(np.zeros((16,1)))
        self.kingDex=np.array(np.zeros((12)))
        self.stateSequence=pd.DataFrame(cG.defaultState)
        self.stateSequence.append(np.array([0]))
        self.side=None
        self.sideCOE=0
    def initWeights(self,weightONE,weightTWO):
        self.wOne=pd.load_pickle("frankWONE")
        self.wTwo=pd.load_pickle("frankWTWO")
        #gotta figure out a rudimentary saving function for this
    def setSide(self,side):
        self.side=side
        if(side==0):
            self.sideCOE=-1
        if(side==1):
            self.sideCOE=1
    def getStateSequence(self):
        return self.stateSequence
    def addToSeq(self,newState,newScore):
        newState=np.append(np.array(newScore))
        self.stateSequence=pd.concat([self.stateSequence,pd.DataFrame(newState)],axis=1)
    def stateDecider(self,currentState):
        providedSet=self.stateProvider(currentState)
        stateValue=self.stateEvaluatorMaster(providedSet)
        best=np.argmax(stateValue)
        self.addToSeq(providedSet[best],stateValue[best])
        return providedSet[best]
    def rowRule(self,iNum,jNum):
        q=iNum%2
        if(q==1):
            return jNum,jNum+1
        else:
            return jNum,jNum-1
    def contains(self, small, big):
        for i in range(0,len(big)-len(small)+1):
            for j in range(0,len(small)):
                if(big[i+j] != small[j]):
                    break
            else:
                return True #i, i+len(small)
        return False
    def constainsTwo(self, small, big):
        for i in range(0,len(big)-len(small)+1):
            for j in range(0,len(small)):
                if(big[i+j] != small[j]):
                    break
            else:
                return i, i+len(small)
        return False
    def stateFarmer(self,iNum,jNum,currentState):
        #Empty Move Checker
        farmSet=[]
        cur=currentState[iNum,jNum]
        jOne,jTwo=self.rowRule(iNum,jNum)
        if(iNum+1*self.sideCOE in range(0,8)):    
            if(currentState[iNum+1*self.sideCOE,jOne]==0):
                newState=np.copy(currentState)
                newState[iNum,jNum]=0
                newState[iNum+1*self.sideCOE,jOne]=cur
                newState=newState.reshape((1,-1))
                farmSet.append(newState)
            if(jTwo in range(0,4)):
                if(currentState[iNum+1*self.sideCOE,jTwo]==0):
                    newState=np.copy(currentState)
                    newState[iNum,jNum]=0
                    newState[iNum+1*self.sideCOE,jTwo]=cur
                    newState=newState.reshape((1,-1))
                    farmSet.append(newState)
        if(cur==2):
            if(iNum-1*self.sideCOE in range(0,8)):    
                if(currentState[iNum-1*self.sideCOE,jOne]==0):
                    newState=np.copy(currentState)
                    newState[iNum,jNum]=0
                    newState[iNum-1*self.sideCOE,jOne]=cur
                    newState=newState.reshape((1,-1))
                    farmSet.append(newState)
                if(jTwo in range(0,4)):
                    if(currentState[iNum-1*self.sideCOE,jTwo]==0):
                        newState=np.copy(currentState)
                        newState[iNum,jNum]=0
                        newState[iNum-1*self.sideCOE,jTwo]=cur
                        newState=newState.reshape((1,-1))
                        farmSet.append(newState)
        return farmSet
    def priorityStateFarmer(self,iNum,jNum,currentState,prioritySet):
        #Priority Move Checker
        if(((iNum==7)and(self.sideCOE==1))or((iNum=0)and(self.sideCOE==-1))):
            currentState[iNum,jNum]=2
        jOne,jTwo=self.rowRule(iNum,jNum)
        cur=currentState[iNum,jNum]
        jOne,jTwo=self.rowRule(iNum,jNum)
        if(cur>=1):
            if(iNum+2*self.sideCOE in range(0,8)):   
                if(currentState[iNum+1*self.sideCOE,jOne]<=-1):
                    #jOneCases
                    if(jOne>jTwo):
                        if(jOne+1 in range(0,4)):
                            if(currentState[iNum+2*self.sideCOE,jOne+1]==0):
                                newState=np.copy(currentState)
                                newState[iNum,jNum]=0
                                newState[iNum+1*self.sideCOE,jOne]=0
                                newState[iNum+2*self.sideCOE,jOne+1]=cur
                                newStateSet=[newState.reshape((1,-1))]
                                if(self.contains(currentState.reshape((1,-1)).tolist(),prioritySet)):
                                    start,end=self.constainsTwo(currentState.reshape((1,-1)).tolist(),prioritySet)
                                    prioritySet.remove(range(start,end))
                                    #removeCurrentStateFromPrioritySet
                                prioritySet.append(self.priorityStateFarmer(iNum+2*self.sideCOE,jOne+1,newState,newStateSet))
                                #newStateFuncwithRecursion
                    else:
                        if(currentState[iNum+2*self.sideCOE,jOne-1]==0):
                            newState=np.copy(currentState)
                            newState[iNum,jNum]=0
                            newState[iNum+1*self.sideCOE,jOne]=0
                            newState[iNum+2*self.sideCOE,jOne-1]=cur
                            newStateSet=[newState.reshape((1,-1))]
                            if(self.contains(currentState.reshape((1,-1)).tolist(),prioritySet)):
                                start,end=self.constainsTwo(currentState.reshape((1,-1)).tolist(),prioritySet)
                                prioritySet.remove(range(start,end))
                                #removeCurrentStateFromPrioritySet
                            prioritySet.append(self.priorityStateFarmer(iNum+2*self.sideCOE,jOne-1,newState,newStateSet))
                            #newStateFuncwithRecursion
                if(jTwo in range(0,4)):
                    #jTwoCases
                    if(currentState[iNum+1*self.sideCOE,jTwo]<=-1):
                        if(jOne>jTwo):
                            if(currentState[iNum+2*self.sideCOE,jTwo-1]==0):
                                newState=np.copy(currentState)
                                newState[iNum,jNum]=0
                                newState[iNum+1*self.sideCOE,jTwo]=0
                                newState[iNum+2*self.sideCOE,jTwo-1]=cur
                                newStateSet=[newState.reshape((1,-1))]
                                if(self.contains(currentState.reshape((1,-1)).tolist(),prioritySet)):
                                    start,end=self.constainsTwo(currentState.reshape((1,-1)).tolist(),prioritySet)
                                    prioritySet.remove(range(start,end))
                                    #removeCurrentStateFromPrioritySet
                                prioritySet.append(self.priorityStateFarmer(iNum+2*self.sideCOE,jTwo-1,newState,newStateSet))
                                #newStateFuncwithRecursion
                        else:
                            if(jTwo+1 in range(0,4)):
                                if(currentState[iNum+2*self.sideCOE,jTwo+1]==0):
                                    newState=np.copy(currentState)
                                    newState[iNum,jNum]=0
                                    newState[iNum+1*self.sideCOE,jTwo]=0
                                    newState[iNum+2*self.sideCOE,jTwo+1]=cur
                                    newStateSet=[newState.reshape((1,-1))]
                                    if(self.contains(currentState.reshape((1,-1)).tolist(),prioritySet)):
                                        start,end=self.constainsTwo(currentState.reshape((1,-1)).tolist(),prioritySet)
                                        prioritySet.remove(range(start,end))
                                        #removeCurrentStateFromPrioritySet
                                    prioritySet.append(self.priorityStateFarmer(iNum+2*self.sideCOE,jTwo+1,newState,newStateSet))
                                    #newStateFuncwithRecursion
            if(cur==2):
                if(iNum-2*self.sideCOE in range(0,8)):
                    if(currentState[iNum-1*self.sideCOE,jOne]<=-1):
                        #jOneCases
                        if(jOne>jTwo):
                            if(jOne+1 in range(0,4)):
                                if(currentState[iNum-2*self.sideCOE,jOne+1]==0):
                                    newState=np.copy(currentState)
                                    newState[iNum,jNum]=0
                                    newState[iNum-1*self.sideCOE,jOne]=0
                                    newState[iNum-2*self.sideCOE,jOne+1]=cur
                                    newStateSet=[newState.reshape((1,-1))]
                                    if(self.contains(currentState.reshape((1,-1)).tolist(),prioritySet)):
                                        start,end=self.constainsTwo(currentState.reshape((1,-1)).tolist(),prioritySet)
                                        prioritySet.remove(range(start,end))
                                        #removeCurrentStateFromPrioritySet
                                    prioritySet.append(self.priorityStateFarmer(iNum-2*self.sideCOE,jOne+1,newState,newStateSet))
                                    #newStateFuncwithRecursion
                        else:
                            if(currentState[iNum-2*self.sideCOE,jOne-1]==0):
                                newState=np.copy(currentState)
                                newState[iNum,jNum]=0
                                newState[iNum-1*self.sideCOE,jOne]=0
                                newState[iNum-2*self.sideCOE,jOne-1]=cur
                                newStateSet=[newState.reshape((1,-1))]
                                if(self.contains(currentState.reshape((1,-1)).tolist(),prioritySet)):
                                    start,end=self.constainsTwo(currentState.reshape((1,-1)).tolist(),prioritySet)
                                    prioritySet.remove(range(start,end))
                                    #removeCurrentStateFromPrioritySet
                                prioritySet.append(self.priorityStateFarmer(iNum-2*self.sideCOE,jOne-1,newState,newStateSet))
                                #newStateFuncwithRecursion
                    if(jTwo in range(0,4)):
                        #jTwoCases
                        if(currentState[iNum-1*self.sideCOE,jTwo]<=-1):
                            if(jOne>jTwo):
                                if(currentState[iNum-2*self.sideCOE,jTwo-1]==0):
                                    #newStateFuncwithRecursion
                                    newState=np.copy(currentState)
                                    newState[iNum,jNum]=0
                                    newState[iNum-1*self.sideCOE,jTwo]=0
                                    newState[iNum-2*self.sideCOE,jTwo-1]=cur
                                    newStateSet=[newState.reshape((1,-1))]
                                    if(self.contains(currentState.reshape((1,-1)).tolist(),prioritySet)):
                                        start,end=self.constainsTwo(currentState.reshape((1,-1)).tolist(),prioritySet)
                                        prioritySet.remove(range(start,end))
                                        #removeCurrentStateFromPrioritySet
                                    prioritySet.append(self.priorityStateFarmer(iNum-2*self.sideCOE,jTwo-1,newState,newStateSet))
                            else:
                                if(jTwo+1 in range(0,4)):
                                    if(currentState[iNum-2*self.sideCOE,jTwo+1]==0):
                                        newState=np.copy(currentState)
                                        newState[iNum,jNum]=0
                                        newState[iNum-1*self.sideCOE,jTwo]=0
                                        newState[iNum-2*self.sideCOE,jTwo+1]=cur
                                        newStateSet=[newState.reshape((1,-1))]
                                        if(self.contains(currentState.reshape((1,-1)).tolist(),prioritySet)):
                                            start,end=self.constainsTwo(currentState.reshape((1,-1)).tolist(),prioritySet)
                                            prioritySet.remove(range(start,end))
                                            #removeCurrentStateFromPrioritySet
                                        prioritySet.append(self.priorityStateFarmer(iNum-2*self.sideCOE,jTwo+1,newState,newStateSet))
                                        #newStateFuncwithRecursion
            return prioritySet
    def stateProvider(self,currentState):
        currentState.reshape((8,4))
        workingSet=[]
        prioritySet=[]
        for i in range(0,8):
            for j in range(0,4):
                if(currentState[i,j]<0):
                    workingSet.append(self.stateFarmer(i,j,currentState))
                    prioritySet.append(currentState.reshape((1,-1)).tolist())
                    prioritySet.append(self.priorityStateFarmer(i,j,currentState,prioritySet))
                    if(prioritySet[range(len(prioritySet)-32],len(prioritySet))]):
                        prioritySet.remove(range(len(prioritySet)-32],len(prioritySet)))
        #implement this big giant if statement
        if(not(currentState.reshape((1,-1)).tolist()==prioritySet[0,31])
            return prioritySet
        else:
            return farmSet
    def activationOne(self,neuronNum,inputZero):
        inputZero.reshape((1,-1))
        return 1/(1+mt.e**-(np.matmul(wOne[:,neuronNum],inputZero)))
    def activationTwo(self,inputOne):
        inputOne.reshape((1,-1))
        return mt.log((1+mt.e**np.matmul(wTwo[:,0],inputOne)))
    def stateEvaluatorMaster(self,providedSet):
        stateValue=np.array(np.zeros(providedSet.shape[1]))
        for i in range(0,providedSet.shape[1]):
            stateValue[i]=self.mainEvaluator(providedSet[:,i])
        return stateValue
    def mainEvaluator(self,potState):
        prodOne=np.array(np.zeros((1,24)))
        for i in range(0,prodOne.shape[1]):
            prodOne[0,i]=self.activationOne(i,potState)
        return self.activationTwo(prodOne)
