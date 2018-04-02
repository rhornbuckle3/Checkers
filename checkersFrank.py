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
        #self.stateSequence=pd.DataFrame()
        #Redo this with a pandas dataframe
        self.side=None
        self.sideCOE=0
        self.gameNum=None
        self.bioFile=None
    def initWeights(self,bioPath):
        bio=np.load(bioPath)
        self.wOne=bio['arr_0']
        self.wTwo=bio['arr_1']
        self.gameNum=bio['arr_2']
        self.bioFile=bioPath
    def saveWeights(self)
        np.savez(self.bioFile,self.wOne,self.wTwo,self.gameNum)
    def setSide(self,side):
        if(side==0):
            self.sideCOE=-1
        if(side==1):
            self.sideCOE=1







    #Rewrite everything below this
    def activationOne(self,neuronNum,inputZero):
        inputZero.reshape((1,-1))
        return mt.log((1+mt.e**np.matmul(wOne[:,0],inputZero)))
    def activationTwo(self,inputOne):
        inputOne.reshape((1,-1))
        prodOne=np.array(np.zeros((1,24)))
        for i in range(0,prodOne.shape[1]):
            prodOne[0,i]=mt.log((1+mt.e**np.matmul(wTwo[:,0],inputOne)))
        return 
    def stateEvaluatorMaster(self,providedSet):
        stateValue=np.array(np.zeros(len(providedSet)))
        for i in range(0,stateValue.shape[0]):
            stateValue[i]=self.mainEvaluator(providedSet[range(i*32,i*32+32)])
        return stateValue
    def mainEvaluator(self,potState):
        #USING TENSORFLOW, ALL OF THE FOLLOWING CODE IS POOP
        prodOne=np.array(np.zeros((1,24)))
        for i in range(0,prodOne.shape[1]):
            prodOne[0,i]=self.activationOne(i,potState)
        return self.activationTwo(prodOne)






    def getStateSequence(self):
        return self.stateSequence
    def addToSeq(self,newState,newScore):
        newState=np.append(newState,np.array(newScore))
        self.stateSequence=pd.concat([self.stateSequence,pd.DataFrame(newState)],axis=1)
        #^up to the consideration of the logging method, likely to be changed
    




    def stateDecider(self,currentState):
        providedSet=self.stateProvider(currentState)
        stateValue=self.stateEvaluatorMaster(providedSet)
        best=np.argmax(stateValue)
        self.addToSeq(providedSet[range(32*best,32*best+32)],stateValue[best])
        return providedSet[range(32*best,32*best+32)]
    
    
    
    def printMoves(self, currentState):
        moveset=self.stateProvider(currentState)
        print(len(moveset))
        for i in range(0,len(moveset)):
            movey=np.array(moveset[i])
            movey=movey.reshape((8,4))            
            print(i,end='')
            print(":")
            for j in range(7,-1,-1):
                print(movey[j,:])
    def rowRule(self,iNum,jNum):
        q=iNum%2
        if(q==1):
            return jNum,jNum+1
        else:
            return jNum,jNum-1
    def stateProvider(self,currentState):
        curList=currentState.tolist()
        currentState=currentState.reshape((8,4))
        workingSet=[]
        prioritySet=[]
        #prioritySet.extend(curList)
        for i in range(0,8):
            for j in range(0,4):
                if(currentState[i,j]>0):
                    stateF=self.StateFarmer(i,j,currentState)
                    if(not(stateF)):
                        pass
                    else:
                        workingSet.extend(stateF)
                    laughingSet=[]
                    prioritySet.extend(self.priorityStateFarmer(i,j,currentState,laughingSet))
                    if(curList  in prioritySet):
                        del prioritySet[prioritySet.index(curList)]
        if(len(prioritySet)>0):
            return prioritySet
        else:
            return workingSet
    def StateFarmer(self,iNum,jNum,currentState):
        #Empty Move Checker
        farmSet=[]
        cur=currentState[iNum,jNum]
        jOne,jTwo=self.rowRule(iNum,jNum)
        if(iNum+1*self.sideCOE in range(0,8)):    
            if(currentState[iNum+1*self.sideCOE,jOne]==0):
                newState=np.copy(currentState)
                newState[iNum,jNum]=0
                newState[iNum+1*self.sideCOE,jOne]=cur
                newState=newState.reshape((1,-1)).tolist()
                farmSet.extend(newState)
            if(jTwo in range(0,4)):
                if(currentState[iNum+1*self.sideCOE,jTwo]==0):
                    newState=np.copy(currentState)
                    newState[iNum,jNum]=0
                    newState[iNum+1*self.sideCOE,jTwo]=cur
                    newState=newState.reshape((1,-1)).tolist()
                    farmSet.extend(newState)
        if(cur==2):
            if(iNum-1*self.sideCOE in range(0,8)):    
                if(currentState[iNum-1*self.sideCOE,jOne]==0):
                    newState=np.copy(currentState)
                    newState[iNum,jNum]=0
                    newState[iNum-1*self.sideCOE,jOne]=cur
                    newState=newState.reshape((1,-1)).tolist()
                    farmSet.extend(newState)
                if(jTwo in range(0,4)):
                    if(currentState[iNum-1*self.sideCOE,jTwo]==0):
                        newState=np.copy(currentState)
                        newState[iNum,jNum]=0
                        newState[iNum-1*self.sideCOE,jTwo]=cur
                        newState=newState.reshape((1,-1)).tolist()
                        farmSet.extend(newState)    
        return farmSet
    def priorityStateFarmer(self,iNum,jNum,currentState,prioritySet):
        #Priority Move Checker
        if(((iNum==7)and(self.sideCOE==1))or((iNum==0)and(self.sideCOE==-1))):
            currentState[iNum,jNum]=2
        curList=currentState.reshape((1,-1)).tolist()
        jOne,jTwo=self.rowRule(iNum,jNum)
        cur=currentState[iNum,jNum]

        if(mt.fabs(cur)>=1):
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
                                newStateSet=[newState.reshape((1,-1)).tolist()]
                                
                                if(curList  in prioritySet):
                                    #print("here")
                                    del prioritySet[prioritySet.index(curList)]
                                    #removeCurrentStateFromPrioritySet
                                prioritySet.extend(self.priorityStateFarmer(iNum+2*self.sideCOE,jOne+1,newState,newStateSet))
                                #newStateFuncwithRecursion
                    else:
                        if(jOne-1 in range(0,4)):
                            if(currentState[iNum+2*self.sideCOE,jOne-1]==0):
                                newState=np.copy(currentState)
                                newState[iNum,jNum]=0
                                newState[iNum+1*self.sideCOE,jOne]=0
                                newState[iNum+2*self.sideCOE,jOne-1]=cur
                                newStateSet=[newState.reshape((1,-1)).tolist()]
                                
                                if(curList in prioritySet):
                                    #print("here")
                                    del prioritySet[prioritySet.index(curList)]
                                    #removeCurrentStateFromPrioritySet
                                prioritySet.extend(self.priorityStateFarmer(iNum+2*self.sideCOE,jOne-1,newState,newStateSet))
                                #newStateFuncwithRecursion
                if(jTwo in range(0,4)):
                    #jTwoCases
                    if(currentState[iNum+1*self.sideCOE,jTwo]<=-1):
                        if(jOne>jTwo):
                            if(currentState[iNum+2*self.sideCOE,jTwo]==0):
                                newState=np.copy(currentState)
                                newState[iNum,jNum]=0
                                newState[iNum+1*self.sideCOE,jTwo]=0
                                newState[iNum+2*self.sideCOE,jTwo]=cur
                                newStateSet=[newState.reshape((1,-1)).tolist()]
                                
                                if(curList  in prioritySet):
                                    #print("here")
                                    del prioritySet[prioritySet.index(curList)]
                                    #removeCurrentStateFromPrioritySet
                                prioritySet.extend(self.priorityStateFarmer(iNum+2*self.sideCOE,jTwo,newState,newStateSet))
                                #newStateFuncwithRecursion
                        else:
                            #if(jTwo in range(0,4)):
                            if(currentState[iNum+2*self.sideCOE,jTwo]==0):
                                newState=np.copy(currentState)
                                newState[iNum,jNum]=0
                                newState[iNum+1*self.sideCOE,jTwo]=0
                                newState[iNum+2*self.sideCOE,jTwo]=cur
                                newStateSet=[newState.reshape((1,-1)).tolist()]
                                
                                if(curList  in prioritySet):
                                    #print("here")
                                    del prioritySet[prioritySet.index(curList)]
                                #removeCurrentStateFromPrioritySet
                                prioritySet.extend(self.priorityStateFarmer(iNum+2*self.sideCOE,jTwo,newState,newStateSet))
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
                                    newStateSet=[newState.reshape((1,-1)).tolist()]
                                    
                                    if(curList  in prioritySet):
                                        del prioritySet[prioritySet.index(curList)]
                                    #removeCurrentStateFromPrioritySet
                                    prioritySet.extend(self.priorityStateFarmer(iNum-2*self.sideCOE,jOne+1,newState,newStateSet))
                                    #newStateFuncwithRecursion
                        else:
                            if(jOne-1 in range(0,4)):
                                if(currentState[iNum-2*self.sideCOE,jOne-1]==0):
                                    newState=np.copy(currentState)
                                    newState[iNum,jNum]=0
                                    newState[iNum-1*self.sideCOE,jOne]=0
                                    newState[iNum-2*self.sideCOE,jOne-1]=cur
                                    newStateSet=[newState.reshape((1,-1)).tolist()]
                                    
                                    if(curList  in prioritySet):
                                        del prioritySet[prioritySet.index(curList)]
                                        #removeCurrentStateFromPrioritySet
                                    prioritySet.extend(self.priorityStateFarmer(iNum-2*self.sideCOE,jOne-1,newState,newStateSet))
                                    #newStateFuncwithRecursion
                    if(jTwo in range(0,4)):
                        #jTwoCases
                        if(currentState[iNum-1*self.sideCOE,jTwo]<=-1):
                            if(jOne>jTwo):
                                if(currentState[iNum-2*self.sideCOE,jTwo]==0):
                                    #newStateFuncwithRecursion
                                    newState=np.copy(currentState)
                                    newState[iNum,jNum]=0
                                    newState[iNum-1*self.sideCOE,jTwo]=0
                                    newState[iNum-2*self.sideCOE,jTwo]=cur
                                    newStateSet=[newState.reshape((1,-1)).tolist()]
                                    
                                    if(curList  in prioritySet):
                                        del prioritySet[prioritySet.index(curList)]
                                    #removeCurrentStateFromPrioritySet
                                    prioritySet.extend(self.priorityStateFarmer(iNum-2*self.sideCOE,jTwo,newState,newStateSet))
                            else:
                                if(jTwo+1 in range(0,4)):
                                    if(currentState[iNum-2*self.sideCOE,jTwo]==0):
                                        newState=np.copy(currentState)
                                        newState[iNum,jNum]=0
                                        newState[iNum-1*self.sideCOE,jTwo]=0
                                        newState[iNum-2*self.sideCOE,jTwo]=cur
                                        newStateSet=[newState.reshape((1,-1)).tolist()]
                                        
                                        if(curList in prioritySet):
                                            #print("here")
                                            del prioritySet[prioritySet.index(curList)]
                                    #removeCurrentStateFromPrioritySet
                                        prioritySet.extend(self.priorityStateFarmer(iNum-2*self.sideCOE,jTwo,newState,newStateSet))
                                        #newStateFuncwithRecursion
            return prioritySet
    
