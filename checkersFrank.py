#Russell Hornbuckle
#2018-2019
#checkers
#currently in the process of moving and rewriting the state farming methods into checkersGame
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
        self.stateSequence=np.copy(cG.defaultState)
        self.stateSequence=self.stateSequence.reshape((-1,1))
        self.stateScores=np.array(np.zeros((1,1)))
        #self.stateSequence=pd.DataFrame(cG.defaultState)
        #self.stateSequence=np.array(cG.defaultState)
        #self.stateSequence=self.stateSequence.reshape((-1,1))
        #self.stateSequence=np.append(self.stateSequence,0,axis=0)
        self.side=None
        self.sideCOE=0
        self.gameNum=None
        self.bioFile=None
    def addToSeq(self,newState,newScore):
        newState=np.array(newState)
        newScore=np.array(newScore)
        newScore=newScore.reshape((1,1))
        #newState=np.append(newState,np.array(newScore))
        newState=newState.reshape((-1,1))
        #print(stateScores.shape)
        #print(newScore.shape)
        try: self.stateSequence=np.append(self.stateSequence,newState,axis=1)
        except NameError:
            self.stateSequence=np.copy(defaultState)
            self.stateSequence=self.stateSequence.reshape((-1,1))
            self.stateSequence=np.append(self.stateSequence,newState,axis=1)
        try: self.stateScores=np.append(self.stateScores,newScore,axis=1)
        except NameError:
            self.stateScores=np.array(np.zeros((1,1)))
            self.stateScores=np.append(self.stateScores,newScore,axis=1)
    def initWeights(self,bioPath):
        bio=np.load(bioPath)
        self.wOne=bio['arr_0']
        self.wTwo=bio['arr_1']
        self.gameNum=bio['arr_2']
        self.bioFile=bioPath
    def saveWeights(self,sOne,sTwo):
        np.savez(self.bioFile,sOne,sTwo,self.gameNum)
    def setSide(self,side):
        if(side==0):
            self.sideCOE=-1
        if(side==1):
            self.sideCOE=1

    def stateEvaluatorMaster(self,provided_set):
        stateValue=np.array(np.zeros(len(provided_set)))
        for i in range(0,stateValue.shape[0]):
            stateValue[i]=self.mainEvaluator(provided_set[i],self.wOne,self.wTwo)
        return stateValue

    def mainEvaluator(self,potState,sOne,sTwo):
        potState=np.array(potState)
        prodOne=np.array(np.zeros((1,16)))
        for i in range(0,prodOne.shape[1]):
            prodOne[0,i]=self.activationOne(i,potState,sOne)
        return self.activationTwo(prodOne,sTwo)
    
    def activationOne(self,neuronNum,inputZero,sOne):
        inputZero=inputZero.reshape((-1,1))
        bingo=1/(1+mt.e**(np.matmul(sOne[:,neuronNum],inputZero)*-1))
        return bingo
        #return mt.log((1+mt.e**np.matmul(sOne[:,neuronNum],inputZero)))

    def activationTwo(self,inputOne,sTwo):
        inputOne=inputOne.reshape((-1,1)) 
        #bingo=mt.log((1+mt.e**np.matmul(self.wTwo[:,0],inputOne)))
        bingo=1/(1+mt.e**(np.matmul(sTwo[:,0],inputOne)*-1))
        #print(bingo)
        return bingo

    def stateDecider(self,currentState):
        currentState=currentState*self.sideCOE
        #provided_set=self.stateProvider(currentState)
        provided_set=cG.state_farmer(currentState)
        stateValue=self.stateEvaluatorMaster(provided_set)
        #if(self.sideCOE==-1):
            #stateValue=stateValue*self.sideCOE
        if(str(stateValue)=='[]'):
            #print(provided_set)
            print(self.printState(currentState))
            print('Player '+str(self.sideCOE)+' Loses')
            currentState
            return currentState, self.mainEvaluator(currentState,self.wOne,self.wTwo)
        else:
            best=np.argmax(stateValue)
        #self.addToSeq(provided_set[range(32*best,32*best+32)],stateValue[best])
        #self.addToSeq(provided_set[best],stateValue[best])
        #print('state values: '+str(stateValue))
        newState=np.array(provided_set[best])
        #print('Player: '+str(self.sideCOE))
        #self.printState(newState)
        #print('Score: '+str(stateValue[best]))
        self.addToSeq(newState,stateValue[best])
        newState=newState*self.sideCOE
        return newState,stateValue[best]

    def gradDesc(self,winner):
        bOne=np.copy(self.wOne)
        bTwo=np.copy(self.wTwo)
        #print(stateSequence)
        #print(score)
        print('Training Player: '+str(self.sideCOE))
        print('State shape: '+str(self.stateSequence.shape[1]))
        #lose condition for both sides if turn limit is reached. Still not sold on this one.
        #if(stateSequence.shape[1]>119):
        #    winner=self.sideCOE*-1
        #winner=winner*self.sideCOE
        if(winner==self.sideCOE):
            winner=1
        else:
            winner=-1
        for i in range(1,self.stateSequence.shape[1]):
            predict=self.stateScores[0,i]
            inputZero=np.copy(self.stateSequence[:,i])
            inputZero=inputZero*self.sideCOE
            prodOne=np.array(np.zeros((1,16)))
            for j in range(0,prodOne.shape[1]):
                prodOne[0,j]=self.activationOne(j,inputZero,bOne)
            learning_step=1e-2
            iterator=0
            while(True):
                iterator+=1
                #second layer
                checkTwo=np.copy(bTwo)
                gradient_upper=self.smallTwo(inputZero,predict,winner,prodOne,bOne,bTwo)
                #gradient_upper - 16x1
                gradient_upper=gradient_upper*np.transpose(prodOne)
                weight_Diff=abs(checkTwo-bTwo)
                checkOne=np.copy(bOne)
                #first layer
                gradient_lower=self.smallOne(inputZero,predict,winner,prodOne,bOne,bTwo)
                #gradient_lower - 32x16
                bOne=np.subtract(bOne,learning_step*gradient_upper)
                bTwo=np.subtract(bTwo,learning_step*gradient_lower)
                #weight_diff_two=abs(checkOne-bOne)
                #weight_diff_two=weight_Diff.flatten()
                #if(weight_Diff[np.argmax(weight_Diff)]<learning_check):
                #    break
                if(iterator>1000):
                    break
            iterator=0
        return bOne, bTwo
    def activationOneD(self, neuronNum,inputZero,sOne,sTwo):
        inputZero=inputZero.reshape((-1,1))
        #return (mt.e**np.matmul(sOne[:,neuronNum],inputZero))/((1+mt.e**np.matmul(sOne[:,neuronNum],inputZero))**2)
        shapeTest=sOne[:,neuronNum].reshape((-1,1))
        shapeTest=np.transpose(shapeTest)
        sigmoid=self.activationOne(neuronNum,inputZero,sOne)
        bingo=sigmoid*(1-sigmoid)
        return bingo

    def smallOne(self,currentState,score,winner,prodOne,sOne,sTwo):
        #result=np.array(np.zeros((1,16)))
        result=np.full((1,16),np.transpose(sTwo)*self.smallTwo(currentState,score,winner,prodOne,sOne,sTwo))
        prodOneD=np.array(np.zeros((1,16)))
        for i in range(0,prodOneD.shape[1]):
            prodOneD[0,i]=self.activationOneD(i,currentState,sOne,sTwo)
        result=np.multiply(result,prodOneD)
        currentState=currentState.reshape((-1,1))
        result=np.dot(currentState,result)
        return result

    def smallTwo(self,currentState,score,winner,prodOne,sOne,sTwo):
        #print('cost: '+str(self.mainEvaluator(currentState,sOne,sTwo)))
        result=np.multiply(((self.mainEvaluator(currentState,sOne,sTwo)-winner)*((mt.e**self.mainEvaluator(currentState,sOne,sTwo))/((1+mt.e**self.mainEvaluator(currentState,sOne,sTwo))**2))),(1/(1+mt.e**(self.mainEvaluator(currentState,sOne,sTwo)*-1))))
        return result

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

    def printState(self,currentState):
        movey=currentState
        movey=movey.reshape((8,4))            
        print("State:")
        for j in range(7,-1,-1):
            print(movey[j,:])
        
    def rowRule(self,iNum,jNum):
        #for determining which movement rules a piece uses dependiong on its row
        q=iNum%2
        if(q==1):
            return jNum,jNum+1
        else:
            return jNum,jNum-1

    #everything below is deprecated

    def stateProvider(self,currentState):
        #print(currentState)
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
        #cur is the current space being checked for pieces
        cur=currentState[iNum,jNum]
        jOne,jTwo=self.rowRule(iNum,jNum)
        if(iNum+1*self.sideCOE in range(0,8)):    
            if(currentState[iNum+1*self.sideCOE,jOne]==0):
                newState=np.copy(currentState)
                newState[iNum,jNum]=0
                newState[iNum+1*self.sideCOE,jOne]=cur
                #king me bit
                if(((iNum+1==7)and(self.sideCOE==1))or((iNum-1==0)and(self.sideCOE==-1))):
                    newState[iNum,jNum]=2
                newState=newState.reshape((1,-1)).tolist()
                farmSet.extend(newState)
            if(jTwo in range(0,4)):
                if(currentState[iNum+1*self.sideCOE,jTwo]==0):
                    newState=np.copy(currentState)
                    newState[iNum,jNum]=0
                    newState[iNum+1*self.sideCOE,jTwo]=cur
                    #king me bit
                    if(((iNum+1==7)and(self.sideCOE==1))or((iNum-1==0)and(self.sideCOE==-1))):
                        newState[iNum,jNum]=2
                    newState=newState.reshape((1,-1)).tolist()
                    farmSet.extend(newState)
        #if cur is a king
        if(cur==2):
            if(iNum-1*self.sideCOE in range(0,8)):
                #there is an error with king movement of some sort that prevents them from moving out of the corners    
                if(currentState[iNum-1*self.sideCOE,jOne]==0):
                    newState=np.copy(currentState)
                    newState[iNum,jNum]=0
                    newState[iNum-1*self.sideCOE,jOne]=cur
                    if(((iNum+1*self.sideCOE==7)and(self.sideCOE==1))or((iNum+1*self.sideCOE==0)and(self.sideCOE==-1))):
                        newState[iNum,jNum]=2
                    newState=newState.reshape((1,-1)).tolist()
                    farmSet.extend(newState)
                if(jTwo in range(0,4)):
                    if(currentState[iNum-1*self.sideCOE,jTwo]==0):
                        newState=np.copy(currentState)
                        newState[iNum,jNum]=0
                        newState[iNum-1*self.sideCOE,jTwo]=cur
                        if(((iNum+1*self.sideCOE==7)and(self.sideCOE==1))or((iNum+1*self.sideCOE==0)and(self.sideCOE==-1))):
                            newState[iNum,jNum]=2
                        newState=newState.reshape((1,-1)).tolist()
                        farmSet.extend(newState)                  
        return farmSet
    def priorityStateFarmer(self,iNum,jNum,currentState,prioritySet):
        #Priority Move Checker
        #the converter to king
        if(((iNum==7)and(self.sideCOE==1))or((iNum==0)and(self.sideCOE==-1))):
            currentState[iNum,jNum]=2
        curList=currentState.reshape((1,-1)).tolist()
        jOne,jTwo=self.rowRule(iNum,jNum)
        cur=currentState[iNum,jNum]
        if(cur>0):
            if(iNum+2*self.sideCOE in range(0,8)):   
                #checking if enemy piece in jumpable spot
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
    
