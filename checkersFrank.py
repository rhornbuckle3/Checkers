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
        self.side=None
        self.sideCOE=0
        self.gameNum=None
        self.bioFile=None
    def addToSeq(self,newState,newScore):
        newState=np.array(newState)
        newScore=np.array(newScore)
        newScore=newScore.reshape((1,1))
        newState=newState.reshape((-1,1))
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
        return 1/(1+mt.e**(np.matmul(sOne[:,neuronNum],inputZero)*-1))

    def activationTwo(self,inputOne,sTwo):
        inputOne=inputOne.reshape((-1,1)) 
        return 1/(1+mt.e**(np.matmul(sTwo[:,0],inputOne)*-1))


    def stateDecider(self,currentState):
        currentState=currentState*self.sideCOE
        #provided_set=self.stateProvider(currentState)
        provided_set=cG.state_farmer(currentState,self.sideCOE)
        #for sub_set in provided_set:
           #for board in sub_set:
                #for sub in board:
                    #print(sub)
        stateValue=self.stateEvaluatorMaster(provided_set)
        #if(self.sideCOE==-1):
            #stateValue=stateValue*self.sideCOE
        if(str(stateValue)=='[]'):
            #print(provided_set)
            print(cG.print_state(currentState))
            print('Player '+str(self.sideCOE)+' Loses')
            currentState
            return currentState, self.mainEvaluator(currentState,self.wOne,self.wTwo)
        else:
            best=np.argmax(stateValue)
        newState=np.array(provided_set[best])
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
                gradient_upper=self.smallTwo(inputZero,predict,winner,prodOne,bOne,bTwo)
                #gradient_upper - 16x1
                gradient_upper=gradient_upper*np.transpose(prodOne)
                checkOne=np.copy(bOne)
                #first layer
                gradient_lower=self.smallOne(inputZero,predict,winner,prodOne,bOne,bTwo)
                #gradient_lower - 32x16
                bOne=np.subtract(bOne,learning_step*gradient_lower)
                bTwo=np.subtract(bTwo,learning_step*gradient_upper)
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
        return sigmoid*(1-sigmoid)

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