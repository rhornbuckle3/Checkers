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
        self.stateSequence=[]
        self.stateSequenceScore=[]
    def initWeights(self):
        self.wOne=pd.load_pickle("frankWONE")
        self.wTwo=pd.load_pickle("frankWTWO")
        #gotta figure out a rudimentary saving function for this
    def 
    def retPositA(self):
        return self.positA
    def retPositE(self):
        return self.positE
    def retCurrentStateA(self):
        return self.genInputState(self.retPositA,self.retPositE)
    def retCurrentStateE(self):
        return self.genInputState(self.retPositE,self.retPositA)
    def updateStateSequence(self,stateInput,stateInputScore):
        #takes 8,4 array as input
        #to be called after a new state(n+1) has been decided by this actor
        self.stateSequence.append(stateInput.tolist())
        self.stateSequenceScore.append(stateInputScore)
    def getStateSequence(self):
        return self.stateSequence
    def getStateSequenceScore(self):
        return self.stateSequenceScore
    def genInputState(self,stateA,stateE):
        inputReadyState=np.array(np.zeros((1,48)))
        flipFlop=True
        k=0
        for l in range(0,1):
            for i in range(0,12):
                for j in range(0,1):
                    if(flipFlop):
                        inputReadyState[0,k]=stateA[j,i]
                        k=k+1
                    else:
                        inputReadyState[0,k]=stateE[j,i]
                        k=k+1
            flipFlop=False    
        return inputReadyState
    def stateDecider(self,currentState):
        providedSet=self.stateProvider(currentState)
        stateValue=self.stateEvaluatorMaster(providedSet)
    def stateProvider(self,currentState):
        providedSet=np.array(np.zeros((32,1)))
        #implement this big giant if statement
        return providedSet
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
        return 

    def mainEvaluator(self,potState):
        prodOne=np.array(np.zeros((1,24)))
        for i in range(0,prodOne.shape[1]):
            prodOne[0,i]=self.activationOne(i,potState)
        return self.activationTwo(prodOne)