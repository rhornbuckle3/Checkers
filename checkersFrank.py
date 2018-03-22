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
    def initWeights(self,weightONE,weightTWO):
        self.wOne=pd.load_pickle("frankWONE")
        self.wTwo=pd.load_pickle("frankWTWO")
    def setSide(self,side):
        self.side=side
        #gotta figure out a rudimentary saving function for this
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
    def stateProvider(self,currentState):
        currentState.reshape((8,4))
        workingSet=[]
        if(side='black'):
            for i in range(0,8):
                for j in range(0,4):
                    if(currentState[i,j]==1):


        else:  

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
        return stateValue
    def mainEvaluator(self,potState):
        prodOne=np.array(np.zeros((1,24)))
        for i in range(0,prodOne.shape[1]):
            prodOne[0,i]=self.activationOne(i,potState)
        return self.activationTwo(prodOne)