import numpy as np
import pandas as pd 
import math as mt
from checkersFrank import checkersFrank as cf
defaultState=np.array((1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1))
#assign sides via binomial random pulls so that neither actor gets too used to playing as black or white
frankOne=None
frankTwo=None
activePlayer=None
stateSequence=np.copy(defaultState)
stateSequence=stateSequence.reshape((-1,1))
stateScores=np.array(np.zeros((1,1)))
#stateSequence=np.append(stateSequence,np.array([0]),axis=0)
currentState=np.copy(defaultState)

#sequence methods
def addToSeq(newState,newScore):
        global stateSequence
        global stateScores
        newState=np.array(newState)
        newScore=np.array(newScore)
        newScore=newScore.reshape((1,1))
        #newState=np.append(newState,np.array(newScore))
        newState=newState.reshape((-1,1))
        #print(stateScores.shape)
        #print(newScore.shape)
        stateSequence=np.append(stateSequence,newState,axis=1)
        stateScores=np.append(stateScores,newScore,axis=1)
def getStateSequence():
    global stateSequence
    return stateSequence

#initialization
def initPlayer():
    global activePlayer
    global frankOne
    #global currentState
    #global defaultState
    coinFlip=np.random.binomial(0,.5)
    frankOne=cf()
    frankOne.setSide(coinFlip)
    frankOne.initWeights("/home/russell/Documents/Frank/bio-One.npz")
    if(coinFlip==1):
        coinFlip=0
    else:
        coinFlip=1
    global frankTwo
    frankTwo=cf()
    frankTwo.setSide(coinFlip)
    frankTwo.initWeights("/home/russell/Documents/Frank/bio-One.npz")
    if(coinFlip==0):
        activePlayer=frankTwo
    else:
        activePlayer=frankOne
    #currentState=defaultState



#Game manager
def playBall():
    global stateSequence
    global currentState
    global frankOne
    global frankTwo
    global activePlayer
    turn=1
    while(True):
        #print(turn)
        #print(currentState)
        newState,newScore=activePlayer.stateDecider(currentState)
        addToSeq(newState,newScore)
        check=endGameCheck(newState)
        if(check):
            break
        if(activePlayer==frankOne):
            activePlayer=frankTwo
        else:
            activePlayer=frankOne
        if(turn>=1200):
            count=np.sum(newState)
            if(count>0):
                print("White Wins")
                break
            if(count<0):
                print("Black Wins")
                break
        currentState=newState
        turn=turn+1
    #print(stateSequence.shape)
    endGame()

#winner values:
#white=1
#black=0

#game ending
def endGameCheck(currentState):
    global stateSequence
    #global currentState
    global frankOne
    global frankTwo
    global activePlayer
    sumBum=0
    currentState=currentState.reshape((-1,1))
    for i in range(0,32):
        sumBum=sumBum+mt.fabs(currentState[i,0])
    if(mt.fabs(np.sum(currentState))==sumBum):
        return True
    return False
    #playOne=frankOne.getStateSequence()
    #playTwo=frankTwo.getStateSequence()
    #combine here, or just have each frank hold onto a complete state sequence, or handle state sequence in this one and record each turn
    #combining here would be the cool thing to do
    #recording here would be the elegant thing to do
def endGame():
    global stateSequence
    global stateScores
    global currentState
    global frankOne
    global frankTwo
    global activePlayer
    print(activePlayer.sideCOE)
    print(currentState)
    #print(stateScores)
    if(np.sum(currentState)>0):
        winner=1
    if(np.sum(currentState)<0):
        winner=0
    if(np.sum(currentState)==0):
        if(activePlayer.sideCOE==1):
            winner=0
        else:
            winner=1
    #for i in range(0,16):
    #    print(frankOne.wOne[:,i])
    #print("new")
    aOne,aTwo=frankOne.gradDesc(stateSequence[:,1:3],stateScores,winner)
    #for i in range(0,16):
    #    print(aOne[:,i])
    #bOne,bTwo=frankTwo.gradDesc(stateSequence,stateScores,winner)
    frankOne.saveWeights(aOne,aTwo)
    #frankTwo.saveWeights(bOne,bTwo)
