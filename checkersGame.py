#Russell Hornbuckle
#2018-2019
#checkers
#In the process of moving and rewriting the state farmer over here from checkersFrank
#will also likely transform this into a class and sever my reliance on globals (2018 me was wild)
import numpy as np
import pandas as pd 
import math as mt
from checkersFrank import checkersFrank as cf
defaultState=np.array((1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1))
#assign sides via binomial random pulls so that neither actor gets too used to playing as black or white
frankOne=None
frankTwo=None
activePlayer=None
#stateSequence=np.append(stateSequence,np.array([0]),axis=0)

#sequence methods
#initialization
def initPlayer():
    global activePlayer
    global frankOne
    global currentState
    currentState=np.copy(defaultState)
    #global currentState
    #global defaultState
    coinFlip=np.random.binomial(1,.5)
    frankOne=cf()
    frankOne.setSide(coinFlip)
    print('FrankOne is side: '+str(coinFlip)+'; 0 is black, 1 is white')
    frankOne.initWeights("./Frank/bio-One.npz")
    if(coinFlip==1):
        coinFlip=0
    else:
        coinFlip=1
    global frankTwo
    frankTwo=cf()
    frankTwo.setSide(coinFlip)
    frankTwo.initWeights("./Frank/bio-Two.npz")
    if(coinFlip==0):
        activePlayer=frankTwo
    else:
        activePlayer=frankOne
    #currentState=defaultState


#Game manager
def playBall():
    global frankOne
    global frankTwo
    global activePlayer
    global currentState
    turn=1
    #print(frankOne.wOne)
    #print(frankOne.wTwo)
    while(True):
        #print(turn)
        #print(currentState)
        check=False
        newState,newScore=activePlayer.stateDecider(currentState)
        check=endGameCheck(newState)
        if(np.array_equal(currentState,newState)):
            if(activePlayer==frankOne):
                activePlayer=frankTwo
            else:
                activePlayer=frankOne
            count=np.sum(newState)
            if(count>0):
                print("White Wins")
            if(count<0):
                print("Black Wins")
            frankOne.printState(newState)
            endGame()
            break
        if(check):
            count=np.sum(newState)
            if(count>0):
                print("White Wins")
            if(count<0):
                print("Black Wins")
            frankOne.printState(newState)
            endGame()
            break
        if(activePlayer==frankOne):
            activePlayer=frankTwo
        else:
            activePlayer=frankOne
        if(turn>120):
            count=np.sum(newState)
            if(count>0):
                print("White Wins")
            if(count<0):
                print("Black Wins")
            frankOne.printState(newState)
            endGame()
            break
        frankOne.printState(newState)
        currentState=newState
        turn=turn+1

#winner values:
#white=1
#black=0

#game ending
def endGameCheck(currentState):
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
    #combine here, or just have each frank hold onto a complete state sequence, or handle state sequence in this one and record each turn
    #combining here would be the cool thing to do
    #recording here would be the elegant thing to do

def endGame():
    global currentState
    global frankOne
    global frankTwo
    global activePlayer
    #print(activePlayer.sideCOE)
    #print(currentState)
    #print(stateScores)
    if(np.sum(currentState)>0):
        winner=1
    if(np.sum(currentState)<0):
        winner=-1
    if(np.sum(currentState)==0):
        if(activePlayer.sideCOE==1):
            winner=-1
        else:
            winner=1
    #for i in range(0,16):
    #    print(frankOne.wOne[:,i])
    #print("new")
    aOne,aTwo=frankOne.gradDesc(winner)
    frankOne.saveWeights(aOne,aTwo)
    #print(aOne)
    #print(aTwo)
    bOne,bTwo=frankTwo.gradDesc(winner)
    frankTwo.saveWeights(bOne,bTwo)
    del frankOne
    del frankTwo
    del activePlayer
    #print(stateSequence)
    #frankTwo.saveWeights(bOne,bTwo)

def board_expand(board_state):
    print(board_state)

