import numpy as np
import pandas as pd 
import math as mt
from checkersFrank import checkersFrank as cf
defaultState=np.array((1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1))
#assign sides via binomial random pulls so that neither actor gets too used to playing as black or white
frankOne=None
frankTwo=None
activePlayer=None
def initPlayer():
    global activePlayer
    global frankOne
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
    frankTwo.initWeights("/home/russell/Documents/Frank/bio-Two.npz")
    if(coinFlip==0):
        activePlayer=frankTwo
    else:
        activePlayer=frankOne
def endGame():
    playOne=frankOne.getStateSequence()
    playTwo=frankTwo.getStateSequence()
    #combine here, or just have each frank hold onto a complete state sequence, or handle state sequence in this one and record each turn
    #combining here would be the cool thing to do
    #recording here would be the elegant thing to do
    #each frank having a complete state sequence would be the easy thing to do
