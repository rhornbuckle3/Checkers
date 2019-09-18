#Russell Hornbuckle
#2019
#checkers
import pandas as pd
import numpy as np   
import math as mt  
import checkersGame as cg 
import pickle as pk
from checkersFrank import checkersFrank as cf
#sideCOE=1
#import checkersFrank as cF
testState=np.array((1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1))
#testState=testState.reshape((8,4))
#print(cG.defaultState)
#out=pd.DataFrame(cg.defaultState)
#gameRes=pd.Series(np.array([.78345]))
#gameRes=gameRes.append(pd.Series(cg.defaultState),ignore_index=True)
#open("testOut.txt",mode='x')
#open("testOut.txt",mode='r')
#out.to_pickle("testOut")
#frank=cf()
#frank.setSide(0)
#frank.initWeights("/home/russell/Documents/Frank/bio-Two.npz")
#frank.initWeights(experiment here)
#testState=testState[:]*-1
#frank.printMoves(testState)
#frank.stateEvaluatorMaster(testState)
#bio=open("/home/russell/Documents/Frank/bio-Two.npz",mode="xb+")
#wOne=np.array(np.random.standard_normal((32,16)))
#wTwo=np.array(np.random.standard_normal((16,1)))
#game=np.array(np.zeros((1,1)))
#bio=np.load("Frank/bio-One.npz")
#wOne=bio['arr_0']
#wTwo=bio['arr_1']
#game=bio['arr_2'] 
##np.savez("/home/russell/Documents/Frank/bio-Two.npz",wOne,wTwo,game)
#frank.printMoves(testState)
#lory=frank.stateProvider(testState)
#bingo=frank.stateEvaluatorMaster(lory)
#print(bingo)
#printy=frank.wTwo
#print(printy.shape)
cg.initPlayer()
cg.playBall()






