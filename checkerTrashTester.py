#Russell Hornbuckle
#2018-2019
#checkers
#Currently in the process of overhauling and rewriting this project and as a result, it is a bit of a mess.
import pandas as pd
import numpy as np   
import math as mt  
import checkersGame as cg 
import pickle as pk
from checkersFrank import checkersFrank as cf

#remove comment symbols on these to reset weights
wOne=np.array(np.random.standard_normal((32,16)))
wTwo=np.array(np.random.standard_normal((16,1)))
game=np.array(np.zeros((1,1)))
np.savez("./Frank/bio-Two.npz",wOne,wTwo,game)
wOne=np.array(np.random.standard_normal((32,16)))
wTwo=np.array(np.random.standard_normal((16,1)))
game=np.array(np.zeros((1,1)))
np.savez("./Frank/bio-One.npz",wOne,wTwo,game)
cg.initPlayer()
cg.playBall()

'''
game=0
while(True):
    game+=1
    print('Game: '+ str(game))
    cg.initPlayer()
    cg.playBall()
'''