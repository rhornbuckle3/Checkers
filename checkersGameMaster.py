#Russell Hornbuckle
#2018-2019
#checkers
#Currently in the process of overhauling and rewriting this project - as a result, it is a bit of a mess.

#RULES
#Original (one piece takes a single other piece) jumps are compulsory, secondary (takes two pieces) and more jumps are not.
#Black goes first
#Game will end in the favor of whoever has a higher score at 121 turns
#Score is determined by the pieces remaining for each player: kings are 2, regular pieces are 1


import numpy as np   
import checkersGame as cg 
from checkersFrank import checkersFrank as cf


#remove triple quotes below to reset weights (or add them to not reset weights)
'''
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


#remove triple quotes to play a series of games -- there appears to be a garbage collection issue here
'''
game=0
while(True):
    game+=1
    print('Game: '+ str(game))
    cg.initPlayer()
    cg.playBall()
'''

cg.initPlayer()
cg.playBall()






#cg.print_state(cg.defaultState)
#cg.board_contract(cg.board_expand(cg.defaultState,False))
#move_set=cg.state_farmer(cg.defaultState,1)
#for i in range(0,len(move_set)):
#    print(move_set[i])
#    type_board=move_set[i]
#    print(type(cg.board_expand(type_board[0],False)))

#14.5
#9.7
#21.8
