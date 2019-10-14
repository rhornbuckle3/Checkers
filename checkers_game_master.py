#Russell Hornbuckle
#2018-2019
#checkers
#Currently in the process of overhauling and rewriting this project - as a result, it has a couple naming schemas and varying quality throughout.

#RULES
#Original (one piece takes a single other piece) jumps are compulsory, secondary (takes two pieces) and more jumps are not.
#Black goes first
#Game will end in the favor of whoever has a higher score at 121 turns
#Score is determined by the pieces remaining for each player: kings are 2, regular pieces are 1

#BUGLIST
#
import numpy as np   
import checkers_game as cg 
from checkers_agent import checkers_agent as ca


#remove triple quotes below to reset weights (or add them to not reset weights)
'''
wOne = np.array(np.random.standard_normal((32,16)))
wTwo = np.array(np.random.standard_normal((16,1)))
game = np.array(np.zeros((1,1)))
np.savez("./Agent/alpha-Two.npz",wOne,wTwo,game)
wOne = np.array(np.random.standard_normal((32,16)))
wTwo = np.array(np.random.standard_normal((16,1)))
game = np.array(np.zeros((1,1)))
np.savez("./Agent/alpha-One.npz",wOne,wTwo,game)
'''

'''
wOne = np.array(np.random.standard_normal((32,16)))
wTwo = np.array(np.random.standard_normal((16,16)))
wThree = wTwo = np.array(np.random.standard_normal((16,1)))
np.savez("./Agent/beta-Two.npz",wOne,wTwo,wThree)
wOne = np.array(np.random.standard_normal((32,16)))
wTwo = np.array(np.random.standard_normal((16,16)))
wThree = np.array(np.random.standard_normal((16,1)))
np.savez("./Agent/beta-One.npz",wOne,wTwo,wThree)
'''
'''

game = 0
while(True):
    game+=1
    print('Game: '+ str(game))
    cg.initPlayer()
    cg.play_game()
'''

cg.initPlayer()
cg.play_game()

'''
import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense
model = Sequential()
model.add(Dense(16,input_dim=32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='relu'))
model.compile(loss='mean_squared_error',optimizer='sgd',metrics=['accuracy'])
model.save_weights('./Agent/gamma_two.hdf')
'''
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
