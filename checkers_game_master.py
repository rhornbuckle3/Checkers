#Russell Hornbuckle
#2019
#checkers


#RULES
#Original (one piece takes a single other piece) jumps are compulsory, secondary (takes two pieces) and more jumps are not.
#Black goes first
#Game will end when one side has no available moves or in the favor of whoever has a higher score at 121 turns
#Score is determined by the pieces remaining for each player: kings are 2, regular pieces are 1

#BUGLIST
#
import numpy as np   
import checkers_game as cg 
import sys
import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense

#Accepted Arguments: 'reset' to reset weights, 'infinity' to play until keyboard interrupt, or a number to play that number of games. If no valid argument is given, one game will be played.

if(len(sys.argv)==2):
    if(str(sys.argv[1])=='reset'):
        print("Resetting Weights")
        for i in range(0,2):
            print(str(i+1))
            model = Sequential()
            model.add(Dense(16,input_dim=32,activation='relu'))
            model.add(Dense(16,activation='relu'))
            model.add(Dense(1,activation='relu'))
            model.compile(loss='mean_squared_error',optimizer='sgd',metrics=['accuracy'])
            if(i == 0):
                model.save_weights('./Agent/gamma_one.hdf')
            else:
                model.save_weights('./Agent/gamma_one.hdf')
            del model
    elif(str(sys.argv[1])=='infinity'):
        game = 0
        while(True):
            game+=1
            print('Game: '+ str(game))
            cg.init_player()
            cg.play_game()
    elif(str(sys.argv[1]).isnumeric()):
        game_total = int(str(sys.argv[1]))
        game = 0
        for i in range(0,game_total):
            game+=1
            print('Game: '+ str(game))
            cg.init_player()
            cg.play_game()
    elif(str(sys.argv[1])=='play'):
        #plays a random agent
        cg.init_player()
        cg.play_human()
    else:
        cg.init_player()
        cg.play_game()
