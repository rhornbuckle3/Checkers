#Russell Hornbuckle
#2019
#checkers

import numpy as np
import math as mt 
import checkers_game as cG
import random
import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense
#checkers-Act
class checkers_agent:
    def __init__(self):
        self.current_state = cG.default_state
        self.wOne = np.array(np.zeros((32,16)))
        self.wTwo = np.array(np.zeros((16,16)))
        self.wThree = np.array(np.zeros((16,1)))
        self.state_sequence = np.copy(cG.default_state)
        self.state_sequence = self.state_sequence.reshape((-1,1))
        self.stateScores = np.array(np.zeros((1,1)))
        self.side = 0
        self.bioFile = None
        #keras network is saved as a json string
        model_file=open('./Agent/Model.txt')
        json_string=model_file.read()
        self.model = model_from_json(json_string)
        self.model.compile(loss='mean_squared_error',optimizer='sgd',metrics=['accuracy'])

    #adds a state to the move sequence for training 
    def add_state(self,newState,newScore):
        newState = np.array(newState)
        newScore = np.array(newScore)
        newScore = newScore.reshape((1,1))
        newState = newState.reshape((-1,1))
        try: self.state_sequence = np.append(self.state_sequence,newState,axis = 1)
        except NameError:
            self.state_sequence = np.copy(cG.default_state)
            self.state_sequence = self.state_sequence.reshape((-1,1))
            self.state_sequence = np.append(self.state_sequence,newState,axis = 1)

        try: self.stateScores = np.append(self.stateScores,newScore,axis = 1)
        except NameError:
            self.stateScores = np.array(np.zeros((1,1)))
            self.stateScores = np.append(self.stateScores,newScore,axis = 1)

    #loads weights into the model, given a viable system path
    def init_weights(self,path):
        self.model.load_weights(path)
    #saves the current weights of the model, given a viable system path
    def save_weights(self,path):
        self.model.save_weights(path)

    #sets the team of the agent
    def set_side(self,side):
        if(side == 0):
            self.side = -1
        if(side == 1):
            self.side = 1

    #returns a list of scores (0.0 to 1.0), given a list of states
    def evaluator_master(self,provided_set):
        state_evaluation = np.array(np.zeros(len(provided_set)))
        for i in range(0,state_evaluation.shape[0]):
            state_evaluation[i] = self.evaluation_network(provided_set[i])
        return state_evaluation

    #feeds a state to the network
    def evaluation_network(self,current_state):
        return self.model.predict(current_state)
    
    #trains the network, given the winner of the game
    def network_train(self,winner):
        if(winner == self.side):
            winner = 1.0
        else:
            winner = 0
        winner_sequence=np.full((1,self.state_sequence.shape[1]),winner)
        training_sequence=self.state_sequence.transpose()
        self.model.fit(training_sequence,winner_sequence[0],verbose=0)

    #determines the next move of the agent, given a state as input
    def state_decider(self,current_state):
        current_state = current_state*self.side
        provided_set = cG.state_farmer(current_state,self.side)
        state_evaluation = self.evaluator_master(provided_set)
        #print(state_evaluation)
        if(str(state_evaluation) == '[]'):
            print(cG.print_state(current_state))
            print('Player '+str(self.side)+' Loses')
            return current_state
        else:            
            if(np.any(state_evaluation)):
                best = np.argmax(state_evaluation)
            else:
                #return random if all states evaluate to 0.
                range_rand = state_evaluation.shape[0]-1
                if(range_rand == 0):
                    best = 0
                else:
                    best = random.randrange(range_rand)
        newState = np.array(provided_set[best])
        self.add_state(newState,state_evaluation[best])
        newState = newState*self.side
        return newState