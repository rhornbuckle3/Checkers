#Russell Hornbuckle
#2018-2019
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
        self.stateSequence = np.copy(cG.default_state)
        self.stateSequence = self.stateSequence.reshape((-1,1))
        self.stateScores = np.array(np.zeros((1,1)))
        self.side = 0
        self.bioFile = None

        #keras network
        model_file=open('./Agent/Model.txt')
        json_string=model_file.read()
        self.model = model_from_json(json_string)
        self.model.compile(loss='mean_squared_error',optimizer='sgd',metrics=['accuracy'])

    def addToSeq(self,newState,newScore):
        newState = np.array(newState)
        newScore = np.array(newScore)
        newScore = newScore.reshape((1,1))
        newState = newState.reshape((-1,1))

        try: self.stateSequence = np.append(self.stateSequence,newState,axis = 1)
        except NameError:
            self.stateSequence = np.copy(cG.default_state)
            self.stateSequence = self.stateSequence.reshape((-1,1))
            self.stateSequence = np.append(self.stateSequence,newState,axis = 1)

        try: self.stateScores = np.append(self.stateScores,newScore,axis = 1)
        except NameError:
            self.stateScores = np.array(np.zeros((1,1)))
            self.stateScores = np.append(self.stateScores,newScore,axis = 1)

    #def init_weights(self,bioPath):
    #    bio = np.load(bioPath)
    #    self.wOne = bio['arr_0']
    #    self.wTwo = bio['arr_1']
    #    self.wThree= bio['arr_2']
    #    self.bioFile = bioPath

    def init_weights(self,path):
        self.model.load_weights(path)
    def save_weights(self,path):
        self.model.save_weights(path)

    #def save_weights(self,sOne,sTwo,sThree):
    #    np.savez(self.bioFile,sOne,sTwo,sThree)

    def set_side(self,side):
        if(side == 0):
            self.side = -1
        if(side == 1):
            self.side = 1

    def evaluator_master(self,provided_set):
        #calls state_evaluator
        state_evaluation = np.array(np.zeros(len(provided_set)))
        for i in range(0,state_evaluation.shape[0]):
            state_evaluation[i] = self.evaluation_network(provided_set[i])
            #state_evaluation[i] = self.state_evaluator(provided_set[i],self.wOne,self.wTwo,self.wThree)
        return state_evaluation

    #goin keras
    def evaluation_network(self,current_state):
        return self.model.predict(current_state)
    
    def network_train(self,winner):
        if(winner == self.side):
            winner = 1.0
        else:
            winner = 0
        #winner_sequence=np.full((0,self.stateSequence.shape[1]),winner)
        #training_sequence=np.array(self.stateSequence)
        #self.model.fit(training_sequence.transpose,winner_sequence)
        for i in range(0,self.stateSequence.shape[1]):
            training_state=self.stateSequence[:,i]
            print(training_state)
            print(np.array(winner))
            self.model.fit(training_state.transpose,np.array(winner))

    def state_evaluator(self,potState,sOne,sTwo,sThree):
        potState = np.array(potState)
        activation_set = np.array(np.zeros((1,16)))
        activation_set_two = np.array(np.zeros((1,16)))
        for i in range(0,activation_set.shape[1]):
            activation_set[0,i]  =  self.activation_one(i,potState,sOne)
        #new stuff here
        for i in range(0,activation_set.shape[1]):
            activation_set_two[0,i]  =  self.activation_one(i,activation_set,sOne)
        result  =  self.activation_two(activation_set,sThree)
        return result

    #re-using this for the first two weighted layers
    def activation_one(self,neuron_id,learning_state,sOne):
        learning_state = learning_state.reshape((-1,1))
        return 1/(1+mt.e**(np.matmul(sOne[:,neuron_id],learning_state)*-1))

    def activation_two(self,inputOne,sTwo):
        inputOne = inputOne.reshape((-1,1)) 
        return 1/(1+mt.e**(np.matmul(sTwo[:,0],inputOne)*-1))

    def state_decider(self,current_state):
        current_state = current_state*self.side
        provided_set = cG.state_farmer(current_state,self.side)
        state_evaluation = self.evaluator_master(provided_set)
        print(state_evaluation)
        if(str(state_evaluation) == '[]'):
            print(cG.print_state(current_state))
            print('Player '+str(self.side)+' Loses')
            current_state
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
        self.addToSeq(newState,state_evaluation[best])
        newState = newState*self.side
        return newState
    
    




    def grad_desc(self,winner):
        bOne = np.copy(self.wOne)
        bTwo = np.copy(self.wTwo)
        print('Training Player: '+str(self.side))
        print('State shape: '+str(self.stateSequence.shape[1]))
        if(winner == self.side):
            winner = 1.0
        else:
            winner = 0
        print(winner)
        for i in range(1,self.stateSequence.shape[1]):
            predict = self.stateScores[0,i]
            learning_state = np.copy(self.stateSequence[:,i])
            learning_state = learning_state*self.side
            activation_set = np.array(np.zeros((1,16)))
            for j in range(0,activation_set.shape[1]):
                activation_set[0,j] = self.activation_one(j,learning_state,bOne)
            learning_step = 1e-2
            iterator = 0
            while(True):
                iterator += 1
                #second layer
                gradient_upper = self.upper_gradient(learning_state,predict,winner,activation_set,bOne,bTwo)
                #gradient_upper - 16x1
                
                #gradient_middle - 16x16
                #gradient_middle = self.middle_gradient(learning_state,predict,winner,activation_set,bOne,bTwo)

                gradient_upper = gradient_upper*np.transpose(activation_set)
                #first layer
                gradient_lower = self.lower_gradient(learning_state,predict,winner,activation_set,bOne,bTwo)
                #gradient_lower - 32x16
                bOne = np.subtract(bOne,learning_step*gradient_lower)
                bTwo = np.subtract(bTwo,learning_step*gradient_upper)
                if(iterator > 100):
                    break
                if((winner - self.state_evaluator(learning_state,bOne,bTwo))**2 == 0):
                    print("0 error happened")
                    break
            iterator = 0
        return bOne, bTwo
    def activation_oneD(self, neuron_id,learning_state,sOne):
        learning_state = learning_state.reshape((-1,1))
        sigmoid = self.activation_one(neuron_id,learning_state,sOne)
        return sigmoid*(1-sigmoid)

    def lower_gradient(self,current_state,score,winner,activation_set,sOne,sTwo,sThree):
        #WILL NEED TO MULTIPLY THIS BY GRADIENT MIDDLE
        result = np.full((1,16),np.transpose(sTwo)*(np.matmul(self.upper_gradient(current_state,score,winner,activation_set,sOne,sTwo,sThree),self.middle_gradient(current_state,activation_set,score,winner,sOne,sTwo,sThree))))
        activation_setD = np.array(np.zeros((1,16)))
        for i in range(0,activation_setD.shape[1]):
            activation_setD[0,i] = self.activation_oneD(i,current_state,sOne)
        result = np.multiply(result,activation_setD)
        current_state = current_state.reshape((-1,1))
        result = np.dot(current_state,result)
        return result

    def middle_gradient(self,current_state,lower_output,score,winner,sOne,sTwo,sThree):
        result = np.full((1,16),np.transpose(sTwo)*self.upper_gradient(current_state,score,winner,lower_output,sOne,sTwo))
        activation_setD = np.array(np.zeros((1,16)))
        for i in range(0,activation_setD.shape[1]):
            activation_setD[0,i] = self.activation_oneD(i,current_state,sTwo)
        result = np.multiply(result,activation_setD)
        lower_output = lower_output.reshape((-1,1))
        result = np.dot(lower_output,result)
        return result

    def upper_gradient(self,current_state,score,winner,activation_set,sOne,sTwo,sThree):
        current_evaluation = self.state_evaluator(current_state,sOne,sTwo,sThree)
        cost  =  winner-current_evaluation
        e_to_x  =  mt.e**current_evaluation
        first  =  cost*(e_to_x/((1+e_to_x)**2))
        second  =  (1/(1+mt.e**(current_evaluation*-1)))
        result  =  np.multiply(first,second)
        return result