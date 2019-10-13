#Russell Hornbuckle
#2018-2019
#checkers
#currently in the process of moving and rewriting the state farming methods into checkersGame
import numpy as np
import math as mt 
import checkers_game as cG
import random
#checkers-Act
class checkers_agent:
    def __init__(self):
        self.current_state = cG.default_state
        self.wOne = np.array(np.zeros((32,16)))
        self.wTwo = np.array(np.zeros((16,1)))
        self.kingDex = np.array(np.zeros((12)))
        self.stateSequence = np.copy(cG.default_state)
        self.stateSequence = self.stateSequence.reshape((-1,1))
        self.stateScores = np.array(np.zeros((1,1)))
        self.side = None
        self.sideCOE = 0
        self.gameNum = None
        self.bioFile = None

    def addToSeq(self,newState,newScore):
        newState = np.array(newState)
        newScore = np.array(newScore)
        newScore = newScore.reshape((1,1))
        newState = newState.reshape((-1,1))
        try: self.stateSequence = np.append(self.stateSequence,newState,axis = 1)
        except NameError:
            self.stateSequence = np.copy(default_state)
            self.stateSequence = self.stateSequence.reshape((-1,1))
            self.stateSequence = np.append(self.stateSequence,newState,axis = 1)
        try: self.stateScores = np.append(self.stateScores,newScore,axis = 1)
        except NameError:
            self.stateScores = np.array(np.zeros((1,1)))
            self.stateScores = np.append(self.stateScores,newScore,axis = 1)

    def initWeights(self,bioPath):
        bio = np.load(bioPath)
        self.wOne = bio['arr_0']
        self.wTwo = bio['arr_1']
        self.gameNum = bio['arr_2']
        self.bioFile = bioPath

    def saveWeights(self,sOne,sTwo):
        np.savez(self.bioFile,sOne,sTwo,self.gameNum)

    def setSide(self,side):
        if(side == 0):
            self.sideCOE = -1
        if(side == 1):
            self.sideCOE = 1

    def evaluator_master(self,provided_set):
        #calls state_evaluator
        state_evaluation = np.array(np.zeros(len(provided_set)))
        for i in range(0,state_evaluation.shape[0]):
            state_evaluation[i] = self.state_evaluator(provided_set[i],self.wOne,self.wTwo)
        return state_evaluation

    def state_evaluator(self,potState,sOne,sTwo):
        potState = np.array(potState)
        activation_set = np.array(np.zeros((1,16)))
        for i in range(0,activation_set.shape[1]):
            activation_set[0,i]  =  self.activation_one(i,potState,sOne)
        result  =  self.activation_two(activation_set,sTwo)
        if(result<0):
            return 0
        else:
            return result
    
    def activation_one(self,neuron_id,learning_state,sOne):
        learning_state = learning_state.reshape((-1,1))
        return 1/(1+mt.e**(np.matmul(sOne[:,neuron_id],learning_state)*-1))

    def activation_two(self,inputOne,sTwo):
        inputOne = inputOne.reshape((-1,1)) 
        return 1/(1+mt.e**(np.matmul(sTwo[:,0],inputOne)*-1))


    def state_decider(self,current_state):
        current_state = current_state*self.sideCOE
        provided_set = cG.state_farmer(current_state,self.sideCOE)
        state_evaluation = self.evaluator_master(provided_set)
        #print(state_evaluation)
        if(str(state_evaluation) == '[]'):
            print(cG.print_state(current_state))
            print('Player '+str(self.sideCOE)+' Loses')
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
        newState = newState*self.sideCOE
        return newState

    def grad_desc(self,winner):
        bOne = np.copy(self.wOne)
        bTwo = np.copy(self.wTwo)
        print('Training Player: '+str(self.sideCOE))
        print('State shape: '+str(self.stateSequence.shape[1]))
        if(winner == self.sideCOE):
            winner = 1.0
        else:
            winner = -1.0
        print(winner)
        for i in range(1,self.stateSequence.shape[1]):
            predict = self.stateScores[0,i]
            learning_state = np.copy(self.stateSequence[:,i])
            learning_state = learning_state*self.sideCOE
            activation_set = np.array(np.zeros((1,16)))
            for j in range(0,activation_set.shape[1]):
                activation_set[0,j] = self.activation_one(j,learning_state,bOne)
            learning_step = 1e-2
            iterator = 0
            while(True):
                iterator += 1
                #bOnePrior = np.copy(bOne)
                #bTwoPrior = np.copy(bTwo)
                #second layer
                gradient_upper = self.upper_gradient(learning_state,predict,winner,activation_set,bOne,bTwo)
                #gradient_upper - 16x1
                gradient_upper = gradient_upper*np.transpose(activation_set)
                checkOne = np.copy(bOne)
                #first layer
                gradient_lower = self.lower_gradient(learning_state,predict,winner,activation_set,bOne,bTwo)
                #gradient_lower - 32x16
                bOne = np.subtract(bOne,learning_step*gradient_lower)
                bTwo = np.subtract(bTwo,learning_step*gradient_upper)
                if(iterator>100):
                    break
            iterator = 0
        return bOne, bTwo
    def activation_oneD(self, neuron_id,learning_state,sOne,sTwo):
        learning_state = learning_state.reshape((-1,1))
        sigmoid = self.activation_one(neuron_id,learning_state,sOne)
        return sigmoid*(1-sigmoid)

    def lower_gradient(self,current_state,score,winner,activation_set,sOne,sTwo):
        result = np.full((1,16),np.transpose(sTwo)*self.upper_gradient(current_state,score,winner,activation_set,sOne,sTwo))
        activation_setD = np.array(np.zeros((1,16)))
        for i in range(0,activation_setD.shape[1]):
            activation_setD[0,i] = self.activation_oneD(i,current_state,sOne,sTwo)
        result = np.multiply(result,activation_setD)
        current_state = current_state.reshape((-1,1))
        result = np.dot(current_state,result)
        return result

    def upper_gradient(self,current_state,score,winner,activation_set,sOne,sTwo):
        cost  =  (winner-self.state_evaluator(current_state,sOne,sTwo))**2
        e_to_x  =  mt.e**self.state_evaluator(current_state,sOne,sTwo)
        first_half  =  cost*(e_to_x/((1+e_to_x)**2))
        second_half  =  (1/(1+mt.e**(self.state_evaluator(current_state,sOne,sTwo)*-1)))
        result  =  np.multiply(first_half,second_half)
        return result