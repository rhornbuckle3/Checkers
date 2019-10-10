#Russell Hornbuckle
#2018-2019
#checkers
#In the process of moving and rewriting the state farmer over here from checkersFrank
import numpy as np
import pandas as pd 
import math as mt
from checkersFrank import checkersFrank as cf
defaultState=np.array((1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1))
#assign sides via binomial random pulls
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
    if(coinFlip==1):
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
            print_state(newState)
            endGame()
            break
        if(check):
            count=np.sum(newState)
            if(count>0):
                print("White Wins")
            if(count<0):
                print("Black Wins")
            print_state(newState)
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
            print_state(newState)
            endGame()
            break
        print_state(newState)
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


def print_state(board_state):
    if(max(board_state.shape)<64):
        board_state=board_expand(board_state,False)
    board_state=board_state.reshape((8,8))
    print(board_state)

def board_expand(board_state,flatten):
    expanded_state=np.zeros((8,8))
    board_iterator=0
    if(type(board_state[0])==np.ndarray):
        board_state=board_state[0]
    for i in range(0,8):
        j=0
        while(j<8):
            if((i+1)%2==1):
                expanded_state[i,j]=0
                expanded_state[i,j+1]=board_state[board_iterator]
                board_iterator=board_iterator+1
                j=j+2
            if((i+1)%2==0):
                expanded_state[i,j]=board_state[board_iterator]
                expanded_state[i,j+1]=0
                board_iterator=board_iterator+1
                j=j+2
    #if flatten is false, returns the 8 by 8 board. Otherwise returns 1,64 board
    if(flatten):
        expanded_state=expanded_state.reshape((1,-1))
        return expanded_state
    return expanded_state
def board_contract(board_state):
    contracted_state=np.zeros((1,32))
    board_iterator=0
    for i in range(0,8):
        j=0
        while(j<8):
            if((i+1)%2==1):
                contracted_state[0,board_iterator]=board_state[i,j+1]
                board_iterator=board_iterator+1
                j=j+2
            if((i+1)%2==0):
                contracted_state[0,board_iterator]=board_state[i,j]
                board_iterator=board_iterator+1
                j=j+2
    return contracted_state 

#board should be multiplied by color coeff before being shipped here. 
#The color coeff is passed for determining if a piece should be king'ed and movement directtion for 1's
def state_farmer(board_state, color_coeff):
    move_set=[]
    jump_set=[]
    board_state=board_expand(board_state,False)
    for i in range(0,8):
        for j in range(0,8):
            if(board_state[i,j]==1 or board_state[i,j]==2):
                #checking for valid forward moves here
                prospect_set=check_moves(board_state,i,j,color_coeff)
                for board in prospect_set:
                    if(not(board==[])):
                        move_set.append(board)
                #print("Move set here: ")
                #print(move_set)
                #jump_set.append(check_jumps(board_state,i,j,color_coeff,jump_set))
                prospect_set=check_jumps(board_state,i,j,color_coeff,[])
                for board in prospect_set:
                    if(not(board==[])):
                        jump_set.append(board)
            if(board_state[i,j]==2):
                #king moves
                prospect_set=check_moves(board_state,i,j,color_coeff*-1)
                for board in prospect_set:
                    if(not(board==[])):
                        move_set.append(board)
                #print("Move set here: ")
                #print(move_set)
                #jump_set.append(check_jumps(board_state,i,j,color_coeff*-1,jump_set))  
                prospect_set=check_jumps(board_state,i,j,color_coeff*-1,[])
                for board in prospect_set:
                    if(not(board==[])):

                        jump_set.append(board)
    if(len(jump_set)>0):
        return jump_set
    else:  
        return move_set

def check_moves(board_state,x,y,color_coeff):
    new_set=[]
    prospect=None
    try: prospect=board_state[x+1*color_coeff,y+1]
    except IndexError:
        pass
    if(prospect==0):
        prospect_board=np.copy(board_state)
        prospect_board[x,y]=0
        if(board_state[x,y]==1):
            if(((x+1*color_coeff)==0 and color_coeff==-1)or((x+1*color_coeff)==7 and color_coeff==1)):
                prospect_board[x+1*color_coeff,y+1]=2
            else: 
                prospect_board[x+1*color_coeff,y+1]=1
        else:
            prospect_board[x+1*color_coeff,y+1]=2
        new_set.append(board_contract(prospect_board))
    prospect=None
    try: prospect=board_state[x+1*color_coeff,y-1]
    except IndexError:
        pass
    if(prospect==0):
        prospect_board=np.copy(board_state)
        prospect_board[x,y]=0
        if(board_state[x,y]==1):
            if(((x+1*color_coeff)==0 and color_coeff==-1)or((x+1*color_coeff)==7 and color_coeff==1)):
                prospect_board[x+1*color_coeff,y-1]=2
            else: 
                prospect_board[x+1*color_coeff,y-1]=1
        else:
            prospect_board[x+1*color_coeff,y-1]=2
        new_set.append(board_contract(prospect_board))
    return new_set

def check_jumps(board_state,x,y,color_coeff,move_set):
    new_set=[]
    prospect=None
    prospect_board=np.copy(board_state)
    #y is +
    try: prospect=board_state[x+2*color_coeff,y+2]
    except IndexError:
        pass
    if(prospect==0):
        if(board_state[x+1*color_coeff,y+1]==-1 or board_state[x+1*color_coeff,y+1]==-2):
            if(board_state[x+2*color_coeff,y+2]==0):
                prospect_board[x,y]=0
                prospect_board[x+1*color_coeff,y+1]=0
                prospect_board[x+2*color_coeff,y+2]=board_state[x,y]            
                if(board_state[x,y]==1):
                    if(((x+2*color_coeff)==0 and color_coeff==-1)or((x+2*color_coeff)==7 and color_coeff==1)):
                        prospect_board[x+2*color_coeff,y+2]=2
                    else: 
                        prospect_board[x+2*color_coeff,y+2]=1
                else:
                    prospect_board[x+2*color_coeff,y+2]=2
                new_set.append(board_contract(prospect_board))
                #recursion for multi jumps
                prospect_recursive=check_jumps(prospect_board,x+2*color_coeff,y+2,color_coeff,[])
                for prospect_state in prospect_recursive:
                    if(not(prospect_state==[])):
                        if(type(prospect_state)==np.ndarray):
                            new_set.append(prospect_state)
                        else:
                            new_set.append(prospect_recursive)
    prospect=None
    #y is -
    try: prospect=board_state[x+2*color_coeff,y-2]
    except IndexError:
        pass
    if(prospect==0):
        if(board_state[x+1*color_coeff,y-1]==-1 or board_state[x+1*color_coeff,y-1]==-2):
            if(board_state[x+2*color_coeff,y-2]==0):
                prospect_board[x,y]=0
                prospect_board[x+1*color_coeff,y-1]=0
                prospect_board[x+2*color_coeff,y-2]=board_state[x,y]            
                if(board_state[x,y]==1):
                    if(((x+2*color_coeff)==0 and color_coeff==-1)or((x+2*color_coeff)==7 and color_coeff==1)):
                        prospect_board[x+2*color_coeff,y-2]=2
                    else: 
                        prospect_board[x+2*color_coeff,y-2]=1
                else:
                    prospect_board[x+2*color_coeff,y-2]=2
                
                new_set.append(board_contract(prospect_board))
                #recursion for multi jumps
                prospect_recursive=check_jumps(prospect_board,x+2*color_coeff,y-2,color_coeff,[])
                for prospect_state in prospect_recursive:
                    if(not(prospect_state==[])):
                        if(type(prospect_state)==np.ndarray):
                            new_set.append(prospect_state)
                        else:
                            new_set.append(prospect_recursive)
    return new_set
#need to write the rules somewhere
#some of the prospect boards are being added wrapped in an array -- needs to be fixed.