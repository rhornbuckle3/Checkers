#Russell Hornbuckle
#2019
#checkers

import numpy as np
import math as mt
from checkers_agent import checkers_agent as ca
from checkers_human_client import human_interface as hi

default_state = np.array((1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1))
agent_one = None
agent_two = None
active_player = None

#considering swapping this over to a class to eliminate my dependence on globals here, 
#I imagine that they could get awkward as the project grows.

#initialization
def init_player():
    global active_player
    global agent_one
    global agent_two
    global current_state
    current_state = np.copy(default_state)
    coin_flip = np.random.binomial(1,.5)
    #assign sides via binomial random pulls
    agent_one = ca()
    agent_one.set_side(coin_flip)
    print('agent_one is side: '+str(coin_flip)+'; 0 is black, 1 is white')
    agent_one.init_weights("./Agent/gamma_one.hdf")
    if(coin_flip == 1):
        coin_flip = 0
    else:
        coin_flip = 1
    agent_two = ca()
    agent_two.set_side(coin_flip)
    agent_two.init_weights("./Agent/gamma_two.hdf")
    if(coin_flip == 1):
        active_player = agent_two
    else:
        active_player = agent_one

#Game manager
def play_human():
    global agent_one
    global agent_two
    global active_player
    global current_state

    #the players
    human = hi()
    play_agent
    coin_flip = np.random.binomial(1,.5)
    if(coin_flip == 1):
        play_agent = agent_one
        del agent_two
    else:
        play_agent = agent_two
        del agent_one
    if(active_player == play_agent):
        pass
    else:
        active_player = human
    if(play_agent.side == 0):
        human.set_side(1)
    else:
        human.set_side(0)
    while(True):
        new_state = active_player.state_decider(current_state)
        check = end_game_check(new_state)
        

def play_game():
    global agent_one
    global agent_two
    global active_player
    global current_state
    turn = 1
    while(True):
        check = False
        new_state  =  active_player.state_decider(current_state)
        check  =  end_game_check(new_state)
        if(np.array_equal(current_state,new_state)):
            if(active_player == agent_one):
                active_player = agent_two
            else:
                active_player = agent_one
            count_board(new_state)
            break
        if(check):
            count_board(new_state)
            break
        if(active_player == agent_one):
            active_player = agent_two
        else:
            active_player = agent_one
        if(turn>120):
            count_board(new_state)
            break
        current_state = new_state
        turn = turn+1
#winner values:
#white = 1
#black = 0

#game ending
def count_board(new_state):
    count = np.sum(new_state)
    if(count>0):
        print("White Wins")
    if(count<0):
        print("Black Wins")
    print_state(new_state)
    end_game()

def end_game_check(current_state):
    global agent_one
    global agent_two
    global active_player
    board_sum = 0
    current_state = current_state.reshape((-1,1))
    for i in range(0,32):
        board_sum = board_sum+mt.fabs(current_state[i,0])
    if(mt.fabs(np.sum(current_state)) == board_sum):
        return True
    else:
        return False

def end_game():
    global current_state
    global agent_one
    global agent_two
    global active_player
    if(np.sum(current_state)>0):
        winner = 1.0
    else:
        if(np.sum(current_state)<0):
            winner = -1.0
        else:
            if(active_player.side == 1):
                winner = -1.0
            else:
                winner = 1.0
    #train here
    agent_one.network_train(winner)
    agent_two.network_train(winner)
    agent_one.save_weights('./Agent/gamma_one.hdf')
    agent_two.save_weights('./Agent/gamma_two.hdf')
    del agent_one
    del agent_two
    del active_player

#prints the board in 8x8 format
def print_state(board_state):
    if(max(board_state.shape)<64):
        board_state = board_expand(board_state)
    board_state = board_state.reshape((8,8))
    print(board_state)

#returns the board as a 8x8
def board_expand(board_state):
    expanded_state = np.zeros((8,8))
    board_iterator = 0
    if(type(board_state[0]) == np.ndarray):
        board_state = board_state[0]
    for i in range(0,8):
        j = 0
        while(j<8):
            if((i+1)%2 == 1):
                expanded_state[i,j] = 0
                expanded_state[i,j+1] = board_state[board_iterator]
                board_iterator = board_iterator+1
                j = j+2
            if((i+1)%2 == 0):
                expanded_state[i,j] = board_state[board_iterator]
                expanded_state[i,j+1] = 0
                board_iterator = board_iterator+1
                j = j+2
    return expanded_state

#returns the board as a 8x4
def board_contract(board_state):
    contracted_state = np.zeros((1,32))
    board_iterator = 0
    for i in range(0,8):
        j = 0
        while(j<8):
            if((i+1)%2 == 1):
                contracted_state[0,board_iterator] = board_state[i,j+1]
                board_iterator = board_iterator+1
                j = j+2
            if((i+1)%2 == 0):
                contracted_state[0,board_iterator] = board_state[i,j]
                board_iterator = board_iterator+1
                j = j+2
    return contracted_state 

#returns the list of legal moves (in the form of board states) given an input board state
#board should be multiplied by color coeff before being shipped here. 
#The color coeff is passed for determining if a piece should be king'ed and movement directtion for 1's
#(potential place for improvement) - maintaining a list of where each peice is for each side will reduce the number of move checks 
#from a(64) to a(12-b), where 'a' is the average legal moves and 'b' is the number of removed pieces
def state_farmer(board_state, color_coeff):
    move_set = []
    jump_set = []
    board_state = board_expand(board_state)
    for i in range(0,8):
        for j in range(0,8):
            if(board_state[i,j] == 1 or board_state[i,j] == 2):
                #checking for valid forward moves here
                prospect_set = check_moves(board_state,i,j,color_coeff)
                for board in prospect_set:
                    if(not(board == [])):
                        move_set.append(board)
                #print("Move set here: ")
                #print(move_set)
                #jump_set.append(check_jumps(board_state,i,j,color_coeff,jump_set))
                prospect_set = check_jumps(board_state,i,j,color_coeff,[])
                for board in prospect_set:
                    if(not(board == [])):
                        jump_set.append(board)
            if(board_state[i,j] == 2):
                #king moves
                prospect_set = check_moves(board_state,i,j,color_coeff*-1)
                for board in prospect_set:
                    if(not(board == [])):
                        move_set.append(board) 
                prospect_set = check_jumps(board_state,i,j,color_coeff*-1,[])
                for board in prospect_set:
                    if(not(board == [])):

                        jump_set.append(board)
    if(len(jump_set)>0):
        return jump_set
    else:  
        return move_set

#returns legal moves given a board state, position, and team (as a 1.0 or -1.0)
def check_moves(board_state,x,y,color_coeff):
    new_set = []
    prospect = None
    try: prospect = board_state[x+1*color_coeff,y+1]
    except IndexError:
        pass
    if(prospect == 0):
        prospect_board = np.copy(board_state)
        prospect_board[x,y] = 0
        if(board_state[x,y] == 1):
            if(((x+1*color_coeff) == 0 and color_coeff == -1)or((x+1*color_coeff) == 7 and color_coeff == 1)):
                prospect_board[x+1*color_coeff,y+1] = 2
            else: 
                prospect_board[x+1*color_coeff,y+1] = 1
        else:
            prospect_board[x+1*color_coeff,y+1] = 2
        new_set.append(board_contract(prospect_board))
    prospect = None
    try: prospect = board_state[x+1*color_coeff,y-1]
    except IndexError:
        pass
    if(prospect == 0):
        prospect_board = np.copy(board_state)
        prospect_board[x,y] = 0
        if(board_state[x,y] == 1):
            if(((x+1*color_coeff) == 0 and color_coeff == -1)or((x+1*color_coeff) == 7 and color_coeff == 1)):
                prospect_board[x+1*color_coeff,y-1] = 2
            else: 
                prospect_board[x+1*color_coeff,y-1] = 1
        else:
            prospect_board[x+1*color_coeff,y-1] = 2
        new_set.append(board_contract(prospect_board))
    return new_set

#returns legal jumps given a board state, position, and team (as a 1.0 or -1.0), and the current move set (for recursive jumps)
def check_jumps(board_state,x,y,color_coeff,move_set):
    new_set = []
    prospect = None
    prospect_board = np.copy(board_state)
    #y is +
    try: prospect = board_state[x+2*color_coeff,y+2]
    except IndexError:
        pass
    if(prospect == 0):
        if(board_state[x+1*color_coeff,y+1] == -1 or board_state[x+1*color_coeff,y+1] == -2):
            if(board_state[x+2*color_coeff,y+2] == 0):
                prospect_board[x,y] = 0
                prospect_board[x+1*color_coeff,y+1] = 0
                prospect_board[x+2*color_coeff,y+2] = board_state[x,y]            
                if(board_state[x,y] == 1):
                    if(((x+2*color_coeff) == 0 and color_coeff == -1)or((x+2*color_coeff) == 7 and color_coeff == 1)):
                        prospect_board[x+2*color_coeff,y+2] = 2
                    else: 
                        prospect_board[x+2*color_coeff,y+2] = 1
                else:
                    prospect_board[x+2*color_coeff,y+2] = 2
                new_set.append(board_contract(prospect_board))
                #recursion for multi jumps
                prospect_recursive = check_jumps(prospect_board,x+2*color_coeff,y+2,color_coeff,[])
                for prospect_state in prospect_recursive:
                    if(not(prospect_state == [])):
                        if(type(prospect_state) == np.ndarray):
                            new_set.append(prospect_state)
                        else:
                            new_set.append(prospect_recursive)
    prospect = None
    #y is -
    try: prospect = board_state[x+2*color_coeff,y-2]
    except IndexError:
        pass
    if(prospect == 0):
        if(board_state[x+1*color_coeff,y-1] == -1 or board_state[x+1*color_coeff,y-1] == -2):
            if(board_state[x+2*color_coeff,y-2] == 0):
                prospect_board[x,y] = 0
                prospect_board[x+1*color_coeff,y-1] = 0
                prospect_board[x+2*color_coeff,y-2] = board_state[x,y]            
                if(board_state[x,y] == 1):
                    if(((x+2*color_coeff) == 0 and color_coeff == -1)or((x+2*color_coeff) == 7 and color_coeff == 1)):
                        prospect_board[x+2*color_coeff,y-2] = 2
                    else: 
                        prospect_board[x+2*color_coeff,y-2] = 1
                else:
                    prospect_board[x+2*color_coeff,y-2] = 2
                
                new_set.append(board_contract(prospect_board))
                #recursion for multi jumps
                prospect_recursive = check_jumps(prospect_board,x+2*color_coeff,y-2,color_coeff,[])
                for prospect_state in prospect_recursive:
                    if(not(prospect_state == [])):
                        if(type(prospect_state) == np.ndarray):
                            new_set.append(prospect_state)
                        else:
                            new_set.append(prospect_recursive)
    return new_set