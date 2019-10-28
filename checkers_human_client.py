#Russell Hornbuckle
#2019
#checkers

import numpy as np
import math as mt 
import checkers_game as cg
import random
import socket
import Tkinter

class human_interface(checkers_agent):

    def __init__(self):
        self.side = 0
        self.socket = socket.socket()
    
    def state_decider(self,current_state):
        provided_set = cg.state_farmer(current_state,self.side)
        #send current state through socket to golang Gui and check recieved states against the provided set
        
    def s_and_r(self,provided_set):
        pass#by
    #def play_game(self, side_coe):
    #client = Tkinter.tk()
    #client.mainloop()
    