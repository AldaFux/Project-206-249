# coding=UTF8
from copy import deepcopy
import Serial
import time
import vision as vs

SER = Serial.Serial()
time.sleep(2) #wait until the serial connection is open
 
class Board:
 
  def __init__(self,other=None):
    self.player = 'X'
    self.opponent = 'O'
    self.empty = '.'
    self.size = 3
    self.fields = {}
    for y in range(self.size):
      for x in range(self.size):
        self.fields[x,y] = self.empty                               #initializing the field
    # copy constructor
    if other:
      self.__dict__ = deepcopy(other.__dict__)                      #when another board is passed at the instantiation of a board,
                                                                    #the variables will be copied over
 
  def move(self,x,y):
    board = Board(self)                                             #make copy of the board, and since an argument is passed, a deepcopy is made
    board.fields[x,y] = board.player                                #make the players move on the given field
    (board.player,board.opponent) = (board.opponent,board.player)   #switch the active player
    return board                                                    #return the modified board -- means that board itself is never modified but new instances of the board are created
    
 
  def __minimax(self, player):
    if self.won():                                                  #checks whether the opponent has won
      if player:                                                    #case that indeed the opponent has won
        return (-1,None)                    
      else:                                                         #case the player has won
        return (+1,None)
    elif self.tied():
      return (0,None)
    elif player:                                                    #noone has won yet - begin iterating over every empty field and start recursion
      best = (-2,None)
      for x,y in self.fields:
        if self.fields[x,y]==self.empty:
          value = self.move(x,y).__minimax(not player)[0]           #a lot is crunched to this line: First, a move is executed. Doing so, the board is copied in a variable. The copied board will be
                                                                    #have a move placed by the current player. The active roles of the board will be canged and afterwards the board will be returned
                                                                    #From the given board the minimax gunction will be executed
          if value>best[0]:
            best = (value,(x,y))
      return best
    else:
      best = (+2,None)
      for x,y in self.fields:
        if self.fields[x,y]==self.empty:
          value = self.move(x,y).__minimax(not player)[0]
          if value<best[0]:
            best = (value,(x,y))
      return best
      
  def best(self):
    if(len(self.emptyFields()) == 9):           #in case the computer makes the first move: spare some long computation and return an optimal result
      return ((1,1))
    else:
      return self.__minimax(True)[1]            #totaly makes sense: The function is called with player is true. that means, that the algorithm should find the path, with the maximum possible outcome
                                                #during the algorithm the position will be changed. __minimax returns a struct that contains value as the first element and 
                                                #as a second element the tuple (x,y). That means, that best returns the x and y coordinate of the best possible move
  def tied(self):                               #when no field has a value, game over
    for (x,y) in self.fields:                   #only called when won has been called 
      if self.fields[x,y]==self.empty:
        return False
    return True
 
  def won(self):                                #only needs to check whether opponent has won since the player has no chance
    # horizontal
    for y in range(self.size):                  #iterate over each row
      winning = []
      for x in range(self.size):                #iterate over each field in the row
        if self.fields[x,y] == self.opponent:   
          winning.append((x,y))
      if len(winning) == self.size:             #when three times in a row the opponent is present
        return winning                          #he has won the game
    # vertical
    for x in range(self.size):                  #iterate over each column
      winning = []                          
      for y in range(self.size):                #iterate over each field in the colum
        if self.fields[x,y] == self.opponent:   
          winning.append((x,y))
      if len(winning) == self.size:
        return winning                          #won if three times in a row the opponent is present
    # diagonal
    winning = []
    for y in range(self.size):          
      x = y
      if self.fields[x,y] == self.opponent:
        winning.append((x,y))
    if len(winning) == self.size:
      return winning
    # other diagonal
    winning = []
    for y in range(self.size):
      x = self.size-1-y
      if self.fields[x,y] == self.opponent:
        winning.append((x,y))
    if len(winning) == self.size:
      return winning
    # default
    return None
    
  def emptyFields(self):
    listing = []
    for y in range(self.size):
      for x in range(self.size):
        if self.fields[x,y] == self.empty:
          field_number= (3*(2-y) + x + 1)#1+x+3*y
          listing.append(field_number)
    return listing
 
  def __str__(self):  #returns field of 3x3 characters
    string = ''
    for y in range(self.size):
      for x in range(self.size):
        string+=self.fields[x,y]
      string+="\n"
    return string
    
#class Board     
    
    
    
 
class GAME:
 
  def __init__(self):
    self.board = Board()
    #self.update()
 
  def reset(self):
    self.board = Board()
    #self.update()

  def getNumberFromXY(self, x, y):
    return (3*(2-y) + x + 1)
    
  def getXYfromNumber(self, number):
    return ((number-1)%3, 2-(number-1)/3 )
      
  def mainloop(self): #Basically the main of the whole part that runs on the computer
    print("Welcome to the optimal Tic Tac Toe player - you will never be able to win agains me!")
    
    with vs.Image() as img: # necessary in order to initialize the connection to the camera and to end it properly
      
        while True:
          moves = 0
          playerHasBegun = True
          print("Who is supposed to start the game? 1 = You; Anything else = The computer")
          temp = raw_input().upper() ##raw_
          if temp == '1':
            playerHasBegun = True
          else:
            playerHasBegun = False
            
          SER.write('F') # ASCII for F -  necessary for python 3 #send the command to draw the field
        
          while(not SER.ser.inWaiting()):
            print("waiting")
            time.sleep(0.5)
          SER.read()
          
          if (not img.calibrated): #Calibrate in order to know the position of the field
            img.calibrate()        #Automatically sets the calibrated flag of img
            img.show_transform()
            
          while moves < 9:
          
            if( (playerHasBegun and (moves % 2 == 0)) or ((not playerHasBegun) and (moves % 2 == 1))):
              print("make your move on the field!!!");
              number = ''
              found = 0
              sign = ''
              while not (number in self.board.emptyFields()):
                  print('Make your move')
                  (found, sign, number) = img.detect_sign(self.board.emptyFields())
              (x,y) = self.getXYfromNumber(number)
              self.board = self.board.move(x,y)
              
            else:
              move = self.board.best()                                        
              if move:
                self.board = self.board.move(*move)                    # "*" makes that the returned tuple "(x,y)" will be passed as "x,y"
                temp = str(self.getNumberFromXY(*move))
                print(temp)
                SER.write(temp)                                        #send the command to draw the field; make sure it is sent as character
                while(not SER.ser.inWaiting()):
                  print("waiting")
                  time.sleep(0.5)
                SER.read()
                
            print(self.board.__str__())
            
            if(self.board.won() != None):
              print("Such a bad luck! You have lost! A new game will be started!")
              moves = 9
            elif(self.board.tied()):
              print("It's something! At least you havn't lost. A new game will be started!")
            moves = moves + 1
          print("")
          self.reset()
      
#class GAME end     
      
      
 
if __name__ == '__main__':
  GAME().mainloop()
