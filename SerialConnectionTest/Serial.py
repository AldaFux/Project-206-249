import serial
import sys
from collections import deque

# Note that bit masks are string representations of hex values
DATA_RECEIVED = 0xff
BUFFER_SIZE = 60 # to be safe
PORT = "/dev/ttyACM0"
BAUDRATE = 9600


""" 
This class  connects to the Arduino via a serial interface. The simple
functionalities for the user is to simply call the read() and write() methods.
All other details should be abstracted away. 

Usage: ALWAYS perform a read before a write in python (empty reads do not hurt).
       
The Arduino side interprets data in binary, but the python side works with strings.
Please refer to the ascii index to send proper data. 

The write argument expects a string. For example, to  send 0x41, or 65 in decimal, 
we can perform a write of 'a' or chr(65). 

The read method also returns a string. We obtain its hex format in these steps:

str = "ff"
hexform = int(str.encode('hex_codec'), 16) = 255
                    

"""
class Serial:
    
    def __init__(self):
        self.bufferSize = 0 # Keeps track of bytes sent to prevent buffer overflow
        
        self.ser = serial.Serial()
        self.ser.port = PORT
        self.ser.baudrate = BAUDRATE
        self.ser.open()
        self.writeBuffer = deque( [] ) # implementation of a queue

    def write(self, data):
        if( self.ser.isOpen() ):
            self.ser.write( data )
            #self.bufferSize += 1
            
        else: #( !sys.set.isOpen() ):
            print("Attempted write when serial port was not open")
        
    def read(self):
        if( self.ser.isOpen() ):
            if( self.ser.inWaiting() ):
                response = self.ser.read() #reads one byte
                """
                if( int( response.encode('hex_codec'), 16)  == DATA_RECEIVED ):
                    self.bufferSize -= 1
                    print("data received")
                    print(int(response.encode('hex_codec'), 16))
                """
                return response
            else:
                print( "No data to read" )
                return ""
        else:
            print("Attempted read when serial port was not open")
            return ""

    def getBufferSize(self):
        return self.bufferSize

    def getWriteBuffer(self):
        return self.writeBuffer

        
