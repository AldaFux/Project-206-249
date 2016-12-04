#!/usr/bin/env python

#import numpy as np
#import scipy as sp
#from scipy import linalg
import math

if __name__ == "__main__":
    while (1):
    
        l3 = float(4.5)
        l1 = float(9)
        l2 = float(9)
        
        x = float (raw_input("please enter the x coordinate: "))
        y = float (raw_input("please enter the y coordinate: "))
        
        alpha = math.atan2(y,x)
        beta  = math.pi - alpha
        
        print("The angle alpha is %f pi" % (alpha/math.pi))
        print("The angle beta  is %f pi" % (beta/math.pi))
        
        c = float(math.sqrt(y*y + x*x))
        
        print("c  is %f" % c)
        
        a = float(math.sqrt( c*c + l3*l3*0.25 - float(c*l3*math.cos(beta))))
        b = float(math.sqrt( c*c + l3*l3*0.25 - float(c*l3*math.cos(alpha))))
        
        print("a  is %f" % a)
        print("b  is %f" % b)
        if(alpha < math.pi/2):
            theta2 = float(math.asin( math.sin(beta) * c/a))
            theta3 = math.pi - float(math.asin( math.sin(alpha)* c/b))
        else:
            theta2 = math.pi -float(math.asin( math.sin(beta) * c/a))
            theta3 = float(math.asin( math.sin(alpha)* c/b))
        
        print("The angle theta2 is %f pi" % (theta2/math.pi))
        print("The angle theta3 is %f pi" % (theta3/math.pi))
        
        theta1 = float(math.acos ((a*a + l1*l1 -l2*l2)/(2*a*l1)))
        theta4 = float(math.acos ((b*b + l1*l1 -l2*l2)/(2*b*l1)))
        
        print("The angle theta1 is %f pi" % (theta1/math.pi))
        print("The angle theta4 is %f pi" % (theta4/math.pi))
        
        theta5 = float(math.pi -(theta1 + theta2))
        theta6 = float(math.pi -(theta3 + theta4))
        
        theta5inpi = float(theta5/math.pi)
        theta6inpi = float(theta6/math.pi)
        
        print("The angle theta 5 is %f pi" % theta5inpi)
        print("The angle theta 6 is %f pi" % theta6inpi)
        
        print("The angle theta 5 is %f" % theta5)
        print("The angle theta 6 is %f" % theta6)
        
        
        
        
