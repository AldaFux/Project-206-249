#include <Servo.h>
#include "math.h"

#define l1 9
#define l2 9
#define l3 4.5
#define pi 3.14159265359

Servo servoM, servoL, servoR;
double theta1, theta2, theta3, theta4;

double servoLsoll, servoRsoll;

//void calculateServoMap(double x, double y, double * servoLsoll , double * servoRsoll);

void setup()
{  
  Serial.begin(9600);
  
  servoM.attach(11);
  servoR.attach(10);
  servoL.attach(9);
  //driveToMontagePosition();
  //calculateServoMap(0,11, &servoLsoll, &servoRsoll);
  //ervoL.write(servoLsoll);
  //servoR.write(servoRsoll);
}

void loop ()
{
  
  //Serial.write((int)servoLsoll);
  //Serial.write((int)servoRsoll);
  //Serial.write(65);
  calculateServoMap(0,11, &servoLsoll, &servoRsoll);
  servoL.write(servoLsoll);
  servoR.write(servoRsoll);
  delay(5000);
  calculateServoMap(0,12, &servoLsoll, &servoRsoll);
  servoL.write(servoLsoll);
  servoR.write(servoRsoll);
  delay(5000);
  calculateServoMap(1,12, &servoLsoll, &servoRsoll);
  servoL.write(servoLsoll);
  servoR.write(servoRsoll);
  delay(5000);
  calculateServoMap(2,12, &servoLsoll, &servoRsoll);
  servoL.write(servoLsoll);
  servoR.write(servoRsoll);
  delay(5000);
  calculateServoMap(3,12, &servoLsoll, &servoRsoll);
  servoL.write(servoLsoll);
  servoR.write(servoRsoll);
  delay(5000);
  
  
}







void calculateServoMap(double x, double y, double * servoLsoll , double * servoRsoll)
{
  /*given the desired xy-coordinates, this function stores the values for the desired angels
  of the servo motors in the variables that are given by their pointers */
  double alpha, beta;
  double a, b, c;
  
  alpha = atan2(y,x);
  beta = pi - alpha;
  
  c = sqrt(y*y + x*x); //c only defined by the distance of the endeffector towards the origin
  
  
  a = sqrt(c*c + l3*l3*0.25 - c*l3*cos(beta)); //use cosine rule to get a
  b = sqrt(c*c + l3*l3*0.25 - c*l3*cos(alpha));//use cosine rule to get b
  
  //use sine rule to get theta2 and theta3, BUT ATTENTION: sine rule is ambiguous - case differentiation is necessary
  if( x > 1.125)
  {
    theta2 = asin(sin(beta) * c/a);
    theta3 = pi - asin(sin(alpha)* c/b);
  } 
  else if( x < -1.125)
  {
    theta2 = pi - asin(sin(beta) * c/a);
    theta3 = asin(sin(alpha)* c/b);
  } 
  else
  {
    theta2 = asin(sin(beta) * c/a);
    theta3 = asin(sin(alpha)* c/b);
  }
  
  
  theta1 = acos((a*a + l1*l1 - l2*l2)/(2 * a * l1));
  theta4 = acos((b*b + l1*l1 - l2*l2)/(2 * b * l1));
 
  *servoLsoll = 161 - (pi - (theta1+theta2) ) * 180/pi; 
  *servoRsoll = (pi - (theta3+theta4)) * 180/pi + 15;
  

}

void driveToMontagePosition()
{
  servoM.write(90);
  servoL.write(16);
  servoR.write(165);
}

