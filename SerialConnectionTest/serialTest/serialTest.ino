int byte1= 0;

void setup()
{ 
  Serial.begin(9600);
}

void loop ()
{
  while( Serial.available() <= 0 ){};
  byte1 = Serial.read();
  Serial.write(byte1+1);
}

