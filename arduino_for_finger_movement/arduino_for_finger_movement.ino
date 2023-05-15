#include <Servo.h>

Servo index, middle, ring_little;  // create servo object to control a servo

int sensorPin1 = 2;  //Replace with the correct pin number.
int sensorPin2 = 3;
int sensorPin3 = 4;
int sensorValue1, sensorValue2, sensorValue3;  //Sensor value from sensorPin1

int pos = 0;  // variable to store the servo position

void setup() {
  index.attach(9);  // attaches the servo on pin 9 to the servo object
  middle.attach(10);
  ring_little.attach(11);
  Serial.begin(9600);
  pinMode(sensorPin1, INPUT);
  pinMode(sensorPin2, INPUT);
  pinMode(sensorPin3, INPUT);
}

void loop() {

  sensorValue1 = digitalRead(sensorPin1);
  sensorValue2 = digitalRead(sensorPin2);
  sensorValue3 = digitalRead(sensorPin3);

  if (sensorValue1 == HIGH) {
    //send which pin to the 'move function' in order to make the correct finger move
    moveFinger(index);
  } else if (sensorValue2 == HIGH) {
    moveFinger(middle);
  } else if (sensorValue3 == HIGH) {
    moveFinger(ring_little);
  }
}

void moveFinger(Servo finger) {
  for (pos = 0; pos <= 160; pos += 2) {  // goes from 0 degrees to 160 degrees in steps of 2 degree
    finger.write(pos);                   // tell servo to go to position in variable 'pos'
    delay(5);                            // delays for 5 ms 
  }
  for (pos = 160; pos >= 0; pos -= 2) {  // goes from 160 degrees to 0 degrees
    finger.write(pos);                   // tell servo to go to position in variable 'pos'
    delay(5);                            // waits 5ms for the servo to reach the position
  }
}