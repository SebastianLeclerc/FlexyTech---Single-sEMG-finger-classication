#include "Servo.h"

Servo index, middle, ring_little;  // create servo object to control a servo

int sensorPin1 = 2;  //Replace with the correct pin number.
int sensorPin2 = 4;
int sensorPin3 = 7;
int sensorValue1, sensorValue2, sensorValue3;  //Sensor value from sensorPin1

int pos = 0;  // variable to store the servo position

void moveFinger(Servo finger);

void setup() {
  index.attach(9);  // attaches the servo on pin 9 to the servo object. All these pins are PWM which is recommended by arduino for running the hand
  middle.attach(10);
  ring_little.attach(11);
  Serial.begin(9600);
  pinMode(sensorPin1, INPUT);
  pinMode(sensorPin2, INPUT);
  pinMode(sensorPin3, INPUT);

  //Reset fingers to 0 (ring_little motor is reversed)
  index.write(0);
  middle.write(0);
  ring_little.write(180); 
}

void loop() {

  sensorValue1 = digitalRead(sensorPin1);
  sensorValue2 = digitalRead(sensorPin2);
  sensorValue3 = digitalRead(sensorPin3);
  
  //The threshold for LOW(logic 0)(Expressed in Volt = 2.35V) is 0.3 Vcc and the threshold for HIGH (Logic 1)(Expressed in Volt = 2.60V) is 0.6Vcc
  //Maxiumum output voltage of the beaglebone green is 3.3V
  if (sensorValue1 == HIGH) {
    //send which pin to the 'move function' in order to make the correct finger move
    moveFingerOne(index);
  } else if (sensorValue2 == HIGH) {
    moveFingerOne(middle);
  } else if (sensorValue3 == HIGH) {
    //For ring and middle finger
    moveFingerTwo(ring_little);
  }
}

void moveFingerOne(Servo finger) {
    for (pos = 0; pos <= 160; pos += 2) {  // goes from 0 degrees to 160 degrees in steps of 2 degree
      finger.write(pos);                   // tell servo to go to position in variable 'pos'
      delay(5);                            // delays for 5 ms 
    }
    for (pos = 160; pos >= 0; pos -= 2) {  // goes from 160 degrees to 0 degrees
      finger.write(pos);                   // tell servo to go to position in variable 'pos'
      delay(5);                            // waits 5ms for the servo to reach the position
    }
}

void moveFingerTwo(Servo finger) {
    for (pos = 160; pos >= 0; pos -= 2) {
      finger.write(pos);
      delay(5);
    }
    for (pos = 0; pos <= 160; pos += 2) {
      finger.write(pos);
      delay(5);
    }
}
