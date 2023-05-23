#!/usr/bin/python3

from time import sleep

PATH = "/sys/bus/gpio/devices/gpiochip0/gpio/"

# GPIO 10 for INDEX
# GPIO 9 for MIDDLE
# GPIO 8 for RING + LITTLE

# Set GPIO pins as output pins
try:
    f1 = open(PATH + "gpio10/direction", "w")
    f1.write("out") 
    f1.close()
    print("P8_31 set as OUTPUT")
    
    f2 = open(PATH + "gpio9/direction", "w")
    f2.write("out") 
    f2.close()
    print("P8_33 set as OUTPUT")
    
    f3 = open(PATH + "gpio8/direction", "w")
    f3.write("out") 
    f3.close()
    print("P8_35 set as OUTPUT")
except:
    print("ERROR setting GPIOs as out")
    pass


# If prediction is INDEX finger 
# Sends the value 1 for one second before sending 0
#if prediction == "IndexFlexion":
try:
    f1 = open(PATH + "gpio10/value", "w")
    f1.write("1") 
    print("P8_31 sent 1")
    f1.close()
    sleep(1)
    f1 = open(PATH + "gpio10/value", "w")
    f1.write("0")
    print("P8_31 sent 0")
    f1.close()
except:
    print("ERROR: P8_31 could not send value")
    pass

#elif prediction == "MiddleFlexion":
try:
    f2 = open(PATH + "gpio9/value", "w")
    f2.write("1") 
    print("P8_33 sent 1")
    f2.close()
    sleep(1)
    f2 = open(PATH + "gpio9/value", "w")
    f2.write("0")
    print("P8_33 sent 0")
    f2.close()
except:
    print("ERROR: P8_33 could not send value")
    pass

#elif prediction == "RingFlexion" or "LittleFlexion":
try:
    f3 = open(PATH + "gpio8/value", "w")
    f3.write("1") 
    print("P8_35 sent 1")
    f3.close()
    sleep(1)
    f3 = open(PATH + "gpio8/value", "w")
    f3.write("0")
    print("P8_35 sent 0")
    f3.close()
except:
    print("ERROR: P8_35 could not send value")
    pass