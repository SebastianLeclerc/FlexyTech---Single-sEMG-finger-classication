#!/usr/bin/python

import socket
from time import sleep
from datetime import datetime

import os

#import Adafruit_BBIO.ADC as ADC
# # Set up ADC
# ADC.setup()

#Server address:port var
serverAddress = ("192.168.7.1", 8080)

#Create a datagram socket
tempSensorSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
pin = "0"
sensor='/sys/bus/iio/devices/iio:device0/in_voltage{}_raw'.format(pin)

data = []
dataCount = 0
f = open(sensor, "r")
timeBefore = datetime.now()

while True:
    f.seek(0) #Init cursor to start of file

    #analog_value = float(f.read()[:-1]) # Read analog value
    #tempStr = str(analog_value)
    analog_value = f.read()[:-1] #test
    #analog_value = os.system('cat /sys/bus/iio/devices/iio:device0/in_voltage0_raw &> /dev/null')
    data.append(analog_value)
    #tempSensorSocket.sendto(tempStr.encode(), serverAddress) #send encoded data
    #response = tempSensorSocket.recv(1024)
    dataCount += 1
    sleep(0.001) #1 ms, 1s = 1000 ms
    if dataCount == 1000:
        break

timeAfter = datetime.now() - timeBefore

ms = timeAfter.total_seconds() * 1000

#we get 100 000 data samples in the 10 s period (right now)

print(ms)
#print(datetime.strptime((timeAfter - timeBefore), "%S"))
f.close()