#!/usr/bin/python

import socket
import datetime
from time import sleep
from datetime import datetime

#Set IP and Port of Actuator node receiving temperature
ip      = "192.168.7.1"
port    = 8080
listeningAddress = (ip, port)

dataCount = 0
timeBefore = 0

dataComplete = []
dataSecond = [0 for i in range(1000)]
filename = "sensorData.txt"

#Make datagram socket using IPv4 addressing scheme and bind it 
datagramSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
datagramSocket.bind(listeningAddress)

#Liten to incoming messages, print  them and send response
while(dataCount != 1000):
    emgVal,_ = datagramSocket.recvfrom(128)
    emgVal = float(emgVal.decode())
    
    #For time stamps
    if dataCount == 1:
        timeBefore = datetime.now()
    dataSecond.append(emgVal)
    
    dataCount += 1
    if dataCount == 1000:
        dataComplete.append(dataSecond)

timeAfter = datetime.now() - timeBefore #comparing time before / after
ms = timeAfter.total_seconds() * 1000 #in ms

print("We received " + str(dataCount) + "samples")

print("It took " + str(ms) + "ms")

with open(filename, 'w') as f: #dumping to a file
    f.write(str(dataComplete))
