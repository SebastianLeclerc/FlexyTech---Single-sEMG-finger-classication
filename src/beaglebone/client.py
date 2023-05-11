#!/usr/bin/python3

'''
This will read a file in the filesystem where sensor data is located which is continuously updated.
It will then send it over a UDP socket with a sampling rate of 1K Hz, approximately

Commented code is for a multithreading approach which in test was slower than doing this sequentially.

WARNING: If this script is kept open - beaglebone needs to be shutdown due to some overflow error!
'''

#import queue
import socket 
#import threading
from time import sleep,time

# Set server address and create a datagram socket
serverAddress = ("192.168.7.1", 8080)
clientSensorSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#count = 0 # For keeping track of packages

# Input to read from file, from a certain pin on the board
pin = 0
file = "/sys/bus/iio/devices/iio:device0/in_voltage{}_raw".format(pin)
f = open(file, "r")

# Read sensor data from file and put in queue
def read_sensor_data():
    #global count
    while(True):
        try:
            f.seek(0)
            data = f.read().strip() #+ "," + str(count) # Last part for pkt counting
            #count += 1
            clientSensorSocket.sendto(data.encode(), serverAddress) #Sending string
            sleep(0.001) # Play around with this value!
        
        except KeyboardInterrupt:
            break
        
read_sensor_data()      
        
# Send sensor data to server from queue
# def send_sensor_data():
#     while(True):
#         try:
            
#             data = data_queue.get()
#             clientSensorSocket.sendto(data.encode(), serverAddress)
#             #sleep(0.0005) #Play around with this value!
#         except KeyboardInterrupt:
#             break

        
    #dataIn,_ = clientSensorSocket.recvfrom(128)#128

      
# Create a queue to hold the sensor data
# data_queue = queue.Queue()

# # Create a thread to read the sensor data
# sensor_thread = threading.Thread(target=read_sensor_data)
# send_thread = threading.Thread(target=send_sensor_data)

# # Start the threads
# send_thread.start()
# sensor_thread.start()

# while(True):
#     try:
#         pass
#     except KeyboardInterrupt:
#         clientSensorSocket.close() #Close socket
#         print("Interrupt!")
        