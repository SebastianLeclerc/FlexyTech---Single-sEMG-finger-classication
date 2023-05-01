#!/usr/bin/python

### using multithreading 
'''
in this code the sensor data is read and saved in the memory where it is continuously updated.
the sensor thread continuosly enqueue the sensor reading at 1Khz sampling frequency, and the send thread will send the data
to the computer/laptop.
it doesn't matter how fast we recieve the data as long as we measure it at the right frequency.
'''
import queue
import socket 
import threading 
import mmap
from time import sleep
import json
import keyboard
import random as rd
# Server address:port var
#serverAddress = ("192.168.7.1", 8080)
serverAddress = ("192.168.0.4", 8080) #test

# Create a datagram socket
tempSensorSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Open the sensor file in read-only mode using a memory-mapped file

pin = 0
filename = "sensorData.json"
dataComplete = []
#file = f"/sys/bus/iio/devices/iio:device0/in_voltage{pin}_raw"
file = 'test.txt'
with open(file, "r") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    # saving data into json file 
    
    def save_data(label):
        global dataComplete
        data_to_save = {'label': label, 'data': dataComplete}
        with open(filename, 'w') as f:  # dumping to a file
            json.dump(data_to_save, f)
        print('Data saved with label:', label,"length=",len(dataComplete))
        dataComplete.clear()

    # Define a function that reads the sensor data
    def read_sensor_data():
        while True:
            # f.seek(0)
            # data = f.read()
            # Read the sensor data from the memory-mapped file
            mm.seek(0)
            data = mm.readline().strip()
            #data = mm.read()
            # Add the data to a queue to be sent over the socket
            dataComplete.append(str(data))

            sleep(0.001)
            
            
    # Define a function that sends the sensor data over the socket
    def send_sensor_data():
        while True:
            # Wait for data to be added to the queue
            print(len(dataComplete))
            sleep(1)
            # Send the data over the socket
            # tempSensorSocket.sendto(data, serverAddress)\

    def listen_keyboard():
        while True:
            if keyboard.is_pressed('1'):# thumb
                # Wait for the key to be released
                dataComplete.clear() # clearing data before saving into it
                #sleep(0.01) # waiting for 10 miliseconds 
                while keyboard.is_pressed('1'): 
                    mm.seek(0)
                    data = mm.readline().strip()
                    dataComplete.append(str(data))
                    print(len(dataComplete))
                    sleep(0.001)
                save_data('1')
    
    # Create a queue to hold the sensor data
    data_queue = queue.Queue()
    
    # Create a thread to read the sensor data
    #sensor_thread = threading.Thread(target=read_sensor_data)
    
    # Create a thread to send the sensor data
    send_thread = threading.Thread(target=send_sensor_data)
    keyboard_thread = threading.Thread(target=listen_keyboard)
    
    # Start the threads
    #sensor_thread.start()
    send_thread.start()
    keyboard_thread.start()