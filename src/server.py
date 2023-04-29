#!/usr/bin/python

import socket
import datetime
import json
import threading
import keyboard
from time import sleep
from datetime import datetime

# Set IP and Port 
ip = "192.168.7.1"
port = 8080
listeningAddress = (ip, port)

filename = "sensorData.json"

# Make datagram socket
datagramSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
datagramSocket.bind(listeningAddress)

# Initialize data variables
dataComplete = []

def receive_data():
    global  dataComplete
    while True:
        emgVal,_ = datagramSocket.recvfrom(128)
        emgVal = float(emgVal.decode())
        dataComplete.append(emgVal)

def save_data(label):
    global dataComplete
    data_to_save = {'label': label, 'data': dataComplete}
    with open(filename, 'w') as f:  # dumping to a file
        json.dump(data_to_save, f)
    print('Data saved with label:', label)
    dataComplete.clear()

if __name__ == '__main__':
    # Start receiving data in a separate thread
    receive_thread = threading.Thread(target=receive_data)
    receive_thread.start()

    # saves data as long as a key is held
    # 1 thumb 2 index 3 middle 4 ring 5 pinky
    while True:
        if keyboard.is_pressed('numpad 1'):# thumb
            # Wait for the key to be released
            dataComplete.clear() # clearing data before saving into it
            sleep(0.01) # waiting for 10 miliseconds 
            while keyboard.is_pressed('numpad 1'): 
                pass
            save_data('1')

        elif keyboard.is_pressed('numpad 2'): # index
            # Wait for the key to be released
            dataComplete.clear() # clearing data before saving into it
            sleep(0.01) # waiting for 10 miliseconds 
            # Wait for the key to be released
            while keyboard.is_pressed('numpad 2'):
                pass
            save_data('2')

        elif keyboard.is_pressed('numpad 3'): # middle

            # Wait for the key to be released
            dataComplete.clear() # clearing data before saving into it
            sleep(0.01) # waiting for 10 miliseconds 
            while keyboard.is_pressed('numpad 3'):
                pass
            save_data('3')
        elif keyboard.is_pressed('numpad 4'): # ring

            # Wait for the key to be released
            dataComplete.clear() # clearing data before saving into it
            sleep(0.01) # waiting for 10 miliseconds 
            while keyboard.is_pressed('numpad 4'):
                pass
            save_data('4')
        elif keyboard.is_pressed('numpad 5'): # pinky

            # Wait for the key to be released
            dataComplete.clear() # clearing data before saving into it
            sleep(0.01) # waiting for 10 miliseconds 
            while keyboard.is_pressed('numpad 5'):
                pass
            save_data('5')
        

    # Join the receive thread
    receive_thread.join()

    # Print the time taken to receive data
    print("We received " + str(dataCount) + " samples")
