#!/usr/bin/python
'''
This will listen to keyboard input (1 - 5) and when pressed collect incomming UDP data from a listening socket.
It will then save the data in separate JSON files, when the sample looks "good" in the resulting graph image, which is based on a JSON data file.
'''

'''
This will listen to keyboard input (1 - 5) and when pressed collect incomming UDP data from a listening socket.
It will then save the data in separate JSON files, when the sample looks "good" in the resulting graph image, which is based on a JSON data file.
'''
from src import socket, json, keyboard, time, pd, plt

# Set IP and Port of listening server
ip = "192.168.7.1"
port = 8080
listeningAddress = (ip, port)

dataComplete = []
time_stamp = []

# Make datagram socket using IPv4 addressing scheme and bind it
datagramSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
datagramSocket.bind(listeningAddress)

name = input("Enter subject name: ")


# Save data in JSON with labels
def save_json(label):
    global dataComplete
    # Could clear terminal/STDIN here to not have "1111111" when recording.
    fingers = ['THUMB', 'INDEX', 'MIDDLE', 'RING', 'LITTLE']
    finger = fingers[int(label) - 1]
    sample = input(f"{finger} sample number (\'q\' to go again): ")
    if (sample == "q"):
        pass
    else:
        filename = str(name) + str(sample) + str(label)  # Will save each finger in separate files

        data_to_save = {'label': label, 'data': dataComplete, 'time': time_stamp}
        with open("../data/" + filename + "data.json", 'w') as f:
            json.dump(data_to_save, f, indent=4)
        plotting(filename)
        print('Data saved with label:', label, "length: ", len(dataComplete))
        dataComplete.clear()
        time_stamp.clear()
        print("\n1 = Thumb, 2 = Index, 3 = Middle, 4 = Ring, 5 = Little")  # Friendly output


# Receive new data to append
def receive_data():
    emgVal, _ = datagramSocket.recvfrom(128)
    emgVal = emgVal.decode()  # .split(",") # Decode incomming traffic, split for packet counting if needed
    dataComplete.append(emgVal)
    time_stamp.append(time.time())
    # return int(emgVal[1]) # For packet counting


"""
# Old working function to alert user if packet is dropped (never saw this during test)
def old_listen_keyboard():
    start, current, i = 0, 0, 0
    if keyboard.is_pressed('1'): # Thumb
        dataComplete.clear() # Clear data before saving
        start = receive_data() # Sanity check, start of UDP transmission counter
        while keyboard.is_pressed('1'): 
            current = receive_data() # Current counter
            i += 1
            if (start + i != current): # If we have dropped, consider restarting the data collection
                print("PACKET DROPPED!") # IS BEING SPAMMED
"""


# Listen to keyboard press, clear data list, get new data and save it
def listen_keyboard():
    if keyboard.is_pressed('1'):  # Thumb
        dataComplete.clear()
        while keyboard.is_pressed('1'):
            receive_data()
        save_json('1')
    elif keyboard.is_pressed('2'):  # Index
        dataComplete.clear()
        while keyboard.is_pressed('2'):
            receive_data()
        save_json('2')
    elif keyboard.is_pressed('3'):  # Middle
        dataComplete.clear()
        while keyboard.is_pressed('3'):
            receive_data()
        save_json('3')
    elif keyboard.is_pressed('4'):  # Ring
        dataComplete.clear()
        while keyboard.is_pressed('4'):
            receive_data()
        save_json('4')
    elif keyboard.is_pressed('5'):  # Little
        dataComplete.clear()
        while keyboard.is_pressed('5'):
            receive_data()
        save_json('5')


# Plot a graph based on JSON data and save it
def plotting(path):
    df = pd.read_json("../data/" + path + "data.json")

    label = df['label'][0]
    x = df['data']
    time = df['time']

    fig, ax = plt.subplots()
    ax.plot(time, x)
    ax.set_title(label)
    ax.set_xlabel('Time')
    ax.set_ylabel('Data')
    ax.grid()

    plt.savefig('../plots/' + path + '.png')


# Some information
print("Hold number key 1-5 to record data in JSON (\'ctrl+c\' to exit)")
print("1 = Thumb, 2 = Index, 3 = Middle, 4 = Ring, 5 = Little")

# Try to listen to the keyboard until interrupted, close the socket
while (True):
    try:
        listen_keyboard()
    except KeyboardInterrupt:
        datagramSocket.close()  # Close socket
        print("Interrupt!")
        break
