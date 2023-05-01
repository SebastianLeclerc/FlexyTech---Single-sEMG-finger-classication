import random as rd
from time import sleep
with open('test.txt','w+') as r:
    while True:
        r.seek(0)
        r.write(str(rd.randint(0,1)))
        sleep(0.0001)