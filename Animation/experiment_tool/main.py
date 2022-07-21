from tkinter import *
from parameters import *
import random
import time
from scipy import signal
import numpy as np

def generate_sequence(t, active_interval, rest_interval):
    return signal.square(2 * np.pi * SQUARE_FREQ * t, duty=DUTY)

def show(side, canvas, left_hand, right_hand, left_foot, right_foot, tongue):
    if (side == 0):
        canvas.itemconfigure(left_hand, state='normal')
        canvas.itemconfigure(right_hand, state='hidden')
        canvas.itemconfigure(left_foot, state='hidden')
        canvas.itemconfigure(right_foot, state='hidden')
        canvas.itemconfigure(tongue, state='hidden')
    elif (side == 1):
        canvas.itemconfigure(left_hand, state='hidden')
        canvas.itemconfigure(right_hand, state='normal')
        canvas.itemconfigure(left_foot, state='hidden')
        canvas.itemconfigure(right_foot, state='hidden')
        canvas.itemconfigure(tongue, state='hidden')
    elif (side == 2):
        canvas.itemconfigure(left_hand, state='hidden')
        canvas.itemconfigure(right_hand, state='hidden')
        canvas.itemconfigure(left_foot, state='normal')
        canvas.itemconfigure(right_foot, state='hidden')
        canvas.itemconfigure(tongue, state='hidden')
    elif (side == 3):
        canvas.itemconfigure(left_hand, state='hidden')
        canvas.itemconfigure(right_hand, state='hidden')
        canvas.itemconfigure(left_foot, state='hidden')
        canvas.itemconfigure(right_foot, state='normal')
        canvas.itemconfigure(tongue, state='hidden')
    elif (side == -1):
        canvas.itemconfigure(left_hand, state='hidden')
        canvas.itemconfigure(right_hand, state='hidden')
        canvas.itemconfigure(left_foot, state='hidden')
        canvas.itemconfigure(right_foot, state='hidden')
        canvas.itemconfigure(tongue, state='hidden')

def start(start_flag):
    start_flag[0] = True
if __name__ == '__main__':
    window = Tk()

    canvas = Canvas(window, width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
    canvas.pack()
    
    start_flag = [False]
    times = []
    target = []
    left_hand = canvas.create_rectangle(300, 300, 400, 400, fill='gray')
    right_hand = canvas.create_rectangle(800, 300, 900, 400, fill='gray')
    left_foot = canvas.create_rectangle(300, 600, 400, 700, fill='blue')
    right_foot = canvas.create_rectangle(800, 600, 900, 700, fill='blue')
    tongue = canvas.create_rectangle(550, 100, 650, 200, fill='red')
    B = Button(window, text ="START", command =lambda: start(start_flag))
    B.pack()
    while(start_flag[0] == False):
        window.update()
        print("waiting to start..")

    rest_time = True
    t_start = time.time()
    while (time.time() - t_start < DURATION):
        window.update()
        square = generate_sequence(time.time(), ACTION_TIME, REST_TIME)
        if (square == -1):
            rest_time = True
            show(square, canvas, left_hand, right_hand, left_foot, right_foot, tongue)
            side = -1
            print(side)
        elif(square == 1 and rest_time):
            rest_time = False
            side = random.randint(0, 3)
            show(side, canvas, left_hand, right_hand, left_foot, right_foot, tongue)
            print(side)
        elif(square == 1):
            show(side, canvas, left_hand, right_hand, left_foot, right_foot, tongue)
            print(side)
        times.append(time.time() - t_start)
        target.append(side)
        time.sleep(1/SAMPLING_FREQ)
    np.savetxt('target.csv', [p for p in zip(times, target)])
    #window.mainloop()