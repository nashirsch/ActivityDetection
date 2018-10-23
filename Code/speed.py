import json
import numpy as np
import matplotlib.pyplot as plt

# this is the file our model determined was driving
trace_file = "../Data/test/activity-dataset-test2.txt"

def accl_to_speed(trace_file):
    '''
    given a trace file extracts the accl readings and integrates
    to find speed
    '''
    with open(trace_file, 'r') as fopen:
        json_content = fopen.read();
        json_content = json_content.replace('\'', '"')

    activity_data = json.loads(json_content)

    xAccl = []
    yAccl = []
    time = []

    for d in activity_data[0]['seq']:
        xAccl.append(d['data']['xAccl'])
        yAccl.append(d['data']['yAccl'])
        time.append(d['time'])

    xAccl = np.array(xAccl)
    yAccl = np.array(yAccl)
    time = np.array(time)
    init_time = time[0]
    time = time - init_time

    generate_graphs(xAccl, yAccl, time)

    #compute the integrals of the graphs
    xSpeed = np.trapz(xAccl, x=time)
    ySpeed = np.trapz(yAccl, x=time)

    return xAccl, yAccl, time, xSpeed, ySpeed

def generate_graphs(xAccl, yAccl, time):
    '''
    takes in np array for Accl values and time and plots them
    '''
    # first plot the xAccl
    plt.plot(time, xAccl)
    plt.xlabel("time")
    plt.ylabel("xAccl")
    plt.show()

    # plot the yAccl
    plt.plot(time, yAccl)
    plt.xlabel("time")
    plt.ylabel("yAccl")
    plt.show()

xAccl, yAccl, time, xSpeed, ySpeed = accl_to_speed(trace_file)
