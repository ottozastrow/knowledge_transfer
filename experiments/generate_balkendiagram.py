import os
import matplotlib.pyplot as plt 
import numpy as np
import argparse
import os
import matplotlib
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)
parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default="experiments/mobilenetKD.csv")
args = parser.parse_args()

from numpy import genfromtxt
data = genfromtxt(args.file, delimiter=',')

def show_diagram(data):
    x = data[:,0]
    y = data[:,1]
    y = 1-y
    plt.ylabel("val_miou error rate")
    plt.xlabel("number of epochs for teacher training")
    plt.plot(x, y, '.', color='blue')
    plt.plot(x, y, color='blue')

    savename = "experiments/figures/" +  str(args.file).rsplit("/")[1][:-3] + "png"
    plt.savefig(savename)
    plt.show()

show_diagram(data)