import os
import numpy as np
import argparse
import matplotlib
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)

import matplotlib.pyplot as plt 

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default="experiments/mobilenetKD.csv")
args = parser.parse_args()




def plot_learning_curve():
    work_dir = "plain KD/"
    # files = ["shuff lg no KD val miou score.csv", "shuff lg plain KD val miou score.csv"]
    work_dir = "noKD/"

    #files = os.listdir(work_dir)

    print(files)
    import csv
    def csv_to_np(path, index):
        f = open(path, 'rt')
        reader = csv.reader(f)
        data = [] 
        counter = 0
        for r in reader: 
            if(counter >0):
                try:
                    data.append(float(r[index]))
                except:
                    pass
            else:
                column = r[index]
            counter +=1
        f.close()
        return data, column

    # plot_data = (csv_to_np(work_dir + files[1]))
    # plot_data_b = (csv_to_np(work_dir + files[0]))

    infos = {}

    for f in files:
        for i in range(2, 6):
            plot_data, column = csv_to_np(work_dir + f, i)
            x = np.arange(len(plot_data))

            plt.plot(x, plot_data, label=f)
            # print("max val miou for ", f , column ," is ", max(plot_data))
            key = str(f + " " + column)
            infos[key] = max(plot_data)
            print(key, infos[key])

    plt.title(work_dir)
    plt.legend()
    plt.ylabel("val miou score")
    plt.xlabel("epochs")
    plt.show()

def scatter():
    import csv
    import math
    # fig = plt.figure(figsize=(6, 5))

    f = open(args.file, 'rt')
    # print(os.listdir(""))
    reader = csv.reader(f)
    x, y, labels, y_kd = [], [], [], []
    counter = 0
    for r in reader: 
        print(r)
        if(counter >0):
            if(r[3]!="None"):
                
                label = r[0] + " " + r[1]
                x_temp = float(r[2])
                y_temp = float(r[3])
                y_kd_temp = (r[4])

                # order of columns is: modelname, preset, params, val-miou, kd val-miou
                labels.append(label)
                x.append(x_temp)
                y.append(y_temp)
                y_kd.append(y_kd_temp)
                    
        counter +=1
    f.close()

    for i in range(len(x)):
        plt.scatter(x[i]/1000, y[i], color="red", marker="x")
        plt.text(x[i]/1000 + 20, (y[i] -0.00005), labels[i], fontsize=12)

        if("None"!=y_kd[i]):
            dy = (float(str(y_kd[i])) - float(y[i]))
            #plt.arrow(x[i]/1000, y[i], dx=0, dy=dy, length_includes_head=True)
            plt.scatter(x[i]/1000, float(y_kd[i]), color="blue", marker="x")

    # y = np.array(y)
    # plt.ylim(0.85, 0.99)
    #plt.ylim(0.0, 0.99)
    max_x= max(x)/1000*1.5
    plt.xlim(0, max_x)
    x = np.array(x)
    y = np.around(y, decimals = 3)
    #plt.xscale("log")
    #plt.title("Tradeoff between number of parameters and val_miou")
    plt.ylabel("val_miou")
    plt.xlabel("model size measured in x10³ trainable parameters")
    savename = "experiments/figures/" +  str(args.file).rsplit("/")[1][:-3] + "png"
    plt.savefig(savename)
    plt.show()

scatter()