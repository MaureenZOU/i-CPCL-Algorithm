import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas
import sys
import csv

def readData(fileName):
    with open(fileName) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        dataSet = []
        for row in spamreader:
            entry = []
            for data in row:
                entry.append(float(data))
            dataSet.append(entry)

    return dataSet

def update_line(num, data, line):
    line.set_data(data[..., :num])
    return line,

fig1 = plt.figure()
clusterNum = 10
seedNum = int(sys.argv[1])
sigma = 3.5
count = 1
seedCount = 1

fileName = "./data/CPCL_sudoData_Cluster"+str(clusterNum)+"_sigma"+str(sigma)+"_"+str(count)+".csv"
dataPoints = readData(fileName)

for data in dataPoints:
    plt.plot(data[0], data[1], 'k,')

fileName = "./animation/lCPCL_Animation_" +str(clusterNum)+ "_"+str(seedNum)+"_sigma"+str(sigma)+"_"+str(count)+"_"+str(seedCount)+".csv"
seedPoints = readData(fileName)

lengthSeed = []
for i, seed in enumerate(seedPoints):
    seedPoints[i] = np.reshape(np.array(seed), (2, len(seed)/2))
    lengthSeed.append(len(seedPoints[i][0]))

maxLength= max(lengthSeed)

data = [0 for i in range(0, seedNum)]
print(len(seedPoints))

for i, seed in enumerate(seedPoints):
    #plt.plot(seed[0][0], seed[1][0], 'k.')
    data[i] = seed

fileName = "./center/lCPCL_center_" +str(clusterNum)+ "_"+str(seedNum)+"_sigma"+str(sigma)+"_"+str(count)+"_"+str(seedCount)+".csv"
center = readData(fileName)

for cen in center:
    plt.plot(cen[0],cen[1], 'r*')

l = [0 for i in range(0, seedNum)]

for i in range(0, len(l)):
    l[i], =  plt.plot([], [], 'b-')

dataEstimate = np.reshape(dataPoints, (2, len(dataPoints)))

plt.xlim(min(dataEstimate[0]), max(dataEstimate[0])) #min max of x
plt.xlim(min(dataEstimate[1]), max(dataEstimate[1])) #min max of y
plt.xlabel('x') 
plt.title('lCPCL epoch=40 clusterNum=10 seedNum='+str(seedNum))

line_ani = [0 for i in range(0, seedNum)]
for i in range(0, seedNum):
    line_ani[i] = animation.FuncAnimation(fig1, update_line, 1000, fargs=(data[i], l[i]),
                                   interval=25, blit=False)

#plt.show()

figName = fileName = "./figure/lCPCL_fig_" +str(clusterNum)+ "_"+str(seedNum)+"_sigma"+str(sigma)+"_"+str(count)+"_"+str(seedCount)+".png"
plt.savefig(figName)
