import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas
import sys

def readData(fileName):
    dataframe = pandas.read_csv(fileName, engine='python', header=None)
    dataset = dataframe.values
    dataSet = dataset.astype('float32')

    return dataSet

def update_line(num, data, line):
    line.set_data(data[..., :num])
    return line,

fig1 = plt.figure()
clusterNum = 3
seedNum = int(sys.argv[1])
seedCount = 1
sigma = 2
count = 1
epoch = int(sys.argv[2])
#num = int(sys.argv[1])

fileName = "./data/CPCL_sudoData_Cluster"+str(clusterNum)+"_sigma"+str(sigma)+"_"+str(count)+".csv"
#fileName = "./data/s1_data.csv"
dataPoints = readData(fileName)

for data in dataPoints:
    plt.plot(data[0], data[1], 'k,')

fileName = "./animation/CPCL_Animation_" +str(clusterNum)+ "_"+str(seedNum)+"_sigma"+str(sigma)+"_"+str(count)+"_"+str(seedCount)+".csv"
#fileName = "./animation/CPCL_Animation_s1.csv"
seedPoints = readData(fileName)

print(len(seedPoints))
seedPoints = np.reshape(seedPoints, (seedNum, 2, epoch))
data = [0 for i in range(0, seedNum)]

for i, seed in enumerate(seedPoints):
    print(str(seed[0][0])+','+str(seed[1][0]))
    plt.plot(seed[0][0], seed[1][0], 'k.')
    plt.plot(seed[0][epoch-1], seed[1][epoch-1], 'k*')
    data[i] = seed

l = [0 for i in range(0, seedNum)]

for i in range(0, len(l)):
    l[i], =  plt.plot([], [], 'r-')

dataEstimate = np.reshape(dataPoints, (2, len(dataPoints)))

plt.xlim(min(dataEstimate[0]), max(dataEstimate[0])) #min max of x
plt.xlim(min(dataEstimate[1]), max(dataEstimate[1])) #min max of y
plt.xlabel('x') 
plt.title('CPCL epoch='+str(epoch)+' clusterNum=3 seedNum='+str(seedNum))

line_ani = [0 for i in range(0, seedNum)]
for i in range(0, seedNum):
    line_ani[i] = animation.FuncAnimation(fig1, update_line, 1000, fargs=(data[i], l[i]),
                                   interval=25, blit=False)

plt.show()

#figName = fileName = "./figure/CPCL_fig_" +str(clusterNum)+ "_"+str(seedNum)+"_sigma"+str(sigma)+"_"+str(count)+"_"+str(seedCount)+".png"
#plt.savefig(figName)
