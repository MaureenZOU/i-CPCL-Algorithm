import numpy as np
import csv
import sys
from random import shuffle
import pandas

class data:
    def __init__(self, point):
        ##point: numpy.array
        self.point = point

    def __repr__(self):
        return str(self.point)

def generateData(clusterNum):
    fileName = "./data/CPCL_sudoData_Cluster" + str(clusterNum) + "_sigma"+str(sigma)+"_"+str(dataCount)+".csv"
#fileName = "./data/s1_data.csv"
    dataSet = readData(fileName)

    output = []
    for i in range(0, len(dataSet)):
        output.append(data(dataSet[i]))
    
    return output

def readData(fileName):
    dataframe = pandas.read_csv(fileName,
                                engine='python', header=None)
    dataset = dataframe.values
    dataSet = dataset.astype('float32')

    return dataSet

def generateSeed(n, dataSet):
    maxX = -10000000
    maxY = -10000000
    minX = sys.maxsize
    minY = sys.maxsize

    for data in dataSet:
        if data.point[0] >= maxX:
            maxX = data.point[0]
        elif data.point[0] < minX:
            minX = data.point[0]

        if data.point[1] >= maxY:
            maxY = data.point[1]
        elif data.point[1] < minY:
            minY = data.point[1]

    seedX = np.random.random(n) * (maxX - minX) + minX
    seedY = np.random.random(n) * (maxY - minY) + minY

    output =[]
    for i in range(0, len(seedX)):
        output.append([seedX[i], seedY[i]])

    return output

def writeFile(writeMatrix, fileName):
    with open(fileName, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in writeMatrix:
            spamwriter.writerow(row)

sigma = 2
clusterNum = 3
seedNum = int(sys.argv[1])
dataCount = 1
seedCount = 1
dataSet = generateData(clusterNum)  ##cluster points
seeds = generateSeed(seedNum, dataSet)


fileName = './seed/seed_' +str(clusterNum)+ "_"+str(seedNum)+ "_sigma"+str(sigma)+"_"+str(seedCount)+".csv"
#fileName = "./seed/"+str(seedNum)+"_s1_seed.csv"
writeFile(seeds, fileName)



