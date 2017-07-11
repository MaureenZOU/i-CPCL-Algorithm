import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
import sys
import csv
import json
import pandas
import time
import copy

################################################
##Original CPCL with cluster adjustment#########
################################################

## the class is about data point, point is the x, y location of the data point
class data:
    def __init__(self, point):
        ##point: numpy.array
        self.point = point

    def __repr__(self):
        return str(self.point)


## the class is about the seed point, which capture the information of previous location of the point, current location
## of the point, and distance between current seed and win seed
class seed:
    def __init__(self, prePoint, curPoint, seedWinNum, seedWinD, positionNum):
        self.prePoint = prePoint  ##numpy.array
        self.curPoint = curPoint  ##numpy.array
        self.seedWinNum = seedWinNum  ##integer
        self.seedWinD = seedWinD  ##float, distance between current seed and win seed
        self.positionNum = positionNum

    def setCurPoint(self, point):
        self.prePoint = self.curPoint
        self.curPoint = point

    def setSeedWinD(self, value):
        self.seedWinD = value

    def setSeedDataD(self, value):
        self.seedDataD = value

    def updateSeedWinNum(self):
        self.seedWinNum = self.seedWinNum + 1

    def setPositionNum(self, value):
        self.positionNum = value

    def __repr__(self):
        return str(self.prePoint) + " " + str(self.curPoint) + " " + str(self.seedWinNum) + " " + str(self.seedWinD)
        
def generateData(clusterNum):
    fileName = "./data/CPCL_sudoData_Cluster" + str(clusterNum) + "_sigma"+str(sigma)+"_"+str(count)+".csv"
#fileName = "./data/s1_data.csv"
    dataSet = readData(fileName)
    
    output = []
    x = []
    y = []
    for i in range(0, len(dataSet)):
        plt.plot(dataSet[i][0], dataSet[i][1], 'k,')
        x.append(dataSet[i][0])
        y.append(dataSet[i][1])
        output.append(data(dataSet[i]))

    d = ((max(x)-min(x))**(2)+(max(y)-min(y))**(2))**(.5)
    
    return output, d 

def generateSeed(clusterNum):
    fileName = './seed/seed_' +str(clusterNum)+ "_"+str(seedNum)+ "_sigma"+str(sigma)+"_"+str(seedCount)+".csv"
#fileName = './seed/'+str(seedNum)+'_s1_seed.csv'

    seeds = readData(fileName)

    output = []

    for i in range(0, len(seeds)):
        obj = seed(np.array([0, 0]), np.array(seeds[i]), 1, 0, i)
        output.append(obj)

    return output
    
# def generateSeed(n, dataSet):
#     maxX = -10000000
#     maxY = -10000000
#     minX = sys.maxsize
#     minY = sys.maxsize

#     for data in dataSet:
#         if data.point[0] >= maxX:
#             maxX = data.point[0]
#         elif data.point[0] < minX:
#             minX = data.point[0]

#         if data.point[1] >= maxY:
#             maxY = data.point[1]
#         elif data.point[1] < minY:
#             minY = data.point[1]

#     seedX = np.random.random(n) * (maxX - minX) + minX
#     seedY = np.random.random(n) * (maxY - minY) + minY

#     output = []

#     for i in range(0, len(seedX)):
#         obj = seed(np.array([0, 0]), np.array([seedX[i], seedY[i]]), 1, 0, i)
#         output.append(obj)

#     return output


def winSeed(inputData, seedPoints):
    distanceVector = []

    for i in range(len(seedPoints)):
        distanceVector.append(seedPoints[i].curPoint - inputData)

    distanceValue = []

    sumSeedWinNum = 0

    for data in seedPoints:
        sumSeedWinNum = sumSeedWinNum + data.seedWinNum

    for i in range(len(distanceVector)):
        Gama = seedPoints[i].seedWinNum / sumSeedWinNum
        value = (distanceVector[i][0] * distanceVector[i][0] + distanceVector[i][1] * distanceVector[i][
            1]) * Gama
        distanceValue.append(value)

    min = sys.maxsize
    loc = 0

    for i in range(len(distanceValue)):
        if distanceValue[i] < min:
            min = distanceValue[i]
            loc = i

    return loc


def updateCooperative(McObject, MuObject, XtObject, T):
    Mc = McObject.curPoint
    Mu = MuObject.curPoint
    Xt = XtObject.point

    VMcXt = Mc - Xt
    VMuXt = Mu - Xt

    DMcXt = (VMcXt[0] ** (2) + VMcXt[1] ** (2)) ** (0.5)
    DMuXt = (VMuXt[0] ** (2) + VMuXt[1] ** (2)) ** (0.5)

    row = DMcXt / max(DMcXt, DMuXt)
    locVector = MuObject.curPoint + (0.0016) * row * (XtObject.point - MuObject.curPoint)
    MuObject.setCurPoint(locVector)


def updatePenalize(McObject, MpObject, XtObject, T):
    Mc = McObject.curPoint
    Mp = MpObject.curPoint
    Xt = XtObject.point

    VMcXt = Mc - Xt
    VMpXt = Mp - Xt

    DMcXt = (VMcXt[0] ** (2) + VMcXt[1] ** (2)) ** (0.5)
    DMpXt = (VMpXt[0] ** (2) + VMpXt[1] ** (2)) ** (0.5)

    locVector = MpObject.curPoint - (0.0016) * (DMcXt / DMpXt) * (XtObject.point - MpObject.curPoint)
    MpObject.setCurPoint(locVector)


def calError(seedPoints):
    E = 0

    for data in seedPoints:
        dis = data.curPoint - data.prePoint
        E = E + (dis[0] ** (2) + dis[1] ** (2)) ** (0.5)

    return E

def readData(fileName):
    dataframe = pandas.read_csv(fileName,
                                engine='python', header=None)
    dataset = dataframe.values
    dataSet = dataset.astype('float32')

    return dataSet

def writeFile(writeMatrix, fileName):
    with open(fileName, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in writeMatrix:
            spamwriter.writerow(row)

def Algorithm(clusterNum, seedNum, seedList, epoch, RMSE, realEpoch):
    dataSet, d = generateData(clusterNum)  ##cluster points
    seedPoints = generateSeed(clusterNum)

    e = 0.0001  ##standard error rate
    Tmax = 10000  ##max iteration
    T = 1  ##actual iteration
    E = sys.maxsize  ##actual error

    w, h = len(seedPoints), len(seedPoints)
    edgeMatrix = [[0 for x in range(w)] for y in range(h)]
    oldEdgeMatrix = [[0 for x in range(w)] for y in range(h)]

    edgeMatrix = np.array(edgeMatrix)
    color = "mx"
    
    start = time.time()
    for k in range(0, realEpoch):
        print("epoch: "+str(k))
        for i in range(0, len(dataSet)):

            winLoc = winSeed(dataSet[i].point, seedPoints)

            for j in range(0, len(seedPoints)):
                vectorWin = seedPoints[winLoc].curPoint - seedPoints[
                    j].curPoint  ## the vector between win point and seed points j
                valueWin = (vectorWin[0] * vectorWin[0] + vectorWin[1] * vectorWin[1]) ** (.5)
                seedPoints[j].setSeedWinD(valueWin)

            WinSeedData = seedPoints[winLoc].curPoint - dataSet[i].point
            WinSeedDataDistance = (WinSeedData[0] * WinSeedData[0] + WinSeedData[1] * WinSeedData[1]) ** (.5)

            Sc = []  ##all the data that falls in the territory region

            for data in seedPoints:
                if data.seedWinD != 0 and data.seedWinD < WinSeedDataDistance:
                    Sc.append(data)

            Sc = sorted(Sc, key=lambda Data: Data.seedWinD)

            Q = len(Sc)
            lRate = 0.005
            Qu = int(Q * min(1, lRate * seedPoints[winLoc].seedWinNum))

            Su = []  ##cooperation data points
            Sp = []  ##penalized data points

            Su = Sc[0:Qu]
            Sp = Sc[Qu:]

            for data in Su:
                updateCooperative(seedPoints[winLoc], data, dataSet[i], T)
                edgeMatrix[winLoc][data.positionNum] = edgeMatrix[winLoc][data.positionNum] + 1

            for data in Sp:
                updatePenalize(seedPoints[winLoc], data, dataSet[i], T)
                edgeMatrix[winLoc][data.positionNum] = edgeMatrix[winLoc][data.positionNum] - 1

            locVector = seedPoints[winLoc].curPoint + 0.001 * (
                dataSet[i].point - seedPoints[winLoc].curPoint)  ##location vector of new win seed point
            seedPoints[winLoc].setCurPoint(locVector)

            seedPoints[winLoc].updateSeedWinNum()
            E = calError(seedPoints)
            T = T + 1

        for i in range(0, len(seedPoints)):
            seedList[i][0].append(seedPoints[i].curPoint[0])
            seedList[i][1].append(seedPoints[i].curPoint[1])
        
        rmse = 0
        for i, seed in enumerate(seedPoints):
            rmse = rmse + ((seed.curPoint[0] - seed.prePoint[0])**(2) + (seed.curPoint[0] - seed.prePoint[0])**(2))**(0.5)
        RMSE.append([k,rmse/(len(seedPoints)*d)])
        CMSE.append([time.time()-start,rmse/(len(seedPoints)*d)])

    end = time.time()
    print("Execution Time: "+str(end-start)+" secs")
    return seedList, dataSet, seedPoints, d

def appendFile(line, fileName):
    with open(fileName, "a") as myfile:
    	myfile.write(line)

# clusterNum = int(sys.argv[1])
# seedPointNum = int(sys.argv[2])
# epoch = int(sys.argv[3])

clusterNum = 3
seedNum = int(sys.argv[1])
epoch = int(sys.argv[2])
sigma = 2
count = 1
seedCount = 1


RMSE = []
CMSE = []
realEpoch = epoch
seedList = [[[],[]] for i in range(0, seedNum)] #store the movement of each seed point 
seedList, dataSet, seedPoints, d = Algorithm(clusterNum, seedNum, seedList, epoch, RMSE, realEpoch)

fileName = "./centroid/CPCL_centroid_Cluster" + str(clusterNum) + "_sigma"+str(sigma)+"_"+str(count)+".csv"
#fileName = "./centroid/s1_center.csv"
centerData = readData(fileName)

distance = 0
for seed in seedPoints:
    plt.plot(seed.curPoint[0], seed.curPoint[1], 'r*')
    minDistance = sys.maxsize
    for cen in centerData:
        curDistance = np.linalg.norm(cen-np.array([seed.curPoint[0], seed.curPoint[1]]))
        if curDistance < minDistance:
            minDistance = curDistance
    distance = distance + minDistance

copySeed = copy.copy(seedPoints)
for seed in seedPoints:
    count = 0
    for i, copy in enumerate(copySeed):
        if np.linalg.norm(seed.curPoint - copy.curPoint) > 0.1 and np.linalg.norm(seed.curPoint - copy.curPoint) < 4000:
            copySeed.pop(i-count)
            count = count + 1



print(len(copySeed))
print(distance/(len(seedPoints)*d))

fileName = "./animation/CPCL_Animation_" +str(clusterNum)+ "_"+str(seedNum)+"_sigma"+str(sigma)+"_"+str(count)+"_"+str(seedCount)+".csv"
#fileName = "./animation/CPCL_Animation_s1.csv"
seedList = np.array(seedList)
seedList = np.reshape(seedList, (seedNum, realEpoch*2))
writeFile(np.array(seedList), fileName)


fileName = "./rmse/CPCL_RMSE_" +str(clusterNum)+ "_"+str(seedNum)+"_sigma"+str(sigma)+"_"+str(count)+"_"+str(seedCount)+".csv"
#fileName = "./epoch/CPCL_RMSE_s1_"+str(seedNum)+".csv"
writeFile(RMSE, fileName)
#fileName = "./cpu/CPCL_RMSE_s1_"+str(seedNum)+".csv"
#writeFile(CMSE, fileName)

line = "CPCL_RMSE_s1_"+str(seedNum)+": "+ str(distance/(len(seedPoints)*d))+"\n"
fileName = "result.log"
#appendFile(line, fileName)

plt.show()




