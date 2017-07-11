import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
import sys
import csv
import time
import pandas

################################
##Cooperation improvement CPCL##
################################
GLOBLE_DELETE_SEED = []
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
    def __init__(self, prePoint, curPoint, seedWinNum, seedWinD, positionNum, trace):
        self.prePoint = prePoint  ##numpy.array
        self.curPoint = curPoint  ##numpy.array
        self.seedWinNum = seedWinNum  ##integer
        self.seedWinD = seedWinD  ##float, distance between current seed and win seed
        self.positionNum = positionNum
        self.trace = trace

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
    #fileName = "./data/CPCL_sudoData_Cluster" + str(clusterNum) + "_sigma"+str(sigma)+"_"+str(count)+".csv"
    fileName = "./data/s2_data.csv"
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


def generateSeed():
    #fileName = './seed/seed_' +str(clusterNum)+ "_"+str(seedNum)+ "_sigma"+str(sigma)+"_"+str(seedCount)+".csv"
    fileName = './seed/'+str(seedNum)+'_s2_seed.csv'
    seeds = readData(fileName)

    output = []

    for i in range(0, len(seeds)):
        obj = seed(np.array([0, 0]), np.array(seeds[i]), 1, 0, i, [[],[]])
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

#     initSeed = []
#     for i in range(0, len(seedX)):
#         obj = seed(np.array([0, 0]), np.array([seedX[i], seedY[i]]), 1, 0, i)
#         initSeed.append(obj.curPoint)
#         output.append(obj)

#     return output, initSeed


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

def readData(fileName):
    dataframe = pandas.read_csv(fileName,
                                engine='python', header=None)
    dataset = dataframe.values
    dataSet = dataset.astype('float32')

    return dataSet

def updateCooperative(McObject, MuObject, XtObject, p1, p2, edgeMatrix):
    Mc = McObject.curPoint
    Mu = MuObject.curPoint
    Xt = XtObject.point

    VMcXt = Mc - Xt
    VMuXt = Mu - Xt

    DMcXt = (VMcXt[0] ** (2) + VMcXt[1] ** (2)) ** (0.5)
    DMuXt = (VMuXt[0] ** (2) + VMuXt[1] ** (2)) ** (0.5)

    l = 0.001
    #print(l)
    #l = 0.001

    row = DMcXt / max(DMcXt, DMuXt)
    locVector = MuObject.curPoint + l * row * (XtObject.point - MuObject.curPoint)
    MuObject.setCurPoint(locVector)


def updatePenalize(McObject, MpObject, XtObject, p1, p2, edgeMatrix):
    Mc = McObject.curPoint
    Mp = MpObject.curPoint
    Xt = XtObject.point

    VMcXt = Mc - Xt
    VMpXt = Mp - Xt

    DMcXt = (VMcXt[0] ** (2) + VMcXt[1] ** (2)) ** (0.5)
    DMpXt = (VMpXt[0] ** (2) + VMpXt[1] ** (2)) ** (0.5)

    l = 0.001
    #print(l)
    #l = 0.001

    locVector = MpObject.curPoint - l * (DMcXt / DMpXt) * (XtObject.point - MpObject.curPoint)
    MpObject.setCurPoint(locVector)


def calError(seedPoints):
    E = 0

    for data in seedPoints:
        dis = data.curPoint - data.prePoint
        E = E + (dis[0] ** (2) + dis[1] ** (2)) ** (0.5)

    return E


def mergePoint(seedPoints, loc1, loc2):
    seedPoints[loc1] = seed((seedPoints[loc1].prePoint + seedPoints[loc2].prePoint) / 2,
                            (seedPoints[loc1].curPoint + seedPoints[loc2].curPoint) / 2,
                            (seedPoints[loc1].seedWinNum + seedPoints[loc2].seedWinNum) / 2, 0, loc1, seedPoints[loc1].trace)

    GLOBLE_DELETE_SEED.append(seedPoints.pop(loc2))

    return seedPoints


def updateMatrix(matrix, seedPoints, loc1, loc2):
    matrix[loc1][loc2] = 0
    matrix[loc2][loc1] = 0

    for i in range(0, len(seedPoints) + 1):
        matrix[loc1][i] = matrix[loc1][i] + matrix[loc2][i]
        matrix[i][loc1] = matrix[i][loc1] + matrix[i][loc2]

    newMatrix = np.array(matrix)

    newMatrix = np.delete(newMatrix, (loc2), axis=0)
    newMatrix = np.delete(newMatrix, (loc2), axis=1)

    return newMatrix


def updateDict(matrix, seedPoints, matrixSeedDict):
    count = 0

    for data in seedPoints:
        matrixSeedDict[data.positionNum] = count
        count = count + 1

    return matrixSeedDict


def updateSeed(seedPoints, edgeMatrix, matrixSeedDict):
    lRate = 0.00000001

    for datai in seedPoints:
        for dataj in seedPoints:
            datai.curPoint = datai.curPoint + lRate * edgeMatrix[matrixSeedDict[datai.positionNum]][
                matrixSeedDict[dataj.positionNum]] * (
                                                  dataj.curPoint - datai.curPoint)

    return seedPoints

def writeFile(writeMatrix, fileName):
    with open(fileName, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in writeMatrix:
            spamwriter.writerow(row)

def calSimilarity(matrix,p,q):
    matrix[q][q]=matrix[q][p]
    matrix[q][p]=0

    distance=np.linalg.norm(matrix[q]-matrix[p])

    return distance

def Algorithm(clusterNum, seedPointNum, epoch, combine):
    dataSet, d = generateData(clusterNum)  ##cluster points
    seedPoints = generateSeed()

    matrixSeedDict = {}
    for data in seedPoints:
        matrixSeedDict[data.positionNum] = data.positionNum

    e = 0.0001  ##standard error rate
    E = sys.maxsize  ##actual error
    T = 1  ##actual iteration

    w, h = len(seedPoints), len(seedPoints)
    edgeMatrix = np.array([[0 for x in range(w)] for y in range(h)])
    oldEdgeMatrix = np.array([[0 for x in range(w)] for y in range(h)])
    finalEpoch = 0

    start = time.time()
    for k in range(0, epoch):
        #print(edgeMatrix)
        dMatrix = edgeMatrix - oldEdgeMatrix
        # RMSE.append([dMatrix.sum()])
        #print(dMatrix.sum())

        ##calculate similarity list
        minValue=sys.maxsize
        locP=0
        locQ=0

        # if k != 0 and np.count_nonzero(dMatrix) == 0:
        #     print("exit")
        #     break

        flag = False
        for p in range(0, len(seedPoints)):
            for q in range(0, len(seedPoints)):
                if edgeMatrix[p][q] > combine:
                    if p < q:
                        loc1 = p
                        loc2 = q
                    else:
                        loc2 = p
                        loc1 = q

                    print("combine location "+"p: "+str(p)+"q: "+str(q))

                    seedPoints = mergePoint(seedPoints, loc1, loc2)
                    edgeMatrix = updateMatrix(edgeMatrix, seedPoints, loc1, loc2)
                    matrixSeedDict = updateDict(edgeMatrix, seedPoints, matrixSeedDict)
                    color = "kx"
                    flag = True
                    break

            if flag == True:
                break

        seedPoints = updateSeed(seedPoints, dMatrix, matrixSeedDict)
        oldEdgeMatrix = edgeMatrix.tolist()

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
                updateCooperative(seedPoints[winLoc], data, dataSet[i], winLoc, matrixSeedDict[data.positionNum], dMatrix)
                edgeMatrix[winLoc][matrixSeedDict[data.positionNum]] = edgeMatrix[winLoc][
                                                                           matrixSeedDict[data.positionNum]] + 1

            for data in Sp:
                updatePenalize(seedPoints[winLoc], data, dataSet[i], winLoc, matrixSeedDict[data.positionNum], dMatrix)
                edgeMatrix[winLoc][matrixSeedDict[data.positionNum]] = edgeMatrix[winLoc][
                                                                           matrixSeedDict[data.positionNum]] - 1

            locVector = seedPoints[winLoc].curPoint + 0.001 * (
                dataSet[i].point - seedPoints[winLoc].curPoint)  ##location vector of new win seed point
            seedPoints[winLoc].setCurPoint(locVector)

            seedPoints[winLoc].updateSeedWinNum()
            E = calError(seedPoints)
            T = T + 1

        for seed in seedPoints:
            seed.trace[0].append(seed.curPoint[0])
            seed.trace[1].append(seed.curPoint[1])

        rmse = 0
        for i, seed in enumerate(seedPoints):
            rmse = rmse + ((seed.curPoint[0] - seed.prePoint[0])**(2) + (seed.curPoint[0] - seed.prePoint[0])**(2))**(0.5)
        RMSE.append([k,rmse/(len(seedPoints)*d)])
        CMSE.append([time.time()-start,rmse/(len(seedPoints)*d)])

        finalEpoch = k
        print("epoch: "+str(k))

    end = time.time()
    print("Execute Time: " + str(end-start)+" sec")

    return seedPoints, finalEpoch, d

def appendFile(line, fileName):
    with open(fileName, "a") as myfile:
        myfile.write(line)

epoch = int(sys.argv[2])
clusterNum = 15
seedNum = int(sys.argv[1])
combine = 4000
sigma = 3.5
count = 1
seedCount = 1
RMSE = []
CMSE = []
seedPoints, finalEpoch, d = Algorithm(clusterNum, seedNum, epoch, combine)
seedList = []
center = []
centerSeed = 0


for seed in GLOBLE_DELETE_SEED:
    seed = np.array(seed.trace)
    seed = np.reshape(seed, (len(seed)*len(seed[0])))
    seedList.append(seed)

#fileName = "./centroid/CPCL_centroid_Cluster" + str(clusterNum) + "_sigma"+str(sigma)+"_"+str(count)+".csv"
fileName = "./centroid/s2_center.csv"
centerData = readData(fileName)


for seed in seedPoints:
    plt.plot(seed.curPoint[0], seed.curPoint[1], 'r*')
    center.append([seed.curPoint[0],seed.curPoint[1]])
    seed = np.array(seed.trace)
    seed = np.reshape(seed, (len(seed)*len(seed[0])))
    seedList.append(seed)

distance = 0
for seed in seedPoints:
    minDistance = sys.maxsize
    for cen in centerData:
        curDistance = np.linalg.norm(cen-np.array([seed.curPoint[0], seed.curPoint[1]]))
        if curDistance < minDistance:
            minDistance = curDistance
    distance = distance + minDistance

print(distance/(len(seedPoints)*d))

#fileName = "./animation/sCPCL_Animation_" +str(clusterNum)+ "_"+str(seedNum)+"_sigma"+str(sigma)+"_"+str(count)+"_"+str(seedCount)+".csv"
fileName = "./animation/s-CPCL_Animation_s2.csv"
writeFile(seedList, fileName)

#fileName = "./center/sCPCL_center_" +str(clusterNum)+ "_"+str(seedNum)+"_sigma"+str(sigma)+"_"+str(count)+"_"+str(seedCount)+".csv"
fileName = "./center/s-CPCL_center_s2.csv"
writeFile(center, fileName)

#fileName = "./rmse/sCPCL_RMSE_" +str(clusterNum)+ "_"+str(seedNum)+"_sigma"+str(sigma)+"_"+str(count)+"_"+str(seedCount)+".csv"
fileName = "./epoch/s-CPCL_RMSE_s2_"+str(seedNum)+".csv"
writeFile(RMSE, fileName)

fileName = "./cpu/s-CPCL_RMSE_s2_"+str(seedNum)+".csv"
writeFile(CMSE, fileName)

#fileName = './seed/sCPCL_seed_' +str(clusterNum)+ "_"+str(seedNum)+ "_sigma"+str(sigma)+"_"+str(count)+".csv"
#writeFile(initSeed, fileName)

line = "s-CPCL_RMSE_s2_"+str(seedNum)+": "+ str(distance/(len(seedPoints)*d))+"\n"
fileName = "result.log"
appendFile(line, fileName)

plt.show()






