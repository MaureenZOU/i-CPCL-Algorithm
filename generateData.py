import numpy as np
import csv
import sys
from random import shuffle
import matplotlib
import matplotlib.pyplot as plt

def generateData(n, sigma):
    N = 500  ##number of data points in one cluster

    centerX = []
    centerY = []

    for i in range(0, n):
        centerX.append(1 + 5 * i)
        centerY.append(1 + 5 * i)

    shuffle(centerX)
    shuffle(centerY)


    x = []
    y = []

    centroid = [[] for i in range(0, len(centerX))]
    print(centerX)
    print(centerY)


    for i, value in enumerate(centerX):
        array=np.random.normal(0, sigma, N) + value
        x=x+array.tolist()
        centroid[i].append(value)

    for i, value in enumerate(centerY):
        array=np.random.normal(0, sigma, N) + value
        y=y+array.tolist()
        centroid[i].append(value)

    print(centroid)

    ##draw data points
    plt.plot(x, y, 'k,')

    output = []

    for i in range(0, len(x)):
        output.append(np.array([x[i], y[i]]))

    return output, centroid

def writeFile(writeMatrix, fileName):
    with open(fileName, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in writeMatrix:
            spamwriter.writerow(row)

sigma = 2
clusterNum = 3
count = 1
dataSet, centroid = generateData(clusterNum, sigma)  ##cluster points

fileName = "./data/CPCL_sudoData_Cluster"+str(clusterNum)+"_sigma"+str(sigma)+"_"+str(count)+".csv"
writeFile(dataSet, fileName)

fileName = "./centroid/CPCL_centroid_Cluster"+str(clusterNum)+"_sigma"+str(sigma)+"_"+str(count)+".csv"
writeFile(centroid, fileName)

plt.show()


