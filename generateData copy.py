import numpy as np
import csv
import sys
from random import shuffle
import matplotlib
import matplotlib.pyplot as plt
import pandas

def readData(fileName):
    dataframe = pandas.read_csv(fileName,
                                engine='python', header=None)
    dataset = dataframe.values
    dataSet = dataset.astype('float32')

    return dataSet

def generateData():
    #fileName = "./data/CPCL_sudoData_Cluster" + str(clusterNum) + "_sigma"+str(sigma)+"_"+str(count)+".csv"
    fileName = "./data/s2_d:q!ata.csv"
    dataSet = readData(fileName)

    output = []
    x = []
    y = []
    for i in range(0, len(dataSet)):
        plt.plot(dataSet[i][0], dataSet[i][1], 'k,')


generateData()
plt.show()


