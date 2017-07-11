import matplotlib.pyplot as plt
import numpy as np
import csv

def read(fileName):
    dataset = []
    with open(fileName) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            line = []
            for data in row:
                line.append(float(data))
            dataset.append(line)

    return dataset


fileName = './rmse/CPCL_RMSE_3_7_sigma2_0_1.csv'
dataset = read(fileName)

x1 = []
y1 = []

for data in dataset:
    x1.append(data[0])
    y1.append(data[1])

# fileName = './epoch/CPCL_RMSE_d1_15.csv'
# dataset = read(fileName)

# x1 = []
# y1 = []

# for data in dataset:
# 	x1.append(data[0])
# 	y1.append(data[1])

# fileName = './epoch/CPCL_RMSE_d1_20.csv'
# dataset = read(fileName)

# x2 = []
# y2 = []

# for data in dataset:
#     x2.append(data[0])
#     y2.append(data[1])

# fileName = './epoch/s-CPCL_RMSE_d1_30.csv'
# dataset = read(fileName)

# x3 = []
# y3 = []

# for data in dataset:
#     x3.append(data[0])
#     y3.append(data[1])

# fileName = './epoch/l1-CPCL_RMSE_d1_30.csv'
# dataset = read(fileName)

# x4 = []
# y4 = []

# for data in dataset:
#     x4.append(data[0])
#     y4.append(data[1])

plt.plot(x1, y1, label = 'CPCL')
# plt.plot(x2, y2, '--', label = 'i-CPCL')
# plt.plot(x3[0:110], y3[0:110], ':', label = 's-CPCL')
# plt.plot(x4[0:110], y4[0:110], '-.', label = 'l1-CPCL')

plt.legend(bbox_to_anchor=(0.95, 0.95), loc=1, borderaxespad=0.)

plt.xlabel('Epoch')
plt.ylabel('Converging Step(e)')
plt.title('Cluster# = 3, seed# = 7', fontsize = 25)
plt.grid(True)
#plt.savefig("test.png")
plt.show()
