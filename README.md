# An improved Cooperative and Penalized Competitive Learning Algorithm for Data Clustering
> This work was introduced in the summer research program supervised by Prof. Cheung Yiu Ming hosted by Hong Kong Baptist University after I finish my first year study in computer science department.

The i-CPCL algorithm(Improved Cooperative and Penalized Competitive Learning Algorithm for Data Clustering) is base on the previous work done by Dr.Hong Jia (CPCL algorithm). It is based on K-means algorithm, however, it could find the cluster centroid and cluster number without the preknowledge of the data distribution. However, for the previous method, although it could find the cluster centroid accurately very often, however, after the seed points get into the same cluster, the converging speed is quite slow. In our method, we find a way to detect the intra-cluster seed points and help them to converge.

## Installation

Python with regular pip install packages

## Usage

i-CPCL.py The improved CPCL algorithm

l1-CPCL.py CPCL algorithm with dynamic learning rate

s-CPCL.py CPCL algorithm with signed network approach

CPCL.py Original CPCL algorithm

show_***.py Show the animation of the seed points

```sh
python i-CPCL.py
python show_iCPCL.py
```

## Experiment Result

Copare of converging speed of S dataset.

![alt tag](https://raw.githubusercontent.com/MaureenZOU/i-CPCL-Algorithm/master/sample.png)



CPU

![alt tag](https://raw.githubusercontent.com/MaureenZOU/i-CPCL-Algorithm/master/cpu.png)

Epoch

![alt tag](https://raw.githubusercontent.com/MaureenZOU/i-CPCL-Algorithm/master/epoch.png)

Comparison of i-CPCL, l-CPCL and s-CPCL algorithm with 10 Gaussian clusters.

![alt tag](https://raw.githubusercontent.com/MaureenZOU/i-CPCL-Algorithm/master/compare.png)
