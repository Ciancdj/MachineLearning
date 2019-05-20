#!/usr/bin/env python
# -*- coding:utf-8 -*-
import math
import numpy

def distance(x1,y1,x2,y2) :
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def KNN(trainData, type, textData, K) :
    s = []
    T = {}
    for i in range(len(trainData)):
        s.append(distance(trainData[i][0],trainData[i][1],textData[0],textData[1]))
    arg = numpy.argsort(s)
    for i in range(K):
        T[type[arg[i]]] = 0
    for i in range(K):
        T[type[arg[i]]] += 1
    max = 0
    maxType = None
    for type, index in T.items():
        if (max < index):
            max = index
            maxType = type
    return type

if __name__ == "__main__" :
    trainData = [
        [1,1],
        [1,0],
        [0,0],
        [10,2],
        [10,1],
    ]
    type = ['A','A','A','B','B']
    textData = [2,2]
    K = 2
    print(KNN(trainData,type,textData,K))