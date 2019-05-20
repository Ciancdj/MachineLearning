import numpy
import os

def img2Vector(Data):
    vec = numpy.zeros([1,1024])
    index = 0
    for i in range(32):
        line = Data.readline().split("\n")[0]
        for i in line:
            vec[0][index] = int(i)
            index += 1
    return vec

def readText(Path):
    file = open(Path,'r')
    for i in file.readlines():
        print(i, end="")
    file.close()
    file = open(Path, 'r')
    return img2Vector(file)

def readTrain(Path):
    list = os.listdir(Path)
    index = 0
    lent = list.__len__()
    vec = numpy.zeros([lent,1024])
    label = []
    for i in list:
        file = open(Path + "\\" + i)
        label.append(int(i[0]))
        vec[index] = img2Vector(file)
        index += 1
    return vec, label, lent

if __name__ == "__main__":
    K = 10
    Path = "..//Data//digits"
    trainingPath = Path + "//trainingDigits"
    testPath = Path + "//testDigits//3_2.txt"
    TestVec = readText(testPath)
    TrainVec,label, lent = readTrain(trainingPath)
    # KNN
    V = numpy.tile(TestVec,(lent,1)) - TrainVec
    V = V**2
    distances = V.sum(axis=1)
    distances = distances**0.5
    sortdistancesIndex = numpy.argsort(distances)
    labledis = {}
    for i in range(K):
        if label[sortdistancesIndex[sortdistancesIndex.__len__() - i - 1]] not in labledis.keys():
            labledis[label[sortdistancesIndex[i]]] = 0
        labledis[label[sortdistancesIndex[i]]] += 1
    label = set(label)
    maxIndex = 0
    maxNum = 0
    for i in label:
        # print("maxNum:{}, maxIndex:{}".format(maxNum, maxIndex))
        if i not in labledis.keys():
            continue
        if maxNum < labledis[i]:
            maxNum = labledis[i]
            maxIndex = i
    print(maxIndex)