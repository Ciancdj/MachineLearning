import numpy

def pred_Load(Path):
    file = open(Path,'r')
    lineNum = 0
    for i in file.readlines():
        lineNum += 1
    file.close()
    return lineNum

def read(Path):
    lineNum = pred_Load(Path)
    TypeData = []
    Data = numpy.zeros([lineNum, 3], dtype=float)
    file = open(Path, 'r')
    for index in range(lineNum):
        line = file.readline().split("\t")
        TypeData.append(line[-1].split("\n")[0])
        Data[index][0], Data[index][1], Data[index][2] = float(line[0]), float(line[1]), float(line[2])
    file.close()
    maxData = Data.max(axis=0)
    for i in range(len(maxData)):
        Data[i] = Data[i] / maxData[i]
    return Data, lineNum, TypeData

if __name__ == "__main__":
    Path = "..//Data//datingTestSet.txt"
    Data, lineNum, TypeData = read(Path)
    Type = set(TypeData)
    Percentage = 0.75
    K = 20
    limitNum = int(lineNum * Percentage)
    TrainData = numpy.zeros([limitNum, 3], dtype=float)
    TestData = numpy.zeros([lineNum - limitNum, 3], dtype=float)
    Error = 0
    for index in range(lineNum):
        if (index >= limitNum):
            TestData[index - limitNum] = Data[index]
        else:
            TrainData[index] = Data[index]
    index = 0
    for line in TestData:
        temp = numpy.tile(line,(TrainData.shape[0],1)) - TrainData
        temp = temp ** 2
        distance = temp.sum(axis=1)
        distance = distance ** 0.5
        sortdistancesIndex = numpy.argsort(distance)
        labledis = {}
        for i in range(K):
            if TypeData[limitNum + index] not in labledis.keys():
                labledis[TypeData[limitNum + index]] = 0
            labledis[TypeData[limitNum + index]] += 1
        maxType = 0
        maxNum = 0
        for i in Type:
            # print("maxNum:{}, maxIndex:{}".format(maxNum, maxIndex))
            if i not in labledis.keys():
                continue
            if maxNum < labledis[i]:
                maxNum = labledis[i]
                maxType = i
        if TypeData[limitNum + index] != maxType:
            Error += 1
        print("Data:{}, predict:{}".format(TypeData[limitNum + index], maxType))
        index += 1
    print("错误率为：{}".format(Error/lineNum))