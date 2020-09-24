from enum import Enum
import pandas as pd
import numpy as np
import math
import subprocess as sp
import matplotlib.pyplot as plt
from IPython.utils.tests.test_wildcard import y

filename = 'dataset.csv'
superDistance = 'manhattan'
superKernel = 'triweight'
resultForFName = './resultForF.txt'
inputForFName = './inputForF.txt'
uniqClassesAmount = 3


class Data:
    def __init__(self, signs, category):
        self.signs = signs
        self.category = category


def copyData(data):
    return Data(np.copy(data.signs), data.category)


def printData(dates):
    for el in dates:
        print(el.signs, '\tclass = ' + el.category.__str__())


def readAsData(dataset):
    newList = []
    for row in dataset.values:
        newList.append(Data(row[:-1], int(row[-1] - 1)))
    return newList


def minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0]) - 1):
        value_min = dataset[:, i].min()
        value_max = dataset[:, i].max()
        minmax.append([value_min, value_max])
    return minmax


def normalize(dates, minMaxList):
    for el in dates:
        for i in range(len(el.signs)):
            el.signs[i] = (el.signs[i] - minMaxList[i][0]) / (minMaxList[i][1] - minMaxList[i][0])


class Distance(Enum):
    MANHATTAN = 'manhattan'
    EUCLIDEAN = 'euclidean'
    CHEBYSHEV = 'chebyshev'


class Kernel(Enum):
    UNIFORM = 'uniform'
    TRIANGULAR = 'triangular'
    EPANECHNIKOV = 'epanechnikov'
    QUARTIC = 'quartic'
    TRIWEIGHT = 'triweight'
    TRICUBE = 'tricube'
    GAUSSIAN = 'gaussian'
    COSINE = 'cosine'
    LOGISTIC = 'logistic'
    SIGMOID = 'sigmoid'


class Window(Enum):
    FIXED = 'fixed'
    VARIABLE = 'variable'


def calculateDistance(dist, v1, v2):
    if dist == Distance.MANHATTAN:
        result = 0
        for (x1, x2) in zip(v1, v2):
            result += abs(x2 - x1)
        return result
    elif dist == Distance.EUCLIDEAN:
        result = 0
        for (x1, x2) in zip(v1, v2):
            result += (x2 - x1) ** 2
        return math.sqrt(result)
    elif dist == Distance.CHEBYSHEV:
        result = 0
        for (x1, x2) in zip(v1, v2):
            result = max(result, abs(x2 - x1))
        return result


def calculateKernel(kernel, u):
    if kernel == Kernel.UNIFORM:
        if u < 1:
            return 1 / 2
        else:
            return 0
    elif kernel == Kernel.TRIANGULAR:
        if u < 1:
            return 1 - u
        else:
            return 0
    elif kernel == Kernel.EPANECHNIKOV:
        if u < 1:
            return (3 / 4) * (1 - (u ** 2))
        else:
            return 0
    elif kernel == Kernel.QUARTIC:
        if u < 1:
            return (15 / 16) * ((1 - (u ** 2)) ** 2)
        else:
            return 0
    elif kernel == Kernel.TRIWEIGHT:
        if u < 1:
            return (35 / 32) * ((1 - (u ** 2)) ** 3)
        else:
            return 0
    elif kernel == Kernel.TRICUBE:
        if u < 1:
            return (70 / 81) * ((1 - (u ** 3)) ** 3)
        else:
            return 0
    elif kernel == Kernel.GAUSSIAN:
        return (1 / (math.sqrt(2 * math.pi))) * math.exp((-1 / 2) * (u ** 2))
    elif kernel == Kernel.COSINE:
        if u < 1:
            return (math.pi / 4) * math.cos((math.pi * u) / 2)
        else:
            return 0
    elif kernel == Kernel.LOGISTIC:
        return 1 / (math.exp(u) + 2 + math.exp(-u))
    elif kernel == Kernel.SIGMOID:
        return (2 / math.pi) * (1 / (math.exp(u) + math.exp(-u)))


def calculateAverage2(dates, investigated):
    result = 0
    newDates = list(filter(lambda _data: _data.signs == investigated, dates))
    for data in newDates:
        result += data.category
    return result / len(newDates)


def calculateAverage(dates):
    result = 0
    for data in dates:
        result += data.category
    return result / len(dates)


def knn(investigated, dates, distance, kernel, window, windowsSize):
    def setDistance(el):
        el.distance = calculateDistance(distance, el.signs, investigated)
        return el

    maxDistance = h = windowsSize
    dates = list(map(setDistance, dates))
    dates = list(sorted(dates, key=lambda _data: _data.distance))

    if window == Window.VARIABLE:
        maxDistance = dates[h].distance

    if maxDistance == 0:
        if len(list(filter(lambda _data: _data.signs == investigated, dates))) != 0:
            return calculateAverage2(dates, investigated)
        else:
            return calculateAverage(dates)
    else:
        x = 0
        y = 0
        for data in dates:
            kerVal = calculateKernel(kernel, data.distance / maxDistance)
            x += data.category * kerVal
            y += kerVal
        if y == 0:
            return calculateAverage(dates)
        else:
            return x / y


def creteTestCase(dates, n):
    newFirstList = list()
    newLastList = list()
    for el in dates[:n]:
        newFirstList.append(copyData(el))
    for el in dates[(n + 1):]:
        newLastList.append(copyData(el))
    newTarget = copyData(dates[n])

    return (newFirstList + newLastList, newTarget)


def investigatedCategory(dates, n):
    return dates[n].category


def creteDates(dates):
    newList = list()
    for el in dates:
        newList.append(copyData(el))
    return newList


def createTestCaseForOneHot(dates, needClass):
    newDates = creteDates(dates)
    for el in newDates:
        if (el.category == needClass):
            el.category = 1
        else:
            el.category = 0
    return newDates


def mathRound(num):
    return int(num + 0.5)


def createEmptyMatrix(n):
    return [[0 for _ in range(n)] for _ in range(n)]


def getMatrixErrorNative(window, dates, dist, kernel, windowH):
    matrix = createEmptyMatrix(len(dates[0].signs))
    for i in range(len(dates)):
        testDates, investigated = creteTestCase(dates, i)
        res = knn(investigated.signs, testDates, Distance(dist), Kernel(kernel), Window(window), windowH)
        resultClass = mathRound(res)
        matrix[investigated.category][resultClass] += 1
    return matrix


def getMatrixErrorOneHot(window, dates, dist, kernel, windowH):
    matrix = createEmptyMatrix(len(dates[0].signs))
    for i in range(len(dates)):
        bestResult = -1
        bestClassId = -1
        for classId in range(uniqClassesAmount):
            upgradedDates = createTestCaseForOneHot(dates, classId)
            testDates, investigated = creteTestCase(upgradedDates, i)
            res = knn(investigated.signs, testDates, Distance(dist), Kernel(kernel), Window(window), windowH)
            if (bestResult < res):
                bestResult = res
                bestClassId = classId
        resultClass = bestClassId
        matrix[investigatedCategory(dates, i)][resultClass] += 1
    return matrix


def computeF(matrix):
    f = open(inputForFName, "a")
    f.truncate(0)
    f.write(len(matrix).__str__() + '\n')
    f.write('\n'.join([' '.join(item.__str__() for item in row) for row in matrix]))
    f.close()
    sp.call("./countF.sh")
    with open(resultForFName) as f:
        macro = float(f.readline())
        micro = float(f.readline())
    return macro, micro


def findOptimalWindowSize(window, dates, maxWindowH, maxDistanse, step=1):
    x = []
    y1 = []
    y2 = []
    if (window == Window.VARIABLE):
        print('Native Variable:')
        for i in range(maxWindowH - 1):
            print('i = ' + i.__str__())
            matrixError = getMatrixErrorNative(window, dates, Distance(superDistance), Kernel(superKernel), i)
            macro, micro = computeF(matrixError)
            x.append(i)
            y1.append(macro)
            y2.append(micro)
    else:
        print('Native Fixed:')
        i = 0.01
        while i <= maxDistanse:
            print('i = ' + i.__str__())
            matrixError = getMatrixErrorNative(window, dates, Distance(superDistance), Kernel(superKernel), i)
            macro, micro = computeF(matrixError)
            x.append(i)
            y1.append(macro)
            y2.append(micro)
            i += step

    return [x, y1, y2]


def findOptimalWindowSizeOneHot(window, dates, maxWindowH, maxDistanse, step=1):
    x = []
    y1 = []
    y2 = []
    if (window == Window.VARIABLE):
        print('OneHot Variable:')
        for i in range(maxWindowH - 1):
            print('i = ' + i.__str__())
            matrixError = getMatrixErrorOneHot(window, dates, Distance(superDistance), Kernel(superKernel), i)
            macro, micro = computeF(matrixError)
            x.append(i)
            y1.append(macro)
            y2.append(micro)
    else:
        print('OneHot Fixed:')
        i = 0.01
        while i <= maxDistanse:
            print('i = ' + i.__str__())
            matrixError = getMatrixErrorOneHot(window, dates, Distance(superDistance), Kernel(superKernel), i)
            macro, micro = computeF(matrixError)
            x.append(i)
            y1.append(macro)
            y2.append(micro)
            i += step
    return [x, y1, y2]


def findOptimalKernelDistance(window, dates, fixWindowH):
    kernels = ['uniform', 'triangular', 'epanechnikov', 'quartic', 'triweight', 'tricube', 'gaussian', 'cosine', 'logistic', 'sigmoid']
    distances = ['manhattan', 'euclidean', 'chebyshev']
    maxScore = 0
    bestCase = ('=(', "=(")
    for kernel in kernels:
        for dist in distances:
            matrixError = getMatrixErrorNative(window, dates, dist, kernel, fixWindowH)
            score = np.trace(matrixError) / len(dates)
            if (score > maxScore):
                maxScore = score
                bestCase = kernel, dist
    print(maxScore)
    return bestCase

def findOptimalKernelDistanceOneHot(window, dates, fixWindowH):
    kernels = ['uniform', 'triangular', 'epanechnikov', 'quartic', 'triweight', 'tricube', 'gaussian', 'cosine', 'logistic', 'sigmoid']
    distances = ['manhattan', 'euclidean', 'chebyshev']
    maxScore = 0
    bestCase = ('=(', "=(")
    for kernel in kernels:
        for dist in distances:
            matrixError = getMatrixErrorOneHot(window, dates, dist, kernel, fixWindowH)
            score = np.trace(matrixError) / len(dates)
            if (score > maxScore):
                maxScore = score
                bestCase = kernel, dist
    print(maxScore)
    return bestCase



def tryFindBestDisKer(dates, step=1):
    nIter = 15
    maxSizeWindow = len(dates)
    maxDistanse = len(dates[0].signs)
    dictKernel = {}
    dictKDist = {}
    i = 0.01
    counter = 1
    print('Searching for Native Variable')
    stepCustom = maxDistance / nIter
    while i <= maxDistanse:
        print('i = ' + i.__str__())
        kernel, dist = findOptimalKernelDistance(Window.FIXED, dates, i)
        if (dictKernel.get(kernel) != None):
            dictKernel.update([(kernel, dictKernel.get(kernel) + 1)])
        else:
            dictKernel.update([(kernel, 1)])

        if (dictKDist.get(dist) != None):
            dictKDist.update([(dist, dictKDist.get(dist) + 1)])
        else:
            dictKDist.update([(dist, 1)])
        i += stepCustom
        # i += step * (counter ** 2)
        # counter += 1

    best_kernel = max(dictKernel, key=dictKernel.get)
    best_dist = max(dictKDist, key=dictKDist.get)
    print('best kernel for Fixed Native: \'' + best_kernel + '\'')
    print('best distance for Fixed Native: \'' + best_dist + '\'')

    dictKernel = {}
    dictKDist = {}
    print('Searching for Native Variable')
    i = 1
    while (i < maxSizeWindow - 1):
        print('i = ' + i.__str__())
        kernel, dist = findOptimalKernelDistance(Window.VARIABLE, dates, i)
        if (dictKernel.get(kernel) != None):
            dictKernel.update([(kernel, dictKernel.get(kernel) + 1)])
        else:
            dictKernel.update([(kernel, 1)])

        if (dictKDist.get(dist) != None):
            dictKDist.update([(dist, dictKDist.get(dist) + 1)])
        else:
            dictKDist.update([(dist, 1)])
        i = i + int(i * 2/3) + 4

    best_kernel = max(dictKernel, key=dictKernel.get)
    best_dist = max(dictKDist, key=dictKDist.get)
    print('best kernel for Variable Native: \'' + best_kernel + '\'')
    print('best distance for Variable Native: \'' + best_dist + '\'')

    dictKernel = {}
    dictKDist = {}
    i = 0.01
    counter = 1
    print('Searching for OneHot Fixed')
    while i <= maxDistanse:
        print('i = ' + i.__str__())
        kernel, dist = findOptimalKernelDistanceOneHot(Window.FIXED, dates, i)
        if (dictKernel.get(kernel) != None):
            dictKernel.update([(kernel, dictKernel.get(kernel) + 1)])
        else:
            dictKernel.update([(kernel, 1)])

        if (dictKDist.get(dist) != None):
            dictKDist.update([(dist, dictKDist.get(dist) + 1)])
        else:
            dictKDist.update([(dist, 1)])
        i += stepCustom
        # i += step * (counter ** 2)
        # counter += 1

    best_kernel = max(dictKernel, key=dictKernel.get)
    best_dist = max(dictKDist, key=dictKDist.get)
    print('best kernel for Fixed OneHot: \'' + best_kernel + '\'')
    print('best distance for Fixed OneHot: \'' + best_dist + '\'')


    dictKernel = {}
    dictKDist = {}
    print('Searching for OneHot Variable')
    i = 1
    while (i < maxSizeWindow - 1):
        print('i = ' + i.__str__())
        kernel, dist = findOptimalKernelDistanceOneHot(Window.VARIABLE, dates, i)
        if (dictKernel.get(kernel) != None):
            dictKernel.update([(kernel, dictKernel.get(kernel) + 1)])
        else:
            dictKernel.update([(kernel, 1)])

        if (dictKDist.get(dist) != None):
            dictKDist.update([(dist, dictKDist.get(dist) + 1)])
        else:
            dictKDist.update([(dist, 1)])
        i = i + int(i * 2/3) + 4

    best_kernel = max(dictKernel, key=dictKernel.get)
    best_dist = max(dictKDist, key=dictKDist.get)
    print('best kernel for Variable OneHot: \'' + best_kernel + '\'')
    print('best distance for Variable OneHot: \'' + best_dist + '\'')


    # print('fixed 0.2: ' + findOptimalKernelDistance(Window.FIXED, dates, 0.2).__str__())
    # print('fixed 1: ' + findOptimalKernelDistance(Window.FIXED, dates, 1).__str__())
    # print('fixed 2.5: ' + findOptimalKernelDistance(Window.FIXED, dates, 2.5).__str__())
    # print('fixed 5: ' + findOptimalKernelDistance(Window.FIXED, dates, 5).__str__())
    # print('fixed 10: ' + findOptimalKernelDistance(Window.FIXED, dates, 10).__str__())
    # print('Variable 0: ' + findOptimalKernelDistance(Window.VARIABLE, dates, 0).__str__())
    # print('Variable 10: ' + findOptimalKernelDistance(Window.VARIABLE, dates, 10).__str__())
    # print('Variable //3: ' + findOptimalKernelDistance(Window.VARIABLE, dates, maxSizeWindow // 3).__str__())
    # print('Variable //2: ' + findOptimalKernelDistance(Window.VARIABLE, dates, maxSizeWindow // 2).__str__())
    # print('Variable max: ' + findOptimalKernelDistance(Window.VARIABLE, dates, maxSizeWindow - 2).__str__())


def plotForNative(folder, prefix, window, dates, maxWindowH, maxDistanse, step=1):
    (n_x, n_y1, n_y2) = findOptimalWindowSize(window, dates, maxWindowH, maxDistanse, step=step)

    fileName = prefix + window.name.__str__() + '_' + superDistance + '_' + superKernel + '_Native'
    plt.title('Native', fontsize=20)
    plt.plot(n_x, n_y1)
    plt.plot(n_x, n_y2)
    plt.xlabel('h')
    plt.ylabel('F')
    plt.legend(('Macro', 'Micro'),
               loc='upper right')
    plt.savefig(folder + '/' + fileName + '.png')
    plt.show()


def plotForOneHot(folder, prefix, window, dates, maxWindowH, maxDistanse, step=1):
    (n_x, n_y1, n_y2) = findOptimalWindowSizeOneHot(window, dates, maxWindowH, maxDistanse, step=step)

    fileName = prefix + window.name.__str__() + '_' + superDistance + '_' + superKernel + '_OneHot'
    plt.title('OneHot', fontsize=20)
    plt.plot(n_x, n_y1)
    plt.plot(n_x, n_y2)
    plt.xlabel('h')
    plt.ylabel('F')
    plt.legend(('Macro', 'Micro'),
               loc='upper right')
    plt.savefig(folder + '/' + fileName + '.png')
    plt.show()


def plotForNativeOneHotFixedWindow(folder, prefix, dates, maxWindowH, maxDistanse, step=1):
    (n_x, n_y1, n_y2) = findOptimalWindowSize(Window.FIXED, dates, maxWindowH, maxDistanse, step=step)
    (on_x, on_y1, on_y2) = findOptimalWindowSizeOneHot(Window.FIXED, dates, maxWindowH, maxDistanse, step=step)

    fileName = prefix + 'FIXED_' + superDistance + '_' + superKernel + '_Macro_NativeVSOneHot'
    plt.title('Macro', fontsize=20)
    plt.plot(n_x, n_y1)
    plt.plot(n_x, on_y1)
    plt.xlabel('h')
    plt.ylabel('F')
    plt.legend(('Native', 'OneHot'),
               loc='upper right')
    plt.savefig(folder + '/' + fileName + '.png')
    plt.show()

    fileName = prefix + 'FIXED_' + superDistance + '_' + superKernel + '_Micro_NativeVSOneHot'
    plt.title('Micro', fontsize=20)
    plt.plot(n_x, n_y2)
    plt.plot(n_x, on_y2)
    plt.xlabel('h')
    plt.ylabel('F')
    plt.legend(('Native', 'OneHot'),
               loc='upper right')
    plt.savefig(folder + '/' + fileName + '.png')
    plt.show()


def plotForNativeOneHotVariableWindow(folder, prefix, dates, maxWindowH, maxDistanse, step=1):
    (n_x, n_y1, n_y2) = findOptimalWindowSize(Window.VARIABLE, dates, maxWindowH, maxDistanse, step=step)
    (on_x, on_y1, on_y2) = findOptimalWindowSizeOneHot(Window.VARIABLE, dates, maxWindowH, maxDistanse, step=step)

    fileName = prefix + 'VARIABLE_' + superDistance + '_' + superKernel + '_Macro_NativeVSOneHot'
    plt.title('Macro', fontsize=20)
    plt.plot(n_x, n_y1)
    plt.plot(n_x, on_y1)
    plt.xlabel('h')
    plt.ylabel('F')
    plt.legend(('Native', 'OneHot'),
               loc='upper right')
    plt.savefig(folder + '/' + fileName + '.png')
    plt.show()

    fileName = prefix + 'VARIABLE_' + superDistance + '_' + superKernel + '_Micro_NativeVSOneHot'
    plt.title('Micro', fontsize=20)
    plt.plot(n_x, n_y2)
    plt.plot(n_x, on_y2)
    plt.xlabel('h')
    plt.ylabel('F')
    plt.legend(('Native', 'OneHot'),
               loc='upper right')
    plt.savefig(folder + '/' + fileName + '.png')
    plt.show()


if __name__ == '__main__':
    dataset = pd.read_csv(filename)
    # dataset = dataset.head(60)  # limit of dataset
    dates = readAsData(dataset)
    normalize(dates, minmax(dataset.values))
    maxDistance = len(dates[0].signs)

    # tryFindBestDisKer(dates, step=0.1)
    maxSizeWindow = len(dates)
    # (n_x, n_y1, n_y2) = findOptimalWindowSize(Window.VARIABLE, dates, maxSizeWindow, maxDistance)
    # (oh_x, oh_y1, oh_y2) = findOptimalWindowSizeOneHot(Window.VARIABLE, dates, maxSizeWindow)

    # (n_x, n_y1, n_y2) = findOptimalWindowSize(Window.FIXED, dates, maxSizeWindow, maxDistance, step=3)
    folder = 'test'
    plotForNative(folder, '', Window.FIXED, dates, maxSizeWindow, maxDistance, step=0.01)
    plotForNative(folder, '', Window.VARIABLE, dates, maxSizeWindow, maxDistance)
    plotForOneHot(folder, '', Window.FIXED, dates, maxSizeWindow, maxDistance, step=0.01)
    plotForOneHot(folder, '', Window.VARIABLE, dates, maxSizeWindow, maxSizeWindow)
    plotForNativeOneHotFixedWindow(folder, '', dates, maxSizeWindow, maxDistance, step=0.01)
    plotForNativeOneHotVariableWindow(folder, '', dates, maxSizeWindow, maxSizeWindow)
