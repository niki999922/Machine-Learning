import random
import math
import datetime
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import pandas as pd
import numpy as np

shuffleOn = False


class Data:
    def __init__(self, signs, category):
        self.signs = signs
        self.category = category


def copyData(data):
    return Data(np.copy(data.signs), data.category)


def copyDates(dates):
    newList = list()
    for el in dates:
        newList.append(copyData(el))
    return newList


def printData(dates):
    for el in dates:
        print(el.signs, '\tclass = ' + el.category.__str__())


def multiplyVectors(v1, v2):
    x_len = len(v1)
    res = 0
    for i in range(x_len):
        res += (v1[i] * v2[i])
    return res


def linearKernel(v1, v2, c):
    return multiplyVectors(v1, v2) + c


def polynomialKernel(v1, v2, c, alpha, d):
    return (multiplyVectors(v1, v2) * alpha + c) ** d


def gaussianKernel(x_i, x_j, sigma):
    x_len = len(x_i)
    t = 0
    for i in range(x_len):
        t += (x_i[i] - x_j[i]) ** 2
    return math.exp(-t / (2. * (sigma ** 2)))


def predict(i, coefs, b, dates):
    res = 0.0
    for j in range(len(dates)):
        res += dates[i].signs[j] * dates[j].category * coefs[j]
    res += b
    return res


def modernPredictForLinear(checkedDate, coefs, b, dates, c):
    res = b
    for i in range(len(dates)):
        res += (dates[i].category * coefs[i] * linearKernel(checkedDate.signs, dates[i].signs, c))
    return 1 if res > 0 else -1


def modernPredictForPolynomial(checkedDate, coefs, b, dates, c, alpha, d):
    res = 0.0
    for i in range(len(dates)):
        res += dates[i].category * coefs[i] * polynomialKernel(checkedDate.signs, dates[i].signs, c, alpha, d)
    res += b
    return 1 if res > 0 else -1


def modernPredictForGaussian(checkedDate, coefs, b, dates, sigma):
    res = 0.0
    for i in range(len(dates)):
        res += dates[i].category * coefs[i] * gaussianKernel(checkedDate.signs, dates[i].signs, sigma)
    res += b
    return 1 if res > 0 else -1


def svm(dates, c, amountIter=1000):
    eps = 1e-6
    orderArr = []
    resultAlpha = [0.0 for _ in range(len(dates[0].signs))]
    resultB = 0
    for i in range(len(dates)):
        orderArr.append(i)

    counter = 0
    while counter < amountIter:
        random.shuffle(orderArr)
        for i in range(len(orderArr)):
            e_i = predict(i, resultAlpha, resultB, dates) - dates[i].category
            if (dates[i].category * e_i < -eps and resultAlpha[i] < c) or (dates[i].category * e_i > eps and resultAlpha[i] > 0):
                if i == orderArr[i]:
                    continue
                j = orderArr[i]
                e_j = predict(j, resultAlpha, resultB, dates) - dates[j].category
                a_i_old = resultAlpha[i]
                a_j_old = resultAlpha[j]
                if dates[i].category != dates[j].category:
                    L = max(0.0, resultAlpha[j] - resultAlpha[i])
                    H = min(float(c), float(c) + resultAlpha[j] - resultAlpha[i])
                else:
                    L = max(0.0, resultAlpha[i] + resultAlpha[j] - float(c))
                    H = min(float(c), resultAlpha[i] + resultAlpha[j])
                if abs(L - H) < eps:
                    continue
                nu = 2 * dates[i].signs[j] - dates[i].signs[i] - dates[j].signs[j]
                if nu >= 0:
                    continue

                a_j = resultAlpha[j] - (dates[j].category * (e_i - e_j)) / nu
                if a_j > H:
                    a_j = H
                elif a_j < L:
                    a_j = L
                if abs(a_j - a_j_old) < eps:
                    continue
                resultAlpha[i] = resultAlpha[i] + dates[i].category * dates[j].category * (a_j_old - a_j)
                b1 = resultB - e_i - dates[i].category * (resultAlpha[i] - a_i_old) * dates[i].signs[i] - dates[j].category * (a_j - a_j_old) * dates[i].signs[j]
                b2 = resultB - e_j - dates[i].category * (resultAlpha[i] - a_i_old) * dates[i].signs[j] - dates[j].category * (a_j - a_j_old) * dates[j].signs[j]
                if 0 < resultAlpha[i] < c:
                    resultB = b1
                elif 0 < a_j < c:
                    resultB = b2
                else:
                    resultB = (b1 + b2) / 2
                resultAlpha[j] = a_j
        counter += 1
    for i in range(len(resultAlpha)):
        if resultAlpha[i] < 0:
            resultAlpha[i] = 0

    return resultAlpha, resultB


def readDates(fileName):
    def convertY(status):
        return 1 if status == 'P' else -1

    dataset = pd.read_csv(fileName)
    if shuffleOn:
        dataset = shuffle(dataset)
    dates = []
    xCord = []
    yCord = []
    for el in dataset.values:
        xCord.append(float(el[0]))
        yCord.append(float(el[1]))
        dates.append(Data(el[:-1], convertY(el[-1])))
    return dates, xCord, yCord


def creteTestCase(dates, n, kStep=1):
    newFirstList = list()
    newLastList = list()
    newTarget = list()

    startPos = n * kStep
    endPos = min((n + 1) * kStep, len(dates))

    for el in dates[:startPos]:
        newFirstList.append(copyData(el))
    for el in dates[startPos:endPos]:
        newTarget.append(copyData(el))
    for el in dates[endPos:]:
        newLastList.append(copyData(el))

    return (newFirstList + newLastList, newTarget)


# dates with features, return dates with kernel
def transformWithLinear(dates, c):
    tmpArr = [[0.0 for _ in range(len(dates))] for _ in range(len(dates))]
    for i in range(len(dates)):
        for j in range(len(dates)):
            tmpArr[i][j] = linearKernel(dates[i].signs, dates[j].signs, c)
    newDates = []
    for i in range(len(tmpArr)):
        newDates.append(Data(tmpArr[i], dates[i].category))
    return newDates


# dates with features, return dates with kernel
def transformWithPolynomial(dates, c, alpha, d):
    tmpArr = [[0.0 for _ in range(len(dates))] for _ in range(len(dates))]
    for i in range(len(dates)):
        for j in range(len(dates)):
            tmpArr[i][j] = polynomialKernel(dates[i].signs, dates[j].signs, c, alpha, d)
    newDates = []
    for i in range(len(tmpArr)):
        newDates.append(Data(tmpArr[i], dates[i].category))
    return newDates


# dates with features, return dates with kernel
def transformWithGaussian(dates, sigma):
    tmpArr = [[0.0 for _ in range(len(dates))] for _ in range(len(dates))]
    for i in range(len(dates)):
        for j in range(len(dates)):
            tmpArr[i][j] = gaussianKernel(dates[i].signs, dates[j].signs, sigma)
    newDates = []
    for i in range(len(tmpArr)):
        newDates.append(Data(tmpArr[i], dates[i].category))
    return newDates


# ~11 minutes chips stepSize=20
# ~54 minutes geyser stepSize=50
def findBestParametersForLinear(dates, stepSize=20):
    step = stepSize
    cLinArr = [0, 0.01, 0.1, 0.5, 1.5, 8.0, 30.0]
    cSWMArr = [0.1, 0.5, 1.0, 10.0, 25.0, 50.0, 100.0]

    amountSteps = len(cLinArr) * len(cSWMArr)
    progressBar = 0
    bestRes = {"cLin": -1, "cSVM": -1}
    bestAccuracy = 0
    for cLin in cLinArr:
        for cSVM in cSWMArr:
            print('\nPROGRESS_L ' + '{:.2%}'.format(progressBar / amountSteps).__str__(), flush=True)
            print(datetime.datetime.now(), flush=True)
            print('dates: cLIN', cLin, 'cSVM', cSVM, flush=True)
            nice = 0
            for i in range(len(dates)):
                if i * step > len(dates):
                    break
                print('i=' + i.__str__(), 'step=' + (i * step).__str__(), flush=True)
                trainDates, checkDates = creteTestCase(dates, i, kStep=step)
                trainDates2 = transformWithLinear(trainDates, cLin)
                alphas, b = svm(trainDates2, cSVM)
                for testEl in checkDates:
                    yPredicted = modernPredictForLinear(testEl, alphas, b, trainDates, cLin)
                    if yPredicted == testEl.category:
                        nice += 1

            accuracy = nice / len(dates)
            if bestAccuracy < accuracy:
                bestAccuracy = accuracy
                print("Found better accuracy", accuracy, flush=True)
                bestRes = {"cLin": cLin, "cSVM": cSVM}
            else:
                print("No change", accuracy, ", best", bestAccuracy, flush=True)
            progressBar += 1
    print('\nBest result was found cLin=' + bestRes["cLin"].__str__() + ' cSVM=' + bestRes["cSVM"].__str__(), flush=True)
    return bestRes


# for  chips ~42 minutes
# for  geysen ~212 minutes
def findBestParametersForPolynomial(dates, stepSize=20):
    step = stepSize
    # alphaPolArr = [0.1, 0.5, 1, 3, 10] for chips 212 minutes, more for other, let just [1]

    cPolArr = [0, 0.01, 0.5, 1.5, 8.0, 30.0]
    alphaPolArr = [1.0]
    dPolArr = [2, 3, 4, 5]
    cSWMArr = [0.01, 1.0, 7.0, 25.0, 100.0]

    amountSteps = len(cPolArr) * len(alphaPolArr) * len(dPolArr) * len(cSWMArr)
    progressBar = 0
    bestRes = {"cPol": -1, "alphaPol": -1, "dPol": -1, "cSVM": -1}
    bestAccuracy = 0
    for cPol in cPolArr:
        for alphaPol in alphaPolArr:
            for dPol in dPolArr:
                for cSVM in cSWMArr:
                    print('\nPROGRESS_P ' + '{:.2%}'.format(progressBar / amountSteps).__str__(), flush=True)
                    print(datetime.datetime.now(), flush=True)
                    print('dates: cPol', cPol, 'alphaPol', alphaPol, 'dPol', dPol, 'cSVM', cSVM, flush=True)
                    nice = 0
                    for i in range(len(dates)):
                        if i * step > len(dates):
                            break
                        print('i=' + i.__str__(), 'step=' + (i * step).__str__(), flush=True)
                        trainDates, checkDates = creteTestCase(dates, i, kStep=step)
                        trainDates2 = transformWithPolynomial(trainDates, cPol, alphaPol, dPol)
                        alphas, b = svm(trainDates2, cSVM)
                        for testEl in checkDates:
                            yPredicted = modernPredictForPolynomial(testEl, alphas, b, trainDates, cPol, alphaPol, dPol)
                            if yPredicted == testEl.category:
                                nice += 1
                    accuracy = nice / len(dates)
                    if bestAccuracy < accuracy:
                        bestAccuracy = accuracy
                        print("Found better accuracy", accuracy, flush=True)
                        bestRes = {"cPol": cPol, "alphaPol": alphaPol, "dPol": dPol, "cSVM": cSVM}
                    else:
                        print("No change", accuracy, ", best", bestAccuracy, flush=True)
                    progressBar += 1
    print('\nBest result was found cPol=' + bestRes["cPol"].__str__() + ' alphaPol=' + bestRes["alphaPol"].__str__() + ' dPol=' + bestRes["dPol"].__str__() + ' cSVM=' + bestRes["cSVM"].__str__(), flush=True)
    return bestRes


# for chips ~10 minutes
# for geysen ~23 minutes
def findBestParametersForGaussian(dates, stepSize=20):
    step = stepSize

    sigmaGauArr = [0.01, 0.1, 0.5, 1, 5, 25]
    cSWMArr = [0.01, 0.3, 1.0, 7.0, 25.0, 50.0, 100.0]

    amountSteps = len(sigmaGauArr) * len(cSWMArr)
    progressBar = 0
    bestRes = {"sigmaGau": -1, "cSVM": -1}
    bestAccuracy = 0
    for sigmaGau in sigmaGauArr:
        for cSVM in cSWMArr:
            print('\nPROGRESS_G ' + '{:.2%}'.format(progressBar / amountSteps).__str__(), flush=True)
            print(datetime.datetime.now(), flush=True)
            print('dates: sigmaGau', sigmaGau, 'cSVM', cSVM, flush=True)
            nice = 0
            for i in range(len(dates)):
                if i * step > len(dates):
                    break
                print('i=' + i.__str__(), 'step=' + (i * step).__str__(), flush=True)
                trainDates, checkDates = creteTestCase(dates, i, kStep=step)
                trainDates2 = transformWithGaussian(trainDates, sigmaGau)
                alphas, b = svm(trainDates2, cSVM)
                for testEl in checkDates:
                    yPredicted = modernPredictForGaussian(testEl, alphas, b, trainDates, sigmaGau)
                    if yPredicted == testEl.category:
                        nice += 1
            accuracy = nice / len(dates)
            if bestAccuracy < accuracy:
                bestAccuracy = accuracy
                print("Found better accuracy", accuracy, flush=True)
                bestRes = {"sigmaGau": sigmaGau, "cSVM": cSVM}
            else:
                print("No change", accuracy, ", best", bestAccuracy, flush=True)
            progressBar += 1
    print('\nBest result was found sigmaGau=' + bestRes["sigmaGau"].__str__() + ' cSVM=' + bestRes["cSVM"].__str__(), flush=True)
    return bestRes


def superPredictForLinear(field, dates, cLin, cSVM):
    trainDates = transformWithLinear(dates, cLin)
    alphas, b = svm(trainDates, cSVM)
    globalArr = []
    progressBar = 0
    for el in field:
        print('Progress ' + '{:.2%}'.format(progressBar / len(field)).__str__())
        predictY = modernPredictForLinear(Data(el, 1), alphas, b, dates, cLin)
        globalArr.append(predictY)
        progressBar += 1
    return globalArr


def superPredictForPolynomial(field, dates, cPol, alphaPol, dPol, cSVM):
    trainDates = transformWithPolynomial(dates, cPol, alphaPol, dPol)
    alphas, b = svm(trainDates, cSVM)
    globalArr = []
    progressBar = 0
    for el in field:
        print('Progress ' + '{:.2%}'.format(progressBar / len(field)).__str__())
        predictY = modernPredictForPolynomial(Data(el, 1), alphas, b, dates, cPol, alphaPol, dPol)
        globalArr.append(predictY)
        progressBar += 1
    return globalArr

def superPredictForGaussian(field, dates, sigmaGau, cSVM):
    trainDates = transformWithGaussian(dates, sigmaGau)
    alphas, b = svm(trainDates, cSVM)
    globalArr = []
    progressBar = 0
    for el in field:
        print('Progress ' + '{:.2%}'.format(progressBar / len(field)).__str__())
        predictY = modernPredictForGaussian(Data(el, 1), alphas, b, dates, sigmaGau)
        globalArr.append(predictY)
        progressBar += 1
    return globalArr


def printPlotsLinear(dates, xArr, yArr, cLin, cSVM):
    h = 0.1
    x_min, x_max = min(xArr) - 1, max(xArr) + 1
    y_min, y_max = min(yArr) - 1, max(yArr) + 1
    xxx, yyy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))

    field = np.c_[xxx.ravel(), yyy.ravel()]

    globalArr = superPredictForLinear(field, dates, cLin, cSVM)
    z = (np.array(globalArr)).reshape(xxx.shape)
    plt.contourf(xxx, yyy, z, alpha=1)

    elMinus = [[], []]
    elPlus = [[], []]
    for el in dates:
        if el.category == 1:
            elPlus[0].append(el.signs[0])
            elPlus[1].append(el.signs[1])
        else:
            elMinus[0].append(el.signs[0])
            elMinus[1].append(el.signs[1])

    plt.scatter(elMinus[0], elMinus[1], c='red')
    plt.scatter(elPlus[0], elPlus[1], c='green')
    plt.show()


def printPlotsPolynomial(dates, xArr, yArr, cPol, alphaPol, dPol, cSVM):
    h = 0.1
    x_min, x_max = min(xArr) - 1, max(xArr) + 1
    y_min, y_max = min(yArr) - 1, max(yArr) + 1
    xxx, yyy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))

    field = np.c_[xxx.ravel(), yyy.ravel()]

    globalArr = superPredictForPolynomial(field, dates, cPol, alphaPol, dPol, cSVM)
    z = (np.array(globalArr)).reshape(xxx.shape)
    plt.contourf(xxx, yyy, z, alpha=1)

    elMinus = [[], []]
    elPlus = [[], []]
    for el in dates:
        if el.category == 1:
            elPlus[0].append(el.signs[0])
            elPlus[1].append(el.signs[1])
        else:
            elMinus[0].append(el.signs[0])
            elMinus[1].append(el.signs[1])

    plt.scatter(elMinus[0], elMinus[1], c='red')
    plt.scatter(elPlus[0], elPlus[1], c='green')
    plt.show()


def printPlotsGaussian(dates, xArr, yArr, sigmaGau, cSVM):
    h = 0.1
    x_min, x_max = min(xArr) - 1, max(xArr) + 1
    y_min, y_max = min(yArr) - 1, max(yArr) + 1
    xxx, yyy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))

    field = np.c_[xxx.ravel(), yyy.ravel()]

    globalArr = superPredictForGaussian(field, dates, sigmaGau, cSVM)
    z = (np.array(globalArr)).reshape(xxx.shape)
    plt.contourf(xxx, yyy, z, alpha=1)

    elMinus = [[], []]
    elPlus = [[], []]
    for el in dates:
        if el.category == 1:
            elPlus[0].append(el.signs[0])
            elPlus[1].append(el.signs[1])
        else:
            elMinus[0].append(el.signs[0])
            elMinus[1].append(el.signs[1])

    plt.scatter(elMinus[0], elMinus[1], c='red')
    plt.scatter(elPlus[0], elPlus[1], c='green')
    plt.show()


if __name__ == '__main__':
    shuffleOn = False
    dates, xCord, yCord = readDates('chips.csv')
    # dates, xCord, yCord = readDates('geyser.csv')
    # printData(dates)

    # print for chips.csv
    # printPlotsLinear(dates, xCord, yCord, 1.5, 100)
    # printPlotsPolynomial(dates, xCord, yCord, 8, 1.0, 3, 7.0)
    # printPlotsGaussian(dates, xCord, yCord, 0.5, 50)

    # print for geyser.csv
    # printPlotsLinear(dates, xCord, yCord, 0, 25)
    # printPlotsPolynomial(dates, xCord, yCord, 0.01, 1, 2, 0.01)
    # printPlotsGaussian(dates, xCord, yCord, 5, 50)

    # linear
    # for chips
    # findBestParametersForLinear(dates)
    # for geyser
    # findBestParametersForLinear(dates, stepSize=50)

    # polynomial
    # for chips
    # findBestParametersForPolynomial(dates)
    # for geyser
    # findBestParametersForPolynomial(dates, stepSize=70)

    # gaussian
    # for chips
    # findBestParametersForGaussian(dates)
    # for geyser
    # findBestParametersForGaussian(dates, stepSize=50)
