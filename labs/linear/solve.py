import math
import random
import os
from typing import List, Any

import numpy as np
import matplotlib.pyplot as plt
from geneticalgorithm import geneticalgorithm as ga
from numpy import array
from numpy.linalg import svd
from numpy import zeros
from numpy import diag

class Data:
    def __init__(self, signs, category):
        self.signs = signs
        self.category = category


def printData(dates):
    for el in dates:
        print(el.signs, '\tclass = ' + el.category.__str__())


def minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0]) - 1):
        value_min = dataset[0][i]
        value_max = dataset[0][i]
        for j in range(len(dataset) - 1):
            value_min = min(dataset[j + 1][i], value_min)
            value_max = max(dataset[j + 1][i], value_max)
        minmax.append([value_min, value_max])
    value_min = dataset[0][-1]
    value_max = dataset[0][-1]
    for j in range(len(dataset) - 1):
        value_min = min(dataset[j + 1][-1], value_min)
        value_max = max(dataset[j + 1][-1], value_max)
    minmax.append([value_min, value_max])
    return minmax


def normalize(dates, minMaxList):
    for el in dates:
        for i in range(len(el.signs)):
            if (minMaxList[i][0] == minMaxList[i][1]):
                el.signs[i] = 0
            else:
                el.signs[i] = (el.signs[i] - minMaxList[i][0]) / (minMaxList[i][1] - minMaxList[i][0])
        el.category = (el.category - minMaxList[len(el.signs)][0]) / (minMaxList[len(el.signs)][1] - minMaxList[len(el.signs)][0])


def predict(dateSigns, coefficients):
    y = coefficients[0]
    for i in range(len(dateSigns)):
        y += coefficients[i + 1] * dateSigns[i]
    return y


def predict_gen(dateSigns, coefficients):
    y = 0
    for i in range(len(coefficients)):
        y += coefficients[i] * dateSigns[i]
    return y


def smape(expYResult, predictYResult):
    n = len(expYResult)
    sum = 0
    for i in range(n):
        sum += abs(expYResult[i] - predictYResult[i]) / (abs(expYResult[i]) + abs(predictYResult[i]))
    return sum / n


def evalError(date, coef):
    return 2 * (predict(date.signs, coef) - date.category)


def step(dates, coef, iter):
    randomDate = dates[random.randint(0, len(dates) - 1)]
    error = evalError(randomDate, coef)
    dQ = [0.0 for _ in range(len(coef))]
    dQ[0] = error
    for i in range(len(coef) - 1):
        dQ[i + 1] = error * randomDate.signs[i]

    d = predict(randomDate.signs, dQ)
    t = 0.003
    if d != 0:
        t = (predict(randomDate.signs, coef) - randomDate.category) / d
    else:
        t = 0
    nu = 0.1 / iter
    for i in range(len(coef)):
        # coef[i] -= 0.1 * nu * dQ[i]
        coef[i] -= 0.003 * t * dQ[i]


def dist(v1, v2):
    sum = 0
    for i in range(len(v1)):
        sum += abs(v1[i] - v2[i])
    return math.sqrt(sum)


def coefficients_sgd(dates, maxIter, checkDates, dataset, mm, withNorm=False):
    def randomCoef(size):
        return (random.random() / size) - (1 / (2 * size))

    def normCoef(co, mm):
        for i in range(len(co) - 1):
            if mm[i][1] != mm[i][0]:
                co[i + 1] = co[i + 1] / (mm[i][1] - mm[i][0])
            else:
                co[i + 1] = 0
        return co

    coef = [randomCoef(len(dates[0].signs)) for _ in range(len(dates[0].signs) + 1)]
    # coefOld = [1 for _ in range(len(dates[0].signs) + 1)]
    counter = 1
    inp = list()
    res = list()
    # eps = 0.005
    while counter < maxIter + 1:
        print('Iteration ' + (counter).__str__() + ' of ' + (maxIter - 1).__str__())
        # while dist(coef, coefOld) > eps:
        #     coefOld = list(coef)
        step(dates, coef, counter)
        counter += 1
        inp.append(counter)
        normcoef = normCoef(list(coef), mm)
        # smape = evalSMAPE(normcoef, copyDates(checkDates), dataset, withNorm)
        smape = evalNRMSE(coef, copyDates(checkDates), dataset, withNorm)
        res.append(smape)
    return coef, inp, res


def testOnSGD(filePath, maxIter, checkDates2, dataset2, fileName, withNorm=False):
    file = open(filePath, 'r')
    file.readline()
    n = int(file.readline())

    teachDates = []
    dataset = list()
    for _ in range(n):
        tmp = list(map(int, file.readline().split()))
        dataset.append(list(tmp))
        teachDates.append(Data(tmp[:-1], tmp.pop()))
    mmTeach = minmax(dataset)
    normalize(teachDates, mmTeach)

    n = int(file.readline())
    # checkDates = []
    dataset = list()
    for _ in range(n):
        tmp = list(map(int, file.readline().split()))
        dataset.append(list(tmp))
        # checkDates.append(Data(tmp[:-1], tmp.pop()))
    mm = minmax(dataset)
    # normalize(checkDates, mm)

    coef, inp, res = coefficients_sgd(teachDates, maxIter, checkDates2, dataset2, mm, withNorm)
    printPlot(fileName, type, inp, res, folder='nHren', prefix='smape_')  # nrmse smape

    # for i in range(len(coef) - 1):
    #     if mmTeach[i][1] != mmTeach[i][0]:
    #         coef[i + 1] = coef[i + 1] / (mmTeach[i][1] - mmTeach[i][0])
    #     else:
    #         coef[i + 1] = 0

    # predictYResult = list()
    # expYResult = list()
    # for date in checkDates:
    #     predictYResult.append(predict(date.signs, coef))
    #     expYResult.append(date.category)
    #
    # smapeRes = smape(expYResult, predictYResult)
    # print('Smape = %.5f' % smapeRes)
    # print(coef)
    return coef


def getInverseModel2(data, y):
    y_np = np.array(y)
    F = np.array(data)
    U, s, VT = svd(F)
    d = 1.0 / s
    D = zeros(F.shape)
    D[:F.shape[1], :F.shape[1]] = diag(d)
    B = VT.T.dot(D.T).dot(U.T)
    # F_t = np.transpose(F)
    # tau = 0.1
    # m = len(data[0])
    # to_inv = (F_t @ F) + tau * np.eye(m, 1)
    # model_t = np.linalg.inv(to_inv) @ F_t @ y_np
    # model_t = np.linalg.pinv(F) @ y_np
    model_t = B @ y_np
    return model_t.T


def getInverseModel(data, y):
   y_np = np.array(y)
   F = np.array(data)
   # F_t = np.transpose(F)
   # tau = 0.1
   # m = len(data[0])
   # to_inv = (F_t @ F) + tau * np.eye(m, 1)
   # model_t = np.linalg.inv(to_inv) @ F_t @ y_np
   model_t = np.linalg.pinv(F) @ y_np
   return model_t.T


genetic_inp_global = list()
genetic_res_global = list()
genetic_res_iter = 1

def testOnGenetic(teachDates, teachY, checkDates2, dataset2, iterAr):
    copeD = list()
    copeTeachY = list()
    for el in checkDates2:
        el.signs = [1] + el.signs
    for el in teachDates:
        el.signs = [1] + el.signs

    for el in dataset2:
        copeD.append([1] + el)

    for el in teachDates:
        copeTeachY.append(el.signs + [el.category])
    global genetic_inp_global, genetic_res_global, genetic_res_iter
    genetic_inp_global = list()
    genetic_res_global = list()
    genetic_res_iter = 1

    def f(X):
        global genetic_inp_global, genetic_res_global, genetic_res_iter
        smape = evalSMAPE_gen(X, teachDates, copeTeachY)
        genetic_inp_global.append(genetic_res_iter)
        genetic_res_global.append(smape)
        genetic_res_iter += 1
        return smape
        

    ln = len(checkDates2[0].signs) + 1
    varbound = np.array([[-5, 5]] * ln)


    inp = list()
    res = list()

    for iter in iterAr:
        inp.append(iter)
        print('iter ' + iter.__str__() + ' of ' + iterAr[-1].__str__())
        algorithm_param = {'max_num_iteration': iter,
                           'population_size': 100,
                           'mutation_probability': 0.1,
                           'elit_ratio': 0.01,
                           'crossover_probability': 0.5,
                           'parents_portion': 0.3,
                           'crossover_type': 'uniform',
                           'max_iteration_without_improv': None}
        model = ga(function=f, dimension=ln, variable_type='real', variable_boundaries=varbound, algorithm_parameters=algorithm_param)
        model.run()
        coefs = model.output_dict['variable']
        resT = evalSMAPE_gen(coefs, checkDates2, copeD)
        res.append(resT)

    return inp, res


def copyData(data):
    return Data(np.copy(data.signs), data.category)


def copyDates(dates):
    newFirstList = list()
    for i in range(len(dates)):
        newFirstList.append(copyData(dates[i]))

    return newFirstList


# withNorm for SGD
def evalSMAPE(coef, checkDates, dataset, withNorm=False):
    if withNorm:
        mm = minmax(dataset)
        normalize(checkDates, mm)
    predictYResult = list()
    expYResult = list()
    for date in checkDates:
        print('___')
        # printData(date)
        print(date)
        print(coef)
        print('___')
        predictYResult.append(predict(date.signs, coef))
        expYResult.append(date.category)
    print(predictYResult)
    return smape(expYResult, predictYResult)

# withNorm for SGD
def evalSMAPE_sq(coef, checkDates, dataset, withNorm=False):
    if withNorm:
        mm = minmax(dataset)
        normalize(checkDates, mm)
    predictYResult = list()
    expYResult = list()
    for date in checkDates:
        print('___')
        print(date)
        print(coef)
        print('___')
        predictYResult.append(predict(date.signs, coef))
        # predictYResult.append(predict_gen(date.signs, coef))
        expYResult.append(date.category)
    print(predictYResult)
    return smape(expYResult, predictYResult)


# withNorm for SGD
def evalSMAPE_gen(coef, checkDates, dataset, withNorm=False):
    if withNorm:
        mm = minmax(dataset)
        normalize(checkDates, mm)
    predictYResult = list()
    expYResult = list()
    for date in checkDates:
        predictYResult.append(predict(date.signs, coef))
        # predictYResult.append(predict_gen(date.signs, coef))
        expYResult.append(date.category)
    return smape(expYResult, predictYResult)


def evalNRMSE(coef, checkDates, dataset, withNorm=False):
    if withNorm:
        mm = minmax(dataset)
        normalize(checkDates, mm)
    predictYResult = list()
    expYResult = list()
    for date in checkDates:
        predictYResult.append(predict(date.signs, coef))
        expYResult.append(date.category)
    return nrmse(expYResult, predictYResult)


def nrmse(abs_result, cur_result):
    n = len(abs_result)
    sum = 0
    for i in range(n):
        sum += ((abs_result[i] - cur_result[i]) ** 2)
    sum /= n
    sum = math.sqrt(sum)
    max_v = max(abs_result)
    min_v = min(abs_result)
    return sum / (max_v - min_v)


def getCheckDates(filePath):
    file = open(filePath, 'r')
    file.readline()
    n = int(file.readline())
    teachDates = []
    teachY = []
    for _ in range(n):
        tmp = list(map(int, file.readline().split()))
        teachY.append(tmp[-1])
        teachDates.append(Data(tmp[:-1], tmp.pop()))

    n = int(file.readline())
    checkDates = []
    dataset = list()
    for _ in range(n):
        tmp = list(map(int, file.readline().split()))
        dataset.append(list(tmp))
        checkDates.append(Data(tmp[:-1], tmp.pop()))
    file.close()
    return teachDates, teachY, checkDates, dataset


def printPlot(path, type, x, y, folder='results', prefix=''):
    if type == 1:
        nameM = 'Squares'
    elif type == 2:
        nameM = 'SGD'
    elif type == 3:
        nameM = 'Genetic'
    else:
        nameM = 'other'

    fileName = prefix + path + '_' + nameM
    plt.title(nameM, fontsize=20)
    plt.plot(x, y)
    plt.xlabel('i')
    plt.ylabel('SMAPE')
    # plt.ylabel('NRMSE')
    plt.legend([nameM], loc='upper right')
    plt.savefig(folder + '/' + fileName + '.png')
    plt.show()
    # plt.show()


# 1 squares, 2 SGD, 3 genetic
def testOnFile(filePath, type, iter=10):
    print('File path with dataset: ' + filePath)
    fileName = os.path.basename(filePath)[:-4]
    teachDates, teachY, checkDates, dataset = getCheckDates(filePath)
    if type == 1:
        newAr = []
        for el in teachDates:
            newAr.append([1] + el.signs)

        inp = list()
        res = list()

        coef = getInverseModel(newAr, teachY)
        print('old_______')
        print(coef)
        print('new_______')
        print(getInverseModel2(newAr, teachY))
        print('_____aaaa_______')

        cp = copyDates(checkDates)
        for el in cp:
            el.signs = [1] + el.signs
        smape = evalSMAPE_sq(coef, cp, dataset, withNorm=False)
        # smape = evalNRMSE(coef, copyDates(checkDates), dataset, withNorm=False)
        inp.append(1)
        inp.append(2)
        res.append(smape)
        res.append(smape)
        # for i in range(iter):
        #     coef = testOnSquares(filePath, i)
        #     smape = evalSMAPE(coef, copyDates(checkDates), dataset, withNorm=True)
        #     inp.append(i + 1)
        #     res.append(smape)
        printPlot(fileName, type, inp, res, folder='nHren', prefix='square_')
    elif type == 2:
        # inp = list()
        # res = list()
        # for i in range(iter):
        # print('Iteration ' + (i + 1).__str__() + ' of ' + iter.__str__())
        # print('Iteration ' + (iter + 1).__str__() + ' of ' + iter.__str__())
        testOnSGD(filePath, iter + 1, copyDates(checkDates), dataset, fileName, withNorm=True)
        # smape = evalNRMSE(coef, copyDates(checkDates), dataset, withNorm=True)

        # smape = evalSMAPE(coef, copyDates(checkDates), dataset, withNorm=True)

        # inp.append(iter + 1)
        # res.append(smape)
        # printPlot(fileName, type, inp, res, folder='nHren')
    elif type == 3:
        arr = [(n + 1) * 100 for n in range(iter)]
        inp, res = testOnGenetic(copyDates(teachDates), teachY, copyDates(checkDates), dataset, arr)
        # coef = testOnGenetic(fileName, copyDates(checkDates), dataset)
        #
        # smape = evalSMAPE(coef, copyDates(checkDates), dataset, withNorm=True)
        # inp.append(1)
        # inp.append(2)
        # res.append(smape)
        # res.append(smape)
        printPlot(fileName, type, inp, res, folder='gen', prefix='new_square_')

    else:
        print('choose type [1-3]')
        return


# scypi for genetic algorithm
if __name__ == '__main__':
    # iter = 100 #for other
    iter = 1 #for gen
    type = 3

    # testOnFile("LR/0.txt", type, iter)
    # testOnFile("LR/1.txt", type, iter)
    # testOnFile("LR/2.txt", type, iter)
    # testOnFile("LR/3.txt", type, iter)
    testOnFile("LR/4.txt", type, iter)
    # testOnFile("LR/5.txt", type, iter)
    # testOnFile("LR/6.txt", type, iter)
    # testOnFile("LR/7.txt", type, iter)
