import math
import os
import matplotlib.pyplot as plt
from operator import itemgetter


def printTable(table):
    for i in range(len(table)):
        print(i.__str__() + ": " + table[i].__str__(), end="\n")


def evalProbability(classes, words, k, alpha):
    def evalCountWordEachClasses():
        wordsTmp = {word: {i: 0 for i in range(k)} for word in words}
        for key in classes:
            arrays = classes[key]
            for array in arrays:
                wasTouched = set()
                for el in array:
                    if not wasTouched.__contains__(el):
                        wasTouched.add(el)
                        wordsTmp[el][key] = wordsTmp[el][key] + 1
        return wordsTmp

    result = {i: {word: 0 for word in words} for i in range(k)}
    wordsStats = evalCountWordEachClasses()
    for i in range(k):
        for word in words:
            result[i][word] = ((wordsStats[word][i] + alpha), (len(classes[i]) + alpha * 2))
    return result


# input like "2 lala la"
# lambdas [1, 1] size of k
# c_w_p, words, c_count, n from evaluateBayes
# return 1 spam, 2 legit
def predict(inputArr, lambdas, c_w_p, words, c_count, n):
    k = 2
    count, *wordsTmp_ = inputArr
    foundWords = set(wordsTmp_)
    notFoundWords = words - foundWords
    res_p = [0 for _ in range(k)]
    for i in range(k):
        resTmp = math.log(lambdas[i]) - math.log(n)
        resTmp += math.log(c_count[i] if c_count[i] != 0 else 0.001)
        for word in foundWords:
            if c_w_p[i].__contains__(word):
                resTmp += math.log(c_w_p[i][word][0]) - math.log(c_w_p[i][word][1])
        for word in notFoundWords:
            resTmp += math.log(c_w_p[i][word][1] - c_w_p[i][word][0]) - math.log(c_w_p[i][word][1])
        res_p[i] = resTmp
    sumRes = sum(res_p)
    maxRes = max(res_p)
    helper = -sumRes / len(res_p)
    # add_val = sum_ln / len(products)
    if maxRes + helper > 15:
        helper = maxRes + 10
    for el in res_p:
        el = math.exp(el + helper)
    newSum = sum(res_p)
    # products = list(map(lambda x: exp(x - add_val), products))
    # sum_products = sum(products)
    # all_results.append(list(map(lambda x: x / sum_products, products)))

    if res_p[0] > res_p[1]:
        return 1, res_p[1] / newSum
    else:
        return 2, res_p[1] / newSum


# k = 2; 1 spam, 2 legit
# alpha = 1
# return c_w_p, words, c_count, n for predict
def evaluateBayes(alpha, inputArrays):
    k = 2
    n = len(inputArrays)
    classes = {i: list() for i in range(k)}
    c_count = {i: 0 for i in range(k)}
    words = set()
    for i in range(n):
        inp = inputArrays[i]
        classTmp = int(inp[0])
        c_count[classTmp - 1] = c_count[classTmp - 1] + 1
        count = int(inp[1])
        arr = list()
        for j in range(count):
            word = inp[2 + j]
            words.add(word)
            arr.append(word)
        classes[classTmp - 1].append(arr)
    c_w_p = evalProbability(classes, words, k, alpha)
    return c_w_p, words, c_count, n


# test /Users/nikita/Machine-Learning/labs/bayes/12spmsg12.txt
# 1 spam, 2 legit
# nGram - how much in one pair : [1, 2, 3]
# return pair (class, array)
def readFile(fileName, nGram):
    f = open(fileName, "r")
    name = os.path.splitext(os.path.basename(fileName))[0]
    if name.__contains__("legit"):
        category = 2
    else:
        category = 1
    line1 = f.readline()[8:].strip().split(" ")
    f.readline()
    inputArr = line1 + f.readline().split(" ")
    res = list()
    for i in range(len(inputArr) - nGram + 1):
        res.append("_".join(inputArr[i:i + nGram]))
    return category, res


def evaluateAccuracy(directoryName, nGram=1, alpha=1, lambdas=[1, 1]):
    def getInputArraysForBayes(teachDirectories):
        inputArray = list()
        for directory in teachDirectories:
            directoryWithMessages = os.path.join(directoryName, directory)
            messages = os.listdir(directoryWithMessages)
            for message in messages:
                category, arr = readFile(os.path.join(directoryWithMessages, message), nGram)
                inputArray.append([category.__str__(), len(arr).__str__()] + arr)
        return inputArray

    # good, all
    def testOnDirectory(testDirectory, c_w_p, words, c_count, n):
        all = 0
        good = 0
        allLegit = 0
        goodLegit = 0
        directoryWithTests = os.path.join(directoryName, testDirectory)
        messages = os.listdir(directoryWithTests)
        for message in messages:
            category, arr = readFile(os.path.join(directoryWithTests, message), nGram)
            predictedCategory, _ = predict([len(arr).__str__()] + arr, lambdas, c_w_p, words, c_count, n)
            if category == predictedCategory:
                good += 1
            if category == predictedCategory and category == 2:
                goodLegit += 1
            if category == 2:
                allLegit += 1
            all += 1
        return all, good, allLegit, goodLegit

    subDirectories = os.listdir(directoryName)
    subDirectories.sort()
    all = 0
    good = 0
    allLegit = 0
    goodLegit = 0
    for numDir in range(len(subDirectories)):
        teachDirectories = subDirectories[:numDir] + subDirectories[numDir + 1:]
        testDirectory = subDirectories[numDir]
        print("\ttesting on \".../" + testDirectory.__str__() + "\"/")
        inputArray = getInputArraysForBayes(teachDirectories)
        c_w_p, words, c_count, n = evaluateBayes(alpha, inputArray)
        allTmp, goodTmp, allLegitTmp, goodLegitTmp = testOnDirectory(testDirectory, c_w_p, words, c_count, n)
        all += allTmp
        good += goodTmp
        allLegit += allLegitTmp
        goodLegit += goodLegitTmp
        # print(f'good:{goodTmp:4} ___ good_legit:{goodLegitTmp:4}')
        # print(f'all: {allTmp:4} ___ all_legit: {allLegitTmp:4}')
    accuracy = good / all
    falseNegative = goodLegit / allLegit
    # print("accuracy: " + accuracy.__str__())
    print("falseNegative: " + falseNegative.__str__())
    return accuracy


def printAccuracy():
    def printImage(x_arr, y_arr):
        # plt.title("Accuracy", fontsize=20)
        # plt.plot(x_arr, y_arr)
        # plt.xlabel('Penalty for legitimate messages')
        # plt.ylabel('Accuracy')
        # plt.legend(["Accuracy"], loc='upper right')
        # plt.show()

        plt.figure(figsize=(16, 9))
        plt.grid(linestyle='--')
        plt.plot(x_arr, y_arr)
        # plt.semilogx(x_arr, y_arr, linestyle='-', marker='.', color='r', label='Accuracy')
        plt.legend(["Accuracy"], loc='upper right')
        plt.xlabel('Penalty for legitimate messages')
        plt.ylabel('Accuracy')
        # plt.legend()
        plt.show()

    x_arr, y_arr = [], []
    # penArr = [1, 10, 50, 100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]
    penArr = [1] + [(i + 1) * 5 for i in range(30)]  # 150
    # penArr = [1] + [(i + 1) * 30 for i in range(25)] # 150
    # penArr = [1E5, 1, 500, 1000, 4000, 20_000, 50_000, 100_000]
    # penArr = [1, 1_000, 100_000]
    # penArr = [1]
    iters = 0
    for pen in penArr:
        iters += 1
        print(f'iteration number {iters} of {len(penArr)}: pen: {pen}')
        accuracy = evaluateAccuracy("/Users/nikita/Machine-Learning/labs/bayes/messages", nGram=1, alpha=0.0001, lambdas=[1, pen])
        x_arr.append(pen)
        y_arr.append(accuracy)
    printImage(x_arr, y_arr)


def printRock(directoryName, nGram=1, alpha=1, lambdas=[1, 1], dir_num=7):
    def getInputArraysForBayes(teachDirectories):
        inputArray = list()
        for directory in teachDirectories:
            directoryWithMessages = os.path.join(directoryName, directory)
            messages = os.listdir(directoryWithMessages)
            for message in messages:
                category, arr = readFile(os.path.join(directoryWithMessages, message), nGram)
                inputArray.append([category.__str__(), len(arr).__str__()] + arr)
        return inputArray

    def testOnDirectory(testDirectory, c_w_p, words, c_count, n):
        resTable = list()
        directoryWithTests = os.path.join(directoryName, testDirectory)
        messages = os.listdir(directoryWithTests)
        for message in messages:
            category, arr = readFile(os.path.join(directoryWithTests, message), nGram)

            predictedCategory, p = predict([len(arr).__str__()] + arr, lambdas, c_w_p, words, c_count, n)

            resTable.append((p, category))
        return resTable

    def printROC(resTable):
        x_step, y_step = 1, 1
        prev_p = -1
        x_res, y_res = [0], [0]
        x_global, y_global = 0, 0
        for el in resTable:
            if (prev_p != el[0]):
                prev_p = el[0]
                x_res.append(x_global)
                y_res.append(y_global)
            if el[1] == 2:
                x_global += x_step
                # y_global += y_step
            else:
                y_global += y_step
                # x_global += x_step
        x_res.append(x_global)
        y_res.append(y_global)

        plt.figure(figsize=(16, 9))
        plt.grid(linestyle='--')
        plt.plot(x_res, y_res)
        plt.legend(["Accuracy"], loc='upper right')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    subDirectories = os.listdir(directoryName)
    subDirectories.sort()
    teachDirectories = subDirectories[:dir_num] + subDirectories[dir_num + 1:]
    testDirectory = subDirectories[dir_num]
    print("\tpainting on \".../" + testDirectory.__str__() + "\"/")
    inputArray = getInputArraysForBayes(teachDirectories)
    c_w_p, words, c_count, n = evaluateBayes(alpha, inputArray)
    resTable = testOnDirectory(testDirectory, c_w_p, words, c_count, n)
    resTable = sorted(resTable, key=itemgetter(0), reverse=True)

    # delete
    # resTable = [(0.6, 2),
    #             (0.5, 1),
    #             (0.3, 2),
    #             (0.2, 1),
    #             (0.2, 2),
    #             (0.1, 1),
    #             (0.0, 1)]
    # delete

    printROC(resTable)

if __name__ == '__main__':
    evaluateAccuracy("/Users/nikita/Machine-Learning/labs/bayes/messages", nGram=1, alpha=0.0001, lambdas=[1, 1E5])
    # printAccuracy()
    # printRock("/Users/nikita/Machine-Learning/labs/bayes/messages", nGram=1, alpha=0.0001, lambdas=[1, 1E5], dir_num=6)
