import math
import os
import sys
import matplotlib.pyplot as plt


freeId = 1


class Node:
    featureNumber = 0
    b = -100_000
    classAns = -1

    leftChild = None
    rightChild = None

    def suggest(self, elem):
        if self.hightOst == 1 or self.classAns != -1:
            return self.classAns
        else:
            if elem[self.featureNumber] < self.b:
                return self.leftChild.suggest(elem)
            else:
                return self.rightChild.suggest(elem)

    # list of teach
    # hightOst of visota do 0
    def __init__(self, teach, hightOst, k):
        global freeId
        self.id = freeId
        freeId += 1
        print(f'teaching id: {self.id}, train size: {len(teach)}', end=" ", flush=True)
        self.k = k
        self.hightOst = hightOst
        self.buildNode(teach, hightOst)

    def initClassesMap(self, k):
        m = {}
        for i in range(k):
            m[i] = 0
        return m

    def findMaxClassByChastota(self, teach):
        tmp = self.initClassesMap(self.k)
        for el in teach:
            tmp[el[1]] += 1
        bestKey = 0
        for key in tmp:
            if tmp[key] > tmp[bestKey]:
                bestKey = key
        return bestKey

    def buildNode(self, teach, hightOst):
        if hightOst == 1:
            print(f'| create leaf', flush=True)
            self.classAns = self.findMaxClassByChastota(teach)
            return
        bestFeature = 0
        bestB = 100_000
        bestScore = 100_000
        leftListBEst = list()
        rightListBEst = list()
        print(f'| feature {len(teach[0][0])}:', end=" ", flush=True)
        for featureI in range(len(teach[0][0])):
            print(featureI, end=" ", flush=True)
            for teachEl in teach:
                bTest = teachEl[0][featureI]
                leftClassesMap = self.initClassesMap(self.k)
                rightClassesMap = self.initClassesMap(self.k)
                leftList = list()
                rightList = list()
                for teachElem in teach:
                    if teachElem[0][featureI] < bTest:
                        leftList.append(teachElem)
                        leftClassesMap[teachElem[1]] += 1
                    else:
                        rightList.append(teachElem)
                        rightClassesMap[teachElem[1]] += 1
                score = self.evalEntrop(leftClassesMap) * sum(leftClassesMap.values()) + self.evalEntrop(rightClassesMap) * sum(rightClassesMap.values())
                if bestScore > score:
                    bestScore = score
                    bestB = bTest
                    bestFeature = featureI
                    leftListBEst = leftList
                    rightListBEst = rightList
        if len(leftListBEst) == 0 or len(rightListBEst) == 0:
            print(f'| create leaf', flush=True)
            self.classAns = self.findMaxClassByChastota(teach)
            return
        print(flush=True)
        leftNode = Node(leftListBEst, hightOst - 1, self.k)
        rightNode = Node(rightListBEst, hightOst - 1, self.k)
        self.b = bestB
        self.featureNumber = bestFeature
        self.leftChild = leftNode
        self.rightChild = rightNode

    # list size of k different classes and count objects
    def evalEntrop(self, childs):
        elemsSum = sum(childs.values())
        res = 0
        for key in childs:
            x = childs[key]
            if x != 0:
                res -= (x * 1.0 / elemsSum) * math.log(x * 1.0 / elemsSum)
        return res

    def printNode(self, listRes):
        if self.hightOst == 1 or self.classAns != -1:
            # listRes.append((self.id, f'C {self.classAns + 1}'))
            listRes.append((self.id, f'C {self.classAns}'))
        else:
            # listRes.append((self.id, f'Q {self.id} {self.b} {self.leftChild.id} {self.rightChild.id}'))
            listRes.append((self.id, f'Q {self.featureNumber + 1} {self.b} {self.leftChild.id} {self.rightChild.id}'))
            self.leftChild.printNode(listRes)
            self.rightChild.printNode(listRes)


class Tree:
    head = None

    def __init__(self, teach, maxH, k):
        self.buildTree(teach, maxH + 1, k)

    def buildTree(self, teach, maxH, k):
        self.head = Node(teach, maxH, k)

    def suggest(self, elem):
        # for correctshift on 1 for 1,2,3 -> 0,1,2
        return self.head.suggest(elem)

    def printTree(self):
        listRes = list()
        self.head.printNode(listRes)
        listRes = sorted(listRes, key=lambda el: el[0])

        print(freeId - 1, flush=True)
        for el in listRes:
            print(el[1], flush=True)


# on who teach:
# ([] - feachers values, class)
# class from 0: 0, 1, 2, 3 (because mapping)
# maxH - 7, hight of tree
# k - amount different classes
# teach [([1, 1, 2, 2, 3], 1), ..]
def teachTree(teach, maxH, k):
    global freeId
    freeId = 1
    print('teaching tree', flush=True)
    return Tree(teach, maxH, k)


def getDatesFromFile(fileName):
    f = open(fileName, "r")
    name = os.path.splitext(os.path.basename(fileName))[0]
    print(f'open on read fileName: \"{name}\"', flush=True)
    m, k = map(int, f.readline().split(" "))
    n = int(f.readline())
    teach = list()
    for i in range(n):
        inp = list(map(int, f.readline().split(" ")))
        features = inp[:m]
        clazz = inp[m]
        teach.append((features, clazz - 1))
    return teach, k


def testOnTree(tree, testDates):
    sum, succ = 0, 0
    tmpInd = 1
    for test in testDates:
        print(f'{tmpInd} of {len(testDates)}', flush=True)
        tmpInd += 1
        classSuggest = tree.suggest(test[0])
        readClass = test[1]
        sum += 1
        if classSuggest == readClass:
            succ += 1
    return succ, sum

def testOnTrees(trees, testDates, teachK):
    sum, succ = 0, 0
    tmpInd = 1
    for test in testDates:
        print(f'{tmpInd} of {len(testDates)}', flush=True)
        tmpInd += 1

        m = {}
        for i in range(teachK):
            m[i] = 0
        for tree in trees:
            classSuggestTmp = tree.suggest(test[0])
            m[classSuggestTmp] += 1

        bestKey = 0
        for key in m:
            if m[key] > m[bestKey]:
                bestKey = key
                bestKey = key
        classSuggest = bestKey



        readClass = test[1]
        sum += 1
        if classSuggest == readClass:
            succ += 1
    return succ, sum


def printGraph(workingDirectory, filePrefix, maxH, stepSize=20, fileName='h_acc'):
    teachFileName = os.path.join(workingDirectory, f'{filePrefix}_train.txt')
    testFileName = os.path.join(workingDirectory, f'{filePrefix}_test.txt')
    teachDates, teachK = getDatesFromFile(teachFileName)
    testDates, testK = getDatesFromFile(testFileName)
    hArr = [1]
    cVal = 1 + stepSize
    while cVal < maxH:
        hArr.append(cVal)
        cVal = cVal + stepSize
    if not hArr.__contains__(maxH):
        hArr.append(maxH)
    print(hArr)
    xx = hArr.copy()
    yy = list()
    for h in hArr:
        tree = teachTree(teachDates, h, teachK)
        succ, sum = testOnTree(tree, testDates + teachDates)
        yy.append(succ / sum)
        accur = '{:.2%}'.format(succ / sum)
        print(f'accuracy = {accur} <- {succ}/{sum} for h={h}', flush=True)
    plt.figure(figsize=(16, 9))
    plt.grid(linestyle='--')
    plt.plot(xx, yy)
    plt.legend(["Accuracy"], loc='upper right')
    plt.xlabel('Max hight tree')
    plt.ylabel('Accuracy')
    print('Saving picture to /Users/nikita/Machine-Learning/labs/dt/results/graphs/{0}.png'.format(fileName))
    plt.savefig('/Users/nikita/Machine-Learning/labs/dt/results/graphs/{0}.png'.format(fileName))
    print('Saved')
    plt.show()


# workingDirectory - path to "dt/teach"
# filePrefix = 01_test.txt -> 01
def evalOptimalHight(workingDirectory, filePrefix):
    teachFileName = os.path.join(workingDirectory, f'{filePrefix}_train.txt')
    testFileName = os.path.join(workingDirectory, f'{filePrefix}_test.txt')
    teachDates, teachK = getDatesFromFile(teachFileName)
    testDates, testK = getDatesFromFile(testFileName)
    hArr = [2, 10, 50]
    bestAccur = 0
    bestH = hArr[0]
    for maxH in hArr:
        tree = teachTree(teachDates, maxH, teachK)
        succ, sum = testOnTree(tree, testDates)
        accTmp = succ / sum
        if bestAccur < accTmp:
            print(f'Found better accuracy = {accTmp} > {bestAccur} for h = {maxH}', flush=True)
            bestAccur = accTmp
            bestH = maxH
        accur = '{:.2%}'.format(succ / sum)
        print(f'accuracy = {accur} <- {succ}/{sum}', flush=True)
    print('-----------------------------------------------', flush=True)
    accur = '{:.2%}'.format(bestAccur)
    print(f'Best h = {bestH} <-> accuracy = {accur}', flush=True)
    print('-----------------------------------------------', flush=True)


def evalWithForest(workingDirectory, filePrefix, splitOn=5, grouping=3):
    teachFileName = os.path.join(workingDirectory, f'{filePrefix}_train.txt')
    testFileName = os.path.join(workingDirectory, f'{filePrefix}_test.txt')
    teachDates, teachK = getDatesFromFile(teachFileName)
    testDates, testK = getDatesFromFile(testFileName)
    splitOnDel = len(teachDates) // splitOn
    teachDatesArrTmp = [teachDates[i:i + splitOnDel] for i in range(0, len(teachDates), splitOnDel)]
    teachDatesArrTmp2 = [teachDatesArrTmp[i:i + grouping] for i in range(len(teachDatesArrTmp) - grouping + 1)]
    teachDatesArr = list()
    for arr in teachDatesArrTmp2:
        newArr = list()
        for subArr in arr:
            for subArr1 in subArr:
                newArr.append(subArr1)
        teachDatesArr.append(newArr)
    trees = list()
    maxH = 100_000
    for teach in teachDatesArr:
        trees.append(teachTree(teach, maxH, teachK))
    succ, sum = testOnTrees(trees, testDates + teachDates, teachK)
    accur = '{:.2%}'.format(succ / sum)
    print('-----------------------------------------------', flush=True)
    print(f'Forest accuracy = {accur} <- {succ}/{sum}', flush=True)
    print('-----------------------------------------------', flush=True)


if __name__ == '__main__':
    testName = str(sys.argv[1])

    #for script
    # evalOptimalHight('/Users/nikita/Machine-Learning/labs/dt/teach', testName)

    #for local
    # evalOptimalHight('/Users/nikita/Machine-Learning/labs/dt/teach', '10')


    #for printGraph
    # printGraph('/Users/nikita/Machine-Learning/labs/dt/teach', '03', 12, 1, 'dataset03_best-2_from-1_to-12_step-1')
    # printGraph('/Users/nikita/Machine-Learning/labs/dt/teach', '14', 51, 4, 'dataset14_best-50_from-1_to-51_step-4')

    #for debug print graph
    # printGraph('/Users/nikita/Machine-Learning/labs/dt/teach', '10', 12, 20, 'dataset10_best-1_from-1_to-12_step-1')


    # for script
    # evalWithForest('/Users/nikita/Machine-Learning/labs/dt/teach', testName, splitOn=7, grouping=3)

    # for local
    evalWithForest('/Users/nikita/Machine-Learning/labs/dt/teach', '00', splitOn=7, grouping=3)

    print('End Analysing!!!')
