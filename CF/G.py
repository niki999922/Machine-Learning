import math

freeId = 1

is_gini = False

class Node:
    featureNumber = 0
    b = -100_000
    classAns = -1

    leftChild = None
    rightChild = None


    # list of teach
    # hightOst of visota do 0
    def __init__(self, teach, hightOst, k):
        global freeId
        self.id = freeId
        freeId += 1
        self.k = k
        self.hightOst = hightOst
        self.buildNode(teach, hightOst)

    def initClassesMap(self, k):
        return {i: 0 for i in range(k)}

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
            self.classAns = self.findMaxClassByChastota(teach)
            return
        bestFeature = 0
        bestB = 100_000
        bestScore = 100_000
        leftListBEstInd = list()
        rightListBEstInd = list()
        for featureI in range(len(teach[0][0])):
            last_score = bestScore
            hello = []
            for i in range(len(teach)):
                hello.append((teach[i][0][featureI], i))
            hello = sorted(hello)
            left_ind = 0
            left_best_ind = 0
            leftListInd = list()
            rightListInd = list()
            leftClassesMap = self.initClassesMap(self.k)
            rightClassesMap = self.initClassesMap(self.k)
            left_classes_count_con = 0
            right_classes_count_con = 0
            for el in hello:
                rightListInd.append(el[1])
                rightClassesMap[teach[el[1]][1]] += 1
                right_classes_count_con += 1

            if is_gini:
                score = self.evalGini(leftClassesMap, left_classes_count_con) * left_classes_count_con + self.evalGini(rightClassesMap, right_classes_count_con) * right_classes_count_con
            else:
                score = self.evalEntrop(leftClassesMap, left_classes_count_con) * left_classes_count_con + self.evalEntrop(rightClassesMap, right_classes_count_con) * right_classes_count_con
            if bestScore > score:
                bestScore = score
                left_best_ind = left_ind
                bestB = hello[left_ind][0]
                bestFeature = featureI

            while left_ind < len(hello):
                znash = hello[left_ind][0]
                while left_ind < len(hello) and znash == hello[left_ind][0]:
                    left_classes_count_con += 1
                    right_classes_count_con -= 1
                    leftClassesMap[teach[hello[left_ind][1]][1]] += 1
                    rightClassesMap[teach[hello[left_ind][1]][1]] -= 1
                    leftListInd.append(hello[left_ind][1])
                    rightListInd.remove(hello[left_ind][1])
                    left_ind += 1

                if is_gini:
                    score = self.evalGini(leftClassesMap, left_classes_count_con) * left_classes_count_con + self.evalGini(rightClassesMap, right_classes_count_con) * right_classes_count_con
                else:
                    score = self.evalEntrop(leftClassesMap, left_classes_count_con) * left_classes_count_con + self.evalEntrop(rightClassesMap, right_classes_count_con) * right_classes_count_con
                if bestScore > score:
                    bestScore = score
                    left_best_ind = left_ind
                    bestB = hello[left_ind][0]
                    bestFeature = featureI
            if bestScore < last_score:
                leftListBEstInd = list(map(lambda e: e[1], hello[:left_best_ind]))
                rightListBEstInd = list(map(lambda e: e[1], hello[left_best_ind:]))

        leftListBEst = [teach[ind] for ind in leftListBEstInd]
        rightListBEst = [teach[ind] for ind in rightListBEstInd]
        leftNode = Node(leftListBEst, hightOst - 1, self.k)
        rightNode = Node(rightListBEst, hightOst - 1, self.k)
        self.b = bestB
        self.featureNumber = bestFeature
        self.leftChild = leftNode
        self.rightChild = rightNode

    # list size of k different classes and count objects
    def evalEntrop(self, childs, elemsSum):
        res = 0
        for key in childs:
            x = childs[key]
            if x != 0:
                p = x * 1.0 / elemsSum
                res -= p * math.log(p)
        return res

    def evalGini(self, childs, elemsSum):
        res = 0
        for key in childs:
            x = childs[key]
            if x != 0:
                p = x * 1.0 / elemsSum
                res += p ** 2
        return 1 - res

    def printNode(self, listRes):
        if self.hightOst == 1:
            # listRes.append((self.id, f'C {self.classAns + 1}'))
            listRes.append((self.id, f'C {self.classAns + 1}'))
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

    def printTree(self):
        listRes = list()
        self.head.printNode(listRes)
        listRes = sorted(listRes, key=lambda el: el[0])

        print(freeId - 1)
        for el in listRes:
            print(el[1])


if __name__ == '__main__':
    m, k, h = map(int, input().split(" "))
    n = int(input())
    teach = list()
    for i in range(n):
        inp = list(map(int, input().split(" ")))
        features = inp[:m]
        clazz = inp[m]
        teach.append((features, clazz - 1))
    if len(teach) > 200:
        is_gini = True
    treeML = Tree(teach, h, k)
    treeML.printTree()

    # print(teach)

'''
Answer:

7
Q 1 2.5 2 5
Q 2 2.5 3 4
C 1
C 4
Q 2 2.5 6 7
C 2
C 3
'''
