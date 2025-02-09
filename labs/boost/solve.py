import pandas as pd
import numpy as np
from sklearn.metrics import zero_one_loss
from sklearn.tree import DecisionTreeClassifier
from math import exp, log
import matplotlib.pyplot as plt

eps = 0.00000000001

def readDates(file):
    dates = pd.read_csv(file)
    y = list(map(lambda it: 1 if it == 'P' else -1, dates["class"].values.tolist()))
    dates = dates.drop(columns="class")
    return np.array(dates), np.array(y)


def adaBoost(X, y, t):
    trees, alphas = list(), list()
    weights = [1 / len(y) for _ in range(len(y))]
    for _ in range(t):
        tree = DecisionTreeClassifier(max_depth=3).fit(X, y, sample_weight=np.array(weights, copy=True))
        trees.append(tree)
        y_predict = tree.predict(X)
        sum_e = zero_one_loss(y, y_predict, normalize=False, sample_weight=weights)
        if abs(sum_e) < eps or abs(1 - sum_e) < eps:
            if abs(sum_e) < eps:
                trees = [tree]
                alphas = [1]
            else:
                trees = [tree]
                alphas = [-1]
            break
        alpha = 0.5 * log((1 - sum_e) / sum_e)
        alphas.append(alpha)
        for i in range(len(y)):
            weights[i] *= exp(-alpha * y[i] * y_predict[i])
        sumW = sum(weights)
        weights = list(map(lambda x: x / sumW, weights))
    return trees, alphas


# predict for array of n objects with k features
# return n predicted results
def predict(trees, alphas, xArr):
    tmpArr, results = list(), list()
    for tree in trees:
        tmpArr.append(tree.predict(xArr))
    for i in range(len(xArr)):
        res = 0
        for j in range(len(trees)):
            res += (alphas[j] * tmpArr[j][i])
        results.append(np.sign(res))
    return results


def evalAccuracy(x, y, xTest, yTest, t):
    trees, alphas = adaBoost(x, y, t)
    yPredicted = predict(trees, alphas, xTest)

    good = 0
    for i in range(len(yPredicted)):
        if yTest[i] == yPredicted[i]:
            good += 1
    return good / len(yTest)


def paintAccuracyGraph(file, upperBound, outputFileName):
    x, y = readDates(file)
    xArr = [i for i in range(upperBound + 1)]
    yArr = list(map(lambda i: evalAccuracy(x, y, x, y, i), xArr))

    plt.plot(xArr, yArr)
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.title(outputFileName)
    plt.savefig(f"results/{outputFileName}.png")
    plt.show()


def paintSolve(file, steps, outputFileName, stepSize=0.1):
    x, y = readDates(file)
    trees, alphas = adaBoost(x, y, steps)
    dates = pd.read_csv(file)
    xx, yy = dates['x'].values.tolist(), dates['y'].values.tolist()
    colors = list(map(lambda x: 'green' if x == 1 else 'red', y))

    x_min, x_max = min(xx) - 1, max(xx) + 1
    y_min, y_max = min(yy) - 1, max(yy) + 1
    xxx, yyy = np.meshgrid(np.arange(x_min, x_max, stepSize), np.arange(y_min, y_max, stepSize))
    xArr = np.c_[xxx.ravel(), yyy.ravel()]
    yPredicted = predict(trees, alphas, xArr)

    z = (np.array(yPredicted)).reshape(xxx.shape)
    plt.contourf(xxx, yyy, z, alpha=1)
    plt.scatter(xx, yy, c=colors)
    plt.title(outputFileName)
    plt.savefig(f"results/{outputFileName}.png")
    plt.show()


if __name__ == '__main__':
    paintAccuracyGraph("testDates/geyser.csv", 55, "geyser_acc_55")
    paintAccuracyGraph("testDates/chips.csv", 55, "chips_acc_55")

    iters = [1, 2, 3, 5, 8, 13, 21, 34, 55]
    for iter in iters:
        paintSolve("testDates/geyser.csv", iter, f"geyser_{iter}", stepSize=0.01)
        paintSolve("testDates/chips.csv", iter, f"chips_{iter}", stepSize=0.01)
