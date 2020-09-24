from enum import Enum
import math


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


class Data:
    def __init__(self, signs, category):
        self.signs = signs
        self.category = category


if __name__ == '__main__':
    def setDistance(el):
        el.distance = calculateDistance(distance, el.signs, investigated)
        return el


    n, m = map(int, input().split())
    dates = []
    investigated = []
    for _ in range(n):
        tmp = list(map(int, input().split()))
        dates.append(Data(tmp[:-1], tmp.pop()))
    investigated = list(map(int, input().split()))
    distance = Distance(input())
    kernel = Kernel(input())
    window = Window(input())
    maxDistance = h = int(input())
    dates = list(sorted(list(map(setDistance, dates)), key=lambda _data: _data.distance))

    if window == Window.VARIABLE:
        maxDistance = dates[h].distance

    if maxDistance == 0:
        if len(list(filter(lambda _data: _data.signs == investigated, dates))) != 0:
            print(calculateAverage2(dates, investigated))
        else:
            print(calculateAverage(dates))
    else:
        x = 0
        y = 0
        for data in dates:
            kerVal = calculateKernel(kernel, data.distance / maxDistance)
            x += data.category * kerVal
            y += kerVal
        if y == 0:
            print(calculateAverage(dates))
        else:
            print(x / y)
