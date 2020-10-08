import random


class Data:
    def __init__(self, signs, category):
        self.signs = signs
        self.category = category


def printData(dates):
    for el in dates:
        print(el.signs, '\tclass = ' + el.category.__str__())


def predict(i, coefs, b, dates):
    res = 0.0
    for j in range(len(dates)):
        res += dates[i].signs[j] * dates[j].category * coefs[j]
    res += b
    return res


# return coefs alphas and b
# dates signs contain core
# link http://cs229.stanford.edu/materials/smo.pdf
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


# input n (elements and fetchers -> matrix core) and after c 1
if __name__ == '__main__':
    n = int(input())
    dates = []
    for _ in range(n):
        tmp = list(map(int, input().split()))
        dates.append(Data(tmp[:-1], tmp.pop()))
    c = int(input())
    alphas, b = svm(dates, c, 1000)

    for alpha in alphas:
        print(alpha)
    print(b)
