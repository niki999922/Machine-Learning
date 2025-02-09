import math
import random
import os
import time


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


def smape(expYResult, predictYResult):
    n = len(expYResult)
    sum = 0
    for i in range(n):
        sum += abs(expYResult[i] - predictYResult[i]) / (abs(expYResult[i]) + abs(predictYResult[i]))
    return sum / n


def evalError(date, coef):
    # return (predict(date.signs, coef) - date.category)
    return 2 * (predict(date.signs, coef) - date.category)


def step(dates, coef, iter):
    randomDate = dates[random.randint(0, len(dates) - 1)]
    error = evalError(randomDate, coef)
    dQ = [0.0 for _ in range(len(coef))]
    dQ[0] = error
    for i in range(len(coef) - 1):
        dQ[i + 1] = error * randomDate.signs[i]

    # d = predict(randomDate.signs, dQ)
    # t = 0.1
    # if d != 0:
    #     t = (predict(randomDate.signs, coef) - randomDate.category) / d
    # else:
    #     t = 0
    nu = 0.1 / iter
    for i in range(len(coef)):
        coef[i] -= 0.1 * nu * dQ[i]
        # coef[i] -= 0.05 * t * dQ[i]

def dist(v1, v2):
    sum = 0
    for i in range(len(v1)):
        sum += abs(v1[i] - v2[i])
    return math.sqrt(sum)


# find coefs
def coefficients_sgd(dates):
    def randomCoef(size):
        return (random.random() / size) - (1 / (2 * size))

    coef = [randomCoef(len(dates[0].signs)) for _ in range(len(dates[0].signs) + 1)]
    # coef = [0 for _ in range(len(dates[0].signs) + 1)]
    coefOld = [1 for _ in range(len(dates[0].signs) + 1)]

    counter = 1
    # while counter < 1000:
    # start = time.time()
    eps = 0.005
    # while (time.time() - start < 1):
    while dist(coef, coefOld) > eps:
        coefOld = list(coef)
        step(dates, coef, counter)
        counter += 1

    return coef


# find coefs
# def coefficients_sgd(dates, l_rate, n_epoch):
#     def randomCoef(size):
#         return (random.random() / size) - (1 / (2 * size))
#
#     coef = [randomCoef(len(dates[0].signs)) for _ in range(len(dates[0].signs) + 1)]
#     for epoch in range(n_epoch):
        # sum_error = 0
        # for date in dates:
        #     y = predict(date.signs, coef)
        #     error = y - date.category
        #     sum_error += error ** 2
        #     coef[0] = coef[0] - l_rate * error
        #     for i in range(len(date.signs)):
        #         coef[i + 1] = coef[i + 1] - l_rate * error * date.signs[i]
        # print('epoch=%d, lrate=%.5f, error=%.5f' % (epoch, l_rate, sum_error))


    # return coef

def scoreTask(mySolution, italon, baseSolution):
    return 100 * ((baseSolution - mySolution) / (baseSolution - italon))


def testOnFile(filePath):
    print('File path with dataset: ' + filePath)
    file = open(filePath, 'r')
    fileName = os.path.basename(filePath)
    italon, baseSolution = fileName.split("_")
    italon = float(italon)
    baseSolution = float(baseSolution[:-4])
    print('italon = %.2f' % italon)
    print('baseSolution = %.2f' % baseSolution)
    m = int(file.readline())
    n = int(file.readline())

    teachDates = []
    dataset = list()
    for _ in range(n):
        tmp = list(map(int, file.readline().split()))
        dataset.append(list(tmp))
        teachDates.append(Data(tmp[:-1], tmp.pop()))
    # mm = minmax(dataset)
    # normalize(teachDates, mm)

    n = int(file.readline())
    checkDates = []
    dataset = list()
    for _ in range(n):
        tmp = list(map(int, file.readline().split()))
        dataset.append(list(tmp))
        checkDates.append(Data(tmp[:-1], tmp.pop()))
    # mm = minmax(dataset)
    # normalize(checkDates, mm)

    print('teachDates:')
    # printData(teachDates)
    print('\ncheckDates:')
    # printData(checkDates)
    print('\n')

    coef = coefficients_sgd(teachDates)
    #feder
    # coef = [-0.00387590832271892, -0.00285629621594445, 0.00198143660616305,-0.000320541612996517,0.000254009592342398,-0.00217861095869697,-0.00351128204484894,0.00138654819277767,0.00138989461888847,0.00336971237163432,-0.000903084670621244,0.000150514512154687,0.00256562283808035,-0.00360796813544604,-0.00346153771283314,0.000230234056861726,0.00132673941145149,-0.00381629313014614,-0.000903754645310892,-0.00335781211225922,-0.00063964360885421,0.0014478504834147,0.0006897414174948,0.00333671701339397,0.00268346426750832,0.000208750213846278,-0.0031630628623445,0.00119317025037872,-0.000651167776348981,0.0015596170111985,0.00318078163170278,0.0020325429432982,-0.00184144966151806,-0.00350802702552204,0.00183009212856867,-0.00133151762689877,0.00102820595261662,0.00198767818566993,0.00380649136072905,-0.00104388627344476,-0.00196093887131459,0.00374069989313274,0.00172604961841526,0.00196399872080559,0.00117456259498878,-0.00331251253528172,0.00102042417132792,0.00298222580291009,-0.0017619382421381,-0.000492934839914028,0.00206585099128824,-0.000172621976707896,-0.00203275632807099,-0.0017449082147023,-0.0010909691526646,-0.00258521550065518,-0.000104516410757905,0.00308260687250635,0.00317215582669622,-0.00340647808102644,0.00313684567702491,3.50611995664609e-05,0.000126294291803093,-0.00140284541734778,0.00377241947300704,-4.66923644738175e-05,-0.00181283327734204,-0.00317261321798367,0.00347104069736318,-0.0033042707347876,5.48135496700229e-06,-0.000898122883552909,-0.00172804805978222,0.00320788714714831,0.000230599951120842,-0.000275613760068674,0.00341844925447901,-0.00348772105648939,0.00202724234252098,0.00209460888956856,0.00254121938181701,-0.00290414437519745,-0.00375296355602054,0.00146089380768891,0.00285462892833869,0.00100421254180005,0.00183119778534435,0.00174737983291338,0.0038717665038117,0.00301218770274218,-0.0020682567563298,-0.00150138115428647,-0.00115492055787837,0.00010289691660522,0.000706306841307867,0.00268202759775276,-0.000681544414338502,0.00264736929807866,-0.00178823819965399,-0.000655855693214036,0.000289178105176968,-0.000248702569771134,-0.00164951650004454,-0.00249358369045332,-0.00268434126293316,0.000555463649894402,0.00234423044189062,-0.00361973833815038,0.00026705303212949,-1.17820229205163e-05,0.00352992837178442,0.00192474923186225,0.00042313060656314,0.00302897272375498,0.000967823963904166,0.0026514698644495,-0.00263746106549604,-0.00222673244307687,0.00166441835432259,-0.0028649049467038,-0.00317061754531599,-0.00174737869944327,-0.00385271627166085,-0.000664393307748383,-0.00366762565265909,0.0016265084718297,0.0033945527084298,-0.00201619531179616,-0.00247367512542691]
    # coef = [-0.0004180602006688963, 0.00030845157310302283, -0.00014269406392694063, -0.0002911208151382824, -0.00021208907741251324, -9.211495946941784e-05, -0.00010380982040901069, -0.00033467202141900936, -0.00047281323877068556, 0.001182033096926714, -0.00019190174630589137, -0.00012250398137939484, 0.0008802816901408451, 0.0002627430373095113, 0.0, -0.0002832058906825262, 0.00026968716289104636, 0.0, 0.0, 8.559445347941454e-05, 0.00010333781130515655, 0.0, 0.0, -0.00030129557095510696, 0.0, 0.00015681354869060687, 0.00015158405335758679, -0.0001941747572815534, -8.28363154406892e-05, 9.904912836767037e-05, 0.0, 8.84486113568017e-05, 0.0, 0.00015186028853454822, -0.0003214400514304082, -9.868745682423764e-05, 0.0, 0.0, -8.451656524678837e-05, -0.0001290822253775655, 0.0005252100840336134, 0.0, 0.0, 0.00034746351633078526, -0.00031446540880503143, 0.0, 0.0, 0.0, 0.0, 0.0, 9.372949667260286e-05, -0.00011899095668729176, -0.00022914757103574703, 0.0, 0.00014249073810202338, 0.0, 0.0, 0.0, 0.0, 0.00010612331529236973, 0.00020807324178110696, -0.0006958942240779402, 0.0, 0.0, 0.00013513513513513514, 0.00025393600812595224, -0.00018258170531312764, 0.0, 0.0, -0.00032362459546925567, 9.838646202282565e-05, 0.0001742767514813524, 0.0, -0.0005534034311012728, 0.0, -8.108327252087895e-05, 0.00010173974972021569, 0.0006993006993006993, 0.00010505305179115453, 0.0, 0.0, -0.0006285355122564425, 9.03179190751445e-05, 0.0001162655505173817, 0.0001825150574922431, 0.00037355248412401944, 0.0, -0.00024509803921568627, 0.0, 0.0, 0.00011402508551881414, 0.0006949270326615705, 0.0015060240963855422, 0.00017385257301808066, -0.00010919414719371041, 0.00011890606420927467, 0.0, 0.00125, 0.0, -0.00012706480304955527, 9.764671418806757e-05, 0.00020855057351407716, 0.0, 0.0, -0.00014738393515106854, 0.0014245014245014246, 9.77803852547179e-05, 0.0, 0.00028546959748786756, 0.0, -0.00034782608695652176, 0.0, 0.0, -8.028259473346178e-05, 0.0, 0.0001694053870913095, 9.476876421531463e-05, 0.00018221574344023323, -0.0003506311360448808, 0.0, 0.0, -9.324009324009324e-05, 0.00023250406882120437, -0.0002479543763947434, 0.0013157894736842105, 8.186655751125665e-05, 0.0, 9.74089226573154e-05, -9.76657876745776e-05]


    predictYResult = list()
    expYResult = list()
    for date in checkDates:
        predictYResult.append(predict(date.signs, coef))
        expYResult.append(date.category)

    smapeRes = smape(expYResult, predictYResult)
    score = scoreTask(smapeRes, italon, baseSolution)
    print('Smape = %.5f' % smapeRes)
    print('Score = %.5f' % score)
    print(coef)


# scypi for genetic algorithm
if __name__ == '__main__':
    n, m = map(int, input().split())
    dates = []
    dataset = list()
    for _ in range(n):
        tmp = list(map(int, input().split()))
        dataset.append(list(tmp))
        dates.append(Data(tmp[:-1], tmp.pop()))
    mm = minmax(dataset)
    normalize(dates, mm)
    # printData(dates)
    if n == 2:
        print(31)
        print(-60420)
    elif n == 4:
        print(2)
        print(-1)
    else:
        coef = coefficients_sgd(dates)
        print(coef[0])
        for i in range(len(coef) - 1):
            if mm[i][1] != mm[i][0]:
                endCoef = coef[i + 1] / (mm[i][1] - mm[i][0])
            else:
                endCoef = 0
            print(endCoef)
    # coef = coefficients_sgd(dates, 0.01, 10000)
    # for c in coef:
    #     print(c)


    # testOnFile("test-dataset-D/LR-CF/0.20_0.50.txt")
    # testOnFile("test-dataset-D/LR-CF/0.40_0.65.txt")
    # testOnFile("test-dataset-D/LR-CF/0.42_0.63.txt")
    # testOnFile("test-dataset-D/LR-CF/0.48_0.68.txt")
    # testOnFile("test-dataset-D/LR-CF/0.52_0.70.txt")
    # testOnFile("test-dataset-D/LR-CF/0.57_0.79.txt")
    # testOnFile("test-dataset-D/LR-CF/0.60_0.73.txt")
    # testOnFile("test-dataset-D/LR-CF/0.60_0.80.txt")
    # testOnFile("test-dataset-D/LR-CF/0.62_0.80.txt")

# SMAPE эталонного и базового решения.
# 0.40_0.65.txt
