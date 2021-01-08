import math


if __name__ == '__main__':
    n = int(input())
    features = []
    top = n * 1.0
    for _ in range(n):
        elems = list(map(int, input().split(" ")))
        features.append(elems)
    s = 0.0
    xs = 0.0
    xsk = 0.0
    ys = 0.0
    ysk = 0.0
    for el in features:
        s += el[0] * el[1]
        xs += el[0]
        xsk += el[0] ** 2
        ys += el[1]
        ysk += el[1] ** 2
    top *= s
    top = top - xs * ys
    bottom = (n * xsk - xs ** 2) * (n * ysk - ys ** 2)
    if bottom < 0.000000001:
        print(0)
    else:
        print(top / math.sqrt(bottom))




