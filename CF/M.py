if __name__ == '__main__':
    n = int(input())
    xs, ys = [], []
    for i in range(n):
        args = list(map(int, input().split(" ")))
        xs.append([args[0], i])
        ys.append([args[1], i])
    xxs = sorted(xs)
    yys = sorted(ys)
    top = 0.0
    for i in range(n):
        xs[xxs[i][1]][1] = i
        ys[yys[i][1]][1] = i
    for i in range(n):
        d = (xs[i][1] - ys[i][1]) ** 2
        top += d
    top = 6 * top
    top = 1 - top / (n * (n ** 2 - 1))
    print(top)