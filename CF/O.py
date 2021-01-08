if __name__ == '__main__':
    k = int(input())
    n = int(input())
    xs, ys = [], []
    for i in range(n):
        args = list(map(int, input().split(" ")))
        xs.append(args[0])
        ys.append(args[1])

    ey2 = 0
    for i in range(len(xs)):
        ey2 += (ys[i] ** 2) / n

    eeyx2 = 0
    px, eyx = [0 for _ in range(k)], [0 for _ in range(k)]
    for i in range(len(xs)):
        px[xs[i] - 1] += 1 / n
        eyx[xs[i] - 1] += ys[i] / n
    for i in range(k):
        if px[i] != 0:
            eeyx2 += (eyx[i] ** 2) / px[i]
    ans = ey2 - eeyx2
    print(ans)
