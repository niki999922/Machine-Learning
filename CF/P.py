if __name__ == '__main__':
    k1, k2 = map(int, input().split(" "))
    n = int(input())
    xs, ys = [], []
    ex = [0 for _ in range(k1)]
    ey = [0 for _ in range(k2)]
    ans = n
    m = {}
    for _ in range(n):
        x, y = map(int, input().split(" "))
        ex[x - 1] += 1
        ey[y - 1] += 1
        if not m.__contains__(f'{x-1} {y-1}'):
            m[f'{x - 1} {y - 1}'] = 0
        m[f'{x-1} {y-1}'] += 1
        xs.append(x)
        ys.append(y)
    for i in range(k1):
        ex[i] /= n
    for i in range(k2):
        ey[i] /= n
    for key in m:
        x, y = map(int, key.split(" "))
        ek = n * ex[x] * ey[y]
        ans -= ek
        ans += (m[key] - ek) ** 2 / ek
    print(ans)


