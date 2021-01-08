import math

if __name__ == '__main__':
    k1, k2 = map(int, input().split(" "))
    n = int(input())
    xs, ys = [], []
    ans = 0
    p = [0 for _ in range(k1)]
    m = {}
    for _ in range(n):
        x, y = map(int, input().split(" "))
        p[x - 1] += 1 / n
        if not m.__contains__(f'{x - 1} {y - 1}'):
            m[f'{x - 1} {y - 1}'] = 0
        m[f'{x - 1} {y - 1}'] += 1 / n
    for key in m:
        x, _ = map(int, key.split(" "))
        ans += -m[key] * math.log(m[key] / p[x])
    print(ans)
