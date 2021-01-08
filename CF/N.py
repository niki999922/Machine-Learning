if __name__ == '__main__':
    k = int(input())
    n = int(input())
    inner = 0
    outer = 0
    inp = []
    suf_ar_m, pref_ar_m = {}, {}
    m = {}
    for _ in range(n):
        x, y = map(int, input().split(" "))
        if not m.__contains__(y):
            m[y] = []
        m[y].append(x)
        inp.append((x, y))

    inp = sorted(inp)
    for key in m:
        m[key] = sorted(m[key])

    for key in m:
        arr = m[key]
        suf = sum(arr)
        pref = 0
        for i in range(len(arr)):
            suf -= arr[i]
            pref += arr[i]
            inner += (suf - arr[i] * (len(arr) - i - 1)) + (arr[i] * (i + 1) - pref)


    suf = 0
    pref = 0
    for el in inp:
        x, y = el
        if not suf_ar_m.__contains__(y):
            suf_ar_m[y] = (0, 0)
        if not pref_ar_m.__contains__(y):
            pref_ar_m[y] = (0, 0)
        suf_ar_m[y] = (suf_ar_m[y][0] + x, suf_ar_m[y][1] + 1)
        suf += x

    for i in range(len(inp)):
        x, y = inp[i]

        suf -= x
        pref += x
        suf_ar_m[y] = (suf_ar_m[y][0] - x, suf_ar_m[y][1] - 1)
        pref_ar_m[y] = (pref_ar_m[y][0] + x, pref_ar_m[y][1] + 1)

        outer += (suf - suf_ar_m[y][0]) - ((n - i - 1) - suf_ar_m[y][1]) * x + ((i + 1) - pref_ar_m[y][1]) * x - (pref - pref_ar_m[y][0])

    print(inner)
    print(outer)
