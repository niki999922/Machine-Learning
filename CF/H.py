if __name__ == '__main__':
    m = int(input())
    answers = []
    for _ in range(2 ** m):
        answers.append(input())
    print(f'2\n{2 ** m} 1')
    for i in range(2 ** m):
        b = 0.5
        for xi in range(m):
            x = i & (1 << xi)
            if x > 0:
                b -= 1
                print(1, end=" ")
            else:
                print(-10_000, end=" ")
        print(b)
    print(" ".join(answers) + " -0.5")
