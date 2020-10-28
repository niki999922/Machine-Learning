import math


def printTable(table):
    for i in range(len(table)):
        print(i.__str__() + ": " + table[i].__str__(), end="\n")


def evalProbability(classes, words, k, alpha):
    def evalCountWordEachClasses():
        wordsTmp = {word: {i: 0 for i in range(k)} for word in words}
        for key in classes:
            arrays = classes[key]
            for array in arrays:
                wasTouched = set()
                for el in array:
                    if not wasTouched.__contains__(el):
                        wasTouched.add(el)
                        wordsTmp[el][key] = wordsTmp[el][key] + 1
        return wordsTmp

    result = {i: {word: 0 for word in words} for i in range(k)}
    wordsStats = evalCountWordEachClasses()
    for i in range(k):
        for word in words:
            result[i][word] = ((wordsStats[word][i] + alpha), (len(classes[i]) + alpha * 2))
    return result



if __name__ == '__main__':
    k = int(input())
    lambdas = [int(el) for el in input().split(" ")]
    alpha = int(input())
    n = int(input())
    classes = {i: list() for i in range(k)}
    c_count = {i: 0 for i in range(k)}
    words = set()
    for i in range(n):
        inp = input().split(" ")
        classTmp = int(inp[0])
        c_count[classTmp - 1] = c_count[classTmp - 1] + 1
        count = int(inp[1])
        arr = list()
        for j in range(count):
            word = inp[2 + j]
            words.add(word)
            arr.append(word)
        classes[classTmp - 1].append(arr)
    c_w_p = evalProbability(classes, words, k, alpha)
    # printTable(c_w_p)

    # need c_count and c_w_p
    m = int(input())
    for _ in range(m):
        count, *wordsTmp_ = input().split()
        foundWords = set(wordsTmp_)
        notFoundWords = words - foundWords
        res_p = [0 for i in range(k)]
        for i in range(k):
            resTmp = lambdas[i] * c_count[i] / n
            for word in foundWords:
                if c_w_p[i].__contains__(word):
                    resTmp *= c_w_p[i][word][0] / c_w_p[i][word][1]
            for word in notFoundWords:
                resTmp *= (c_w_p[i][word][1] - c_w_p[i][word][0]) / c_w_p[i][word][1]
            res_p[i] = resTmp
        for i in range(k):
            print(res_p[i] / sum(res_p), end=" ")
        print("")

    # need c_count and c_w_p with logs
    # m = int(input())
    # for _ in range(m):
    #     count, *wordsTmp_ = input().split()
    #     foundWords = set(wordsTmp_)
    #     notFoundWords = words - foundWords
    #     res_p = [0 for i in range(k)]
    #     for i in range(k):
    #         resTmp = math.log(lambdas[i] * c_count[i]) - math.log(n)
    #         for word in foundWords:
    #             if c_w_p[i].__contains__(word):
    #                 resTmp += math.log(c_w_p[i][word][0]) - math.log(c_w_p[i][word][1])
    #         for word in notFoundWords:
    #             resTmp += math.log(c_w_p[i][word][1] - c_w_p[i][word][0]) - math.log(c_w_p[i][word][1])
    #         res_p[i] = math.exp(resTmp)
    #     for i in range(k):
    #         print(res_p[i] / sum(res_p), end=" ")
    #     print("")

