import numpy as np
import math

nodes = []


class Tnh:
    def __init__(self, ind):
        self.node_ind = ind

    def main_calc(self):
        def foreach_h(x_arr):
            for i in range(len(x_arr)):
                for j in range(len(x_arr[i])):
                    x_arr[i][j] = math.tanh(x_arr[i][j])
            return x_arr

        self.array = foreach_h(np.copy(nodes[self.node_ind].array))
        self.back_array = np.zeros(shape=self.array.shape)
        self.back_start_array = np.zeros(shape=self.array.shape)

    def back_calc(self):
        def foreach_h(x_arr):
            for i in range(len(x_arr)):
                for j in range(len(x_arr[i])):
                    x_arr[i][j] = 1 - x_arr[i][j] ** 2
            return x_arr

        arr_tmp = foreach_h(np.copy(self.array))
        self.back_array += self.back_start_array * arr_tmp


        nodes[self.node_ind].back_start_array += np.copy(self.back_array)


class Had:
    def __init__(self, indexes):
        self.node_indexes = indexes
        self.array = None
        self.back_array = None
        self.back_start_array = None

    def main_calc(self):
        def foreach_f(x_arr_1, x_arr_2):
            for i in range(len(x_arr_1)):
                for j in range(len(x_arr_1[i])):
                    x_arr_1[i][j] = x_arr_1[i][j] * x_arr_2[i][j]
            return x_arr_1

        tmp = np.ones(shape=nodes[self.node_indexes[0]].array.shape)
        for ind in self.node_indexes:
            tmp = foreach_f(tmp, nodes[ind].array)
        self.array = tmp
        self.back_array = np.zeros(shape=self.array.shape)
        self.back_start_array = np.zeros(shape=self.array.shape)


    def back_calc(self):
        self.back_array += self.back_start_array

        for i in range(len(nodes[self.node_indexes[0]].array)):
            for j in range(len(nodes[self.node_indexes[0]].array[0])):
                for k in range(len(self.node_indexes)):
                    mult = 1.0
                    for z in range(len(self.node_indexes)):
                        if z != k:
                            mult *= nodes[self.node_indexes[z]].array[i][j]
                    nodes[self.node_indexes[k]].back_start_array[i][j] += mult * self.back_array[i][j]


class Var:
    def __init__(self, row, column):
        self.array = np.zeros(shape=(row, column))
        self.back_array = np.zeros(shape=(row, column))
        self.back_start_array = np.zeros(shape=(row, column))

    def main_calc(self):
        pass

    def back_calc(self):
        self.back_array = self.back_start_array


class Rlu:
    def __init__(self, alpha, ind):
        self.node_ind = ind
        self.alpha = alpha
        self.beta = 1.0 / alpha
        self.array = None

    def main_calc(self):
        def foreach_h(x_arr):
            for i in range(len(x_arr)):
                for j in range(len(x_arr[i])):
                    if x_arr[i][j] > 0:
                        x_arr[i][j] = x_arr[i][j]
                    else:
                        x_arr[i][j] = self.beta * x_arr[i][j]
            return x_arr


        self.array = foreach_h(np.copy(nodes[self.node_ind].array))
        self.back_array = np.zeros(shape=self.array.shape)
        self.back_start_array = np.zeros(shape=self.array.shape)

    def back_calc(self):
        def foreach_h(x_arr):
            for i in range(len(x_arr)):
                for j in range(len(x_arr[i])):
                    if x_arr[i][j] >= 0:
                        x_arr[i][j] = 1
                    else:
                        x_arr[i][j] = self.beta
            return x_arr

        # arr_tmp = np.vectorize(h)(self.array)
        arr_tmp = foreach_h(np.copy(self.array))
        # self.back_array = self.back_start_array.dot(arr_tmp)
        self.back_array += self.back_start_array * arr_tmp


        nodes[self.node_ind].back_start_array += np.copy(self.back_array)



class Mul:
    def __init__(self, ind_x1, ind_x2):
        self.node_ind_x1 = ind_x1
        self.node_ind_x2 = ind_x2
        self.array = None
        self.back_array = None
        self.back_start_array = None

    def main_calc(self):
        self.array = nodes[self.node_ind_x1].array.dot(nodes[self.node_ind_x2].array)
        self.back_array = np.zeros(shape=self.array.shape)
        self.back_start_array = np.zeros(shape=self.array.shape)

    def back_calc(self):
        self.back_array += self.back_start_array
        nodes[self.node_ind_x1].back_start_array += self.back_array.dot(nodes[self.node_ind_x2].array.transpose())
        nodes[self.node_ind_x2].back_start_array += nodes[self.node_ind_x1].array.transpose().dot(self.back_array)



class Sum:
    # list of index in indexes
    def __init__(self, indexes):
        self.node_indexes = indexes
        self.array = None
        self.back_array = None
        self.back_start_array = None

    def main_calc(self):
        tmp = np.zeros(shape=nodes[self.node_indexes[0]].array.shape)
        for ind in self.node_indexes:
            tmp = tmp + nodes[ind].array
        self.array = tmp
        self.back_array = np.zeros(shape=self.array.shape)
        self.back_start_array = np.zeros(shape=self.array.shape)

    def back_calc(self):
        self.back_array += self.back_start_array
        for ind in self.node_indexes:
            nodes[ind].back_start_array += np.copy(self.back_array)

def printCFAns():
    print("_________\nCF ans")
    print("""0.0 -0.1
-3.8 2.0 -1.9
2.0 -0.2
-3.0 0.3
-5.0 0.5
-1.0 0.1""")

def do_main_go():
    for i in range(len(nodes)):
        nodes[i].main_calc()

def do_back_go():
    for i in range(len(nodes) - 1, -1, -1):
        nodes[i].back_calc()

def print_k(k):
    for i in range(k):
        arr = nodes[len(nodes) - k + i].array
        for i1 in range(len(arr)):
            print(" ".join(map(str, arr[i1].tolist())))

def print_m(m):
    for i in range(m):
        arr = nodes[i].back_array
        for i1 in range(len(arr)):
            print(" ".join(map(str, arr[i1].tolist())))

if __name__ == '__main__':
    n, m, k = map(int, input().split(" "))
    var_rows = []
    for _ in range(n):
        args = input().split(" ")
        if args[0] == "var":
            nodes.append(Var(int(args[1]), int(args[2])))
            var_rows.append(int(args[1]))
        elif args[0] == "tnh":
            nodes.append(Tnh(int(args[1]) - 1))
        elif args[0] == "rlu":
            nodes.append(Rlu(int(args[1]), int(args[2]) - 1))
        elif args[0] == "mul":
            nodes.append(Mul(int(args[1]) - 1, int(args[2]) - 1))
        elif args[0] == "sum":
            args.pop(0)
            args.pop(0)
            nodes.append(Sum(list(map(lambda el: int(el) - 1, args))))
        else:
            args.pop(0)
            args.pop(0)
            nodes.append(Had(list(map(lambda el: int(el) - 1, args))))
    nodeNumber = 0
    for i in range(len(var_rows)):
        for j in range(var_rows[i]):
            nodes[nodeNumber].array[j] = np.array(list(map(float, input().split(" "))))
        nodeNumber += 1

    do_main_go()


    for i in range(k):
        array = nodes[len(nodes) - k + i].array
        tmp = np.zeros(shape=array.shape)
        for j in range(array.shape[0]):
            tmp[j] = np.array(list(map(float, input().split(" "))))
        nodes[len(nodes) - k + i].back_start_array += tmp

    do_back_go()

    print_k(k)
    print_m(m)

    # printCFAns()
