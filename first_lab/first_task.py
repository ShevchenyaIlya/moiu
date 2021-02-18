import copy

import numpy as np


def input_matrix(filename):
    with open(filename, "r") as file:
        n, i = map(int, file.readline().split(" "))
        i -= 1

        a = np.zeros((n, n))
        a_inv = np.zeros((n, n))
        x = np.zeros(n)
        for index, line in enumerate(file.readlines()):
            if index < 3:
                a[index:] = list(map(int, line.split(" ")))
            elif index < 6:
                a_inv[index - 3 :] = list(map(int, line.split(" ")))
            else:
                x = list(map(int, line.split(" ")))

        return a, a_inv, x, n, i


def check_answer(a, b, x, i):
    a[:, i] = x
    b_true = np.linalg.inv(a)
    if np.allclose(b, b_true):
        return True

    return False


def inverse_matrix(A_inv, x, i):
    l = np.dot(A_inv, x)
    n = len(A_inv)

    if l[i] == 0:
        print("Error. Matrix A is not invertible !")
        return None

    l_copy = copy.copy(l)
    l_copy[i] = -1

    l_copy = [-1 / l[i] * x for x in l_copy]

    Q = np.eye(n)
    Q[:, i] = l_copy

    return multiply_matrix(Q, A_inv, n, i)


def multiply_matrix(Q, A_inv, n, i):
    R = np.zeros((n, n))

    for k in range(n):
        for j in range(n):
            if i == j:
                R[j][k] = Q[j][i] * A_inv[k][j]
            else:
                R[j][k] = A_inv[j][k] * Q[j][j] + A_inv[i][k] * Q[j][i]

    return R


def np_matrix_multiplication(Q, A):
    return np.dot(Q, A)


if __name__ == "__main__":
    files = ["input.txt", "input_1.txt", "input_2.txt"]

    for index, file in enumerate(files):
        print(f"Test {index}:")
        a, a_inv, x, n, i = input_matrix(file)
        result = inverse_matrix(a_inv, x, i)

        if result is not None:
            print(result)
