import numpy as np
from second_lab.simplex_method import simplex_method


def start_phase(A, b):
    m, n = A.shape
    J_b = [n + i for i in range(m)]

    for index, b_i in enumerate(b):
        if b_i < 0:
            b[index] *= -1
            A[index, :] = [A[index, j] * (-1) for j in range(n)]

    x = [0 for _ in range(n)] + b
    c = [0 for _ in range(n)] + [-1 for _ in range(m)]

    A_new = np.hstack((A, np.eye(m)))
    print("New A: ", A_new)
    print("J_b: ", J_b)
    print("c:", c)
    print("x:", x)
    opt_pl, J_b_ans = simplex_method(A_new, c, x, J_b)

    for j in range(m):
        if x[n + j] != 0:
            print("Задача несовместна!")
            return

    print("Задача совместна")
    print(J_b_ans, " - базисный план полученный симплекс методом для искусственно созданной задачи")
    while True:
        i, artificial_j, J_b_len = -1, -1, len(J_b_ans)
        for j in range(J_b_len):
            if J_b_ans[j] > n - 1:
                i = J_b_ans[j] - n + 1
                artificial_j = j
                break

        if i == -1:
            return x[:n], [i + 1 for i in J_b_ans]

        Ab = np.zeros((J_b_len, J_b_len))
        for k in range(J_b_len):
            Ab[:, k] = A_new[:, J_b_ans[k]]

        next_iteration = False
        for j in range(n):
            if j not in J_b_ans:
                if np.linalg.det(Ab) == 0:
                    break

                l_j = np.dot(np.linalg.inv(Ab), A_new[:, j])
                print(f"l({j}): ", l_j)
                if round(l_j[artificial_j], 10) != 0:
                    J_b_ans[artificial_j] = j
                    next_iteration = True
                    break

        if next_iteration:
            continue

        del J_b_ans[artificial_j]
        A_new = np.delete(A_new, artificial_j, axis=0)


def run_tests():
    A = np.array([
        [-1, 6, 1, 0],
        [4, 1, 0, 1],
        [3, 7, 1, 1],
    ])
    b = [18, 28, 46]

    A = np.array([
        [-2, 7, 1, 0, 0],
        [1, 1, 0, 1, 0],
        [0, -1, 0, 0, 1],
    ])
    b = [7/2, 5, -2]

    A = np.array([
        [-1, 5, 1, 0],
        [5, 1, 0, 1],
        [4, 6, 1, 1],
    ])
    b = [20, 30, 50]

    A = np.array([
        [1, 1, 1],
        [2, 2, 2],
    ])
    b = [0, 0]

    result = start_phase(A, b)

    if result is not None:
        print("Базисный блан:")
        print(result[0])
        print(result[1])


if __name__ == '__main__':
    run_tests()
