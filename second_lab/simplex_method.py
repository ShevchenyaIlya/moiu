import numpy as np
import copy


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

    return np.dot(Q, A_inv)


def simplex_method(A, c, x):
    A_b_inv = None
    j = [i for i, value in enumerate(x) if value > 0]

    print("A = ", A)
    print("c = ", c)
    print("x = ", x)
    print("J = ", j)
    while True:
        A_b, c_b = np.column_stack([A[:, i] for i in j]), np.array([c[i] for i in j])
        A_b_inv = np.linalg.inv(A_b) if A_b_inv is None else inverse_matrix(A_b_inv, A[:, j_0], s)
        print("Inverse A: ", A_b_inv, end="\n\n")

        u = np.dot(c_b, A_b_inv)
        print("u = ", u)
        delta = np.dot(u, A) - c
        print("delta = ", delta)
        J_n = [i for i, k in enumerate(delta) if k < 0 and i not in j]

        if not J_n:
            print("Optimal plan: ", x)
            return x

        j_0 = J_n[0]
        print("j_0 = ", j_0)
        z = A_b_inv.dot(A[:, j_0])
        print("z = ", z)

        omega = [x[value] / z[i] if z[i] > 0 else np.inf for i, value in enumerate(j)]
        omega_0 = np.min(omega)

        if omega_0 == np.inf:
            print("Linear programming task is not limited!")
            return

        s = omega.index(omega_0)

        new_x_values = np.array([x[i] for i in j]) - z.dot(omega_0)
        x[j_0] = omega_0
        for i, element in enumerate(j):
            x[element] = new_x_values[i]

        j[s] = j_0
        print("J_b = ", j)
        print("x = ", x, end="\n\n")


if __name__ == '__main__':
    A = [
        [-1, 1, 1, 0, 0],
        [1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1],
    ]

    c = [1, 1, 0, 0, 0]

    x = [0, 0, 1, 3, 2]

    simplex_method(np.array(A), c, x)
    print()
    print()

    A = [
        [1, -1, 1, 0],
        [-1, 1, 0, 1],
    ]

    c = [1, 0, 0, 0]

    x = [1, 0, 0, 3]

    simplex_method(np.array(A), c, x)
    print()
    print()

    A = [
        [2, 1, 1, 0, 0, 0],
        [2, 9, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1],
    ]

    c = [22, 31, 1, 0, 0, 0]
    x = [0, 0, 22, 54, 10, 5]
    simplex_method(np.array(A), c, x)
    print()
    print()

    A = [
        [4, 1, 0, 1],
        [3, 7, 1, 1],
    ]

    c = [1, 1, 0, 0]

    x = [0, 0, 18, 28]

    simplex_method(np.array(A), c, x)
    print()
