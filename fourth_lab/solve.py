import numpy as np


def double_simplex(c, b, A, j):
    m, n = A.shape
    x_0 = [0 for _ in range(n)]
    j -= 1

    A_inv = np.linalg.inv(A[:, j])
    y = (c[j]).dot(A_inv)

    iter = 1
    while True:
        print(f"Iteration {iter}")
        j_n = np.delete(np.arange(n), j)
        A_inv = np.linalg.inv(A[:, j])
        print('Inverse A:', A_inv)

        kappa = A_inv.dot(b)
        print("Kappa:", kappa)

        if all(kappa >= 0):
            for j, copy in zip(j, kappa):
                x_0[j] = copy

            return x_0

        ind = np.argmin(kappa)
        delta_y = A_inv[ind]
        mu = delta_y.dot(A)

        print("mu:", mu)
        print("y:", y)

        sigma = []
        for i in j_n:
            sigma.append(np.inf) if mu[i] >= 0 else sigma.append((c[i] - A[:, i].dot(y)) / mu[i])

        print("sigma:", sigma)

        sigma_0_ind = j_n[np.argmin(sigma)]
        sigma_0 = min(sigma)

        print(f"sigma_val: {sigma_0}")
        print(f"sigma_index: {sigma_0_ind}")

        if sigma_0 == np.inf:
            print("Прямая задача несовместна")
            return

        y += sigma_0 * delta_y
        j[ind] = sigma_0_ind

        iter += 1


if __name__ == "__main__":
    A = np.array([
        [-2, -1, -4, 1, 0],
        [-2, -2, -2, 0, 1]
    ])
    b = np.array([-1, -1.5])
    c = np.array([-4, -3, -7, 0, 0])
    J = np.array([4, 5])

    result = double_simplex(c, b, A, J)
    print("Result:", result)

    A = np.array([
        [1, -7, 1, 0],
        [-2, -3, 1, 1]
    ])
    b = np.array([-14, -23])
    c = np.array([0, -8, 1, 0])
    J = np.array([3, 4])

    result = double_simplex(c, b, A, J)
    print("Result:", result)
