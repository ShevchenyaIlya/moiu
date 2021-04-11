from copy import copy

import numpy as np


def breadth_first_search(g, m, n, v, direction):
    queue, used = [v], [False for _ in range(m * n)]
    p = [0 for _ in range(m * n)]
    d = copy(p)

    p[v] = -1
    used[v] = True

    while len(queue):
        v = queue[-1]
        queue.pop()

        for x in g[v]:

            if used[x]:
                continue

            used[x] = True
            queue.append(x)
            d[x], p[x] = d[v] + 1, v

    path, v = [], direction

    while v != -1:
        path.append(v)
        v = p[v]

    return path


def border_vertex(i, j, J_b):
    row, col = [pos for pos in J_b if pos[0] == i and pos[1] != j], [
        pos for pos in J_b if pos[1] == j and pos[0] != i
    ]

    min_dist = np.inf
    nearest_row_pos = (i, j)

    for row_pos in row:
        if abs(row_pos[1] - j) < min_dist:
            min_dist = abs(row_pos[1] - j)
            nearest_row_pos = row_pos

    min_dist = np.inf
    nearest_col_pos = (i, j)

    for col_pos in col:
        if abs(col_pos[0] - i) < min_dist:
            min_dist = abs(col_pos[0] - i)
            nearest_col_pos = col_pos

    return nearest_row_pos, nearest_col_pos


def find_path(solution_tree, nearest_row_pos, nearest_col_pos, size):
    m, n = size

    return breadth_first_search(
        solution_tree,
        m,
        n,
        nearest_row_pos[0] * n + nearest_row_pos[1],
        nearest_col_pos[0] * n + nearest_col_pos[1],
    )


def get_side_vertex_positions(path):
    k, new_path = 0, []

    while k < len(path):
        pos = path[k]

        while k < len(path) and path[k][0] == pos[0]:
            k += 1

        if k < len(path):
            pos = path[k]
            new_path.append(path[k - 1])

        while k < len(path) and path[k][1] == pos[1]:
            k += 1

        if k < len(path):
            new_path.append(path[k - 1])

        if k >= len(path) and not (
            path[-1][0] == path[-2][0]
            and path[-1][0] == path[0][0]
            or path[-1][1] == path[-2][1]
            and path[-1][1] == path[0][1]
        ):
            new_path.append(path[-1])
            break

    return new_path


def get_odd_positions(path, path_extended):
    odd_positions = []
    for i in range(1, len(path_extended) // 2 + 1, 2):
        odd_positions.append(path_extended[i])

        if len(path) - i != i:
            odd_positions.append(path_extended[-i])

    print("Odd positions:", odd_positions)

    return odd_positions


def minimal_odd_position(odd_positions, x):
    min_odd_pos_value, min_odd_pos = np.inf, (-1, -1)

    for i, j in odd_positions:
        if x[i][j] < min_odd_pos_value:
            min_odd_pos_value = x[i][j]
            min_odd_pos = (i, j)

    print("Min odd pos value:", min_odd_pos_value)

    return min_odd_pos_value, min_odd_pos


def update_plan(x, path_extended, odd_positions, min_odd_pos_value):
    x_new = np.copy(x)

    for i, j in path_extended:
        if (i, j) in odd_positions:
            x_new[i][j] -= min_odd_pos_value
        else:
            x_new[i][j] += min_odd_pos_value

    return x_new


def update_path(x, path, J_b, new_pos):
    m, n = np.shape(x)
    J_b_new = []
    path_extended = [new_pos] + [(pos // n, pos - n * (pos // n)) for pos in path]
    side_path = get_side_vertex_positions(path_extended)
    path_extended = side_path
    print("Path:", path_extended)
    print("New path:", side_path)

    odd_positions = get_odd_positions(path, path_extended)
    min_odd_pos_value, min_odd_pos = minimal_odd_position(odd_positions, x)
    new_plan = update_plan(x, path_extended, odd_positions, min_odd_pos_value)

    for i, j in J_b:
        if (i, j) != min_odd_pos:
            J_b_new.append((i, j))

    J_b_new.append(new_pos)

    return new_plan, J_b_new


def tree(x, J):
    m, n = np.shape(x)
    tree_matrix = [[] for _ in range(m * n)]

    for i in range(m):
        for j in range(n):
            if (i, j) not in J:
                continue

            for k in range(j + 1, n):
                if (i, k) in J:
                    tree_matrix[n * i + j].append(n * i + k)
                    tree_matrix[n * i + k].append(n * i + j)
                    break

            for k in range(i + 1, m):
                if (k, j) in J:
                    tree_matrix[n * i + j].append(n * k + j)
                    tree_matrix[n * k + j].append(n * i + j)
                    break

    return tree_matrix


def potentials(x, c, J_b):
    m, n = np.shape(x)
    a, b = np.zeros((m + n, m + n)), np.zeros(m + n)
    k = 1

    for i, j in J_b:
        print(f"u[{i}] + v[{j}] = {c[i][j]}")

        a[k][i] = 1
        a[k][j + m] = 1
        b[k] = c[i][j]
        k += 1

    a[0][0] = 1
    solution = np.linalg.solve(a, b)
    print("Matrix representation:", a, sep="\n")
    print("Solution: ", solution)

    return solution[:m], solution[m:]


def check_potentials(c, u, v, J_b):
    for i, u_value in enumerate(u):
        for j, v_value in enumerate(v):
            if (i, j) not in J_b and u_value + v_value > c[i][j]:
                return i, j

    return -1, -1


def north_west_angle(a, b):
    m, n = len(a), len(b)
    x, J_b = np.zeros((m, n)), []
    a_copy, b_copy = np.copy(a), np.copy(b)
    i, j = 0, 0

    while i < m:
        while j < n:
            x[i][j] = min(a_copy[i], b_copy[j])
            a_copy[i] -= x[i][j]
            b_copy[j] -= x[i][j]
            J_b.append((i, j))

            if not a_copy[i]:
                i += 1
                break

            j += 1

    return x, J_b


def check_balance_condition(first, second) -> bool:
    return sum(first) != sum(second)


def matrix_transport_problem(a, b, c):
    m, n = np.shape(c)

    if check_balance_condition(a, b):
        print("Balance condition is not satisfied")
        return

    x, J_b = north_west_angle(a, b)
    print(f"North west method output: \nX = {x}, \nJ = {J_b}")

    while True:
        u, v = potentials(x, c, J_b)
        print(f"u = ({u})")
        print(f"v = ({v})")

        new_i, new_j = check_potentials(c, u, v, J_b)

        if all([new_i == -1, new_j == -1]):
            print(f"Optimal plan: \nX = {x}, \nJ_b = {J_b}")
            return

        nearest_row_pos, nearest_col_pos = border_vertex(new_i, new_j, J_b)
        path = find_path(tree(x, J_b), nearest_row_pos, nearest_col_pos, (m, n))
        x, J_b = update_path(x, path, J_b, (new_i, new_j))
        print(f"New X: {x}", f"New J_b: {J_b}", sep="\n")


if __name__ == "__main__":
    a = [100, 300, 300]
    b = [300, 200, 200]
    c = [[8, 4, 1], [8, 4, 3], [9, 7, 5]]

    matrix_transport_problem(a, b, c)
    print()

    a = [10, 10, 160]
    b = [30, 90, 60]
    c = [[2, 3, 4], [3, 1, 5], [1, 2, 1]]

    matrix_transport_problem(a, b, c)
