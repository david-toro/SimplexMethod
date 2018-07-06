# Simplex method to solve linear programming models, implemented using big M method
# External libraries: Numpy
# Author: David Toro Meneses

import numpy as np
import warnings


def simplex(type, A, B, C, D, M):
    """Calculates an optimal point for the linear programming model given by A*x <= B , Optimize z= C' * x

    Arguments:
    type -- type of optimization, it can be 'max' or 'min'
    A    -- A matrix of the model (numpy array)
    B    -- B matrix of the model, column vector (numpy array)
    C    -- C matrix of the model, column vector (numpy array)
    D    -- column vector with the types of restrictions of the model (numpy array), 1 is <=, 0 is =, -1 is >=
            for <= restrictions do nothing
            for = restrictions add an artificial variables and a big M in the objective
            function (min --> +M , max --> -M)
            for >= restrictions multiply by -1
    M    -- big M value
    """

    # m -- number of restrictions
    # n -- number of variables
    (m, n)= A.shape

    basic_vars = []
    count = n

    # matrix with new variables
    R = np.eye(m)

    # values of the new variables
    P = B

    # artificial variables position indicator
    artificial= []

    for i in range(m):
        if D[i] == 1:
            # add the slack variable to objective function
            C = np.vstack((C, [[0]]))

            # regist the slack variable as basic variable
            count = count + 1
            basic_vars = basic_vars + [count-1]

            artificial = [artificial, 0]

        elif D[i] == 0:
            # add the artificial variable to objective function with the big M value
            if type == 'min':
                C = np.vstack((C, [[M]]))
            else:
                C = np.vstack((C, [[-M]]))

            # regist the artificial variable as basic variable
            count = count + 1
            basic_vars = basic_vars + [count-1]

            artificial = [artificial, 1]
        elif D[i] == -1:
            # add the surplus and artificial variables to objective function
            if type == 'min':
                C = np.vstack((C, [[0], [M]]))
            else:
                C = np.vstack((C, [[0], [-M]]))

            R = repeatColumnNegative(R, count + 1 - n)
            P = insertZeroToCol(P, count + 1 - n)

            # regist the artificial variable as basic variable
            count = count + 2
            basic_vars = basic_vars + [count-1]

            artificial = [artificial, 0, 1]
        else:
            print("invalid case")

    # current vertex
    X = np.vstack((np.zeros((n, 1)), P))

    # add new variables to matrix A
    A = np.hstack((A, R))

    # simplex tableau
    st = np.vstack((np.hstack((-np.transpose(C), np.array([[0]]))), np.hstack((A, B))))

    # number of columns
    (rows, cols) = st.shape

    # basic_vars = ((n + 1):n+m)'

    print('\nsimplex tableau\n')
    print(st)
    print('\ncurrent basic variables\n')
    print(basic_vars)
    print('\noptimal point\n')
    print(X)

    # check if z != 0 (when there are artificial variables)
    z_optimal = np.matmul(np.transpose(C), X)

    print('\ncurrent Z\n\n', z_optimal)

    if z_optimal != 0:
        for i in range(m):
            if D[i] == 0 or D[i] == -1:
                if type == 'min':
                    st[0,:]= st[0,:] + M * st[1+i,:]
                else:
                    st[0,:]= st[0,:] - M * st[1+i,:]

        print('\ncorrected simplex tableau\n')
        print(st)

    iteration = 0
    while True:
    # for zz in range(2):
        if type == 'min':
            # select the more positive value
            w = np.amax(st[0, 0:cols-1])
            iw = np.argmax(st[0, 0:cols-1])
        else:
            # select the more negative value
            w = np.amin(st[0, 0:cols-1])
            iw = np.argmin(st[0, 0:cols-1])

        if w <= 0 and type == 'min':
            print('\nGlobal optimum point\n')
            break
        elif w >= 0 and type == 'max':
            print('\nGlobal optimum point\n')
            break
        else:
            iteration = iteration + 1

            print('\n----------------- Iteration {} -------------------\n'.format(iteration))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                T = st[1:rows, cols-1] / st[1: rows, iw]

            R = np.logical_and(T != np.inf, T > 0)
            (k, ik) = minWithMask(T, R)

            # current z row
            cz = st[[0],:]

            # pivot element
            pivot = st[ik+1, iw]

            # pivot row divided by pivot element
            prow = st[ik+1,:] / pivot

            st = st - st[:, [iw]] * prow

            # pivot row is a special case
            st[ik+1,:]= prow

            # new basic variable
            basic_vars[ik] = iw

            print('\ncurrent basic variables\n')
            print(basic_vars)

            # new vertex
            basic = st[:, cols-1]
            X = np.zeros((count, 1))

            t = np.size(basic_vars)

            for k in range(t):
                X[basic_vars[k]] = basic[k+1]

            print('\ncurrent optimal point\n')
            print(X)

            # new z value
            C = -np.transpose(cz[[0], 0:count])

            z_optimal = cz[0, cols-1] + np.matmul(np.transpose(C), X)
            st[0, cols-1] = z_optimal

            print('\nsimplex tableau\n\n')
            print(st)

            print('\ncurrent Z\n\n')
            print(z_optimal)

    # check if some artificial variable is positive (infeasible solution)
    tv = np.size(artificial)
    for i in range(tv):
        if artificial[i] == 1:
            if X[n + i] > 0:
                print('\ninfeasible solution\n')
                break

    return (z_optimal[0, 0], X)


def minWithMask(x, mask):

    min = 0
    imin = 0

    n = np.size(x)

    for i in range(n):
        if mask[i] == 1:
            if min == 0:
                min = x[i]
                imin = i
            else:
                if min > x[i]:
                    min = x[i]
                    imin = i

    return (min, imin)


def repeatColumnNegative(Mat, h):
    """Repeat column h multiplied by - 1"""
    (r, c) = Mat.shape
    Mat = np.hstack((Mat[:, 0:h-1], -Mat[:, [h-1]], Mat[:, h-1:c]))

    return Mat


def insertZeroToCol(col, h):
    """insert zero to column"""
    k = np.size(col)
    col = np.vstack((col[0:h-1, [0]], np.array([[0]]), col[h-1:k, [0]]))

    return col


if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    (z, x) = simplex('min', np.array([[3, 1], [4, 3], [1, 2]]),
                            np.array([[3], [6], [4]]),
                            np.array([[4], [1]]),
                            np.array([[0], [-1], [1]]),
                     100)
