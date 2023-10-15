import cvxpy as cp
import numpy as np
import time


def LP(solveMathine):
    x = cp.Variable(2)
    p = np.array([3, 4])
    objective = cp.Maximize(p.T @ x)

    con = np.array([[1, 1],
                    [-1, 1],
                    [-1, 0],
                    [0, -1]])
    b = np.array([5, -1, 0, 0])

    constraints = [con @ x <= b]

    prob = cp.Problem(objective, constraints)
    start_time = time.time()
    result = prob.solve(solver=solveMathine)
    end_time = time.time()
    solve_time = (end_time - start_time)*1000
    print("For solver", solver)
    print("The optimal value is: ", prob.value)
    print("The optimal x is: ", x.value)
    print("求解时间: {:.2f} 毫秒".format(solve_time))




def QP(solver):
    x = cp.Variable(3)
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9],
                  [10, 11, 12],
                  [13, 14, 15]
                  ])
    B = np.array([6, 15, 24, 33, 42])
    cost = cp.sum_squares(B - A @ x)
    object = cp.Minimize((1 / 2) * cost)
    constraints = []
    prob = cp.Problem(object, constraints)
    prob.solve(solver = solver)
    # 输出结果
    # Print result.
    print("For solver", solver)
    print("The optimal value is", prob.value)
    print("The optimal x is")
    print(x.value)


def QCQP(solver):
    x = cp.Variable(2)
    P = np.array([[1, 0],
                  [0, 1]])
    q = np.array([2, 4])
    con = np.array([1, 1])
    qCon = np.array([[1, 0],
                     [0, 1]])
    constraints = [cp.quad_form(x, qCon) <= 1, con.T @ x <= 0]
    object = cp.Minimize(cp.quad_form(x, P) + q.T @ x)
    prob = cp.Problem(object, constraints)
    prob.solve(solver = solver)
    print("\nThe optimal value is", prob.value)
    print("The optimal x is")
    print(x.value)



solvers = [cp.ECOS, cp.SCS, cp.OSQP]
for solver in solvers:
    LP(solver)
# q()
# QCQP()

