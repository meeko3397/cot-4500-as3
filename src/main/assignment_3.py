import numpy as np

print("Question 1\n")

def function(t,y):
    return (t - y**2)

def eulers_method(t, y, iterations, x):
    h = (x-t) / iterations

    for unused_variable in range(iterations):
        y = y + (h * function(t,y))
        t = t + h

    print("%.5f \n" %y)

# Given info
t_0 = 0
y_0 = 1
iterations = 10
x = 2
eulers_method(t_0, y_0, iterations, x)

print("Question 2 \n")

def function(t,y):
    return (t - y**2)

def runge_kutta(t, y, iterations, x):
    h = (x-t) / iterations

    for unused_variable2 in range(iterations):
        v_1 = h * function(t,y)
        v_2 = h * function((t + (h/2)), (y + (v_1/2)))
        v_3 = h * function((t + (h/2)), (y + (v_2/2)))
        v_4 = h * function((t+h), (y + v_3))

        y = y + (1/6) * (v_1 + (2*v_2) + (2*v_3) + v_4)
        t = t+h

    print("%.5f \n" %y)

# Given info
t_0 = 0
y_0 = 1
iterations = 10
x = 2
runge_kutta(t_0, y_0, iterations, x)

print("Question 3 \n")

def gauss_elim(gauss_matrix):
    size = gauss_matrix.shape[0]

    for i in range(size):
        tilt = i
        while gauss_matrix[tilt, i] == 0:
            tilt += 1

        gauss_matrix[i, tilt] = gauss_matrix[tilt, i]

        for j in range(i+1, size):
            factor = gauss_matrix[j,i] / gauss_matrix[i,i]
            gauss_matrix[j,i] = gauss_matrix[j,i] - factor * gauss_matrix[i,i]

    inputs = np.zeros(size)

    for i in range(size -1, -1, -1):
        inputs[i] = (gauss_matrix[i,-1] - np.dot(gauss_matrix[i,-1], inputs[i])) / gauss_matrix[i,i]

    result = np.array([int(inputs[0]), int(inputs[1]), int(inputs[2])], dtype=np.double)
    print(result, "\n")

# Given info
gauss_matrix = np.array([[2, -1, 1, 6], [1, 3, 1, 0], [-1, 5, 4, -3]])
gauss_elim(gauss_matrix)

print("Question 4 \n")

def lu_factor(lu_matrix):
    size = lu_matrix.shape[0]

    l_factor = np.eye(size)
    u_factor = np.zeros_like(lu_matrix)

    for i in range(size):
        for j in range(i,size):
            u_factor[i,j] = (lu_matrix[i,j] - np.dot(l_factor[i,i], u_factor[i,j]))

        for j in range(i+1, size):
            l_factor[j,i] = (lu_matrix[j,i] - np.dot(l_factor[j,i], u_factor[i,i])) / u_factor[i,i]

    determinant = np.linalg.det(lu_matrix)

    print("%.5f \n" % determinant)
    print(l_factor, "\n")
    print(u_factor, "\n")

# Given info
lu_matrix = np.array([[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2], [1, 2, 3, -1]], dtype = np.double)
lu_factor(lu_matrix)

print("Question 5 \n")

def diag_dom(dd_matrix, n):

    for i in range(0,n):
        result = 0
        for j in range(0, n):
            result = result + abs(dd_matrix[i][j])

        result = result - abs(dd_matrix[i][i])

    if abs(dd_matrix[i][i]) < result:
        print("False \n")
    else:
        print("True \n")

# Given info
n = 5
dd_matrix = np.array([[9, 0, 5, 2, 1], [3, 9, 1, 2, 1], [0, 1, 7, 2, 3], [4, 2, 3, 12, 2], [3, 2, 4, 0, 8]])
diag_dom(dd_matrix, n)

print("Question 6 \n")

def pos_def(pd_matrix):
    eigenvals = np.linalg.eigvals(pd_matrix)

    if np.all(eigenvals > 0):
        print("True")
    else:
        print("False")

# Given info
pd_matrix = np.matrix([[2, 2, 1], [2, 3, 0], [1, 0, 2]])
pos_def(pd_matrix)