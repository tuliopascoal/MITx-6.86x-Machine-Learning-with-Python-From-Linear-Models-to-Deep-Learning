import numpy
import matplotlib.pyplot as plt
import scipy
import tqdm
from collections import OrderedDict
from matplotlib.pyplot import figure
import numpy as np
from itertools import chain


def isInside(circle_x, circle_y, rad, x, y):
    # Compare radius of circle
    # with distance of its center
    # from given point
    if ((x - circle_x) * (x - circle_x) +
            (y - circle_y) * (y - circle_y) <= rad * rad):
        return True
    else:
        return False

def perceptron(x, y, w0=0, start=0):
    """
    Decision boundary: wx(i) + w0 = 0
    """

    n = x.shape[0]
    print("n:", n)
    w = np.zeros(2)
    weights = []  # updating progress
    while True:
        converged = True
        for i in chain(range(start, n), range(start)):
            print("x[i]:", x[i])
            print("y[i]:", y[i])
            #print("w.dot(x[i]):", w.dot(x[i]))
            #print("w0:", w0)
            #print("result:", y[i] * (w.dot(x[i]) + w0))
            if y[i] * (w.dot(x[i]) + w0) <= 0:  # misclassified
                w += y[i] * x[i]  # update weight
                weights.append(w.tolist())  # capture intermediate weight
                print("theta:", w)
                converged = False

        if converged:
            break

    return weights


def misclassify(x, y, w=None, iters=1000):
    """
    Decision boundary: wx(i) = 0
    w: if None, initialize to 0
    """

    n, d = x.shape
    x = np.column_stack((np.ones(n), x))  # add preceding 1's
    if w is None:
        w = np.zeros(d + 1)  # add preceding offset
    else:
        w = np.array(w, dtype=float)
    print("w:", w)
    counts = np.zeros(n, dtype=int)  # count of mistakes on each point
    mistakes = []  # intermediate mistakes
    weights = []  # intermediate weights
    for k in range(iters):
        converged = True
        for i in range(n):
            if y[i] * w.dot(x[i]) <= 0:  # misclassified
                w += y[i] * x[i]  # update weight
                counts[i] += 1  # update mistake count
                weights.append(w.tolist())  # capture intermediate weight
                mistakes.append(counts.tolist())  # capture intermediate mistakes
                converged = False

        if converged:
            break

    return mistakes, weights

################ 1. Perceptron Mistakes ################

x = np.array([[-1, -1], [1, 0], [-1, 1.5]])
y = np.array([1, -1, 1])
print(perceptron(x, y))
print(perceptron(x, y, start=1))

print("OK \n")
x2 = np.array([[-1,-1], [1,0], [-1, 10]])
y2 = np.array([1, -1, 1])
print(perceptron(x2, y2))
print(perceptron(x2, y2, start=1))

print("OK \n")
plt.scatter(x[:, 0], x[:, 1], color="r", marker='^')
plt.scatter(x2[:, 0], x2[:, 1], color="b", alpha=.4)
print("Look at the plot")
plt.show()
print("OK \n")

################ 2. Perceptron Performance ################

x3 = np.array([[-4, 2], [-2, 1], [-1, -1], [2, 2], [1, -2]])
y3 = np.array([1, 1, -1, -1, -1])
mistakes, weights = misclassify(x3, y3)
print("mistakes:", mistakes)
print("weights:", weights)
print("OK \n")

mistakes, weights = misclassify(x3, y3, w=(-10, -10, 10))
print("mistakes:", mistakes)
print("weights:", weights)
print("OK \n")

################ 3. Decision Boundaries ################

x = 1
y = 1
circle_x = -2
circle_y = -2
rad = 4
if (isInside(circle_x, circle_y, rad, x, y)):
    print("Inside")
else:
    print("Outside")