import numpy as np

def randomization(n):
    """
    Arg:
      n - an integer
    Returns:
      A - a randomly-generated nx1 Numpy array.
    """
    #Your code here

    #A = np.array(n*1)
    A = np.random.random([n, 1])
    return A
    raise NotImplementedError

def operations(h, w):
    """
    Takes two inputs, h and w, and makes two Numpy arrays A and B of size
    h x w, and returns A, B, and s, the sum of A and B.

    Arg:
      h - an integer describing the height of A and B
      w - an integer describing the width of A and B
    Returns (in this order):
      A - a randomly-generated h x w Numpy array.
      B - a randomly-generated h x w Numpy array.
      s - the sum of A and B.
    """
    #Your code here
    A = np.random.random([h, w])
    B = np.random.random([h, w])
    print("A:", A)
    print("B:", B)
    s = A + B
    print(s)
    return A, B, s
    raise NotImplementedError


def norm(A, B):
    """
    Takes two Numpy column arrays, A and B, and returns the L2 norm of their
    sum.

    Arg:
      A - a Numpy array
      B - a Numpy array
    Returns:
      s - the L2 norm of A+B.
    """
    #Your code here
    s = A + B
    print(s)
    s = np.linalg.norm(s)
    print(s)
    return s
    raise NotImplementedError


def neural_network(inputs, weights):
    """
     Takes an input vector and runs it through a 1-layer neural network
     with a given weight matrix and returns the output.

     Arg:
       inputs - 2 x 1 NumPy array
       weights - 2 x 1 NumPy array
     Returns (in this order):
       out - a 1 x 1 NumPy array, representing the output of the neural network
    """
    #Your code here
    print("inputs: ", inputs)
    print("weights: ", weights)
    print("weights.transpose(): ", weights.transpose())
    z = np.tanh(np.matmul(weights.transpose(), inputs))
    print("z: ", z)
    return z
    raise NotImplementedError

def scalar_function(x, y):
    """
    Returns the f(x,y) defined in the problem statement.
    """
    #Your code here
    if x<=y:
        return x*y
    else:
        return x/y
    raise NotImplementedError

def vector_function(x, y):
    """
    Make sure vector_function can deal with vector input x,y 
    """
    #Your code here
    vector_function = np.vectorize(scalar_function)
    return vector_function(x,y)
    raise NotImplementedError


#n = 5
#randomization(n)
#operations(3,3)

#print(np.random.random([4,]))
#norm(np.random.random([4,]), np.random.random([4,]))

A = np.random.random([2, 1])
B = np.random.random([2, 1])
neural_network(A,B)