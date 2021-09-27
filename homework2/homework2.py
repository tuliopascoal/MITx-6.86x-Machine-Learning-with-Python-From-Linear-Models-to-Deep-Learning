import numpy as np

def empirical_risk_with_hinge_loss(theta, y_array=[],x_array=[]):
    loss = []
    z = 0
    for x,y in zip(x_array, y_array):
        print("multi:", np.dot(theta,x))
        z = y - np.dot(theta, x)
        print("agreement z  = (y - multi):", y-np.dot(theta,x))
        if (z) >= 1:
            z = 0
        else:
            z = 1 - z
        print("loss:", z)
        loss.append(z)
    print("hinge loss array:", loss)
    print("np.sum(loss): ", np.sum(loss))
    print("len(x_array):", len(x_array))
    print("(np.sum(loss)/len(x_array)):", (np.sum(loss)/len(x_array)))
    return round(np.sum(loss)/len(x_array),2) #round to 2 decimals

def empirical_risk_with_hinge_loss_squared(theta, y_array=[],x_array=[]):
    loss = []
    z = 0
    for x,y in zip(x_array, y_array):
        print("multi:", np.dot(theta,x))
        z = y - np.dot(theta, x)
        print("agreement z  = (y - multi):", y-np.dot(theta,x))
        squared_z = (z*z) / 2
        print("loss:", squared_z)
        loss.append(squared_z)

    print("hinge loss array:", loss)
    print("np.sum(loss): ", np.sum(loss))
    print("len(x_array):", len(x_array))
    print("(np.sum(loss)/len(x_array)):", (np.sum(loss)/len(x_array)))
    return round(np.sum(loss)/len(x_array),2) #round to 2 decimals

theta = np.array([0,1,2]).T

x1 = np.array([1,0,1]).T
x2 = np.array([1,1,1]).T
x3 = np.array([1,1,-1]).T
x4 = np.array([-1,1,1]).T
x_array = [x1, x2, x3, x4]

y1 = 2
y2 = 2.7
y3 = -0.7
y4 = 2
y_array = [y1, y2, y3, y4]

print("R:", empirical_risk_with_hinge_loss(theta, y_array, x_array))
print("\n\n")
print("R with squared loss:", empirical_risk_with_hinge_loss_squared(theta, y_array, x_array))