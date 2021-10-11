import numpy as np
import math

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

x = np.array([1,0,0]).T
x_prime = np.array([0,1,0]).T
distance = np.linalg.norm(np.subtract(x, x_prime))
print("distance:", distance)
distance = distance * distance
print("Radial basis kernel result:", pow(math.e,-distance/2))


########## HOMEWORK 2 ##########

y = np.array([[5, 0, 7], [0, 2, 0], [4, 0, 0], [0, 3, 6]])
print("Y matrix:", y)
u = np.array([[6], [0], [3], [6]])
v = np.array([4, 2, 1]).T
print("u:", u)
print("v:", v)
uv = u*v
print("u*(v.T): ", uv)
print("uv[0][1]:", uv[0][1])


total_loss = 0
a_count = 0
i_count = 0
### Squared Error term ###
for a in y:
    i_count = 0
    for i in a:
        print("Y_ai:", i)
        print("a_count:", a_count)
        print("i_count:", i_count)
        print("uv[a][i]:", uv[a_count][i_count])
        loss = (i - uv[a_count][i_count]) * (i - uv[a_count][i_count])
        #loss = (i - uv[a_count][i_count]) * (i - uv[a_count][i_count]) / 2
        if (uv[a_count][i_count] == 0):
            loss = 0
        print("loss: ", loss, "\n")
        total_loss = total_loss + loss
        i_count += 1
    a_count += 1
print("total loss:", total_loss)
print("squared error:", total_loss/2)

### Regularization term ###
reg_u_total = 0
for i in u:
    print("i:", i)
    reg_u = i*i
    reg_u_total = reg_u_total + reg_u
print("total reg_u_total:", reg_u_total)
print("reg term u:", reg_u_total/2)

reg_v_total = 0
for i in v:
    print("i:", i)
    reg_v = i*i
    reg_v_total = reg_v_total + reg_v
print("total reg_v_total:", reg_v_total)
print("reg term v:", reg_v_total/2)

print("total regularization term: ", (reg_u_total/2)+reg_v_total/2)