#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#loading dataset
data = pd.read_csv('linear_regression.csv')
data.head()

spliting the dataset into feature and target variable
x = data.iloc[:, -2].values
y = data.iloc[:, -1].values

#feature scaling
s = np.std(x)    
m = np.mean(x)
def feature_scaling(variable):
    global m
    global s
    lst = []
    for i in variable:
        fs_x = (i - m) / s
        lst.append(fs_x)
    variable = np.array(lst)
    
    return variable
    
X = feature_scaling(x)  #Feature scaling x

#cost function
def cost(X, y, t0, t1):   #Function to find the cost
    cost = 0
    m = float(len(X))
    
    for i in range(len(X)):
        cost += ((t0 + t1 * X[i]) - y[i]) ** 2
    
    return cost / (2 * m)

#step-by-step_gradient
def gradient_update(X, y, t0_current, t1_current, alpha):   #single step of gradient descent
    t0_gradient = 0  #Initialising d/d(t0) to 0
    t1_gradient = 0  #Initialising d/d(t1) to 0
    m = float(len(X))  
    
    for i in range(len(X)):
        t0_gradient += 1/m * ((t0_current + t1_current * X[i]) - y[i])
        t1_gradient += 1/m * ((t0_current + t1_current * X[i]) - y[i]) * X[i]
        
    new_t0 = t0_current - alpha * t0_gradient  #Updating for t0
    new_t1 = t1_current - alpha * t1_gradient  #Updating for t1
    
    return new_t0, new_t1
    
#gradient descent
def gradient_descent(X, y, t0, t1, n_iter, alpha):  #function to run single step gradient descent as many iterations we need
       
    for i in range(n_iter):
        t0, t1 = gradient_update(X, y, t0, t1, alpha)  #Updating t0 and t1 iteratively.
    
    return t0, t1

#Start running
t0 = 0  #Initialising t0 as 0
t1 = 0  #Initialising t1 as 0

n_iter = 2000  #Number of iteration (minimum 1000)
alpha = 0.01  #Learning Rate (generally 0.01, 0.1, 0.05)

print('Before running gradient descent \ntheta_0 = {0}, \ntheta_1 = {1}, \nCost = {2}'.format(t0, t1, cost(X, y, t0, t1)))
print('\nRunning...')

t0, t1 = gradient_descent(X, y, t0, t1, n_iter, alpha)  #Running Gradient Descent

print('\nAfter running gradient descent \ntheta_0 = {0}, \ntheta_1 = {1}, \nCost = {2}'.format(t0, t1, cost(X, y, t0, t1)))

#plotting the best-fit line
plt.scatter(X, y, color = 'r', alpha = 0.4)
plt.plot(X, t0 + t1 * X, color = 'b', linewidth = 2)
plt.xlabel('X (feature)')
plt.ylabel('y (target)')
plt.title('X vs y')

#prediction
def predict(pred):
    pred = feature_scaling([pred])
    
    return t0 + t1 * pred

#sample prediction
predict([70, 100, 82, 64.5, 92.38768])  #You can pass list, array, single int or float
