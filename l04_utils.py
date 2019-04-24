# Kasim Terzic (kt54) Feb 2018

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Our linear model defined as a list of theta parameters
# Accepts the samples and thetas and returns the vector of
# predictions
def f(x, theta):
    y_hat = []

    # Calculate our linear function
    # remember, X_0 is 1, so theta_0 is intercept
    for sample in x:
        y_acc = 0
        for i in range(len(sample)):
            y_acc = y_acc + theta[i]*sample[i]
        y_hat.append(y_acc)
    return np.array(y_hat)

# Our squared error loss and MSE loss
def error(y, y_hat):
    return (y-y_hat)

def squaredError(y, y_hat):
    return (y-y_hat)**2

def meanSquaredErrorLoss(y, y_hat):
    return squaredError(y,y_hat).mean()

def gradient(x,y,theta):
    err = error(y, f(x,theta))
    grad = -(1.0/len(x)) * err.dot(x)
    return grad

# Gradient descent to find the best parameters
def gradientDescent(x,y,alpha,theta,stop=.1):
    grad = gradient(x,y,theta)
    MSE_Theta_list = []

    while np.linalg.norm(grad) > stop:
        # Move in the direction of the gradient
        # N.B. this is point-wise multiplication, not a dot product
        theta = theta - grad*alpha
        mse = meanSquaredErrorLoss(y,f(x,theta))
        grad = gradient(x,y,theta)
        MSE_Theta_list.append((mse, theta))
        print(mse)

    print("Gradient descent finished. MSE="+str(mse))
    return theta, mse, MSE_Theta_list

# Finally, a helper to display the model against the data
def plotModel(x, y, y_hat, title='Plot'):
    # Create a dictionary to pass to matplotlib
    # These settings make the plots readable on slides, feel free to change
    # This is an easy way to set many parameters at once
    fontsize = "30";
    params = {'figure.autolayout':True,
              'legend.fontsize': fontsize,
              'figure.figsize': (12, 8),
             'axes.labelsize': fontsize,
             'axes.titlesize': fontsize,
             'xtick.labelsize':fontsize,
             'ytick.labelsize':fontsize}
    plt.rcParams.update(params)
    
    # Create a new figure and an axes objects for the subplot
    # We only have one plot here, but it's helpful to be consistent
    fig, ax = plt.subplots()
    
    # Draw a scatter plot of the first column of x vs second column.
    ax.scatter(x[:,1], y,color='blue', alpha=.8, s=140, marker='v')
    ax.set_xlabel('Height')
    ax.set_ylabel('Weight')
    ax.grid(color='lightgray', linestyle='-', linewidth=1)
    ax.set_axisbelow(True)
    ax.set_title(title)

    x2 = [x[:,1].min(), x[:,1].max()]
    y2 = [y_hat.min(), y_hat.max()]
    ax.plot(x2,y2,color='red', linewidth='3')

# And a function to plot the loss function
def plotLossFunction(x, y, MSE_Theta, title='Plot'):
    # Create a dictionary to pass to matplotlib
    # These settings make the plots readable on slides, feel free to change
    # This is an easy way to set many parameters at once
    fontsize = "20";
    params = {'figure.autolayout':True,
              'legend.fontsize': fontsize,
              'figure.figsize': (12, 8),
             'axes.labelsize': fontsize,
             'axes.titlesize': fontsize,
             'xtick.labelsize':fontsize,
             'ytick.labelsize':fontsize}
    plt.rcParams.update(params)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    res = 30
    xspace = np.linspace(-1, 2, res)
    yspace = np.linspace(-1, 2, res)
    xx, yy = np.meshgrid(xspace, yspace)
    xy = np.c_[xx.ravel(), yy.ravel()]
    L = []
    for theta in xy:
        L.append(meanSquaredErrorLoss(y,f(x,theta)))
    L = np.array(L).reshape(res,res)

    ax.plot_surface(xx, yy, L, rstride=1, cstride=1, cmap='jet', edgecolor='none', alpha=0.7) 
    ax.set_xlabel('Theta 0')
    ax.set_ylabel('Theta 1')
    ax.set_title(title)
    thetaWithContour(MSE_Theta, ax)

def errorOverTime(epochs, MSEs):
    x = epochs
    y = MSEs

    fig, ax = plt.subplots()
    
    ax.scatter(x, y,color='blue')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('MSE')
    ax.set_xlim(-100, 1000)
    ax.grid(color='lightgray', linestyle='-', linewidth=1)
    ax.set_axisbelow(True)
    ax.set_title("MSE vs. Epochs")

def thetaWithContour(thetas, ax):
    theta_zero = [x[1][0] for x in thetas]
    theta_one = [x[1][1] for x in thetas]
    ax.scatter(theta_zero, theta_one, color='black')
