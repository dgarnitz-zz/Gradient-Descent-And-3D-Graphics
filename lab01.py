from l04_utils import *

data = np.loadtxt("l01-data.txt")
factor = np.max(data)                   # NORMALIZE the data. IF you dont include this, the gradient descent algorithm will not run properly

x = data[:, 1] / factor                 # extract the second column (0-indexed), store it in x

x = np.c_[np.ones_like(x), x]           # .c_ is the operator for concatenate
                                        # take a column vector of 1s, same size as column vector x
                                        # concatenate x onto the 1s column vector

y = data[:, 2]  / factor                # extract the third column, store it in y
                                        # will use x to predict y

fig, ax = plt.subplots()                # Get a figure object and an axis object to manipulate

# Create a scatter plot and label the axes
ax.scatter(x[:, 1], y, color='blue', alpha=.8, s=140, marker='o')
#ax.set_xlabel('Height')
#ax.set_ylabel('Weight')
#plt.show()

# Update multiple properties at the same time
'''
fontsize = '20'
params = {'axes.labelsize': fontsize, 'axes.titlesize': fontsize, 'figure.figsize': (12, 8)}
plt.rcParams.update(params)

ax.set_xlim(20, 80)             #this configures the x values shown
ax.set_ylim(0, 100)             #this configures the y values shown
ax.set_title('Linear regression example')
ax.grid(color = 'lightgray', linestyle = '-', linewidth = '1')
ax.set_axisbelow(True)
'''
theta = np.array([0.5, 1])        # these are the starting values for the parameters; intercept of 0.5 and slope of 1
'''
new_x = np.linspace(x[:, 1].min(), x[:, 1].max(), 30)
new_x = np.c_[np.ones_like(new_x), new_x]
ax.plot(new_x, f(new_x, theta), color = 'red')
ax.set_xlabel('Height')
ax.set_ylabel('Weight')
plt.show()
'''
theta_new, loss, MSE_Theta = gradientDescent(x, y, 0.1, theta, 0.001) #1e-8
#print(theta_new, loss)

#print(meanSquaredErrorLoss(y,f(x,theta)))
plotModel(x,y,f(x,theta),title='Random parameters')

#print(meanSquaredErrorLoss(y,f(x,theta_new)))
plotModel(x,y,f(x,theta_new),title='After gradient descent')

plotLossFunction(x,y,MSE_Theta,title='Loss function')

MSEs = [x[0] for x in MSE_Theta]
epoch = list(range(1, len(MSEs)+1))
errorOverTime(epoch, MSEs)

plt.show()



