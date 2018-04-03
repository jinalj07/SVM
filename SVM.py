import math
import numpy as np

#-------------------------------------------------------------------------
'''
Problem: Support Vector Machine (with Linear Kernel)
In this problem, we will implement the SVM classification method.
We will optimize the parameters using subgradient descent method.
Note: Did not use any existing package for SVM, implemented my own version of SVM.
'''

#--------------------------
def predict(X, w, b):
'''
Predicting the labels of data instances.
Input:
X: the feature matrix of the data instances, a numpy matrix of shape n by p
Here n is the number of instances, p is the number of features
w: the weights of the SVM model, a numpy float vector of shape p by 1.
b: the bias of the SVM model, a float scalar.
Output:
y : the labels of the data instances, a numpy vector of shape n by 1.
If the i-th instance is predicted as positive, y[i]= 1, otherwise -1.
'''

y = (X*w)+b
y[y>0]=1
y[y<0]=-1

return y 
#--------------------------
def subgradient(x, y, w, b, l=0.001):
'''
Computing the subgradient of loss function w.r.t. w and b (on one training instance).
Input:
x: the feature vector of a training data instance, a numpy vector of shape p by 1
Here p is the number of features
y: the label of the training data instance, a float scalar (1. or -1.)
w: the current weights of the SVM model, a numpy float vector of shape p by 1.
b: the current bias of the SVM model, a float scalar.
l: (lambda) = 1/ (n C), which is the weight of the L2 regularization term.
Here n is the number of training instances, C is the weight of the hinge loss.
Output:
dL_dw : the subgradient of the weights, a numpy float vector of shape p by 1.
The i-th element is d L / d w[i]
dL_db : the sbugradient of the bias, a float scalar.
'''

fofx=w.T*x+b
cond = 1-y*fofx
if (cond>0) :
    dL_dw = ((l*w)- (y*x))
    dL_db = (-1*y)
else:
    dL_dw = (l*w)
    dL_db = 0

return dL_dw, dL_db 
#--------------------------
def update_w(w, dL_dw, lr=0.01):
'''
Updating the parameter w using the subgradient.
Input:
w: the current weights of the SVM model, a numpy float vector of shape p by 1.
dL_dw : the subgradient of the weights, a numpy float vector of shape p by 1.
The i-th element is d L / d w[i]
lr: the learning rate, a float scalar, controling the speed of gradient descent.
Output:
w: the updated weights of the SVM model, a numpy float vector of shape p by 1.
'''

w = w - (lr*dL_dw)

return w 
#--------------------------
def update_b(b, dL_db, lr=0.01):
'''
Updating the parameter b using the subgradient.
Input:
b: the current weights of the SVM model, a float scalar.
dL_db : the subgradient of the weights, a numpy float vector of shape p by 1.
lr: the learning rate, a float scalar, controling the speed of gradient descent.
Output:
b: the updated bias of the SVM model, a float scalar.
'''

b = b - (lr*dL_db)

return b
#--------------------------
def train(X, Y, lr=0.01,C = 1., n_epoch = 10):
'''
Training the SVM model using Stochastic Gradient Descent (SGD).
Input:
X: the design/feature matrix of the training samples, a numpy matrix of shape n by p
Here n is the number of training samples, p is the number of features
Y : the sample labels, a numpy vector of shape n by 1.
lr: the learning rate, a float scalar, controling the speed of gradient descent.
C: the weight of the hinge loss, a float scalar.
n_epoch: the number of rounds to go through the instances in the training set.
Output:
w: the weights of the SVM model, a numpy float vector of shape p by 1.
b: the bias of the SVM model, a float scalar.
'''
n,p = X.shape

#l: (lambda) = 1/ (n C), which is the weight of the L2 regularization term. 
l = 1./(n * C)

w,b = np.asmatrix(np.zeros((p,1))), 0. # initialize the weight vector as all zeros
for _ in xrange(n_epoch):
    for i in xrange(n):
        x = X[i].T # get the i-th instance in the dataset
        y = float(Y[i]) 

        dL_dw,dL_db=subgradient(x, y, w, b, l)
        w=update_w(w, dL_dw, lr)
        b=update_b(b, dL_db, lr)

return w,b
