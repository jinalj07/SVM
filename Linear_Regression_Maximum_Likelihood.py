import math
import numpy as np

#-------------------------------------------------------------------------
'''
    Problem 1: Linear Regression (Maximum Likelihood)
    In this problem, we will implement the linear regression method based upon maximum likelihood (least square).
    w'x + b = y
   
    Note: Did not use any existing package and implemented my own version of linear regression problem.
'''

#--------------------------
def compute_Phi(x,p):
    '''
        Computing the design matrix Phi of x. We will construct p polynoials a the p features of the data samples. 
        The features of each sample, is x^0, x^1, x^2 ... x^(p-1)
        Input:
            x : a vector of samples in one dimensional space, a numpy vector of shape n by 1.
                Here n is the number of samples.
            p : the number of polynomials/features
        Output:
            Phi: the design/feature matrix of x, a numpy matrix of shape (n by p).
    '''
    
    Phi = np.power(x,range(p))
    return Phi 


#--------------------------
def least_square(Phi, y):
    '''
        Fitting a linear model on training samples. Computing the paramter w using Maximum likelihood (equal to least square).
        Input:
            Phi: the design/feature matrix of the training samples, a numpy matrix of shape n by p
                Here n is the number of training samples, p is the number of features
            y : the sample labels, a numpy vector of shape n by 1.
        Output:
            w: the weights of the linear regression model, a numpy float vector of shape p by 1. 
    '''
    w = np.linalg.inv(np.transpose(Phi)*Phi)*np.transpose(Phi)*y

    return w 


#--------------------------
def ridge_regression(Phi, y, alpha=0.001):
    '''
        Fitting a linear model on training samples. Computing the paramter w using Maximum posterior (equal to least square with L2 regularization).
        min_w sum_i (y_i - Phi_i * w)^2/2 + alpha * w^T * w
        Input:
            Phi: the design/feature matrix of the training samples, a numpy matrix of shape n by p
                Here n is the number of training samples, p is the number of features
            y : the sample labels, a numpy vector of shape n by 1.
            alpha: the weight of the L2 regularization term, a float scalar.
        Output:
            w: the weights of the linear regression model, a numpy float vector of shape p by 1. 
    '''

    w = (np.linalg.inv((np.transpose(Phi)*Phi)+(alpha*np.identity(Phi.shape[1]))))*(np.transpose(Phi)*y)

    return w 
