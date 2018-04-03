import math
import numpy as np

#-------------------------------------------------------------------------
'''
    Problem 3: Support Vector Machine (with non-linear kernels)
    In this problem, we will implement the SVM using SMO method.
    Note: Did not use any existing package for SVM and implemented my own version of SVM.
'''

#--------------------------
def linear_kernel(X1, X2):
    '''
        Computing the linear kernel matrix between data instances in X1 and X2
        Input:
            X1: the feature matrix of the data instances, a numpy matrix of shape n1 by p
                Here n1 is the number of instances, p is the number of features
            X2: the feature matrix of the data instances, a numpy matrix of shape n2 by p
        Output:
            K: the kernel matrix between the data instances in X1 and X2, a numpy float matrxi of shape n1 by n2.
                If the i,j-th elment is the kernel between the i-th instance in X1, and j-th instance in X2.
        Note: please don't use any existing package for computing kernels.
    '''
   
    K = X1*np.transpose(X2)

    return K 

#--------------------------
def polynomial_kernel(X1, X2,d=2):
    '''
        Computing the polynomial kernel matrix between data instances in X1 and X2. 
        Input:
            X1: the feature matrix of the data instances, a numpy matrix of shape n1 by p
                Here n1 is the number of instances, p is the number of features
            X2: the feature matrix of the data instances, a numpy matrix of shape n2 by p
            d: the degree of polynomials, an integer scalar
        Output:
            K: the kernel matrix between the data instances in X1 and X2, a numpy float matrxi of shape n1 by n2.
                If the i,j-th elment is the kernel between the i-th instance in X1, and j-th instance in X2.
        Note: please don't use any existing package for computing kernels.
    '''
  
    K = np.power(1+(X1*np.transpose(X2)),d)

    return K 

#--------------------------
def gaussian_kernel(X1, X2,gamma=1.):
    '''
        Computing the Gaussian (RBF) kernel matrix between data instances in X1 and X2. 
        Input:
            X1: the feature matrix of the data instances, a numpy matrix of shape n1 by p
                Here n1 is the number of instances, p is the number of features
            X2: the feature matrix of the data instances, a numpy matrix of shape n2 by p
            gamma: the degree of polynomials, an integer scalar
        Output:
            K: the kernel matrix between the data instances in X1 and X2, a numpy float matrxi of shape n1 by n2.
                If the i,j-th elment is the kernel between the i-th instance in X1, and j-th instance in X2.
        Note: please don't use any existing package for computing kernels.
    '''
 
    K=np.zeros((X1.shape[0],X2.shape[0]))
    for i,x in enumerate(X1):
        for j,y in enumerate(X2):
            X_norm=np.linalg.norm(x-y)
            K[i,j]=np.exp(-1*(X_norm**2)/(2*gamma**2))
    K = np.mat(K)
   
    return K 


#--------------------------
def predict(K, a, y, b):
    '''
        Predicting the labels of testing instances.
        Input:
            K: the kernel matrix between the testing instances and training instances, a numpy matrix of shape n_test by n_train.
                Here n_test is the number of testing instances.
                n_train is the number of training instances.
            a: the alpha values of the training instances, a numpy float vector of shape n_train by 1. 
            y: the labels of the training instances, a float numpy vector of shape n_train by 1. 
            b: the bias of the SVM model, a float scalar.
        Output:
            y_test : the labels of the testing instances, a numpy vector of shape n_test by 1.
                If the i-th instance is predicted as positive, y[i]= 1, otherwise -1.
    '''
  
    y_test = (K*np.multiply(a,y)+b)
    y_test[y_test>0]=1
    y_test[y_test<0]=-1

    return y_test

#--------------------------
def compute_HL(ai,yi,aj,yj,C):
    '''
        Computing the clipping range of a[i] when pairing with a[j]
        Input:
            ai: the current alpha being optimized (the i-th instance), a float scalar, value: 0<= a_i <= C
            yi: the label of the i-th instance, a float scalar of value -1 or 1. 
            aj: the pairing alpha being optimized (the j-th instance), a float scalar, value: 0<= a_j <= C
            yj: the label of the j-th instance, a float scalar of value -1 or 1. 
            C: the upperbound of both ai and aj, a positive float scalar.
        Output:
            H: the upper-bound of the range of ai, a float scalar, between 0 and C 
            L: the lower-bound of the range of ai, a float scalar, between 0 and C 
    '''
   
    if (yi==yj):
        L = max(0,(ai+aj-C))
        H = min(C,(ai+aj))
    else:
        L = max(0,(ai-aj))
        H = min(C,(ai-aj+C))

    return H, L 


#--------------------------
def compute_E(Ki,a,y,b,i):
    '''
        Computing the error on the i-th instance: Ei = f(x[i]) - y[i] 
        Input:
            Ki: the i-th row of kernel matrix between the training instances, a numpy vector of shape 1 by n_train.
                Here n_train is the number of training instances.
            y: the labels of the training instances, a float numpy vector of shape n_train by 1. 
            a: the alpha values of the training instances, a numpy float vector of shape n_train by 1. 
            b: the bias of the SVM model, a float scalar.
            i: the index of the i-th instance, an integer scalar.
        Output:
            E: the error of the i-th instance, a float scalar.
    '''

    a1= np.multiply(a,y)
    E = float((Ki*a1)+b-y[i])  
    
    return E
 
#--------------------------
def compute_eta(Kii,Kjj,Kij):
    '''
        Computing the eta on the (i,j) pair of instances: eta = 2* Kij - Kii - Kjj
        Input:
            Kii: the kernel between the i,i-th instances, a float scalar 
            Kjj: the kernel between the j,j-th instances, a float scalar 
            Kij: the kernel between the i,j-th instances, a float scalar 
        Output:
            eta: the eta of the (i,j)-th pair of instances, a float scalar.
    '''

    eta = (2*Kij - Kii - Kjj)

    return eta
  
#--------------------------
def update_ai(Ei,Ej,eta,ai,yi,H,L):
    '''
        Updating the a[i] when considering the (i,j) pair of instances.
        Input:
            Ei: the error of the i-th instance, a float scalar.
            Ej: the error of the j-th instance, a float scalar.
            eta: the eta of the (i,j)-th pair of instances, a float scalar.
            ai: the current alpha being optimized (the i-th instance), a float scalar, value: 0<= a_i <= C
            yi: the label of the i-th instance, a float scalar of value -1 or 1. 
            H: the upper-bound of the range of ai, a float scalar, between 0 and C 
            L: the lower-bound of the range of ai, a float scalar, between 0 and C 
        Output:
            ai_new: the updated alpha of the i-th instance, a float scalar, value: 0<= a_i <= C
    '''

    if eta==0:
        ai_new = ai
    else:
            ai_s = (ai - (yi*(Ej-Ei)/eta))
            if (ai_s>H):
                ai_new = H
            elif (ai_s<L):
                ai_new = L
            else:
                ai_new = ai_s

    return ai_new
  
  
#--------------------------
def update_aj(aj,ai,ai_new,yi,yj):
    '''
        Updating the a[j] when considering the (i,j) pair of instances.
        Input:
            aj: the old value of a[j], a float scalar, value: 0<= a[j] <= C
            ai: the old value of a[i], a float scalar, value: 0<= a[i] <= C
            ai_new: the new value of a[i], a float scalar, value: 0<= a_i <= C
            yi: the label of the i-th instance, a float scalar of value -1 or 1. 
            yj: the label of the j-th instance, a float scalar of value -1 or 1. 
        Output:
            aj_new: the updated alpha of the j-th instance, a float scalar, value: 0<= a_j <= C
    '''
  
    aj_new = (aj + ((yi*yj)*(ai-ai_new)))
  
    return aj_new
  
 
#--------------------------
def update_b(b,ai_new,aj_new,ai,aj,yi,yj,Ei,Ej,Kii,Kjj,Kij,C):
    '''
        Updating the bias term.
        Input:
            b: the current bias of the SVM model, a float scalar.
            ai_new: the new value of a[i], a float scalar, value: 0<= a_i <= C
            aj_new: the updated alpha of the j-th instance, a float scalar, value: 0<= a_j <= C
            ai: the old value of a[i], a float scalar, value: 0<= a[i] <= C
            aj: the old value of a[j], a float scalar, value: 0<= a[j] <= C
            yi: the label of the i-th instance, a float scalar of value -1 or 1. 
            yj: the label of the j-th instance, a float scalar of value -1 or 1. 
            Ei: the error of the i-th instance, a float scalar.
            Ej: the error of the j-th instance, a float scalar.
            Kii: the kernel between the i,i-th instances, a float scalar 
            Kjj: the kernel between the j,j-th instances, a float scalar 
            Kij: the kernel between the i,j-th instances, a float scalar 
            C: the upperbound of both ai and aj, a positive float scalar.
        Output:
            b: the new bias of the SVM model, a float scalar.
    '''
    
    b1 = (b - Ei - (yj*(aj_new-aj)*Kij) - (yi*(ai_new-ai)*Kii))
    b2 = (b - Ej - (yj*(aj_new-aj)*Kjj) - (yi*(ai_new-ai)*Kij))
    
    if (ai_new>0 and ai_new<C):
        b = b1
    elif (aj_new>0 and aj_new<C):
        b = b2
    else:
        b = ((b1+b2)/2)
        
    return b 
  

 
#--------------------------
def train(K, y, C = 1., n_epoch = 10):
    '''
        Training the SVM model using simplified SMO algorithm.
        Input:
            K: the kernel matrix between the training instances, a numpy float matrxi of shape n by n.
            y : the sample labels, a numpy vector of shape n by 1.
            C: the weight of the hinge loss, a float scalar.
            n_epoch: the number of rounds to go through the instances in the training set.
        Output:
            a: the alpha of the SVM model, a numpy float vector of shape n by 1. 
            b: the bias of the SVM model, a float scalar.
    '''
    n = K.shape[0]
    a,b = np.asmatrix(np.zeros((n,1))), 0. 
    for _ in xrange(n_epoch):
        for i in xrange(n):
            for j in xrange(n):
                ai = float(a[i])
                aj = float(a[j])
                yi = float(y[i])
                yj = float(y[j])
                
                # compute the bounds of ai (H, L)
                H,L=compute_HL(ai,yi,aj,yj,C)


                #if H==L, no change is needed, skip to next j
                if (H==L):
                    continue
                
                else:
                      
                # compute Ei and Ej
                    Ki=K[i]
                    Kj=K[j]
                    Ei = compute_E(Ki,a,y,b,i)
                    Ej = compute_E(Kj,a,y,b,j)


                # compute eta 
                    Kii=K[i,i]
                    Kij=K[i,j]
                    Kjj=K[j,j]
                    eta=compute_eta(Kii,Kjj,Kij)

                # update ai, aj, and b
                    ai_new=update_ai(Ei,Ej,eta,ai,yi,H,L)
                    aj_new=update_aj(aj,ai,ai_new,yi,yj)
                    b=update_b(b,ai_new,aj_new,ai,aj,yi,yj,Ei,Ej,Kii,Kjj,Kij,C)
                    a[i]=ai_new
                    a[j]=aj_new

               
    return a,b
    
