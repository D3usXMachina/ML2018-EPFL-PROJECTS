# -*- coding: utf-8 -*-
# ==============================================================================
# implementations.py
# ------------------------------------------------------------------------------
# authors:                             Patrick Ley, Joel Fischer
# date:                                10.10.2018
# ==============================================================================
# This code collection contains the functions required for the submission of
# ptoject 1 of the 2018 machine learning course at EPFL as well as some other
# utility functions used to complete the task at hand.
# The functions required for the submission can be found in the first section
# "Trainers".
# ==============================================================================
# -Trainers
#   -> least_squares_GD(y, tx, initial_w, max_iters=100, gamma=0.5, lambda_=0 )
#   -> least_squares_SGD(y, tx, initial_w, max_iters=100, gamma=0.5, batch_size=1, lambda_=0)
#   -> least_squares(y, tx, lambda_=0)
#   -> ridge_regression(y, tx, lambda_, mode = "ls", max_iters=100, gamma=0.5, batch_size=1)
#   -> logistic_regression(y, tx, initial_w, max_iters=100, gamma=0.5, mode="log",lambda_=0)
#   -> reg_logistic_regression(y, tx, lambda_ ,initial_w, max_iters=100, gamma=0.5, mode="log")
# -Utility functions for trainers
#   -> compute_gradient(y, tx, w, lambda_=0, mode="mse")
#   -> compute_sigma(tx,w,lim=100.0)
#   -> batch_iter(y, tx, batch_size=1, num_batches=1, shuffle=True, seed=1)
# -Loss functions
#   -> compute_loss(y, tx, w, mode="mse", lambda_=0)
# -Preprocessing
#   -> standardize(tx)
#   -> split_data(y, tx, ratio, seed=1)
#   -> add_constant(x)
#   -> poly_expansion(x, degree, add_constant=True, mix_features=False)
# -Utility
#   -> generate_data(nsamples,nfeatures,seed=1)
# ==============================================================================
# TODO: -check that implementations work regardless the format of the inpute
#        i.e. (n,) should be treated as (n,1)
#       -implement feature mixing in polynomial feature expansion
#       -check must implement functions
#       -implement stochastic version of logistic_regression/reg_logistic_regression
# ==============================================================================

import numpy as np

# ==============================================================================
# Trainers
# ==============================================================================
# This section contains the training functions required for the submission.
# The training functions are used to compute the optimal weights for a given
# model from a training dataset.
# ------------------------------------------------------------------------------

def least_squares_GD(y, tx, initial_w, max_iters=100, gamma=0.5, lambda_=0 ):
    """
    ----------------------------------------------------------------------------
    Iteratively compute the model weights "w" from "y" and "tx" using the
    gradient descent algorithm starting at "initial_w" with a maximum of
    "max_iters" steps and step size "gamma".
    The function also returns the loss computed as the mean square error (mse).
    ----------------------------------------------------------------------------
    Input:
    - y             "measured" objective function, (nsamples,1) np.array
    - tx            features, (nsamples,nfeatures) np.array
    - initial_w     initial guess of model weights, (nfeatures,1) np.array
    - max_iters     # of iterations after which the procedure will stop, int>0
                    (default=100)
    - gamma         step size, scalar in ]0,1[ (default=0.5)
    - lambda_       regularization parameter, scalar>0 (default = 0)
    Output:
    - w             obtained weights, (nfeatures,1) np.array
    - loss          loss computed as the mean square error, scalar
    ----------------------------------------------------------------------------
    """

    w = initial_w

    for n_iter in range(max_iters):
        w = w -gamma*compute_gradient(y, tx, w, lambda_)

    loss = compute_loss(y,tx,w)

    return w, loss

# ------------------------------------------------------------------------------

def least_squares_SGD(y, tx, initial_w, max_iters=100, gamma=0.5, batch_size=1, lambda_=0):
    """
    ----------------------------------------------------------------------------
    Iteratively compute the model parameters "w" from "y" and "tx" using the
    stochastic gradient descent algorithm starting at "initial_w" with a
    maximum of "max_iters" steps and step size "gamma" and sample size batch_size.
    The function also returns the loss computed as the mean square error.
    ----------------------------------------------------------------------------
    Input:
    - y             "measured" objective function, (nsamples,1) np.array
    - tx            features, (nsamples,nfeatures) np.array
    - initial_w     initial guess of model weights, (nfeatures,1) np.array
    - batch_size    size of the subsamples, positive integer (default=1)
                    ( setting to 0 or >nsamples will result in using standard sg)
    - max_iters     # of iterations after which the procedure will stop, int>0
                    (default=100)
    - gamma         step size, scalar in ]0,1[ (default=0.5)
    - lambda_       regularization parameter, scalar>0 (default = 0)
    Output:
    - w             obtained weights, (nfeatures,1) np.array
    - loss          loss computed as the mean square error, scalar
    ----------------------------------------------------------------------------
    """

    nsamples = tx.shape[0]
    if ( batch_size==0 ) | ( batch_size>nsamples ):
        w, loss = least_squares_GD(y, tx, initial_w, max_iters, gamma, lambda_)
    else:
        w = initial_w
        batches = batch_iter(y, tx, batch_size, max_iters)

        for batch in batches:
            yb = np.transpose(batch[0])
            txb = batch[1]
            w = w -gamma*compute_gradient(yb, txb, w)

        loss = compute_loss(y,tx,w)

    return w, loss

# ------------------------------------------------------------------------------

def least_squares(y, tx, lambda_=0):
    """
    ----------------------------------------------------------------------------
    Compute the model weights "w" from "y" and "tx" using the
    normal equations.
    The function also returns the loss computed as the mean square error (mse).
    ----------------------------------------------------------------------------
    Input:
    - y             "measured" objective function, (nsamples,1) np.array
    - tx            features, (nsamples,nfeatures) np.array
    - lambda_       regularization parameter, scalar>0 (default = 0)
    Output:
    - w             obtained weights, (nfeatures,1) np.array
    - loss          loss computed as the mean square error, scalar
    ----------------------------------------------------------------------------
    """

    if lambda_ == 0:
        a = 0
    else:
        a = 2*tx.shape[0]*lambda_*np.eye(tx.shape[1])

    w = tx.transpose().dot(tx) + a
    w = np.linalg.inv(w)
    w = w.dot(tx.transpose().dot(y))

    loss = compute_loss(y, tx, w)

    return w, loss

# ------------------------------------------------------------------------------

def ridge_regression(y, tx, lambda_, mode = "ls", max_iters=100, gamma=0.5, batch_size=1):
    """
    ----------------------------------------------------------------------------
    Compute the model weights "w" from "y" and "tx" using the
    normal equations with reagularization parameter "lambda_".
    The function also returns the loss computed as the mean square error (mse).
    ----------------------------------------------------------------------------
    Input:
    - y             "measured" objective function, (nsamples,1) np.array
    - tx            features, (nsamples,nfeatures) np.array
    - lambda_       regularization parameter, scalar>0
    - mode          choice of algorithm (0/"ls",1/"gd",2/"sgd"), string/integer
                    (default "ls")
    - max_iters     # of iterations after which the procedure will stop, int>0
                    (default=100)
    - gamma         step size, scalar in ]0,1[ (default=0.5)
    - batch_size    size of the subsamples, positive integer (default=1)
                    ( setting to 0 or >nsamples will result in using standard sg)
    Output:
    - w             obtained weights, (nfeatures,1) np.array
    - loss          loss computed as the mean square error, scalar
    ----------------------------------------------------------------------------
    """

    if( mode=="gd" ) | ( mode==1 ): # gradient_descent
        w, loss = gradient_descent(y, tx, initial_w, max_iters, gamma, lambda_)
    elif ( mode=="sgd" ) | ( mode==2 ): # stochastic_gradient_descent
        w, loss = stochastic_gradient_descent(y,tx,initial_w,batch_size,max_iters,gamma,lambda_)
    else: # least_squares
        w, loss = least_squares(y, tx, lambda_)

    return w, loss

# ------------------------------------------------------------------------------

def logistic_regression(y, tx, initial_w, max_iters=100, gamma=0.5, mode="log",lambda_=0):
    """
    ----------------------------------------------------------------------------
    Iteratively computes the model weights "w" from "y" and "tx" using
    logistic regression with an optional reagularization parameter "lambda_"
    in up "max_iters" iterations.
    The function also returns the loss computed according to the logistic
    regression loss function.
    ----------------------------------------------------------------------------
    Input:
    - y             "measured" objective function, (nsamples,1) np.array
    - tx            features, (nsamples,nfeatures) np.array
    - initial_w     initial guess of model weights, (nfeatures,1) np.array
    - max_iters     # of iterations after which the procedure will stop,
                    int>0 (default=100)
    - gamma         step size, scalar in ]0,1[ (default=0.5)
    - mode          choice of algorithm (1/"log",2/"newton"), string/integer
                    (default=1/"log")
    - lambda_       regularization parameter, scalar>0
    Output:
    - w             obtained weights, (nfeatures,1) np.array
    - loss          loss computed as the mean square error, scalar
    ----------------------------------------------------------------------------
    """

    if (mode != "newton") & (mode != 2 ):
        mode = "log"

    w = initial_w

    for i in range(max_iters):
        w = w -gamma*compute_gradient(y, tx, w, lambda_, mode)

    loss = compute_loss(y,tx,w,"log")

    return w, loss

# ------------------------------------------------------------------------------

def reg_logistic_regression(y, tx, lambda_ ,initial_w, max_iters=100, gamma=0.5, mode="log"):
    """
    ----------------------------------------------------------------------------
    Iteratively computes the model weights "w" from "y" and "tx" using
    regularized logistic regression with reagularization parameter "lambda_"
    in up "max_iters" iterations.
    The function also returns the loss computed according to the logistic
    regression loss function.
    ----------------------------------------------------------------------------
    Input:
    - y             "measured" objective function, (nsamples,1) np.array
    - tx            features, (nsamples,nfeatures) np.array
    - lambda_       regularization parameter, scalar>0
    - initial_w     initial guess of model weights, (nfeatures,1) np.array
    - max_iters     # of iterations after which the procedure will stop,
                    int>0 (default=100)
    - gamma         step size, scalar in ]0,1[ (default=0.5)
    - mode          choice of algorithm (1/"log",2/"newton"), string/integer
                    (default=1/"log")
    Output:
    - w             obtained weights, (nfeatures,1) np.array
    - loss          loss computed as the mean square error, scalar
    ----------------------------------------------------------------------------
    """

    w, loss = logistic_regression(y,tx,initial_w,max_iters,gamma,mode,lambda_)

    return w, loss

# ==============================================================================
# Utility functions for trainers
# ==============================================================================
# Accesory function required for the different training functions.
# ------------------------------------------------------------------------------

def compute_gradient(y, tx, w, lambda_=0, mode="mse"):
    """
    ----------------------------------------------------------------------------
    Compute the gradient of the mean square error loss function evaluated at
    (y,tx,w) with respect to the weights "w" with or without taking into account
    the regularization parameter "lambda_".
    Optionally this function can also be used to compute the gradient with
    respect to the loss function used for logistic regression or even to compute
    one step of the second order logistic regression algorithm (Newton method).
    ----------------------------------------------------------------------------
    Input:
    - y             "measured" objective function, (nsamples,1) np.array
    - tx            features, (nsamples,nfeatures) np.array
    - w             current weights, (nfeatures,1) np.array
    - lambda_       regularization parameter, scalar>0 (default = 0)
    - mode          choice of regression type for which the gradient is used
                    (0/"mse",1/"log",2/"newton"), int/string (default=0/"mse")
    Output:
    - grad          gradient, (tx.shape[0],1) np.array
    ----------------------------------------------------------------------------
    MSE Gradient:   L(w) = (y-x^t*w)^2/(2*N) + lambda_*w^2
                    grad = -(y-x^t*w)^t*w/N + 2*lambda_*w
                         = -(y-x^t*w-N*lambda_/2)^t*w
    Log Gradient:   L(w) = lambda_/2*w^2 + sum_n^N ln[1+exp(x_n^t*w)]-y_n*x_n^t*w
                    grad = lambda_*w + x^t*[sigma(x*w)-y]
    Newton (Log):   H    = x^t*S*x, S_nn = sigma(x_n*w)*[1-sigma(x_n^t*w)]
                    grad2= H^-1*grad
    ----------------------------------------------------------------------------
    """
    if (lambda_ == 0 ):
        a = 0
    else:
        a = lambda_*w

    if (( mode=="log" ) | ( mode=="1" )) | (( mode=="newton" ) | ( mode=="2" )):

        sigma = compute_sigma(tx,w)
        grad = sigma-y
        grad = tx.transpose().dot(grad)+a

        if ( mode=="newton" ) | ( mode=="2" ):
            S = sigma*(1-sigma)
            H = tx.transpose().dot(tx*S)
            grad = np.linalg.inv(H).dot(grad)
    else:
        grad = -tx.transpose().dot(y-tx.dot(w))/tx.shape[0] + 2*a

    return grad

# ------------------------------------------------------------------------------

def compute_sigma(tx,w,lim=100.0):
    """
    ----------------------------------------------------------------------------
    Computes sigma=exp(tx*w)/(1+exp(tx*w)) as needed for logistic regression.
    ----------------------------------------------------------------------------
    Input:
    - tx            features, (nsamples,nfeatures) np.array
    - w             current weights, (nfeatures,1) np.array
    - lim           limit of the exponent after which sigma isn't computed
                    explicitly anymore (instead it will be simply set to 0 or 1)
                    , scalar>0 (default=100.0)
    Output:
    - sigma         sigma=exp(tx*w)/(1+exp(tx*w)), (tx.shape[0],1) np.array
    ----------------------------------------------------------------------------
    """

    x = tx.dot(w)

    large = x>lim
    small = x<-lim
    neither = np.logical_not(large)*np.logical_not(small)

    x[neither] = np.exp(x[neither])

    sigma = np.divide(x,1+x)

    sigma[large] = 1
    sigma[small] = 0

    return sigma

# ------------------------------------------------------------------------------

def batch_iter(y, tx, batch_size=1, num_batches=1, shuffle=True, seed=1):
    """
    ----------------------------------------------------------------------------
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the
    input data 'tx') and outputs an iterator which gives mini-batches of
    `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing
    with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    Note: this implementation was provided by the teachers/TAs and has since
    been slightly modified to allow for more iterations. However iterations that
    exceed the batch_num*batch_size>data_size will reuse the same batches (i.e
    without additional shuffeling).
    ----------------------------------------------------------------------------
    Input:
    - y             "measured" objective function, (nsamples,1) np.array
    - tx            features, (nsamples,nfeatures) np.array
    - batch_size    subsample size, integer>0 (default=1)
    - num_batches   number of batches, integer>0 (default=1)
    - shuffle       random order if True, retains order otherwise, boolean
    - seed          seed for np.random, integer
    Output:
    - batches       subsamples,
                    generator num_batches*[(batch_size,1),(batch_size,nfeatures)]
    ----------------------------------------------------------------------------
    """
    nsamples = tx.shape[0]

    if shuffle:
        np.random.seed(seed)
        shuffle_indices = np.random.permutation(np.arange(nsamples))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx

    for batch_num in range(num_batches):
        start_index = (batch_num*batch_size)%nsamples
        end_index = min(start_index+batch_size, nsamples)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

# ==============================================================================
# Loss functions
# ==============================================================================
# Functions used to compute the loss of between the prediction created by a
# model and the actual "measurements"/reality.
# ------------------------------------------------------------------------------

def compute_loss(y, tx, w, mode="mse", lambda_=0):
    """
    ----------------------------------------------------------------------------
    Computes the loss according to the specified loss function. The available
    loss functions are mean square error (mse), root mean square error (rmse),
    mean absolute error (mae) and the loss function used for logistic regression
    (log). Furthermore the regularization parameter "lambda_" can be taken
    account of if desired.
    ----------------------------------------------------------------------------
    Input:
    - y             "measured" objective function, (nsamples,1) np.array
    - tx            features, (nsamples,nfeatures) np.array
    - mode          choice of loss function (0/"mse",1/"rmse",2/"msa",3/"log"),
                    string/integer (default = 0/"mse")
    - lambda_       regularization parameter, scalar>0, (default=0)
    Output:
    - loss          loss computed using the loss function of choice, scalar
    ----------------------------------------------------------------------------
    """

    # compute error
    if (mode == "msa") | (mode == 2):
        e = y-tx.dot(w)
        e = np.absolute(e).sum()/e.shape[0]/2
    elif (mode == "log") | (mode == 3):
        e = tx.dot(w)
        e = np.log(1+np.exp(e)) - y*e
    else: # mse or rmse loss
        e = y-tx.dot(w)
        e = (np.transpose(e).dot(e)/(2*e.shape[0]))
        if (mode == "rmse") | (mode == 1):
            e = np.sqrt(2*e)

    loss = np.sum(e)

    # compute regularization contribution
    if lambda_ == 0:
        a = 0
    else:
        a = lambda_*np.sum(w*w)

    return loss + a

# ==============================================================================
# Preprocessing
# ==============================================================================
# Preprocessing functions to prepare data for training and/or prediction.
# ------------------------------------------------------------------------------

def standardize(tx):
    """
    ----------------------------------------------------------------------------
    standardize data (i.e substract mean and divide by standard deviation)
    ----------------------------------------------------------------------------
    Input:
    - tx            features, (nsamples,nfeatures) np.array
    Output:
    - tx_std        features standardized to mean 0 and std 1,
                    (nsamples,nfeatures) np.array
    - mean          mean value of features, (1,nfeatures) np.array
    - std           standrad deviation of features, (1,nfeatures) np.array
    ----------------------------------------------------------------------------
    """

    mean = np.mean(tx,0)
    std = np.std(tx,0)
    tx_std = np.divide(tx-mean,std)

    return tx_std, mean, std

# ------------------------------------------------------------------------------

def split_data(y, tx, ratio, seed=1):
    """
    ----------------------------------------------------------------------------
    randomly split the dataset "x" based on the split passed ratio.
    ----------------------------------------------------------------------------
    Input:
    - y             "measured" objective function, (nsamples,1) np.array
    - tx            features, (nsamples,nfeatures) np.array
    - ratio         split ratio, scalar in [0,1]
    - seed          seed for np.random, integer
    Output:
    - y1            "measured" objective function, (nsamples*ratio,1) np.array
    - x1            features, (nsamples*ratio,nfeatures) np.array
    - y2            "measured" objective function,
                    (nsamples*(1-ratio),nfeatures) np.array
    - x2            features, (nsamples*(1-ratio),nfeatures) np.array
    ----------------------------------------------------------------------------
    """
    np.random.seed(seed)

    nsamples = tx.shape[0]
    indices = np.indices([nsamples])[0]
    np.random.shuffle(indices)

    split1 = indices[0:np.floor(nsamples*ratio)]
    split2 = indices[np.floor(nsamples*ratio):nelem]

    return y[split1],x[split1,:],y[split2],x[split2,:]

# ------------------------------------------------------------------------------

def add_constant(x):
    """
    ----------------------------------------------------------------------------
    Adds a constant feature to the features present in the data array "x".
    ----------------------------------------------------------------------------
    Input:
    - x             features, (nsamples,nfeatures) np.array
    Output:
    - tx            features, (nsamples,nfeatures+1) np.array
    ----------------------------------------------------------------------------
    """

    tx = np.column_stack((np.ones([x.shape[0],1]),x))

    return tx

# ------------------------------------------------------------------------------

# mix_features is WIP
def poly_expansion(x, degree, add_constant=True, mix_features=False):
    """
    ----------------------------------------------------------------------------
    Performs a polynomial feature expansion of "x" up to the degree "degree",
    assuming that each column of "x" represents a feature. Note that to avoid
    undesired side effects the array "x" should not contain a constant feature
    since it would carry over in the expansion.
    This function also allows to directly add a constant feature if
    desired.
    In the future this function will be extended to also allow for feature
    mixing (fe) in the polynomial expansion. However caution is adviced since the
    size of the expansion will grow as O(nfeatures^degree) instead of
    O(nfeatures*degree) if feature mixing is enabled.
    ----------------------------------------------------------------------------
    Input:
    - x             features, (nsamples,nfeatures) np.array
    - degree        degree of the polynomial, integer>0
    - add_constant  if enabled a constant feature will be added, boolean
                    (default=True)
    - mix_features  if enabled features will be mixed, boolean (default=False) - WIP
    Output:
    - tx            the polynomial expansion of "x",
                    (nsamples,nfeatures*degree+add_constant) np.array (w/o fm)
                    (nsamples,?) np.array (w fm)
    ----------------------------------------------------------------------------
    """

    nfeatures = x.shape[1]
    nelements = x.shape[0]

    if False: #mix_features:
        # WIP
        nfeatures = nfeatures
        # nftot = (nfeatures**(degree+1)-1)/(nfeatures-1)
        # tx = np.ones([nelements,nftot])
        # for d in range(1,degree):
        #     for i in range(nfeatures):
        #         tx(:,1+i*d*(nfeatures):1+(1+i*d)*(nfeatures)) = tx(:,1:1+nfeatures)*x(:,i)


    else:
        tx = np.ones([nelements,degree*nfeatures+1])
        for n in range(0,nfeatures+1):
            tx[:,n*(degree)+1] = x[:,n]
            for d in range(2,degree+1):
                tx[:,n*(degree)+degree] = tx[:,n*(degree)+degree-1]*x[:,n]

    if not add_constant:
        tx = tx[:,1:]
        #TODO : improve this, fix mix_features

    return tx

# ==============================================================================
# Utility
# ==============================================================================
# Collection of other functions of use.
# ------------------------------------------------------------------------------

def generate_data(nsamples,nfeatures,seed=1):
    """
    generates a random dataset from a random model
    """

    np.random.seed(seed)

    w = np.random.random([nfeatures+1,1])

    x = np.random.random([nsamples,nfeatures])
    tx = np.ones([nsamples,1])
    tx = np.column_stack((tx,x))

    y = tx.dot(w)
    y = y + np.random.normal(0,0.5,[nsamples,1])

    return y, tx, w

# ==============================================================================
# WIP
# ==============================================================================
# Other functions that are still work in progress and not guaranteed to work.
# ------------------------------------------------------------------------------

# def correct_shape(y, tx, w):
#     """
#     atempts to reshape data that isn't supplied in the correct/expected fromat
#     """
#
#     nsamples = len(y)
#     y = y.reshape(nsamples,1)
#     w = w.reshape(nfeatures,1)
#
#     nfeatures = tx.shape
#     if tx.shape[0] == nsamples:
#         nfeatures = tx.shape[1]
#     elif tx.shape[1] == nsamples:
#         nfeatures = tx.shape[0]
#         tx = tx.transpose()
#
#     return y, tx, w, nsamples, nfeatures
#
# def generate_bin_data(nsamples,nfeatures,seed=1):
#     """
#     generate a random dataset with a binary dependent variable.
#     Not tested and no guarantee that this actually works.
#     """
#
#     y,tx,w = generate_data(nsamples,nfeatures,seed)
#
#     y = np.exp(y)
#     y = np.divide(y,1+y)
#     y = np.round(y)
#
#     return y, tx
#
#def analyze_data(tx):
#    return
