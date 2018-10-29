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
# "Required Functions".
# ==============================================================================
# -Required Functions
#   -> least_squares_GD(y, tx, initial_w, max_iters, gamma)
#   -> least_squares_SGD(y, tx, initial_w, max_iters, gamma)
#   -> least_squares(y, tx)
#   -> ridge_regression(y, tx, lambda_)
#   -> logistic_regression(y, tx, initial_w, max_iters, gamma)
#   -> reg_logistic_regression(y, tx, lambda_ ,initial_w, max_iters, gamma)
# -Trainers
#   -> my_least_squares_GD(y, tx, initial_w, max_iters=1000, gamma=0.2, lambda_=0, eps=1e-5)
#   -> my_least_squares_SGD(y, tx, initial_w, max_iters=1000, gamma=0.2, batch_size=1, lambda_=0, eps=1e-1)
#   -> my_least_squares(y, tx, lambda_=0)
#   -> my_ridge_regression(y, tx, lambda_, mode = "ls", max_iters=100, gamma=0.2, batch_size=1, eps=1e-5)
#   -> my_logistic_regression(y, tx, initial_w, max_iters=100, gamma=0.2, mode="log", lambda_=0, eps=1e-5)
#   -> my_reg_logistic_regression(y, tx, lambda_ ,initial_w, max_iters=100, gamma=0.2, mode="log", eps=1e-5)
#   -> my_stoch_logistic_regression(y, tx, initial_w, max_iters=100, gamma=0.2, batch_size=4, mode="log", lambda_=0, eps=1e-5)
# -Utility functions for trainers
#   -> compute_gradient(y, tx, w, mode="mse", lambda_=0)
#   -> compute_sigma(tx,w,lim=100.0)
#   -> batch_iter(y, tx, batch_size=1, num_batches=1, shuffle=True, seed=1)
# -Loss functions
#   -> compute_loss(y, tx, w, mode="mse", lambda_=0)
# -Preprocessing
#   -> standardize(tx,mean_=0,std_=1)
#   -> split_data(y, tx, ratio, seed=1)
#   -> eq_split_data(y, tx, nparts, shuffle=True, seed=1)
#   -> add_constant(x)
#   -> poly_expansion(x, degree, add_constant=True, mix_features=False)
# -Utility
#   -> generate_data(nsamples,nfeatures,seed=1,std)
#   -> generate_bin_data(nsamples,nfeatures,seed=1,std=0.1)
#   -> column_array(y)
#   -> compute_y(tx,w)
#   -> nCr(n,k)
# ==============================================================================
# TODO:
#       -implement feature mixing in polynomial feature expansion
#       -check must implement functions
# ==============================================================================

import numpy as np

# ==============================================================================
# Required Functions
# ==============================================================================
# This section contains the training functions required for the submission
# without optional arguments.
# The training functions are used to compute the optimal weights for a given
# model from a training dataset.
# For the respective source code, see the functions in the section "Trainers"
# (same names but with prefix "my_").
# ------------------------------------------------------------------------------

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
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
    - gamma         step size, scalar in ]0,1[
    Output:
    - w             obtained weights, (nfeatures,1) np.array
    - loss          loss computed as the mean square error, scalar
    ----------------------------------------------------------------------------
    """

    return my_least_squares_GD(y, tx, initial_w, max_iters, gamma)

# ------------------------------------------------------------------------------

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    ----------------------------------------------------------------------------
    Iteratively compute the model parameters "w" from "y" and "tx" using the
    stochastic gradient descent algorithm starting at "initial_w" with a
    maximum of "max_iters" steps and step size "gamma".
    The function also returns the loss computed as the mean square error.
    ----------------------------------------------------------------------------
    Input:
    - y             "measured" objective function, (nsamples,1) np.array
    - tx            features, (nsamples,nfeatures) np.array
    - initial_w     initial guess of model weights, (nfeatures,1) np.array
    - max_iters     # of iterations after which the procedure will stop, int>0
    - gamma         step size, scalar in ]0,1[
    Output:
    - w             obtained weights, (nfeatures,1) np.array
    - loss          loss computed as the mean square error, scalar
    ----------------------------------------------------------------------------
    """

    return my_least_squares_SGD(y, tx, initial_w, max_iters, gamma)

# ------------------------------------------------------------------------------

def least_squares(y, tx):
    """
    ----------------------------------------------------------------------------
    Compute the model weights "w" from "y" and "tx" using the
    normal equations.
    The function also returns the loss computed as the mean square error (mse).
    ----------------------------------------------------------------------------
    Input:
    - y             "measured" objective function, (nsamples,1) np.array
    - tx            features, (nsamples,nfeatures) np.array
    Output:
    - w             obtained weights, (nfeatures,1) np.array
    - loss          loss computed as the mean square error, scalar
    ----------------------------------------------------------------------------
    """

    return my_least_squares(y, tx)

# ------------------------------------------------------------------------------

def ridge_regression(y, tx, lambda_):
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
    Output:
    - w             obtained weights, (nfeatures,1) np.array
    - loss          loss computed as the mean square error, scalar
    ----------------------------------------------------------------------------
    """

    return my_ridge_regression(y, tx, lambda_)

# ------------------------------------------------------------------------------

def logistic_regression(y, tx, initial_w, max_iters, gamma):
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
    - max_iters     # of iterations after which the procedure will stop, int>0
    - gamma         step size, scalar in ]0,1[
    Output:
    - w             obtained weights, (nfeatures,1) np.array
    - loss          loss computed as the mean square error, scalar
    ----------------------------------------------------------------------------
    """

    return my_logistic_regression(y, tx, initial_w, max_iters, gamma)

# ------------------------------------------------------------------------------

def reg_logistic_regression(y, tx, lambda_ ,initial_w, max_iters, gamma):
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
    - max_iters     # of iterations after which the procedure will stop, int>0
    - gamma         step size, scalar in ]0,1[
    Output:
    - w             obtained weights, (nfeatures,1) np.array
    - loss          loss computed as the mean square error, scalar
    ----------------------------------------------------------------------------
    """

    return my_reg_logistic_regression(y, tx, lambda_ ,initial_w, max_iters, gamma)

# ==============================================================================
# Trainers
# ==============================================================================
# This section contains the training functions required for the submission with
# optional arguments.
# The training functions are used to compute the optimal weights for a given
# model from a training dataset.
# ------------------------------------------------------------------------------

def my_least_squares_GD(y, tx, initial_w, max_iters=100, gamma=0.2, lambda_=0, eps=1e-3):
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
    - eps           end condition, scalar>0 (default=1e-5)
    Output:
    - w             obtained weights, (nfeatures,1) np.array
    - loss          loss computed as the mean square error, scalar
    ----------------------------------------------------------------------------
    """

    y = column_array(y)
    nsamples = tx.shape[0]
    nfeatures = tx.shape[1]
    w = initial_w
    err = 1

    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w,"mse",lambda_)
        w = w -gamma*grad
        if ((n_iter+1)%10 == 0):
            err = np.sum(np.abs(grad))/nfeatures
            if err < eps :
                print("Terminated least_squares_GD after ",n_iter," iterations.")
                break

    loss = compute_loss(y,tx,w)

    return w, loss

# ------------------------------------------------------------------------------

def my_least_squares_SGD(y, tx, initial_w, max_iters=1000, gamma=0.2, batch_size=10, lambda_=0, eps=1e-1):
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
    - batch_size    size of the subsamples, positive integer (default=4)
                    ( setting to 0 or >=nsamples will result in using standard sg)
    - max_iters     # of iterations after which the procedure will stop, int>0
                    (default=1000)
    - gamma         step size, scalar in ]0,1[ (default=0.5)
    - lambda_       regularization parameter, scalar>0 (default = 0)
    - eps           end condition, scalar>0 (default=1e-1)
    Output:
    - w             obtained weights, (nfeatures,1) np.array
    - loss          loss computed as the mean square error, scalar
    ----------------------------------------------------------------------------
    """

    y = column_array(y)
    nsamples = tx.shape[0]
    nfeatures = tx.shape[1]
    if ( batch_size==0 ) | ( batch_size>=nsamples ):
        w, loss = my_least_squares_GD(y, tx, initial_w, max_iters, gamma, lambda_,eps)
    else:
        w = initial_w
        batches = batch_iter(y, tx, batch_size, max_iters)
        err = 0
        nerr = min(max((nsamples/50),20),100)

        n_iter = 0
        for batch in batches:
            yb = batch[0]
            txb = batch[1]
            grad = compute_gradient(yb, txb, w)
            err = err + np.sum(np.abs(grad))
            w = w -gamma*grad
            if ((n_iter+1)%nerr == 0):
                if err/(nerr*batch_size*nfeatures) < eps:
                    print("Terminated least_squares_SGD after ",n_iter," iterations.")
                    break
                err = 0
            n_iter = n_iter+1

        loss = compute_loss(y,tx,w)

    return w, loss

# ------------------------------------------------------------------------------

def my_least_squares(y, tx, lambda_=0):
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

    y = column_array(y)

    w = tx.transpose().dot(tx) + a
    w = np.linalg.inv(w)
    w = w.dot(tx.transpose().dot(y))

    loss = compute_loss(y, tx, w)

    return w, loss

# ------------------------------------------------------------------------------

def my_ridge_regression(y, tx, lambda_, mode = "ls", max_iters=1000, gamma=0.2, batch_size=4, eps=1e-3):
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
    - batch_size    size of the subsamples, positive integer (default=4)
                    ( setting to 0 or >=nsamples will result in using standard sg)
    - eps           end condition, scalar>0 (default=1e-5)
    Output:
    - w             obtained weights, (nfeatures,1) np.array
    - loss          loss computed as the mean square error, scalar
    ----------------------------------------------------------------------------
    """

    if( mode=="gd" ) | ( mode==1 ): # gradient_descent
        w, loss = my_least_squares_GD(y, tx, initial_w, max_iters, gamma, lambda_,eps)
    elif ( mode=="sgd" ) | ( mode==2 ): # stochastic_gradient_descent
        eps = 1e-1
        w, loss = my_least_squares_SGD(y,tx,initial_w,max_iters,gamma,batch_size,lambda_,eps)
    else: # least_squares
        w, loss = my_least_squares(y, tx, lambda_)

    return w, loss

# ------------------------------------------------------------------------------

def my_logistic_regression(y, tx, initial_w, max_iters=100, gamma=0.2, mode="log", lambda_=0, eps=1e-3):
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
    - eps           end condition, scalar>0 (default=1e-5)
    Output:
    - w             obtained weights, (nfeatures,1) np.array
    - loss          loss computed as the mean square error, scalar
    ----------------------------------------------------------------------------
    """

    if (mode != "newton") & (mode != 2 ):
        mode = "log"

    y = column_array(y)
    w = initial_w
    nfeatures = tx.shape[1]

    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w, mode, lambda_)
        w = w -gamma*grad
        if ((n_iter+1)%10 == 0):
            err = np.sum(np.abs(grad))
            if err/nfeatures < eps:
                break

    loss = compute_loss(y,tx,w,"log")

    return w, loss

# ------------------------------------------------------------------------------

def my_stoch_logistic_regression(y, tx, initial_w, max_iters=100, gamma=0.2, batch_size=4, mode="log", lambda_=0, eps=1e-1):
    """
    ----------------------------------------------------------------------------
    Iteratively computes the model weights "w" from "y" and "tx" using stochastic
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
    - batch_size    size of the subsamples, positive integer (default=4)
                    ( setting to 0 or >nsamples will result in using standard sg)
    - mode          choice of algorithm (1/"log",2/"newton"), string/integer
                    (default=1/"log")
    - lambda_       regularization parameter, scalar>0
    - eps           end condition, scalar>0 (default=1e-5)
    Output:
    - w             obtained weights, (nfeatures,1) np.array
    - loss          loss computed as the mean square error, scalar
    ----------------------------------------------------------------------------
    """

    if (mode != "newton") & (mode != 2 ):
        mode = "log"

    y = column_array(y)
    w = initial_w

    nsamples = tx.shape[0]
    nfeatures = tx.shape[1]
    if ( batch_size==0 ) | ( batch_size>=nsamples ):
        w, loss = my_logistic_regression(y, tx, initial_w, max_iters, gamma, mode, lambda_, eps)
    else:
        w = initial_w
        batches = batch_iter(y, tx, batch_size, max_iters)
        err = 0
        nerr = min(max((nsamples/50),20),100)

        n_iter = 0
        for batch in batches:
            grad = compute_gradient(y, tx, w, mode, lambda_)
            w = w -gamma*grad
            err = err + np.sum(np.abs(grad))
            if ((n_iter+1)%nerr == 0):
                if err/(nerr*batch_size*nfeatures) < eps:
                    break
                err = 0
            if( np.abs(w).any()>1e+20):
                break
            n_iter = n_iter+1

    loss = compute_loss(y,tx,w,"log")

    return w, loss

# ------------------------------------------------------------------------------

def my_reg_logistic_regression(y, tx, lambda_ ,initial_w, max_iters=100, gamma=0.2, mode="log", eps=1e-5):
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
    - eps           end condition, scalar>0 (default=1e-5)
    Output:
    - w             obtained weights, (nfeatures,1) np.array
    - loss          loss computed as the mean square error, scalar
    ----------------------------------------------------------------------------
    """

    w, loss = my_logistic_regression(y,tx,initial_w,max_iters,gamma,mode,lambda_,eps)

    return w, loss

# ==============================================================================
# Utility functions for trainers
# ==============================================================================
# Accesory function required for the different training functions.
# ------------------------------------------------------------------------------

def compute_gradient(y, tx, w, mode="mse", lambda_=0):
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
    - mode          choice of regression type for which the gradient is used
                    (0/"mse",1/"log",2/"newton"), int/string (default=0/"mse")
    - lambda_       regularization parameter, scalar>0 (default = 0)
    Output:
    - grad          gradient, (tx.shape[0],1) np.array
    ----------------------------------------------------------------------------
    MSE Gradient:   L(w) = (y-x*w)^2/(2*N) + lambda_*w^2
                    grad = -(y-x*w)^t*x/N + 2*lambda_*w
                         = -(y-x*w-N*lambda_/2)^t*w
    Log Gradient:   L(w) = lambda_/2*w^2 + sum_n^N ln[1+exp(x_n*w)]-y_n*x_n*w
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

    y = column_array(y)

    # compute error
    if (mode == "msa") | (mode == 2):
        e = y-tx.dot(w)
        e = np.absolute(e).sum()/e.shape[0]/2
    elif (mode == "log") | (mode == 3):

         e = compute_y(tx,w)
         e = np.abs(y - e)/(2*len(e))

        # e = tx.dot(w)
        #
        # lim = 100.0
        # large = e>lim
        # small = e<-lim
        # neither = np.logical_not(large)*np.logical_not(small)
        #
        # x = e
        # x[neither] = np.log(1+np.exp(e[neither]))
        # x[large] = e[large]
        # x[small] = 0
        #
        # e = (x - y*e)/(2*len(e))

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

def standardize(tx, mean_=0, std_=1):
    """
    ----------------------------------------------------------------------------
    Standardize data (i.e substract mean and divide by standard deviation). If
    non-zero mean and non unitary standard deviation are passed as arguments the
    passed values will be used instead of the mean value and standrad deviation
    of the sample "tx".
    ----------------------------------------------------------------------------
    Input:
    - tx            features, (nsamples,nfeatures) np.array
    - mean_         mean value of features, (1,nfeatures) np.array, (default=0)
    - std_          standrad deviation of features, (1,nfeatures) np.array,
                    (default=1)
    Output:
    - tx_std        features standardized to mean 0 and std 1,
                    (nsamples,nfeatures) np.array
    - mean          mean value of features, (1,nfeatures) np.array
    - std           standrad deviation of features, (1,nfeatures) np.array
    ----------------------------------------------------------------------------
    """

    if (np.sum(mean_==0)) | (np.sum(std_==1)):
        # does not work if there is an actual feature with mean 0 or std 1
        mean = np.mean(tx,0)
        std = np.std(tx,0)
    else:
        mean = mean_
        std = std_

    tx_std = np.divide(tx-mean,std)

    return tx_std, mean, std

# ------------------------------------------------------------------------------

def split_data(y, tx, ratio, seed=1):
    """
    ----------------------------------------------------------------------------
    Randomly split the dataset "(y,tx)" based on the passed split ratio.
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

    split1 = indices[0:int(np.floor(nsamples*ratio))]
    split2 = indices[int(np.floor(nsamples*ratio)):nsamples]


    return y[split1],tx[split1,:],y[split2],tx[split2,:]

# ------------------------------------------------------------------------------

def eq_split_data(y, tx, nparts, shuffle=True, seed=1):
    """
    ----------------------------------------------------------------------------
    Randomly split the dataset "(y,tx)" into "nparts" equally sized portions.
    ----------------------------------------------------------------------------
    Input:
    - y             "measured" objective function, (nsamples,1) np.array
    - tx            features, (nsamples,nfeatures) np.array
    - nparts        number of parts, integer>0
    - shuffle       wheter or not to shuffle the data before splitting, logical
                    (default=True)
    - seed          seed for np.random, integer
    Output:
    - ys            "measured" objective function, (nsamples//nparts,1,nparts) np.array
    - txs           features, (nsamples//nparts,nfeatures,nparts) np.array
    - inds          indices included in each split, (nsamples//nparts,nparts) int np.array
    - boolind       indices included in each split, (nsamples,nparts) logical np.array
    ----------------------------------------------------------------------------
    """

    np.random.seed(seed)
    y = column_array(y)

    nsamples = tx.shape[0]
    nfeatures = tx.shape[1]
    indices = np.indices([nsamples])[0]
    np.random.shuffle(indices)

    nspp = nsamples//nparts #number of samples per part

    ys = np.zeros([nspp,nparts])
    txs = np.zeros([nspp,nfeatures,nparts])
    inds = np.zeros([nspp,nparts],"int")
    boolind = np.zeros([nsamples,nparts],"bool")
    for part in range(nparts):
        inds[:,part] = indices[part*nspp:(part+1)*nspp]
        ys[:,part] = y[inds[:,part],0]
        txs[:,:,part] = tx[inds[:,part]]
        boolind[inds[:,part],part] = True

    return ys, txs, inds, boolind

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
    - mix_features  if enabled features will be mixed up to the second degree,
                    boolean (default=False)
    Output:
    - tx            the polynomial expansion of "x",
                    (nsamples,nfeatures*degree+add_constant) np.array (w/o fm)
                    (nsamples,?) np.array (w fm)
    ----------------------------------------------------------------------------
    """

    nfeatures = x.shape[1]
    nsamples = x.shape[0]

    # if mix_features:
    #     nftot = nCr(nfeatures+degree,degree)
    #     tx = np.ones([nsamples,nftot])
    #
    #     for d in range(1,degree):
    #         for i in range(nfeatures):
    #             tx(:,1+i*d*(nfeatures):1+(1+i*d)*(nfeatures)) = tx(:,1:1+nfeatures)*x(:,i)
    #
    # else:
    tx = np.ones([nsamples,degree*nfeatures+1])
    for n in range(0,nfeatures):
        tx[:,n*degree+1] = x[:,n]
        for d in range(2,degree+1):
            tx[:,n*(degree)+d] = tx[:,n*(degree)+d-1]*x[:,n]

    if mix_features:
        tx_add = np.zeros([nsamples,int(nfeatures*(nfeatures-1)/2)])
        ind = 0
        for i in range(nfeatures):
            for j in range(i+1,nfeatures):
                tx_add[:,ind] = x[:,i]*x[:,j]
                ind = ind+1

        tx = np.column_stack([tx,tx_add])

    if not add_constant:
        tx = tx[:,1:]
        #TODO : improve this, fix mix_features

    return tx

# ==============================================================================
# Utility
# ==============================================================================
# Collection of other functions of use.
# ------------------------------------------------------------------------------

def generate_data(nsamples,nfeatures,seed=1,std=0.1):
    """
    ----------------------------------------------------------------------------
    This procedure generates a random data set following a linear model
    with random coefficients. Additionally normal distributed error is then
    added to the dataset.
    ----------------------------------------------------------------------------
    Input:
    - nsamples      number of samples, integer>0
    - nfeatures     number of features, integer>0
    - seed          seed for np.random, integer (default=1)
    - std           standrad deviation of normal distrubted noise, scalar>0,
                    (default=0.1)
    Output:
    - y             objective function, (nsamples,1) np.array
    - x             features, (nsamples,nfeatures+1) np.array
    - w             actual weights, (nfeatures+1,1) np.array
    ----------------------------------------------------------------------------
    """

    np.random.seed(seed)

    w = np.random.random([nfeatures+1,1])*2-1

    x = np.random.random([nsamples,nfeatures])
    tx = np.ones([nsamples,1])
    tx = np.column_stack((tx,x))

    y = tx.dot(w)
    y = y + np.random.normal(0,std,[nsamples,1])

    return y, tx, w

# ------------------------------------------------------------------------------

def generate_bin_data(nsamples,nfeatures,seed=1,std=0.1):
    """
    ----------------------------------------------------------------------------
    This procedure generates a random data set following a linear model
    with random coefficients and a binary objective function.
    ----------------------------------------------------------------------------
    Input:
    - nsamples      number of samples, integer>0
    - nfeatures     number of features, integer>0
    - seed          seed for np.random, integer (default=1)
    - std           standrad deviation of normal distrubted noise, scalar>0,
                    (default=0.1)
    Output:
    - y             binary objective function, (nsamples,1) np.array
    - x             features, (nsamples,nfeatures+1) np.array
    - w             actual weights, (nfeatures+1,1) np.array
    ----------------------------------------------------------------------------
    """

    z,x,w = generate_data(nsamples,nfeatures,seed,std)

    y = compute_y(z,1)

    return y, x, w

# ------------------------------------------------------------------------------

def column_array(y):
    """
    ----------------------------------------------------------------------------
    Reshape y to a (len,1) np.array.
    ----------------------------------------------------------------------------
    Input:
    - y             y, (nelem,) np.array
    Output:
    - yr            yr, (nelem,1) np.array
    ----------------------------------------------------------------------------
    """

    if len(y.shape) != 2:
        nelem = len(y)
        yr = np.zeros([nelem,1])
        yr[:,0] = y
    else:
        yr = y

    return yr

# ------------------------------------------------------------------------------

def compute_y(tx,w):
    """
    ----------------------------------------------------------------------------
    Convert {0,1} array to {-1,1} array
    ----------------------------------------------------------------------------
    Input:
    - tx            features, (nsamples,nfeatures) np.array
    - w             current weights, (nfeatures,1) np.array
    Output:
    - y             predicted objective function, (nelem,1) np.array {-1,1}
    ----------------------------------------------------------------------------
    """

    y = np.where(tx.dot(w)<0,-1,1)

    return y

# ------------------------------------------------------------------------------

def nCr(n, k):
    """
    ----------------------------------------------------------------------------
    Compute binomial coefficients.
    ----------------------------------------------------------------------------
    Input:
    - n             integer > 0
    - k             integer > 0
    Output:
    - ncr           nCR(n,k)
    ----------------------------------------------------------------------------
    """
    if 0 <= k <= n:
        ncr = 1
        n_ = n
        for i in range(1,k+1):
            ncr *= n_/i
            n_ = n_-1
    else:
        ncr = 0

    return ncr

# ==============================================================================
# WIP
# ==============================================================================
# Other functions that are still work in progress and not guaranteed to work.
# ------------------------------------------------------------------------------

# def example(arg1, arg2):
#     """
#     ----------------------------------------------------------------------------
#     ----------------------------------------------------------------------------
#     Input:
#     - arg1          argument 1, type (default=?)
#     - arg2          argument 2, type (default=?)
#     Output:
#     - rv            return value, type
#     ----------------------------------------------------------------------------
#     """
#
#     y = x
#
#     return y
