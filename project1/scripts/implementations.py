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
#   -> gradient_descent(y, tx, initial_w, max_iters, gamma)
#   -> stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma)
#   -> least_squares(y, tx)
#   -> ridge_regression(y, tx, lambda_)
#   -> logistic_regression(y, tx, initial_w,max_iters, gamma, batch_size=1, mode="gradient")
#   -> reg_logistic_regression(y, tx, lambda_ ,initial_w, max_iters, gamma, mode="gradient")
# -Loss functions
#   -> compute_loss(y, tx, w, mode="mse", lambda_=0)
# -Preprocessing
#   -> standardize(tx)
#   -> split_data(x, y, ratio, seed=1)
#   -> add_constant(x)
#   -> build_polynomial(x, degree, add_constant_feature=True, mix_features=False)
# -Helpers
#   -> compute_gradient(y, tx, w, mode=0, lambda_=0)
#   -> compute_sigma(tx,w)
#   -> batch_iter(y, tx, batch_size, num_batches=1, shuffle=True)
# ==============================================================================

import numpy as np
import math as math

# ==============================================================================
# trainers
# ==============================================================================

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """
    Gradient descent algorithm.
    """

    w = initial_w

    for n_iter in range(max_iters):
        w = w -gamma*compute_gradient(y, tx, w)

    loss = compute_loss(y,tx,w)

    return w, loss

# ------------------------------------------------------------------------------

def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """
    Stochastic gradient descent algorithm.
    """

    w = initial_w

    batches = batch_iter(y, tx, batch_size, min(max_iters,(y.shape[0]+batch_size-1)//batch_size))
    n_iter = 0

    for batch in batches:
        yb = np.transpose(batch[0])
        txb = batch[1]
        w = w -gamma*compute_gradient(yb, txb, w)
        n_iter = n_iter + 1

    loss = compute_loss(y,tx,w)

    return w, loss

# ------------------------------------------------------------------------------

def least_squares(y, tx):
    """
    Computes the least squares solution.
    """

    w = tx.transpose().dot(tx)
    w = np.linalg.inv(w)
    w = w.dot(tx.transpose())
    w = w.dot(y)

    loss = compute_loss(y, tx, w)

    return w, loss

# ------------------------------------------------------------------------------

def ridge_regression(y, tx, lambda_):
    """
    Computes ridge regression solutions.
    """

    w = tx.transpose().dot(tx)+2*tx.shape[0]*lambda_*np.eye(tx.shape[1])
    w = np.linalg.inv(w)
    w = w.dot(tx.transpose().dot(y))

    loss = compute_loss(y,tx,w)

    return w, loss

# ------------------------------------------------------------------------------

def logistic_regression(y, tx, initial_w,max_iters, gamma, batch_size=1, mode="gradient"):
    """
    Computes logistic regression solutions.
    """

    w = initial_w

    batches = batch_iter(y, tx, batch_size, min(max_iters,(y.shape[0]+batch_size-1)//batch_size))
    n_iter = 0

    for batch in batches:
        yb = np.transpose(batch[0])
        txb = batch[1]
        w = w -gamma*compute_log_step(yb, txb, w, mode)
        n_iter = n_iter + 1

    loss = compute_loss(y,tx,w,"log")

    return w, loss

# ------------------------------------------------------------------------------

def reg_logistic_regression(y, tx, lambda_ ,initial_w, max_iters, gamma, mode="gradient"):
    """
    Computes regularized linear regression
    """

    w = initial_w

    batches = batch_iter(y, tx, batch_size, min(max_iters,(y.shape[0]+batch_size-1)//batch_size))
    n_iter = 0

    for batch in batches:
        yb = np.transpose(batch[0])
        txb = batch[1]
        w = w -gamma*compute_log_step(yb, txb, w, mode, lambda_)
        n_iter = n_iter + 1

    loss = compute_loss(y,tx,w,"log")

    return w, loss

# ==============================================================================
# Loss functions
# ==============================================================================

def compute_loss(y, tx, w, mode="mse", lambda_=0):
    """
    Computes MSE/MSA/RMSE loss.
    """

    # compute error
    if (mode == "msa") | (mode == 2):
        e = y-tx.dot(w)
        e = np.absolute(e).sum()/e.shape[0]/2
    else if (mode == "log") | (mode == 3):
        e = tx.transpose().dot(w)
        e = np.log(1+np.exp(e)) - y*e
    else:
        e = y-tx.dot(w)
        e = (np.transpose(e).dot(e)/(2*e.shape[0]))
        if (mode == "rmse") | (mode == 1):
            e = np.sqrt(2*e)

    loss = np.sum(e)

    # compute regularization contribution
    if lambda_ == 0 then:
        a = 0
    else:
        a = lambda_*np.sum(w*w)

    return loss + a

# ==============================================================================
# Preprocessing
# ==============================================================================

def standardize(tx):
    """
    standardize data (i.e substract mean and divide by standard deviation)
    """

    mean = np.mean(tx,0)
    std = np.std(tx,0)
    tx_std = np.divide(tx-mean,std)

    return tx_std, mean, std

# ------------------------------------------------------------------------------

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing
    """
    np.random.seed(seed)

    nelem = len(y)
    indices = np.indices([nelem])[0]
    np.random.shuffle(indices)

    split1 = indices[0:math.floor(nelem*ratio)]
    split2 = indices[math.floor(nelem*ratio):nelem]

    return x[split1,:],y[split1],x[split2,:],y[split2]

# ------------------------------------------------------------------------------

def add_constant(x):
    """
    add a constant feature
    """

    tx = np.column_stack(np.ones([x.shape[0],1]),x)

    return tx

# ------------------------------------------------------------------------------

# mix_features is WIP
def build_polynomial(x, degree, add_constant_feature=True, mix_features=False):
    """
    polynomial basis functions for input data x (one feature, i.e.
    x is of the form nelem*1), for j=0 up to j=degree. Note that this feature
    expansion by default adds a constant feature and does not mix features.
    """

    nfeatures = x.shape[1]
    nelements = x.shape[0]

    if False: #mix_features:
        nfeatures = nfeatures
        nftot = (nfeatures**(degree+1)-1)/(nfeatures-1)
        tx = np.ones([nelements,nftot])
        for d in range(1,degree):
            for i in range(nfeatures):
                tx(:,1+i*d*(nfeatures):1+(1+i*d)*(nfeatures)) = tx(:,1:1+nfeatures)*x(:,i)
        # WIP
        nfeatures = nfeatures

    else:
        tx = np.ones([nelements,degree*nfeatures+1])
        for n in range(0,nfeatures+1):
                tx[:,n*(degree)+1] = x[:,n]
            for d in range(2,degree+1):
                tx[:,n*(degree)+degree] = tx[:,n*(degree)+degree-1]*x[:,n]

    if not add_constant_feature:
        tx = tx[:,1:]
        #TODO : improve this, fix mix_features

    return tx

# ==============================================================================
# helpers for trainers
# ==============================================================================

def compute_gradient(y, tx, w, mode=0, lambda_=0):
    """
    Compute the gradient (w.r.t. the mean square error).
    """

    grad = -tx.transpose().dot(y-tx.dot(w))/y.shape[0]

    return grad

# ------------------------------------------------------------------------------

def compute_sigma(tx,w):
    """
    computes sigma=exp(tx*w)/(1+exp(tx*w)) for logisitic regression
    """

    x = tx.dot(w)

    large = x>100
    small = x<-100
    neither = (not large)*(not small)

    x[neither] = np.exp(x[neither])

    sigma = np.divide(x,1+x)

    sigma[large] = 1
    sigma[small] = 0

    return sigma

# ------------------------------------------------------------------------------

def compute_log_gradient(y, tx, w, lambda_=0):
    """
    computes the gradient for logistic regression
    """

    if (lambda_ == 0 ):
        a = 0
    else:
        a = tx.shape[0]*lambda_*w

    grad = compute_sigma(tx,w)-y
    grad = tx.transpose().dot(grad)+a

    return grad

# ------------------------------------------------------------------------------

def compute_log_newton_step(y, tx, w, lambda_):
    """
    computes one step of logistic regression (using compute_log_gradient)
    """

    sigma = compute_sigma(tx,w)
    S_tmp = sigma*(1-sigma)
    n_elem = S_tmp.shape[0]
    S = np.zeros([n_elem,n_elem])
    for i in range(n_elem):
        S[i,i] = S_tmp[i]

    H = tx.transpose().dot(S).dot(tx)
    grad = compute_log_gradient(y,tx,w,lambda_)

    return np.linalg.inv(H).dot(grad)

# ------------------------------------------------------------------------------

def compute_log_step(y, tx, w, mode="newton", lambda_=0):
    """
    wrapper function for the differen step methods.
    """

    if ( mode == "newton" ) | ( mode == 0 ):
        return compute_log_newton_step(y,tx,w,lambda_)
    else if ( mode == "gradient" ) | ( mode == 1 ):
        return compute_log_gradient(y,tx,w,lambda_)
    else:
        return compute_log_gradient(y,tx,w,lambda_)

# ------------------------------------------------------------------------------

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx

    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

# ==============================================================================
# WIP
# ==============================================================================

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
#     else if tx.shape[1] == nsamples:
#         nfeatures = tx.shape[0]
#         tx = tx.transpose()
#
#     return y, tx, w, nsamples, nfeatures
#
# def generate_data(nsamples,nfeatures,seed=1):
#     """
#     generates a random dataset from a random model
#     """
#
#     np.random.seed(seed)
#
#     w = np.random.random([nfeatures+1,1])
#
#     x = np.random.random([nsamples,nfeatures])
#     tx = np.ones([nsamples,1])
#     tx = np.column_stack((tx,x))
#
#     y = tx.dot(w)
#     y = y + np.random.normal(0,0.5,[nsamples,1])
#
#     return y, tx, w
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
