# -*- coding: utf-8 -*-
import numpy as np
import math as math

# compute loss functions

def build_poly1(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""

    nelem = len(x)
    tx = np.ones([nelem,degree+1])
    for j in range(1,degree+1):
        tx[:,j] = tx[:,j-1]*x

    return tx

def build_polynomial(x, degrees):
    """polynominal basis functions for input data x with more than one feature (i.e. x 2D array)"""
    """degrees has to be an array/list specifing the degree to which each feature should be developed"""

    nelem = x.shape[0]
    nfeatures = x.shape[1]

    if nfeatures > 1:
        if( len(degrees) != nfeatures ):
            raise ValueError('Degrees has to be an array/list with nfeatures elements - Thank you python for not being type safe.')
        else:
            tx = build_poly1(x[:,0],degrees[0])
            for i, degree in enumerate(degrees[1:]):
                tx = np.concatenate([tx,build_poly1(x[:,i],degree)[:,1:]],1)
    else:
        tx = build_poly1(x,degrees)

    return tx

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

def generate_data(nsamples,nfeatures,seed=1):

    np.random.seed(seed)
    w = np.random.random([nfeatures+1,1])
    x = np.random.random([nsamples,nfeatures])
    tx = np.ones([nsamples,1])
    tx = np.column_stack((tx,x))
    y = tx.dot(w)
    y = y + np.random.normal(0,0.5,[nsamples,1])

    return y, tx, w

def compute_loss_mse(y, tx, w):
    """Computes MSE loss."""

    e = y-tx.dot(w)
    loss = (np.transpose(e).dot(e)/(2*e.shape[0]))

    return loss

def compute_loss_mae(y, tx, w):
    """Computes MAE loss."""

    e = y-tx.dot(w)
    loss = np.absolute(e).sum()/e.shape[0]/2

    return loss


def compute_loss_rmse(y, tx, w):
    """Computes RMSE loss."""

    loss = math.sqrt(2*compute_loss_mse(y,tx,w))

    return loss


def compute_loss(y, tx, w, mode="mse"):
    """Computes MSE/MSA/RMSE loss."""

    if (mode == "msa") | (mode == 1):
        loss = compute_loss_mae(y, tx, w)
    elif (mode == "rmse") | (mode == 2):
        loss = compute_loss_rmse(y, tx, w)
    else:
        loss = compute_loss_mse(y, tx, w)

    return loss[0][0]

# trainers

def correct_shape(y, tx, w):

    nsamples = len(y)
    y = y.reshape(nsamples,1)

    nfeatures = tx.shape
    if tx.shape[0] == nsamples:
        nfeatures = tx.shape[1]
    elif tx.shape[1] == nsamples:
        nfeatures = tx.shape[0]
        tx = tx.transpose()

    w = w.reshape(nfeatures,1)

    return y, tx, w, nsamples, nfeatures

def compute_gradient(y, tx, w):
    """Compute the gradient."""

    grad = -tx.transpose().dot(y-tx.dot(w))/y.shape[0]
    return grad

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""

    w = initial_w
    for n_iter in range(max_iters):

        w = w -gamma*compute_gradient(y, tx, w)

    loss = compute_loss(y,tx,w)

    return w, loss

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""

    grad =  compute_gradient(y, tx, w)
    return grad

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

def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""

    w = initial_w

    batches = batch_iter(y, tx, batch_size, min(max_iters,(y.shape[0]+batch_size-1)//batch_size))
    n_iter = 0

    for batch in batches:

        yb = np.transpose(batch[0])
        txb = batch[1]
        w = w -gamma*compute_stoch_gradient(yb, txb, w)
        n_iter = n_iter + 1

    loss = compute_loss(y,tx,w)

    return w, loss

def least_squares(y, tx):
    """Computes the least squares solution."""

    w = tx.transpose().dot(tx)
    w = np.linalg.inv(w)
    w = w.dot(tx.transpose())
    w = w.dot(y)

    loss = compute_loss(y, tx, w)

    return w, loss

def ridge_regression(y, tx, lambda_):
    """Computes ridge regression solutions."""

    w = tx.transpose().dot(tx)+2*tx.shape[0]*lambda_*np.eye(tx.shape[1])
    w = np.linalg.inv(w)
    w = w.dot(tx.transpose().dot(y))

    loss = compute_loss(y,tx,w)

    return w, loss

#logistic_regression(y, tx, initial w,max iters, gamma)
#reg_logistic_regression(y, tx, lambda ,initial w, max iters, gamma)
