# -*- coding: utf-8 -*-
# ==============================================================================
# run.py
# ------------------------------------------------------------------------------
# authors:                             Patrick Ley, Joel Fischer
# date:                                18.12.2018
# ==============================================================================
# A python script to reproduce our submission.
# ==============================================================================
# TODO:
#   - nobody ain't got time for that!
# ==============================================================================

# import numpy
import random as rd
import numpy as np

# import scikit-surprise classes
from surprise import SVD
from surprise import kNNWithZScore
from surprise import Dataset

# import costum stuff
from utility import *

# assure reproducability (hopefully...)
rd.seed(10)

# import training and quiz data
datafilepath = "../data/data_train.csv"
data_df = loadData2df(infilepath=datafilepath)

predinfilepath = "../data/sample_submission.csv"
pred_df = loadData2df(infilepath=predinfilepath)

data_ds = loadData2ds(data_df)
full_train_ds = data_ds.build_full_trainset()

# initialize model

# #svd option 1
# algo = SVD(\
# n_factors=125,\
# lr_all=0.007,\
# reg_bu=0.05,\
# reg_bi=0.02,\
# reg_pu=0.05,\
# reg_qi=0.05,\
# n_epochs=20,\
# biased=True)
#
# #svd option 2
# algo = SVD(\
# n_factors=10,\
# lr_all=0.01,\
# reg_all=0.1,\
# n_epochs=2000,\
# biased=True)

# # knn with zscore
# algo = KNNWithZScore(\
# k=1000,\
# min_k=10,\
# sim_options={'name': 'pearson', 'user_based': False})

# knn with means
algo = KNNWithMeans(\
k=200,\
min_k=10,\
sim_options={'name': 'pearson', 'user_based': False},\
verbose=True)

# train model
algo.fit(full_train_ds)

# compute predictions
generatePredictions(algo,pred_df)

# export predictions
predoutfilepath = "../data/submission.csv"
exportPredictions(pred_df,outfilepath=predoutfilepath)
