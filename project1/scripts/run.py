import numpy as np
from implementations import *
from proj1_helpers import *

path_tr = "../data/train.csv"
yb_tr, data_tr, ids_tr = load_csv_data(path_tr,False)

degree = 7
data_tr0 = poly_expansion(data_tr, degree, False, False)
data_tr0, mean_tr, std_tr = standardize(data_tr0)
data_tr0 = add_constant(data_tr0)
nfeatures = data_tr0.shape[1]
initial_w = np.ones([nfeatures,1])/nfeatures

w_SGD, loss_SGD = my_stoch_logistic_regression(yb_tr, data_tr0, initial_w, 2000, 0.02, 10, mode = "log", lambda_=0, eps=1e-5)

path_te = "../data/test.csv"
yb_te, data_te, ids_te = load_csv_data(path_te,False)
data_te0 = poly_expansion(data_te, degree, False, False)
data_te0, mean_te, std_te = standardize(data_te0, mean_tr, std_tr)
data_te0 = add_constant(data_te0)
yb_te = yb_te.reshape(len(yb_te),1)

y_pred = compute_y(data_te0, w_SGD)

create_csv_submission(ids_te, y_pred, "Submission.csv")