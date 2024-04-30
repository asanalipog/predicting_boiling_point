
import sys
sys.path.append('..')  # to import from GP.kernels and property_predition.data_utils

import gpflow
from gpflow.mean_functions import Constant
from gpflow.utilities import print_summary
import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
import math

from rdkit import Chem
from rdkit.Chem import AllChem

from GP.kernels import Tanimoto
from property_prediction.data_utils import transform_data, TaskDataLoader, featurise_mols


def main():

    data_loader = TaskDataLoader('bp', 'alkene.csv')
    smiles_list, y1 = data_loader.load_property_data()
    X = featurise_mols(smiles_list, 'fingerprints')
    
    X = X.astype(np.float64)
    y = y1.reshape(-1, 1)
    

    m1 = None
    def objective_closure():
        return -m1.log_marginal_likelihood()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    _, y_train, _, y_test, y_scaler = transform_data(X_train, y_train, X_test, y_test)
    X_train = X_train.astype(np.float64)
    X_test = X_test.astype(np.float64)
    
    k = Tanimoto()
    m1 = gpflow.models.GPR(data=(X_train, y_train), mean_function=Constant(np.mean(y_train)), kernel=k, noise_variance=1)
    

    opt = gpflow.optimizers.Scipy()
    opt.minimize(objective_closure, m1.trainable_variables, options=dict(maxiter=1000))
    print_summary(m1)
    

    y_pred, y_var = m1.predict_f(X_test)

    y_pred = y_scaler.inverse_transform(y_pred)
    y_test = y_scaler.inverse_transform(y_test)


    score = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print("\nR^2: {:.3f}".format(score))
    print("RMSE: {:.3f}".format(rmse))
    print("MAE: {:.3f}".format(mae))
    

   
    #X_test = X_test.astype(np.float64) need to do that
    #print(m1.predict_f("CCC(C)CC"))

    my_list = ['CCC(C)CC', 'CCC(C)CC(C)CC', 'CC(C)CC']
    my_train = featurise_mols(my_list, 'fingerprints')
    my_train = my_train.astype(np.float64)
    my_pred, my_var = m1.predict_f(my_train)
    my_pred = y_scaler.inverse_transform(my_pred)
    print(my_pred)


if __name__ == "__main__":
    main()
 
