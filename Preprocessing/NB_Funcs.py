import numpy as np
import pandas as pd
import pickle
from scipy.interpolate import CubicSpline



def cubic_splines(data, xs):
    """ returns the derivatives with respect to xs. Assumes the data is formatted with 3 dimensions. D1 = time. D2 = rows. D2 = columns"""

    t = data.shape[0]
    rows = data.shape[1]
    cols = data.shape[2]

    dXdt = np.empty((t,rows,cols))
    
    for i in range(rows):
        for j in range(cols):

            derivs = CubicSpline(xs,data[:,i,j])(xs,1)
            
            dXdt[:,i,j] = derivs
    
    return dXdt

def pickle_save(filename, data):
    """saves data as a pickle file"""

    with open(filename,'wb') as _save:
        pickle.dump(data,_save)
    
def pickle_load(filename):
    """loads a pickle file and returns it"""

    with open(filename,'rb') as _load:
        var = pickle.load(_load)
    return var

def best_score(model, featureSet):
    """Selects the row from model output that contains the equation with highest score. Also adds featureSet
    as field in the dataframe."""

    model.equations_['feature_set'] = featureSet
    ind = model.equations_['score'].idxmax()
    df = model.equations_.iloc[ind]
    return df

def Null_Model_Errors(true,mean):
    """Get the plate RMSE of the Null Model"""
    
    errs = np.zeros(true.shape)
    for i in range(true.shape[1]):
        for j in range(true.shape[2]):
            errs[:,i,j] = np.sqrt(np.square(mean-true[:,i,j]))
    return errs

def get_errors(true, preds, agg="None"):
    """Get the RMSE of predictions. Can do by space, time or none"""
    
    rows,cols,points = true.shape[0],true.shape[1],true.shape[2]
    rmse = np.sqrt(np.square(preds-true))
    if agg=="None": return rmse
    if agg=="Time": return rmse.mean(axis=(1,2))
    if agg=="Space": return rmse.mean(axis=0)
    else: return "Invalid Agg"

def predict(equation,dataframes):
    """Evaluate the named data from dataframe based on some passed equation"""
    
    rows = len(dataframes); cols = len(dataframes[0]); points = len(dataframes[0][0])
    preds = np.zeros((points,rows,cols))
    for i in range(rows):
        for df in dataframes[i]:
            ii = int(df['i'][0]); jj = int(df['j'][0])
            preds[:,ii,jj] = np.array(df.eval(equation))
    return preds