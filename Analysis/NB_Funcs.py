import numpy as np
import pandas as pd
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