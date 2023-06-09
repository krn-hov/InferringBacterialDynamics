from cmath import nan
from pysr import PySRRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import plotly.express as px
from matplotlib.lines import Line2D
from scipy.stats import pearsonr
import random


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


def concat_samples(list_data, indices):
    """concatenates dataframe elements of list_data together based on indices"""

    beg = pd.DataFrame(columns = list_data[0].columns)

    for index in indices:
        beg = pd.concat([beg,list_data[index]])
        
    return beg


def get_ring_training_data(list_data,ring,N):
    """ Obsolete """

    border_indices = []
    
    for i in range(len(list_data)):
        ring_val = list_data[i]['Ring'].unique()[0]
        if ring_val == ring:
            border_indices.append(i)
        else:
            continue
    training_indices = random.sample(border_indices, N)
    
    training_data = concat_samples(list_data, training_indices)
    
    return (training_indices,training_data)

def Run_SR_Process(Nmodel, features, response):
    
    print(f"running Model")
    Nmodel.fit(features,response)
    
    results = Nmodel.equations_
    best_eq = Nmodel.get_best()['equation']
    
    return (results,best_eq)


def complexityVSloss(model_equations, title = 'Complexity VS Loss of Equation'):
    """plot complexity vs loss of equations in dataframe"""
    df = model_equations
    fig = px.line(df, x = 'complexity', y = 'loss', markers = True, title = title,
                  hover_data = {'loss':':.6f','equation':True})
    fig.show()

def predVSdata(X,Y,xlabel, ylabel ,title):
    fig, ax = plt.subplots(figsize=(8,8))
    plt.scatter(x = X, y = Y, marker = '.')
    ax.plot([0, 1], [0, 1], color='red', alpha = 0.5, transform=ax.transAxes)
    minimum = min(X.min(),Y.min())
    maximum = max(X.max(),Y.max())
    plt.xlim(minimum - 0.01, maximum + 0.01)
    plt.ylim(minimum - 0.01, maximum + 0.01)
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.title(title,fontsize = 16)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    
def inferredVScalculated(equation, og_data, inds, y_var, No_Eval = True, title = 'dNdt - Inferred vs Calculated'):
    """Obsolete"""
    field = y_var + '_eval'
    Neq = field + '= ' + equation
    
    if No_Eval:
        for df in og_data:
            df.eval(Neq,inplace = True)
            
    plotting_data = concat_samples(og_data,inds)
    
    dNdt_R = pearsonr(plotting_data[y_var],plotting_data[field])
    print(f"The pearson correlation coefficient is {dNdt_R}")
    
    fig, ax = plt.subplots(figsize = (12,12))
    plt.scatter(x = plotting_data[y_var],y = plotting_data[field],marker = '.')
    ax.plot([0, 1], [0, 1], color='red', transform=ax.transAxes)
    
    minimum = min(plotting_data[y_var].min(),plotting_data[field].min())
    maximum = max(plotting_data[y_var].max(),plotting_data[field].max())
    
    plt.xlim(minimum - 0.01, maximum + 0.01)
    plt.ylim(minimum - 0.01, maximum + 0.01)
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.title(title,fontsize = 16)
    plt.xlabel('Inferred')
    plt.ylabel('Calculated')
    plt.show()


def predVSreal_curves(reals,preds,X,r_label,p_label,xlabel,ylabel,title):
    
    plt.figure(figsize = (16,10))
    plt.title(title)
    
    for i in range(len(reals)):
        if i == 0:
            plt.plot(X,reals[i],c='b',lw=1,alpha=0.7, label = r_label)
            plt.plot(X,preds[i],c='orange',lw=1,alpha = 0.7, label = p_label)
        else:
            plt.plot(X,reals[i],c='b',lw=1,alpha=0.7)
            plt.plot(X,preds[i],c='orange',lw=1,alpha = 0.7)
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.legend()
    
    plt.show()
    
def fieldVStime(equation,og_data,inds, time, field, title = 'dNdt fits, ', No_Eval = False):

    calc = field + '_eval'
    Neq = calc + ' = ' + equation

    if No_Eval:
        for df in og_data:
            df.eval(Neq,inplace = True)

    legend_elements = [Line2D([0],[0],color= 'b',label = 'Inferred'),
                       Line2D([0],[0],color='orange',label = 'Fitted')]

    plt.figure(figsize = (16,10))
    plt.title(title + Neq)

    plt.legend(handles = legend_elements)

    for_plotting = [og_data[i] for i in inds]

    for df in for_plotting:
        plt.plot(time,df[field],c='b',lw=1,alpha=0.7)
        plt.plot(time,df[calc],c='orange',lw=1,alpha=0.7)


def errors_by_space(arr,title):
    plt.imshow(arr, cmap = 'Reds')
    plt.title(title)
    plt.colorbar()
    plt.show()
    
def plotly_error_by_space(arr,title):
    fig = px.imshow(arr,color_continuous_scale='Reds',aspect='equal',title = title)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.show("notebook")

def errors_by_time(errors, X, title, xlabel, ylabel):
    plt.figure(figsize = (16,10))
    plt.title(title)
    plt.plot(X,errors)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

    
def error_plotting(X,curve_dict):
    curves = curve_dict.values()
    curve_labels = list(curve_dict.keys())
    plt.figure(figsize = (16,10))
    for i, curve in enumerate(curves):
        plt.plot(X, curve, label = curve_labels[i])
    plt.legend()
    plt.title("Error Over Time Comparison")
    plt.xlabel("Time")
    plt.ylabel("RMSE")
    plt.show()