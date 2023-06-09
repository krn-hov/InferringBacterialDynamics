import pandas as pd
import numpy as np
import random
from pysr import PySRRegressor
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import convolve
from sympy import simplify, Symbol, series, latex
import pickle

def pickle_save(filename, data):
    with open(filename,'wb') as _save:
        pickle.dump(data,_save)
    
def pickle_load(filename):
    with open(filename,'rb') as _load:
        var = pickle.load(_load)
    return var

np.random.seed(25)

colony_form_data, ring_form_data, plate_maximums = pickle_load("../../Pickles/Data/AC_S_PreprocessedData")

# training indices include horizontal and vertical midsection for total of 79 colonies
goo = list(zip(list(np.ones(48).astype(int)*15),np.arange(48)))
foo = list(zip(np.arange(32),list(np.ones(32).astype(int)*23)))
tr_inds = set(goo+foo)

#testing indices are all other colonies
te_inds = [x for x in np.ndindex((32,48)) if x not in tr_inds]

#use if we want to use midsections
# training_data = pd.DataFrame(columns = colony_form_data[1][0][0].columns)
# for ind in tr_inds:
#     training_data = pd.concat([training_data, colony_form_data[1][ind[0]][ind[1]]])

#use if we want to do 10% of each ring
training_data = pd.DataFrame(columns = colony_form_data[1][0][0].columns)
for r in ring_form_data[1].keys():
    perm = np.random.permutation(len(ring_form_data[1][r]))
    perm = perm[0:len(perm) // 10]
    for ind in perm:
        training_data = pd.concat([training_data, ring_form_data[1][r][ind]])


training_data = training_data.iloc[::6] #every 4 for midsection, every 6 for 10%ring

# te_inds = random.choices(te_inds,k=30)
# testing_data = pd.DataFrame(columns = colony_form_data[1][0][0].columns)
# for ind in te_inds:
#     testing_data = pd.concat([testing_data, colony_form_data[1][ind[0]][ind[1]]])


Nmodel = PySRRegressor(
    equation_file = "ACS1.csv",
    procs=32,
    early_stop_condition = 2e-09,
    timeout_in_seconds = 80000,
    model_selection="best",
    niterations=1e10,
    binary_operators=["+","-", "*", "/"],
    unary_operators=["exp"],
    constraints={"exp":5},
    nested_constraints={"exp":{"exp":0}},
    populations = 96,
    loss = "L2DistLoss()",
    ncyclesperiteration = 5000,
    maxsize = 20,
    maxdepth = 8,
    parsimony = 4e-10,
    weight_optimize = 0.001,
    turbo = True,
    progress = False
    )

# Feature Sets
FS1 = training_data[['Pop']]; FS2 = training_data[['Pop','Ring']]; FS3 = training_data[['Pop','N_init']]
FS4 = training_data[['Pop','Nbar']]; FS5 = training_data[['Pop','Nbar','Ring']]; FS6 = training_data[['Pop','Nbar','N_init']]
FS7 = training_data[['Pop','Ring','N_init']]; FS8 = training_data[['Pop','Ring','N_init','Nbar']]

FS9 = training_data[['Cum_N']]; FS10 = training_data[['Cum_N','Cum_Nbar']]; FS11 = training_data[['Cum_N','N_init']]
FS12 = training_data[['Cum_N','Ring']]; FS13 = training_data[['Cum_N','Cum_Nbar','N_init']]
FS14 = training_data[['Cum_N','Cum_Nbar','Ring']]; FS15 = training_data[['Cum_N','N_init','Ring']] 
FS16 = training_data[['Cum_N','Cum_Nbar','N_init','Ring']]

FS_All = training_data[['Pop','Nbar','Cum_N','Cum_Nbar','N_init','Ring']]

Nfeats = FS1
Nresponse = training_data['dNovN']

Nmodel.fit(Nfeats,Nresponse)

print("\nChosen Equation\n", simplify(Nmodel.sympy()))

print("\nLatex Format\n", latex(simplify(Nmodel.sympy())))