# Brayden Molinyawe, Luca Toth. Both members of the group contributed equally to the completion of this assignment.

from pyexpat import model
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, mean_squared_error, mean_absolute_error)

# A Config Settings
# Here is our global seed
GLOBAL_SEED = 42

#Here are the five seeds we will use for the "mean"+/- std over 5 seeds" requirement
SEEDS = [0, 123, 99, 202, 7] 

# required subsample sizes: 10,000 training images and 5,000 validation images
N_TRAIN, N_VAL = 10000, 5000

# max_depth values to test for Decision Tree
DEPTHS = [2, 4, 6, 8, 10, 12, 15, 20, None]

# Sampling fractions 
FRACTIONS = [0.1, 0.3, 0.5, 0.7, 1.0]

# Directory to save results
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# set our seed to be the global seed
np.random.seed(GLOBAL_SEED)


# B: Loading and Splitting the Datasets

# Loading the FashionMNIST dataset from OpenML,
#and split it into (pool, locked_test). It will return 4 arrays: X_pool, y_pool, X_test, y_test
# "pool" is the official 60,000-image training set. Each seed later draws its 10k train / 5k validation subsample from it.
# "test" is the official 10,000-image test set
def load_FashionMNIST_raw():
    ds = fetch_openml('Fashion-MNIST', version=1, as_frame=False, parser='auto')

    X = ds.data.astype(np.float32)
    y = ds.target.astype(np.int64)

    return X[:60_000], y[:60_000], X[60_000:], y[60_000:]


# Function to create our splits for the fashionMNIST dataset.
# We will return a dictionary with keys: "X_train", "y_train", "X_val", "y_val", "X_test", "y_test"
def make_fashionMNIST_splits(X_pool, y_pool, seed):

    #first, we carve out the training set. 
        X_tr, X_rest, y_tr, y_rest = train_test_split(
        X_pool, y_pool, train_size=N_TRAIN, stratify=y_pool, random_state=seed)

    # then, we draw thev validation  set from the leftover data. This way,
    #validation can't overlap with training data.
        X_va, _, y_va, _ = train_test_split(
        X_rest, y_rest, train_size=N_VAL, stratify=y_rest, random_state=seed)
 
        return dict(Xtr=X_tr, ytr=y_tr, Xva=X_va, yva=y_va)

# Loading in the california Housing dataset from sklearn,
# splitting it into 60/20/20 which correlates to train,validation, and test splits
# returns a a nested dictionary, and the testing split seperatly
def make_housing_splits(seed):

     # define our x and y
     X, y = fetch_california_housing(return_X_y=True)

     # call fetch_california_housing to get the names of the features
     names = fetch_california_housing().feature_names

     # here is our first split, 60% training (40% leftover for validation and testing)
     X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, random_state=seed)

     # Sectond split, 20% validation, 20% testing (using that 40% leftover from the first split)
     X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=seed)
     
     # Return our nested dictionary, and our testing split seperatly
     return dict(Xtr=X_tr, ytr=y_tr, Xva=X_va, yva=y_va), (X_te, y_te), names


# actually loading the datasets
X_pool, y_pool, _FASHION_TEST_X_LOCKED, _FASHION_TEST_Y_LOCKED = load_fashion_raw()
print("pool:", X_pool.shape, "locked test:", _FASHION_TEST_X_LOCKED.shape)

d = make_fashion_splits(X_pool, y_pool, seed=0)
print("train class counts:", np.bincount(d["ytr"]))

#################################################################################################3
# C: Harness

# a function to return the total number of nodes in a given tree
def count_nodes(tree):
     if hasattr(model, "tree_"):
        return model.tree_.node_count
     # if not a tree,
     return np.nan

# a function to run a single model on a single dataset split, and return a dict of metrics
# args:
    # make_model: a function that takes a seed and return a fresh model thats unfitted
    # data: dict with keys "Xtr", "ytr", "Xva", "yva"
    # seed: passed to make_model
    # task:  "clf" for classification metrics, "reg" for regression metrics
def run_once(make_model, data, seed, task='clf'):
     
    model = make_model(seed)


    # Time the fit. 
    t0 = time.perf_counter()
    model.fit(data["Xtr"], data["ytr"])
    fit_s = time.perf_counter() - t0
 
    # time prediction separately.
    t0 = time.perf_counter()
    pred_va = model.predict(data["Xva"])
    pred_ms_per_1k = (time.perf_counter() - t0) * 1000 / (len(data["Xva"]) / 1000)
 
    # training predictions are needed for the train-vs-validation
    # curves that show overfitting.
    pred_tr = model.predict(data["Xtr"])
 
    # Metrics shared by both task types.
    out = dict(seed=seed, fit_s=fit_s, pred_ms_per_1k=pred_ms_per_1k,
               nodes=count_nodes(model))
 
    if task == "clf":
        out.update(
            train_acc=accuracy_score(data["ytr"], pred_tr),
            val_acc=accuracy_score(data["yva"], pred_va),
            # average="macro": compute F1 for each class, then take an unweighted
            # mean, so small classes count as much as large ones.
            val_macro_f1=f1_score(data["yva"], pred_va, average="macro"),
        )
        # Only single trees have get_depth/get_n_leaves. 
        if hasattr(model, "get_depth"):
            out["actual_depth"] = model.get_depth()
            out["leaves"] = model.get_n_leaves()
    else:
        # RMSE = square root of mean squared error. A small lambda keeps it readable.
        rmse = lambda a, b: np.sqrt(mean_squared_error(a, b))
        out.update(
            train_rmse=rmse(data["ytr"], pred_tr),
            val_rmse=rmse(data["yva"], pred_va),
            val_mae=mean_absolute_error(data["yva"], pred_va),
        )
    return out


# A function that will run "run_once" for each seed in our list
# split_fn(seed) will return a dict
# return a data frame with the mean and std
def run_seeds(make_model, split_fn, seeds=SEEDS, task='clf'):
    rows = [run_once(make_model, split_fn(s), s, task) for s in seeds]
    df = pd.DataFrame(rows)
    return df, df.mean(numeric_only=True), df.std(numeric_only=True)


_split_cache = {}

 # create our fashion split function, which will cache the splits for each seed so we don't have to recompute them
def fashion_split(seed):
    if seed not in _split_cache:
        _split_cache[seed] = make_fashion_splits(X_pool, y_pool, seed)
    return _split_cache[seed]

##############################################################################
# D. plot helpers

# function to draw a line plot with error bars, given a dataframe with mean and std values
def line_with_err(ax, x, mean, std, label):
    ax.errorbar(x, mean, yerr=std, marker="o", capsize=3, label=label)


#add labels, grid and legend to a plot, and save it to a file
def finish(fig, ax, title, xlabel, ylabel, filename, legend=True):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    if legend:
        ax.legend()
    fig.tight_layout()
    fig.savefig(filename)


#this function will draw a 2D grid with the number printed in each cell
def heatmap(values, xticks, yticks, title, xlabel, ylabel, fname, fmt=".3f"):
    fig, ax = plt.subplots()
    # origin="lower" puts row 0 at the BOTTOM, so small fractions start bottom-left
    # like a normal graph instead of top-left.
    im = ax.imshow(values, origin="lower", cmap="viridis")
    # Replace the default 0,1,2... tick labels with the actual parameter values.
    ax.set_xticks(range(len(xticks)), xticks)
    ax.set_yticks(range(len(yticks)), yticks)
    # Write each cell's value on top of it, in white for contrast with viridis.
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, format(values[i, j], fmt), ha="center", va="center", color="w")
    fig.colorbar(im, ax=ax)     # color legend showing what the colors mean
    finish(fig, ax, title, xlabel, ylabel, fname, legend=False)

