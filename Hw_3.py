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
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)   # sklearn deprecation chatter
 
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree, export_text
from sklearn.ensemble import (BaggingClassifier, BaggingRegressor,
                              RandomForestClassifier, RandomForestRegressor,
                              ExtraTreesClassifier, ExtraTreesRegressor,
                              AdaBoostClassifier, AdaBoostRegressor,
                              GradientBoostingRegressor, HistGradientBoostingClassifier,
                              VotingClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.datasets import make_classification


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

# if True, tiny settings for debugging the code path
QUICK = False            
# use all cpu cores wherever supported
N_JOBS = -1             
if QUICK:
    SEEDS = [0, 1]
HEAVY_SEEDS = SEEDS[:3]  # seeds for the expensive sweeps (Parts 2, 3a, 4). Use all 5 if time allows
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# number of pixel features
P = 784                  
 
def rs(make_model, split_fn=None, task="clf", seeds=None):
    """Shortcut for run_seeds with our defaults (Fashion split, SEEDS)."""
    return run_seeds(make_model, split_fn or fashion_split, seeds=seeds or SEEDS, task=task)
 
def save_table(df, name):
    df.to_csv(f"{OUT_DIR}/{name}.csv", index=False)
    return df
 
hsplit = lambda s: make_housing_splits(s)[0]   # housing split fn (train/val dict only)

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
def count_nodes(model):
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


# Part 1 ###################################
# 1A: Impurity Plots and hand computation 

def gini(p):
    p = np.asarray(p, float); return 1.0 - np.sum(p ** 2)
 
def entropy(p):
    p = np.asarray(p, float); p = p[p > 0]; return -np.sum(p * np.log2(p))

#A function to compute the decrease in impurity from a parent node to its children nodes, 
# #given a function f (either gini or entropy)

def impurity_decrease(parent, children, f):
    parent = np.asarray(parent, float); n = parent.sum()
    weighted = sum(np.sum(c) / n * f(np.asarray(c, float) / np.sum(c)) for c in children)
    return f(parent / n) - weighted
 
parent, kids = [25, 15], [[20, 2], [5, 13]]
print("Gini decrease:", impurity_decrease(parent, kids, gini))
print("Information gain (bits):", impurity_decrease(parent, kids, entropy))
 
p = np.linspace(0, 1, 401)
pc = np.clip(p, 1e-12, 1 - 1e-12)
# two-class Gini
G = 2 * p * (1 - p)
# entropy / 2
H_half = 0.5 * -(pc * np.log2(pc) + (1 - pc) * np.log2(1 - pc)) 
# misclassifcation error
ERR = 1 - np.maximum(p, 1 - p)                         
# plot the three measures on a graph
fig, ax = plt.subplots()
ax.plot(p, G, label="Gini G"); ax.plot(p, H_half, label="Entropy H/2")
ax.plot(p, ERR, label="Misclassification error")
finish(fig, ax, "Impurity measures for two classes", "p (proportion of class 1)",
       "Impurity", "p1a_impurity")

# 1B:  criterion comparison
keys = ["train_acc", "val_acc", "fit_s", "leaves", "actual_depth"]
rows = []
for crit in ["gini", "entropy"]:
    for d in DEPTHS:
        mk = lambda s, c=crit, dd=d: DecisionTreeClassifier(criterion=c, max_depth=dd, random_state=s)
        _, m, sd = rs(mk)
        rows.append({"criterion": crit, "max_depth": "None" if d is None else d,
                     **{k + "_mean": m[k] for k in keys}, **{k + "_std": sd[k] for k in keys}})
tab1b = save_table(pd.DataFrame(rows), "part1b_criterion_depth")
 
x = np.arange(len(DEPTHS)); labels = ["None" if d is None else str(d) for d in DEPTHS]
fig, ax = plt.subplots(figsize=(8, 5))
for crit, color in [("gini", "C0"), ("entropy", "C1")]:
    t = tab1b[tab1b.criterion == crit]
    ax.errorbar(x, t.train_acc_mean, yerr=t.train_acc_std, ls="--", marker="o", capsize=3,
                color=color, label=f"{crit} train")
    ax.errorbar(x, t.val_acc_mean, yerr=t.val_acc_std, ls="-", marker="s", capsize=3,
                color=color, label=f"{crit} validation")
ax.set_xticks(x, labels)
finish(fig, ax, "Train/validation accuracy vs. max_depth (mean +/- std, 5 seeds)",
       "max_depth", "Accuracy", "p1b_depth_criterion")
 
# Best depth per criterion (rows are in DEPTHS order, so index into DEPTHS).
best_depth = {}
for crit in ["gini", "entropy"]:
    vals = tab1b[tab1b.criterion == crit]["val_acc_mean"].to_numpy()
    best_depth[crit] = DEPTHS[int(vals.argmax())]
print("best depth:", best_depth)
 
# This function will return the feature index used at each node, 
# #level by level (None where there is no split).
def feats_by_level(clf, levels=3):
    """Feature index used at each node, level by level (None where there is no split)."""
    t = clf.tree_; level = [0]; out = []
    for _ in range(levels):
        out.append([t.feature[n] if (n is not None and t.children_left[n] != -1) else None
                    for n in level])
        nxt = []
        for n in level:
            if n is not None and t.children_left[n] != -1:
                nxt += [t.children_left[n], t.children_right[n]]
            else:
                nxt += [None, None]
        level = nxt
    return out
 
agree = {lvl: [] for lvl in range(3)}
for s in SEEDS:
    dat = fashion_split(s)
    f = {c: feats_by_level(DecisionTreeClassifier(criterion=c, max_depth=3, random_state=s)
                           .fit(dat["Xtr"], dat["ytr"])) for c in ["gini", "entropy"]}
    for lvl in range(3):
        pairs = [(a, b) for a, b in zip(f["gini"][lvl], f["entropy"][lvl])
                 if a is not None and b is not None]
        agree[lvl].append(np.mean([a == b for a, b in pairs]) if pairs else np.nan)
for lvl in range(3):
    print(f"level {lvl} same-feature rate: {np.nanmean(agree[lvl]):.2f}")
 

# 1c: pruning
dat0 = fashion_split(0)
path = DecisionTreeClassifier(random_state=0).cost_complexity_pruning_path(dat0["Xtr"], dat0["ytr"])
alphas = np.unique(path.ccp_alphas[:-1])                    # last alpha prunes to the root
alpha_grid = alphas[np.linspace(0, len(alphas) - 1, 25).astype(int)]   # ~25 candidates
 
path_rows = []
for a in alpha_grid:
    t = DecisionTreeClassifier(ccp_alpha=a, random_state=0).fit(dat0["Xtr"], dat0["ytr"])
    path_rows.append(dict(alpha=a, leaves=t.get_n_leaves(),
                          val_acc=t.score(dat0["Xva"], dat0["yva"])))
pr = pd.DataFrame(path_rows); pr = pr[pr.alpha > 0]         # log axis can't show alpha=0
fig, ax = plt.subplots()
ax.plot(pr.alpha, pr.val_acc, "o-", label="validation accuracy"); ax.set_xscale("log")
ax2 = ax.twinx(); ax2.plot(pr.alpha, pr.leaves, "s--", color="C1", label="leaves")
ax2.set_yscale("log"); ax2.set_ylabel("Number of leaves")
finish(fig, ax, "Cost-complexity pruning path (seed 0)", "ccp_alpha (log)",
       "Validation accuracy", "p1c_pruning_path", legend=False)
 
# Choose alpha by 5-fold CV on the TRAINING data only.
gs = GridSearchCV(DecisionTreeClassifier(random_state=0), {"ccp_alpha": list(alpha_grid)},
                  cv=5, n_jobs=N_JOBS).fit(dat0["Xtr"], dat0["ytr"])
best_alpha = gs.best_params_["ccp_alpha"]; print("CV-chosen alpha:", best_alpha)
 
_, m_p, sd_p = rs(lambda s: DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=s))
_, m_d, sd_d = rs(lambda s: DecisionTreeClassifier(max_depth=best_depth["gini"], random_state=s))
print(f"pruned tree      val acc {m_p['val_acc']:.4f} +/- {sd_p['val_acc']:.4f}, leaves {m_p['leaves']:.0f}")
print(f"best-depth tree  val acc {m_d['val_acc']:.4f} +/- {sd_d['val_acc']:.4f}, leaves {m_d['leaves']:.0f}")
 
msl_rows = []
for msl in [1, 5, 20, 50]:
    _, m, sd = rs(lambda s, k=msl: DecisionTreeClassifier(min_samples_leaf=k, random_state=s))
    msl_rows.append(dict(min_samples_leaf=msl, val_acc=m["val_acc"], val_std=sd["val_acc"],
                         train_acc=m["train_acc"], leaves=m["leaves"]))
print(save_table(pd.DataFrame(msl_rows), "part1c_min_samples_leaf"))
 
# 1D: Regression Trees
REG_DEPTHS = [2, 3, 4, 5, 6, 8, 10, 12, None]
rrows = []
for crit in ["squared_error", "absolute_error", "friedman_mse"]:
    for d in REG_DEPTHS:
        mk = lambda s, c=crit, dd=d: DecisionTreeRegressor(criterion=c, max_depth=dd, random_state=s)
        _, m, sd = rs(mk, split_fn=hsplit, task="reg")
        rrows.append(dict(criterion=crit, depth_label="None" if d is None else d,
                          val_rmse=m["val_rmse"], rmse_std=sd["val_rmse"],
                          val_mae=m["val_mae"], mae_std=sd["val_mae"]))
tab1d = save_table(pd.DataFrame(rrows), "part1d_regression_trees")
 
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5)); xr = np.arange(len(REG_DEPTHS))
for crit in tab1d.criterion.unique():
    t = tab1d[tab1d.criterion == crit]
    axes[0].errorbar(xr, t.val_rmse, yerr=t.rmse_std, marker="o", capsize=3, label=crit)
    axes[1].errorbar(xr, t.val_mae, yerr=t.mae_std, marker="o", capsize=3, label=crit)
for ax, (nm, yl) in zip(axes, [("RMSE", "Validation RMSE"), ("MAE", "Validation MAE")]):
    ax.set_xticks(xr, [str(d) for d in REG_DEPTHS]); ax.set_title(f"{nm} vs. depth")
    ax.set_xlabel("max_depth"); ax.set_ylabel(yl + " ($100k)"); ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/p1d_rmse_mae_depth.png", dpi=150)
 
# Best (criterion, depth) by validation RMSE -> visualize the top three levels.
bi = tab1d.val_rmse.idxmin()
best_reg = (tab1d.loc[bi, "criterion"], REG_DEPTHS[bi % len(REG_DEPTHS)])   # rows are criterion-major
print("best regression tree:", best_reg)
d0, _hou_test, hnames = make_housing_splits(0)              # test set deliberately unused here
bt = DecisionTreeRegressor(criterion=best_reg[0], max_depth=best_reg[1], random_state=0)\
        .fit(d0["Xtr"], d0["ytr"])
fig, ax = plt.subplots(figsize=(16, 7))
plot_tree(bt, max_depth=3, feature_names=hnames, filled=True, fontsize=7, ax=ax)
ax.set_title("Top three levels of the best regression tree"); fig.savefig(f"{OUT_DIR}/p1d_tree.png", dpi=150)
print(export_text(bt, feature_names=hnames, max_depth=2))   # read the first splits in plain words
 
# Step-function demo: a tree using ONLY median income predicts a staircase.
inc = d0["Xtr"][:, [0]]                                     # column 0 = MedInc
st = DecisionTreeRegressor(max_depth=4, random_state=0).fit(inc, d0["ytr"])
grid = np.linspace(inc.min(), inc.max(), 600).reshape(-1, 1)
fig, ax = plt.subplots()
ax.scatter(d0["Xva"][:, 0], d0["yva"], s=4, alpha=0.25, label="actual (validation)")
ax.plot(grid, st.predict(grid), "r-", lw=2, label="tree prediction (depth 4)")
finish(fig, ax, "Regression tree prediction is a step function", "Median income ($10k)",
       "House value ($100k)", "p1d_staircase")


##################################
# Part 2: Sampling-based ensembles
def make_sampler(method, row_frac=1.0, col_frac=1.0, base_depth=None, n_trees=100,
                 seed=0, oob=False):
    """All four methods = BaggingClassifier with different sampling arguments.
    Note: pasting/subspaces at fraction 1.0 give 100 (nearly) identical trees: nothing varies."""
    cfg = {
        "pasting":   dict(bootstrap=False, max_samples=row_frac, max_features=1.0),
        "bagging":   dict(bootstrap=True,  max_samples=row_frac, max_features=1.0),
        "subspaces": dict(bootstrap=False, max_samples=1.0,      max_features=col_frac),
        "patches":   dict(bootstrap=False, max_samples=row_frac, max_features=col_frac),
    }[method]
    return BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=base_depth),
                             n_estimators=n_trees, oob_score=(oob and cfg["bootstrap"]),
                             n_jobs=N_JOBS, random_state=seed, **cfg)

def sampling_sweep(base_depth, seeds, tag):
    """2a (and 2d with base_depth=4). Returns ({method: (means, stds)}, patches_grid)."""
    out = {}
    for method, axis in [("pasting", "rows"), ("bagging", "rows"), ("subspaces", "cols")]:
        means, stds = [], []
        for f in FRACTIONS:
            kw = {"row_frac": f} if axis == "rows" else {"col_frac": f}
            _, m, sd = rs(lambda s, kw=kw, me=method: make_sampler(me, base_depth=base_depth, seed=s, **kw),
                          seeds=seeds)
            means.append(m["val_acc"]); stds.append(sd["val_acc"])
        out[method] = (np.array(means), np.array(stds))
    grid = np.zeros((len(FRACTIONS), len(FRACTIONS)))       # rows: row_frac, cols: col_frac
    for i, rf in enumerate(FRACTIONS):
        for j, cf in enumerate(FRACTIONS):
            _, m, _ = rs(lambda s, rf=rf, cf=cf: make_sampler("patches", rf, cf, base_depth, seed=s),
                         seeds=seeds)
            grid[i, j] = m["val_acc"]
    _, m1, _ = rs(lambda s: DecisionTreeClassifier(max_depth=base_depth, random_state=s), seeds=seeds)

    fig, ax = plt.subplots()
    for method, (mu, sd) in out.items():
        ax.errorbar(FRACTIONS, mu, yerr=np.nan_to_num(sd), marker="o", capsize=3,
                    label=f"{method} ({'rows' if method != 'subspaces' else 'columns'} sampled)")
    ax.axhline(m1["val_acc"], color="gray", ls=":", label="single tree")
    depth_txt = "fully grown" if base_depth is None else f"depth {base_depth}"
    finish(fig, ax, f"Sampling ensembles, 100 {depth_txt} trees", "Sampled fraction",
           "Validation accuracy", f"p2_sampling_{tag}")
    heatmap(grid, FRACTIONS, FRACTIONS, f"Random patches, 100 {depth_txt} trees",
            "Column fraction", "Row fraction", f"p2_patches_heatmap_{tag}")
    pd.DataFrame({k: v[0] for k, v in out.items()}, index=FRACTIONS).to_csv(f"{OUT_DIR}/p2_sampling_{tag}.csv")
    return out, grid

# 2a: fully grown trees
sweep_full, grid_full = sampling_sweep(None, HEAVY_SEEDS, "full")
# Best settings, reused in Parts 5/6.
best_frac = {m: FRACTIONS[int(np.argmax(v[0]))] for m, v in sweep_full.items()}
bi, bj = np.unravel_index(grid_full.argmax(), grid_full.shape)
best_patch = (FRACTIONS[bi], FRACTIONS[bj]); print(best_frac, best_patch)

# 2b: OOB vs validation accuracy as trees are added (bagging, 1..300 trees)
def bagging_curves(seed, n_trees=300):
    dat = fashion_split(seed)
    bag = make_sampler("bagging", n_trees=n_trees, seed=seed).fit(dat["Xtr"], dat["ytr"])
    classes = bag.classes_; n = len(dat["ytr"])
    p_va = np.zeros((len(dat["yva"]), len(classes))); p_oob = np.zeros((n, len(classes)))
    cnt = np.zeros(n); val_c, oob_c = [], []
    for est, idx in zip(bag.estimators_, bag.estimators_samples_):
        p_va += est.predict_proba(dat["Xva"])                # running average over first k trees
        val_c.append(np.mean(classes[p_va.argmax(1)] == dat["yva"]))
        oob = np.ones(n, bool); oob[idx] = False             # rows this tree never saw
        p_oob[oob] += est.predict_proba(dat["Xtr"][oob]); cnt[oob] += 1
        seen = cnt > 0                                       # rows with >= 1 out-of-bag tree
        oob_c.append(np.mean(classes[p_oob[seen].argmax(1)] == dat["ytr"][seen]))
    return np.array(val_c), np.array(oob_c)

curves = [bagging_curves(s) for s in HEAVY_SEEDS]
val_m = np.mean([c[0] for c in curves], 0); oob_m = np.mean([c[1] for c in curves], 0)
fig, ax = plt.subplots(); k = np.arange(1, len(val_m) + 1)
ax.plot(k, val_m, label="validation accuracy"); ax.plot(k, oob_m, label="OOB accuracy")
finish(fig, ax, "Bagging: OOB estimate vs. validation accuracy", "Number of trees",
       "Accuracy", "p2b_oob_vs_val")
print("final gap (OOB - val):", oob_m[-1] - val_m[-1])

# Why pasting has no OOB estimate: sklearn refuses, because without a bootstrap there
# is no principled "left-out" set tied to the sampling scheme. (Think about why, then write it up.)
try:
    BaggingClassifier(DecisionTreeClassifier(), n_estimators=5, bootstrap=False,
                      oob_score=True).fit(dat0["Xtr"][:500], dat0["ytr"][:500])
except ValueError as e:
    print("Pasting + oob_score ->", e)

# 2c: bias-variance by measurement (10 bootstrap resamples of the training set)
N_RESAMPLES, BV_TREES = 10, 25      # ensembles use 25 trees here (not 100) to keep it feasible

def preds_for_resamples(make_model, dat, seed=0):
    rng = np.random.RandomState(seed); P_ = []
    for r in range(N_RESAMPLES):
        idx = rng.randint(0, len(dat["ytr"]), len(dat["ytr"]))      # one outer bootstrap resample
        P_.append(make_model(seed + r).fit(dat["Xtr"][idx], dat["ytr"][idx]).predict(dat["Xva"]))
    return np.array(P_)                                             # (10 models, n_val)

def bv_summary(P_, y):
    counts = np.stack([(P_ == k).sum(0) for k in range(10)])        # votes per class per point
    vote = counts.argmax(0); top = counts.max(0)
    return dict(mean_single_model_err=(P_ != y).mean(),             # average individual error
                majority_vote_err=(vote != y).mean(),               # error of the 10-model vote
                frac_points_any_disagreement=(top < N_RESAMPLES).mean(),   # variance proxy
                avg_minority_share=(1 - top / N_RESAMPLES).mean())         # variance proxy

bv_models = {
    "single tree (unpruned)": lambda s: DecisionTreeClassifier(random_state=s),
    "single tree (depth 4)":  lambda s: DecisionTreeClassifier(max_depth=4, random_state=s),
    "pasting":   lambda s: make_sampler("pasting", row_frac=best_frac["pasting"], n_trees=BV_TREES, seed=s),
    "bagging":   lambda s: make_sampler("bagging", row_frac=best_frac["bagging"], n_trees=BV_TREES, seed=s),
    "subspaces": lambda s: make_sampler("subspaces", col_frac=best_frac["subspaces"], n_trees=BV_TREES, seed=s),
    "patches":   lambda s: make_sampler("patches", best_patch[0], best_patch[1], n_trees=BV_TREES, seed=s),
}
dat0 = fashion_split(0)     # measured on the VALIDATION set (safer than the test set)
bv_tab = pd.DataFrame({name: bv_summary(preds_for_resamples(mk, dat0), dat0["yva"])
                       for name, mk in bv_models.items()}).T
print(save_table(bv_tab.reset_index().rename(columns={"index": "model"}), "part2c_bias_variance"))
# Expected story: disagreement collapses for the ensembles, majority-vote error falls less
# (the remaining errors are shared by all models = systematic / bias).

# 2d: repeat 2a with depth-4 base trees
sweep_d4, grid_d4 = sampling_sweep(4, HEAVY_SEEDS, "depth4")

###################################
# Part 3: random forests and extra trees

mf_list = [1, int(np.sqrt(P)), int(np.log2(P)), int(0.1 * P), int(0.3 * P), P]
mf_names = ["1", "sqrt(p)", "log2(p)", "0.1p", "0.3p", "p (=bagging)"]
rows3 = []
for crit in ["gini", "entropy"]:
    for mf, nm in zip(mf_list, mf_names):
        mk = lambda s, c=crit, k=mf: RandomForestClassifier(n_estimators=300, max_features=k,
                                                            criterion=c, n_jobs=N_JOBS, random_state=s)
        _, m, sd = rs(mk, seeds=HEAVY_SEEDS)
        rows3.append(dict(criterion=crit, max_features=nm, mf_value=mf, val_acc=m["val_acc"],
                          val_std=sd["val_acc"], fit_s=m["fit_s"], fit_std=sd["fit_s"]))
tab3a = save_table(pd.DataFrame(rows3), "part3a_max_features")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for crit in ["gini", "entropy"]:
    t = tab3a[tab3a.criterion == crit]
    axes[0].errorbar(t.mf_value, t.val_acc, yerr=np.nan_to_num(t.val_std), marker="o", capsize=3, label=crit)
    axes[1].errorbar(t.mf_value, t.fit_s, yerr=np.nan_to_num(t.fit_std), marker="o", capsize=3, label=crit)
for ax, yl, ttl in zip(axes, ["Validation accuracy", "Training time (s)"],
                       ["Accuracy vs. max_features", "Training time vs. max_features"]):
    ax.set_xscale("log"); ax.set_xlabel("max_features (log scale)"); ax.set_ylabel(yl)
    ax.set_title(ttl); ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/p3a_max_features.png", dpi=150)
bt3 = tab3a.loc[tab3a.val_acc.idxmax()]
best_rf = dict(max_features=int(bt3.mf_value), criterion=bt3.criterion); print("best RF:", best_rf)

# 3b: random forest vs extra trees
def make_et(s, mf=int(np.sqrt(P))):
    return ExtraTreesClassifier(n_estimators=300, max_features=mf, n_jobs=N_JOBS, random_state=s)
cmp = []
for name, mk in [("Random forest", lambda s: RandomForestClassifier(n_estimators=300, max_features=int(np.sqrt(P)),
                                                                    n_jobs=N_JOBS, random_state=s)),
                 ("Extra trees", make_et)]:
    _, m, sd = rs(mk)                                       # all 5 seeds
    cmp.append(dict(method=name, val_acc=m["val_acc"], val_std=sd["val_acc"],
                    fit_s=m["fit_s"], fit_std=sd["fit_s"]))
print(save_table(pd.DataFrame(cmp), "part3b_rf_vs_et"))

# 3c: feature importance (impurity vs permutation)
def show_map(arr, title, fname):
    fig, ax = plt.subplots(); im = ax.imshow(arr.reshape(28, 28), cmap="hot")
    fig.colorbar(im, ax=ax, label="importance")
    finish(fig, ax, title, "pixel column", "pixel row", fname, legend=False)

rf = RandomForestClassifier(n_estimators=300, max_features=best_rf["max_features"],
                            criterion=best_rf["criterion"], n_jobs=N_JOBS, random_state=0)\
        .fit(dat0["Xtr"], dat0["ytr"])
show_map(rf.feature_importances_, "Impurity-based importance (Fashion-MNIST RF)", "p3c_impurity_map")
sub = slice(0, 2000)         # permutation importance is slow: 784 features x repeats x predict()
perm = permutation_importance(rf, dat0["Xva"][sub], dat0["yva"][sub], n_repeats=3,
                              random_state=0, n_jobs=N_JOBS)
show_map(perm.importances_mean, "Permutation importance on validation set", "p3c_permutation_map")

# California Housing: both importances side by side.
dh, _hou_test, hnames = make_housing_splits(0)
rfh = RandomForestRegressor(n_estimators=300, max_features=0.5, n_jobs=N_JOBS, random_state=0)\
          .fit(dh["Xtr"], dh["ytr"])
ph = permutation_importance(rfh, dh["Xva"], dh["yva"], n_repeats=10, random_state=0, n_jobs=N_JOBS)
xb = np.arange(len(hnames)); fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(xb - 0.2, rfh.feature_importances_, 0.4, label="impurity")
ax.bar(xb + 0.2, ph.importances_mean, 0.4, yerr=ph.importances_std, label="permutation (val)")
ax.set_xticks(xb, hnames, rotation=30)
finish(fig, ax, "California Housing: impurity vs. permutation importance", "Feature", "Importance", "p3c_housing_importance")

########### 
# Part 4: Boosting

# 4a: AdaBoost
ADA_ROUNDS = 100 if QUICK else 500
ADA_DEPTHS = [1, 2, 4, 8]; ADA_LRS = [0.1, 0.5, 1.0]

def make_ada(depth, lr, n_rounds, seed):
    kw = dict(estimator=DecisionTreeClassifier(max_depth=depth), n_estimators=n_rounds,
              learning_rate=lr, random_state=seed)
    try:                       # older sklearn defaults to SAMME.R; request SAMME explicitly
        return AdaBoostClassifier(algorithm="SAMME", **kw)
    except TypeError:          # newer sklearn removed the argument (SAMME is the only option)
        return AdaBoostClassifier(**kw)

ada_curves = {}
for depth in ADA_DEPTHS:
    for lr in ADA_LRS:
        mdl = make_ada(depth, lr, ADA_ROUNDS, 0).fit(dat0["Xtr"], dat0["ytr"])
        tr = [np.mean(p != dat0["ytr"]) for p in mdl.staged_predict(dat0["Xtr"])]
        va = [np.mean(p != dat0["yva"]) for p in mdl.staged_predict(dat0["Xva"])]
        ada_curves[(depth, lr)] = (np.array(tr), np.array(va))     # may stop early if train err hits 0
        print(f"depth {depth} lr {lr}: final train err {tr[-1]:.3f}, best val err {min(va):.3f} @ round {int(np.argmin(va)) + 1}")

fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
for ax, depth in zip(axes.ravel(), ADA_DEPTHS):
    for i, lr in enumerate(ADA_LRS):
        tr, va = ada_curves[(depth, lr)]
        ax.plot(np.arange(1, len(tr) + 1), tr, "--", color=f"C{i}", label=f"train lr={lr}")
        ax.plot(np.arange(1, len(va) + 1), va, "-", color=f"C{i}", label=f"val lr={lr}")
    ax.set_title(f"Base tree depth {depth}"); ax.set_xlabel("Boosting round")
    ax.set_ylabel("Error rate"); ax.grid(alpha=0.3)
axes[0, 0].legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{OUT_DIR}/p4a_adaboost_staged.png", dpi=150)

(best_ada_depth, best_ada_lr), (_, va_best) = min(ada_curves.items(), key=lambda kv: kv[1][1].min())
best_ada_rounds = int(np.argmin(ada_curves[(best_ada_depth, best_ada_lr)][1])) + 1
best_ada = dict(depth=best_ada_depth, lr=best_ada_lr, rounds=best_ada_rounds); print("best AdaBoost:", best_ada)

# 4b: Sample weights

ada50 = make_ada(best_ada["depth"], best_ada["lr"], 50, 0).fit(dat0["Xtr"], dat0["ytr"])
# sklearn does not keep the final sample weights, so rebuild them with the SAMME update
# (estimator_weights_ already includes the learning rate): w <- w * exp(alpha * [wrong]), renormalize.
w = np.ones(len(dat0["ytr"])) / len(dat0["ytr"]); max_diff = 0.0
for est, alpha, err in zip(ada50.estimators_, ada50.estimator_weights_, ada50.estimator_errors_):
    wrong = est.predict(dat0["Xtr"]) != dat0["ytr"]
    max_diff = max(max_diff, abs(np.sum(w * wrong) / w.sum() - err))   # sanity check vs sklearn's own error
    w = w * np.exp(alpha * wrong); w /= w.sum()
print("reconstruction check (should be ~0):", max_diff)

top = np.argsort(w)[::-1][:20]; fin_pred = ada50.predict(dat0["Xtr"][top])
fig, axes = plt.subplots(4, 5, figsize=(10, 9))
for ax, i, pr_ in zip(axes.ravel(), top, fin_pred):
    ax.imshow(dat0["Xtr"][i].reshape(28, 28), cmap="gray"); ax.axis("off")
    ax.set_title(f"y={CLASS_NAMES[dat0['ytr'][i]]}\npred={CLASS_NAMES[pr_]}", fontsize=7)
fig.suptitle("20 highest-weight training images after 50 AdaBoost rounds")
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/p4b_top_weights.png", dpi=150)

def flip_labels(y, frac, seed):
    rng = np.random.RandomState(seed); y2 = y.copy()
    idx = rng.choice(len(y), int(frac * len(y)), replace=False)
    y2[idx] = (y[idx] + rng.randint(1, 10, len(idx))) % 10      # always a DIFFERENT class
    return y2

noisy_split = lambda s: {**fashion_split(s), "ytr": flip_labels(fashion_split(s)["ytr"], 0.10, s)}
noise_rows = []
for name, mk in [("AdaBoost", lambda s: make_ada(best_ada["depth"], best_ada["lr"], best_ada["rounds"], s)),
                 ("Random forest", lambda s: RandomForestClassifier(n_estimators=300, max_features=best_rf["max_features"],
                                                                    n_jobs=N_JOBS, random_state=s))]:
    _, mc, _ = rs(mk, seeds=HEAVY_SEEDS)
    _, mn, _ = rs(mk, split_fn=noisy_split, seeds=HEAVY_SEEDS)   # val labels stay clean
    noise_rows.append(dict(method=name, clean_val_acc=mc["val_acc"], noisy_val_acc=mn["val_acc"],
                           drop=mc["val_acc"] - mn["val_acc"]))

# 4c: Gradient boosting

hgb_rows, hgb_models = [], {}
for lr in [0.03, 0.1, 0.3]:
    for depth in [3, 6, None]:
        t0 = time.perf_counter()
        # early_stopping=True holds out 10% of the TRAINING data internally; the test set is never touched.
        m = HistGradientBoostingClassifier(learning_rate=lr, max_depth=depth, max_iter=500,
                                           early_stopping=True, validation_fraction=0.1,
                                           n_iter_no_change=10, random_state=0)\
                .fit(dat0["Xtr"], dat0["ytr"])
        hgb_models[(lr, depth)] = m
        hgb_rows.append(dict(lr=lr, depth=str(depth), rounds_used=m.n_iter_,
                             val_acc=m.score(dat0["Xva"], dat0["yva"]), fit_s=time.perf_counter() - t0))
tab4c = save_table(pd.DataFrame(hgb_rows), "part4c_hgb_grid"); print(tab4c)
bh = tab4c.loc[tab4c.val_acc.idxmax()]
best_hgb = dict(lr=float(bh.lr), depth=None if bh.depth == "None" else int(bh.depth)); print("best HGB:", best_hgb)

fig, ax = plt.subplots()      # learning-rate vs. rounds trade-off at the best depth
for lr in [0.03, 0.1, 0.3]:
    m = hgb_models[(lr, best_hgb["depth"])]
    ax.plot(-np.asarray(m.validation_score_), label=f"lr={lr} (stopped at {m.n_iter_})")
finish(fig, ax, "HistGradientBoosting: learning rate vs. rounds (internal validation loss)",
       "Boosting round", "Validation log-loss", "p4c_lr_vs_rounds")
###################################################
gb_rows = []
for lr in [0.02, 0.05, 0.1, 0.2]:
    for depth in [2, 3, 4, 6]:
        g = GradientBoostingRegressor(n_estimators=1000, learning_rate=lr, max_depth=depth,
                                      n_iter_no_change=20, validation_fraction=0.1, random_state=0)\
                .fit(dh["Xtr"], dh["ytr"])
        gb_rows.append(dict(lr=lr, depth=depth, rounds_used=g.n_estimators_,
                            val_rmse=np.sqrt(np.mean((g.predict(dh["Xva"]) - dh["yva"]) ** 2))))
tab4c_r = save_table(pd.DataFrame(gb_rows), "part4c_gbr_grid"); print(tab4c_r)
bg = tab4c_r.loc[tab4c_r.val_rmse.idxmin()]
best_gbr = dict(lr=float(bg.lr), depth=int(bg.depth)); print("best GBR:", best_gbr)

fig, ax = plt.subplots()      # classic trade-off plot: val RMSE vs. rounds for several learning rates
for lr in [0.02, 0.05, 0.1, 0.2]:
    g = GradientBoostingRegressor(n_estimators=600, learning_rate=lr, max_depth=best_gbr["depth"],
                                  random_state=0).fit(dh["Xtr"], dh["ytr"])
    ax.plot([np.sqrt(np.mean((p - dh["yva"]) ** 2)) for p in g.staged_predict(dh["Xva"])], label=f"lr={lr}")
finish(fig, ax, "Gradient boosting regression: learning rate vs. rounds", "Boosting round",
       "Validation RMSE ($100k)", "p4c_gbr_lr_rounds")

reg_cmp = []
for name, mk in [("Gradient boosting", lambda s: GradientBoostingRegressor(
                      n_estimators=1000, learning_rate=best_gbr["lr"], max_depth=best_gbr["depth"],
                      n_iter_no_change=20, validation_fraction=0.1, random_state=s)),
                 ("AdaBoost.R2 (depth 3)", lambda s: AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=3),
                                                                      n_estimators=200, random_state=s)),
                 ("AdaBoost.R2 (depth 6)", lambda s: AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=6),
                                                                      n_estimators=200, random_state=s))]:
    _, m, sd = rs(mk, split_fn=hsplit, task="reg")
    reg_cmp.append(dict(method=name, val_rmse=m["val_rmse"], rmse_std=sd["val_rmse"],
                        val_mae=m["val_mae"], mae_std=sd["val_mae"]))
print(save_table(pd.DataFrame(reg_cmp), "part4c_regression_comparison"))

#######################################
# Part 5: Combining classifiers (stacking)
def best_members(s):
    return [("tree", DecisionTreeClassifier(criterion="gini", max_depth=best_depth["gini"], random_state=s)),
            ("rf", RandomForestClassifier(n_estimators=300, max_features=best_rf["max_features"],
                                          criterion=best_rf["criterion"], n_jobs=N_JOBS, random_state=s)),
            ("ada", make_ada(best_ada["depth"], best_ada["lr"], best_ada["rounds"], s)),
            ("hgb", HistGradientBoostingClassifier(learning_rate=best_hgb["lr"], max_depth=best_hgb["depth"],
                                                   max_iter=500, early_stopping=True, random_state=s))]

probas, preds = {}, {}
for name, m in best_members(0):
    m.fit(dat0["Xtr"], dat0["ytr"]); probas[name] = m.predict_proba(dat0["Xva"])
    preds[name] = probas[name].argmax(1)                    # classes are 0..9, so argmax = label
yv = dat0["yva"]
acc = lambda p: np.mean(p == yv)

votes = np.stack([(np.stack(list(preds.values())) == k).sum(0) for k in range(10)])
hard = votes.argmax(0)                                      # ties go to the lower class index
soft = np.mean(list(probas.values()), axis=0).argmax(1)
# cv=3 (instead of 5) keeps stacking affordable; the meta-learner sees only out-of-fold predictions.
stack = StackingClassifier(best_members(0), final_estimator=LogisticRegression(max_iter=2000),
                           cv=3, stack_method="predict_proba").fit(dat0["Xtr"], dat0["ytr"])
res5 = {**{k: acc(v) for k, v in preds.items()}, "hard voting": acc(hard),
        "soft voting": acc(soft), "stacking": acc(stack.predict(dat0["Xva"]))}
print(save_table(pd.DataFrame(res5.items(), columns=["model", "val_acc"]), "part5_combining"))

E = np.stack([preds[k] != yv for k in preds]).astype(float)          # 1 = member is wrong
corr = pd.DataFrame(np.corrcoef(E), index=list(preds), columns=list(preds)); print(corr.round(2))
print("all members wrong on", f"{E.all(0).mean():.3f}", "of points; at least one right on", f"{1 - E.all(0).mean():.3f}")
heatmap(corr.to_numpy(), list(preds), list(preds), "Correlation of member errors (validation)",
        "Model", "Model", "p5_error_corr", fmt=".2f")

#################################
# Part 6: Final comparision and report
CONFIRM_FINAL = False   # flip to True only when ALL tuning is finished. The test set is touched once.

def final_run(make_model, Xtr_, ytr_, Xte_, yte_, seed, task):
    mdl = make_model(seed)
    t0 = time.perf_counter(); mdl.fit(Xtr_, ytr_); fit_s = time.perf_counter() - t0
    t0 = time.perf_counter(); pred = mdl.predict(Xte_)
    pms = (time.perf_counter() - t0) * 1000 / (len(Xte_) / 1000)
    out = dict(seed=seed, train_s=fit_s, predict_ms_per_1k=pms, nodes=count_nodes(mdl))
    if task == "clf":
        out.update(test_acc=100 * accuracy_score(yte_, pred), macro_f1=f1_score(yte_, pred, average="macro"))
    else:
        out.update(rmse=np.sqrt(mean_squared_error(yte_, pred)), mae=mean_absolute_error(yte_, pred))
    return out, pred

FINAL_CLF = {
    "Tree (Gini)":   lambda s: DecisionTreeClassifier(criterion="gini", max_depth=best_depth["gini"], random_state=s),
    "Tree (entropy)": lambda s: DecisionTreeClassifier(criterion="entropy", max_depth=best_depth["entropy"], random_state=s),
    "Pasting":       lambda s: make_sampler("pasting", row_frac=best_frac["pasting"], seed=s),
    "Bagging":       lambda s: make_sampler("bagging", row_frac=best_frac["bagging"], seed=s),
    "Random subspaces": lambda s: make_sampler("subspaces", col_frac=best_frac["subspaces"], seed=s),
    "Random patches": lambda s: make_sampler("patches", best_patch[0], best_patch[1], seed=s),
    "Random forest": lambda s: RandomForestClassifier(n_estimators=300, max_features=best_rf["max_features"],
                                                      criterion=best_rf["criterion"], n_jobs=N_JOBS, random_state=s),
    "Extra trees":   lambda s: make_et(s, best_rf["max_features"]),
    "AdaBoost":      lambda s: make_ada(best_ada["depth"], best_ada["lr"], best_ada["rounds"], s),
    "Gradient boosting": lambda s: HistGradientBoostingClassifier(
        learning_rate=best_hgb["lr"], max_depth=best_hgb["depth"], max_iter=500, early_stopping=True, random_state=s),
    "Voting (soft)": lambda s: VotingClassifier(best_members(s), voting="soft"),
    "Stacking":      lambda s: StackingClassifier(best_members(s), final_estimator=LogisticRegression(max_iter=2000),
                                                  cv=3, stack_method="predict_proba"),
}
FINAL_REG = {   # NOTE: bagging/RF/ET settings below are un-tuned defaults. Tune them on validation first if time allows.
    "Tree": lambda s: DecisionTreeRegressor(criterion=best_reg[0], max_depth=best_reg[1], random_state=s),
    "Bagging": lambda s: BaggingRegressor(DecisionTreeRegressor(), n_estimators=100, n_jobs=N_JOBS, random_state=s),
    "Random forest": lambda s: RandomForestRegressor(n_estimators=300, max_features=0.5, n_jobs=N_JOBS, random_state=s),
    "Extra trees": lambda s: ExtraTreesRegressor(n_estimators=300, max_features=0.5, n_jobs=N_JOBS, random_state=s),
    "AdaBoost.R2": lambda s: AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=6), n_estimators=200, random_state=s),
    "Gradient boosting": lambda s: GradientBoostingRegressor(
        n_estimators=1000, learning_rate=best_gbr["lr"], max_depth=best_gbr["depth"],
        n_iter_no_change=20, validation_fraction=0.1, random_state=s),
}
ms = lambda mean, std, nd=2: f"{mean:.{nd}f} +/- {std:.{nd}f}"

if CONFIRM_FINAL:
    Xte, yte = _FASHION_TEST_X_LOCKED, _FASHION_TEST_Y_LOCKED
    table6, saved_preds = [], {}
    for name, mk in FINAL_CLF.items():
        runs = []
        for s in SEEDS:
            d_ = fashion_split(s)               # train on the seed's 10k subsample
            r, pr_ = final_run(mk, d_["Xtr"], d_["ytr"], Xte, yte, s, "clf"); runs.append(r)
            if s == SEEDS[0]: saved_preds[name] = pr_
        df = pd.DataFrame(runs)
        table6.append({"Method": name,
                       "Test accuracy (%)": ms(df.test_acc.mean(), df.test_acc.std()),
                       "Macro-F1": ms(df.macro_f1.mean(), df.macro_f1.std(), 3),
                       "Train time (s)": ms(df.train_s.mean(), df.train_s.std()),
                       "Predict (ms/1k)": ms(df.predict_ms_per_1k.mean(), df.predict_ms_per_1k.std()),
                       "Nodes": int(df.nodes.mean()), "_acc": df.test_acc.mean()})
    t6 = pd.DataFrame(table6); best_name = t6.loc[t6._acc.idxmax(), "Method"]
    save_table(t6.drop(columns="_acc"), "part6_fashion_final"); print(t6.drop(columns="_acc"))
    ConfusionMatrixDisplay.from_predictions(yte, saved_preds[best_name], display_labels=CLASS_NAMES,
                                            xticks_rotation=45, cmap="Blues")
    plt.title(f"Confusion matrix: {best_name} (test set, seed {SEEDS[0]})"); plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/p6_confusion_matrix.png", dpi=150)

    reg_table = []
    for name, mk in FINAL_REG.items():
        runs = []
        for s in SEEDS:
            dd, (Xte_h, yte_h), _ = make_housing_splits(s)
            runs.append(final_run(mk, dd["Xtr"], dd["ytr"], Xte_h, yte_h, s, "reg")[0])
        df = pd.DataFrame(runs)
        reg_table.append({"Method": name, "RMSE": ms(df.rmse.mean(), df.rmse.std(), 3),
                          "MAE": ms(df.mae.mean(), df.mae.std(), 3),
                          "Train time (s)": ms(df.train_s.mean(), df.train_s.std()),
                          "Predict (ms/1k)": ms(df.predict_ms_per_1k.mean(), df.predict_ms_per_1k.std()),
                          "Nodes": int(df.nodes.mean())})
    print(save_table(pd.DataFrame(reg_table), "part6_housing_final"))

# %% ====================================================================
# BONUS (b): multiclass SAMME AdaBoost on library stumps, logging sample weights
# ======================================================================
def samme_fit(X, y, n_rounds, K, seed=0):
    n = len(y); w = np.ones(n) / n
    stumps, alphas, w_log = [], [], [w.copy()]              # w_log[t] = weights used to train round t
    for t in range(n_rounds):
        stump = DecisionTreeClassifier(max_depth=1, random_state=seed).fit(X, y, sample_weight=w)
        wrong = stump.predict(X) != y
        err = np.sum(w * wrong) / np.sum(w)
        if err >= 1 - 1 / K:                                # no better than random guessing: stop
            break
        err = max(err, 1e-10)
        alpha = np.log((1 - err) / err) + np.log(K - 1)     # the extra log(K-1) is what makes it multiclass
        w = w * np.exp(alpha * wrong); w /= w.sum()         # up-weight the mistakes
        stumps.append(stump); alphas.append(alpha); w_log.append(w.copy())
    return stumps, np.array(alphas), np.array(w_log)

def samme_predict(X, stumps, alphas, K):
    votes = np.zeros((len(X), K))
    for s, a in zip(stumps, alphas):
        votes[np.arange(len(X)), s.predict(X)] += a         # each stump votes with weight alpha
    return votes.argmax(1)

# 2-feature, 3-class toy problem so the decision boundaries can be compared visually.
Xt, yt = make_classification(n_samples=600, n_features=2, n_informative=2, n_redundant=0,
                             n_classes=3, n_clusters_per_class=1, class_sep=1.0, random_state=0)
stumps, alphas, w_log = samme_fit(Xt, yt, 50, K=3)
sk = make_ada(1, 1.0, 50, 0).fit(Xt, yt)
print("mine acc:", np.mean(samme_predict(Xt, stumps, alphas, 3) == yt), " sklearn acc:", sk.score(Xt, yt))
print("first 3 alphas  mine:", alphas[:3].round(3), " sklearn:", sk.estimator_weights_[:3].round(3))

gx, gy = np.meshgrid(np.linspace(Xt[:, 0].min() - 1, Xt[:, 0].max() + 1, 300),
                     np.linspace(Xt[:, 1].min() - 1, Xt[:, 1].max() + 1, 300))
gp = np.c_[gx.ravel(), gy.ravel()]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, zz, ttl in [(axes[0], samme_predict(gp, stumps, alphas, 3), "From-scratch SAMME"),
                    (axes[1], sk.predict(gp), "sklearn AdaBoost (SAMME)")]:
    ax.contourf(gx, gy, zz.reshape(gx.shape), alpha=0.3); ax.scatter(Xt[:, 0], Xt[:, 1], c=yt, s=8)
    ax.set_title(ttl); ax.set_xlabel("feature 1"); ax.set_ylabel("feature 2")
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/bonus_samme_boundaries.png", dpi=150)
np.save(f"{OUT_DIR}/bonus_samme_weight_log.npy", w_log)     # sample weights at every round
