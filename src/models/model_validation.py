import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import lars_path
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.tools import add_constant
from sklearn.preprocessing import StandardScaler

def find_consistent_train_test_split(X, y, test_size, display_top_ten=False):
    """
    Takes in two data frames, test_size as decimal, and returns best random state
    """
    attempts = []

    for i in range(100):
        X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=i)
        total_dev_off = sum([((X_train[col].mean() - X_test[col].mean()) / X_train[col].std())**2 for col in X_train.columns])
        attempts.append((i, total_dev_off))
    top_ten = sorted(attempts, key=lambda e: e[1])[:10]

    if display_top_ten:
        print(top_ten)

    return top_ten[0][0]

def validate_train_test_split(X_train, X_test, y_train, y_test):
    print('***** Diff Btwn Mean of Train vs. Test (in std) *****')
    print('***** Features *****')
    for col in X_train.columns:
        dev_off = (X_train[col].mean() - X_test[col].mean()) / X_train[col].std()
        print(f'{col}: {dev_off}')

    print('***** Ind Var *****')
    print((y_train.mean() - y_test.mean()) / y_train.std())

def print_k_fold_metrics(metrics, show_all=False):
    for label, values in metrics:
        print(f'{label}: {np.mean(values):.5f} +- {np.std(values):.5f}')
        if show_all:
            print(f'{label}_all: ', values)

def run_stats_models(X, y, show_summary=True):
    lm_sm = sm.OLS(y, add_constant(X))
    fit_sm = lm_sm.fit()
    if show_summary:
        print(fit_sm.summary())
    return fit_sm

def generate_resid_plots(sm_fit):
    # diagnose/inspect residual normality using qqplot:
    stats.probplot(sm_fit.resid, dist="norm", plot=plt)
    plt.title("Normal Q-Q plot")

    plt.figure(figsize=(12,8))
    plt.scatter(sm_fit.predict(), sm_fit.resid)
    plt.title('Predicted vs. Residuals')

def print_coef(columns, coefs):
    for col, coef in sorted(zip(columns, coefs), key=lambda x: -np.abs(x[1])):
        print(f'{col}: {coef}')

def cross_val(X, y, model, drop_cols=None, standardize=False, alpha=None, random_state=71, n_splits=5, show_all=False, sm_summary=False, resid_plots=False):
    """
    Implements K Fold cross validation, prints metrics, and returns the last model


    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    is_lm_r2s, oos_lm_r2s, oos_lm_mse,oos_lm_mae, is_lm_aic = [], [], [], [], [] #collect the validation results

    last_lm = None
    if drop_cols:
        X = X.drop(columns=drop_cols)

    X_train_np = np.array(X)
    y_train_np = np.array(y)

    for train_ind, val_ind in kf.split(X_train_np, y_train_np):

        X_tr, y_tr = X_train_np[train_ind], y_train_np[train_ind]
        X_val, y_val = X_train_np[val_ind], y_train_np[val_ind]

        if standardize:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_val = scaler.transform(X_val)

        # run linear regression
        lm = model(alpha=alpha) if alpha else model()
        fit = lm.fit(X_tr, y_tr)

        in_sample_preds = fit.predict(X_tr)
        out_sample_preds = fit.predict(X_val)

        is_lm_r2s.append(metrics.r2_score(y_true=y_tr, y_pred=in_sample_preds))
        oos_lm_r2s.append(metrics.r2_score(y_true=y_val, y_pred=out_sample_preds))
        oos_lm_mse.append(metrics.mean_squared_error(y_true=y_val, y_pred=out_sample_preds))
        oos_lm_mae.append(metrics.mean_absolute_error(y_true=y_val, y_pred=out_sample_preds))

        # get AIC
        regr = sm.OLS(y_tr, add_constant(X_tr)).fit()
        is_lm_aic.append(regr.aic)

        last_lm = lm

    print_k_fold_metrics([
        ('R2 (in sample)', is_lm_r2s),
        ('R2', oos_lm_r2s),
        ('Mean Sq Error', oos_lm_mse),
        ('Mean abs error', oos_lm_mae),
        ('AIC (in sample)', is_lm_aic),
    ], show_all=show_all)

    if sm_summary or resid_plots:
        if standardize:
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        fit = run_stats_models(X, y, show_summary=sm_summary)

        if resid_plots:
            generate_resid_plots(fit)

    return last_lm

def generate_LARS_plot(X, y):
    X_lars = StandardScaler().fit_transform(X.values)
    _, _, coefs_lars = lars_path(X_lars, y.values, method='lasso')

    # plotting the LARS path
    plt.rcParams['image.cmap'] = 'gray'

    xx = np.sum(np.abs(coefs_lars.T), axis=1)
    xx /= xx[-1]

    plt.figure(figsize=(10,10))
    plt.plot(xx, coefs_lars.T)
    ymin, ymax = plt.ylim()
    plt.vlines(xx, ymin, ymax, linestyle='dashed')
    plt.xlabel('|coef| / max|coef|')
    plt.ylabel('Coefficients')
    plt.title('LASSO Path')
    plt.axis('tight')
    plt.legend(X.columns)
    plt.show()
