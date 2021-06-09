##### IMPORTS #####
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from math import sqrt

import matplotlib.pyplot as plt
import seaborn as sns


def plot_residuals(df, x, y):

    residual = df['yhat'] - y
    residual_baseline = df['yhat_baseline'] - y
    
    fig, ax = plt.subplots(2, figsize=(10, 8))
    
    ax[0].scatter(x, residual)
    ax[0].set_title(label='Model Residuals')
    ax[0].axhline(y=0, ls=':', color='grey')
    
    ax[1].scatter(x, residual_baseline)
    ax[1].set_title(label='Baseline Residuals')
    ax[1].axhline(y=0, ls=':', color='grey')

    plt.show()
    
def regression_errors(df, y):
    
    sse = mean_squared_error(y, df.yhat) * len(df)
    print("Model SSE: ", sse)
    
    ess = ((df.yhat - y.mean())**2).sum()
    print("Model ESS: ", ess)
    
    tss = ((df.tip - y.mean())**2).sum()
    print("Model TSS: ", tss)

    mse = mean_squared_error(y, df.yhat)
    print("Model MSE: ", mse)

    rmse = sqrt(mean_squared_error(y, df.yhat))
    print("Model RMSE: ", rmse)
    
def baseline_mean_errors(df, y):

    sse_baseline = mean_squared_error(y, df.yhat_baseline) * len(df)
    print("Baseline SSE: ", sse_baseline)

    mse_baseline = mean_squared_error(y, df.yhat_baseline)
    print("Baseline MSE: ", mse_baseline)

    rmse_baseline = sqrt(mean_squared_error(y, df.yhat_baseline))
    print("Baseline RMSE: ", rmse_baseline)
    
def better_than_baseline(df, y):
    
    rmse = sqrt(mean_squared_error(y, df.yhat))
    rmse_baseline = sqrt(mean_squared_error(y, df.yhat_baseline))
    
    print("RMSE: ", rmse)
    print("Baseline RMSE: ", rmse_baseline)

    if rmse < rmse_baseline:
        print(f'\nBased on RMSE, the model({rmse:.4f}) performs better than the baseline({rmse_baseline:.4f}).')
    else:
        print(f'\nBased on RMSE, the baseline({rmse_baseline:.4f}) performs better than the model ({rmse:.4f}).')
        

def residuals(actual, predicted):
    return actual - predicted

def sse(actual, predicted):
    return (residuals(actual, predicted) ** 2).sum()

def mse(actual, predicted):
    n = actual.shape[0]
    return sse(actual, predicted) / n

def rmse(actual, predicted):
    return math.sqrt(mse(actual, predicted))

def ess(actual, predicted):
    return ((predicted - actual.mean()) ** 2).sum()

def tss(actual):
    return ((actual - actual.mean()) ** 2).sum()

def r2_score(actual, predicted):
    return ess(actual, predicted) / tss(actual)


def plot_residuals(actual, predicted):
    residuals = actual - predicted
    plt.hlines(0, actual.min(), actual.max(), ls=':')
    plt.scatter(actual, residuals)
    plt.ylabel('residual ($y - \hat{y}$)')
    plt.xlabel('actual value ($y$)')
    plt.title('Actual vs Residual')
    return plt.gca()

def regression_errors(actual, predicted):
    return pd.Series({
        'sse': sse(actual, predicted),
        'ess': ess(actual, predicted),
        'tss': tss(actual),
        'mse': mse(actual, predicted),
        'rmse': rmse(actual, predicted),
        'r2': r2_score(actual, predicted),
    })

def baseline_mean_errors(actual):
    predicted = actual.mean()
    return {
        'sse': sse(actual, predicted),
        'mse': mse(actual, predicted),
        'rmse': rmse(actual, predicted),
    }

def better_than_baseline(actual, predicted):
    sse_baseline = sse(actual, actual.mean())
    sse_model = sse(actual, predicted)
    return sse_model < sse_baseline