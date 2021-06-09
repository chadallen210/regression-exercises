import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr


def plot_variable_pairs(df, target):
    
    columns = df[list(df.select_dtypes(exclude='O').columns)].drop(columns=target)
    
    for col in columns:
        
        sns.lmplot(x=col, y=target, data=df, line_kws={'color': 'red'})
        plt.show()
        

def pairplot_variable_pairs(df):
    
    columns = df[list(df.select_dtypes(exclude='O').columns)]
    
    sns.pairplot(columns, kind="reg", plot_kws={'line_kws':{'color':'red'}}, corner=True)
    plt.show()
    
    
def plot_categorical_and_continuous_vars(df, catagorical_var, continuous_var):
    
    sns.barplot(data=df, x=catagorical_var, y=continuous_var)
    plt.show()
    sns.swarmplot(data=df, x=catagorical_var, y=continuous_var)
    plt.show()
    sns.boxplot(data=df, x=catagorical_var, y=continuous_var)
    plt.show()
    

    
##### FROM LESSON EXERCISES

def months_to_years(df):
    """
    Takes in the telco df and returns the df with new 
    categorical feature 'tenure_years'
    """
    df['tenure_years'] = round(df.tenure // 12)
    df['tenure_years'] = df.tenure_years.astype('object')
    return df


def plot_variable_pairs(train, cols, hue=None):
    '''
    This function takes in a df, a list of cols to plot, and default hue=None 
    and displays a pairplot with a red regression line.
    '''
    plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.7}}
    sns.pairplot(train[cols], hue=hue, kind="reg",plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.1}})
    plt.show()