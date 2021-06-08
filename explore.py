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
    
