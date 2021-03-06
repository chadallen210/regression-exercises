##### IMPORTS #####
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from env import host, username, password

##### DB CONNECTION #####
def get_db_url(db, username=username, host=host, password=password):
    
    return f'mysql+pymysql://{username}:{password}@{host}/{db}'

def wrangle_grades():
    '''
    Read student_grades csv file into a pandas DataFrame,
    drop student_id column, replace whitespaces with NaN values,
    drop any rows with Null values, convert all columns to int64,
    return cleaned student grades DataFrame.
    '''
    # Acquire data from csv file.
    grades = pd.read_csv('student_grades.csv')

    # Replace white space values with NaN values.
    grades = grades.replace(r'^\s*$', np.nan, regex=True)

    # Drop all rows with NaN values.
    df = grades.dropna()

    # Convert all columns to int64 data types.
    df = df.astype('int')

    return df
    
    
# Generic splitting function for continuous target.

def split_continuous(df):
    '''
    Takes in a df
    Returns train, validate, and test DataFrames
    '''
    # Create train_validate and test datasets
    train_validate, test = train_test_split(df, 
                                        test_size=.2, 
                                        random_state=123)
    # Create train and validate datsets
    train, validate = train_test_split(train_validate, 
                                   test_size=.3, 
                                   random_state=123)

    # Take a look at your split datasets

    print(f'train -> {train.shape}')
    print(f'validate -> {validate.shape}')
    print(f'test -> {test.shape}')
    return train, validate, test

    
    ##### TELCO FUNCTIONS #####
def new_telco_data():
    '''
    gets telco_churn information from CodeUp db and creates a dataframe
    '''

    # SQL query
    telco_query = '''SELECT customer_id, monthly_charges, tenure, total_charges
                     FROM customers
                     WHERE contract_type_id = 3'''
    
    # reads SQL query into a DataFrame            
    df = pd.read_sql(telco_query, get_db_url('telco_churn'))
    
    return df

def wrangle_telco():
    '''
    checks for existing telco_churn csv file and loads if present,
    otherwise runs new_telco_data function to acquire data
    '''
    
    # checks for existing file and loads
    if os.path.isfile('telco_churn.csv'):
        
        df = pd.read_csv('telco_churn.csv', index_col=0)
        
    else:
        
        # pull in data and creates csv file if not already present
        df = new_telco_data()
        
        df.to_csv('telco_churn.csv')
        
    # replace symbols, etc with NaN's
    df = df.replace(r'^\s*$', np.nan, regex=True)
    
    # replace NaN's in total_charges with value from monthly_charges
    df.total_charges = df.total_charges.fillna(df.monthly_charges)
    
    # change total_charges data type to float
    df = df.astype({'total_charges': 'float64'})
    
    return df

def split_telco(df):
    '''
    this function takes in the telco_churn data 
    '''
    
    # split df into 20% test, 80% train_validate
    train_validate, test = train_test_split(df, test_size=0.2, random_state=1234)
    
    # split train_validate into 30% validate, 70% train
    train, validate = train_test_split(train_validate, test_size=0.3, random_state=1234)
    
    return train, validate, test

def scale_telco(train, validate, test):
    
    scaler = sklearn.preprocessing.MinMaxScaler()

    scaler.fit(train[['monthly_charges']])

    
    

##### ZILLOW FUNCTIONS #####

def new_zillow_data():
    '''
    gets zillow information from CodeUp db and creates a dataframe
    '''

    # SQL query
    zillow_query = '''SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
                      FROM properties_2017
                      WHERE propertylandusetypeid = 261'''
    
    # reads SQL query into a DataFrame            
    df = pd.read_sql(zillow_query, get_db_url('zillow'))
    
    return df

def wrangle_zillow():
    '''
    checks for existing zillow csv file and loads if present,
    otherwise runs new_zillow_data function to acquire data
    '''
    
    # checks for existing file and loads
    if os.path.isfile('zillow.csv'):
        
        df = pd.read_csv('zillow.csv')
        
    else:
        
        # pull in data and creates csv file if not already present
        df = new_zillow_data()
        
        df.to_csv('zillow.csv')
        
    # drop column 'Unnamed: 0'
    df = df.drop(columns='Unnamed: 0')

    # replace symbols, etc with NaN's
    df = df.replace(r'^\s*$', np.nan, regex=True)
    
    # drop nulls
    df = df.dropna()
        
    return df

def split_zillow(df):
    '''
    this function takes in the zillow data 
    '''
    
    # split df into 20% test, 80% train_validate
    train_validate, test = train_test_split(df, test_size=0.2, random_state=1234)
    
    # split train_validate into 30% validate, 70% train
    train, validate = train_test_split(train_validate, test_size=0.3, random_state=1234)
    
    return train, validate, test