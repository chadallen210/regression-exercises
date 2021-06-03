##### IMPORTS #####
import pandas as pd
import numpy as np
import os
from env import host, username, password

##### DB CONNECTION #####
def get_db_url(db, username=username, host=host, password=password):
    
    return f'mysql+pymysql://{username}:{password}@{host}/{db}'

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
        
        df = pd.read_csv('zillow.csv', index_col=0)
        
    else:
        
        # pull in data and creates csv file if not already present
        df = new_zillow_data()
        
        df.to_csv('zillow.csv')
        
    # replace symbols, etc with NaN's
    df = df.replace(r'^\s*$', np.nan, regex=True)
    
    # drop nulls
    df = df.dropna()
        
    return df