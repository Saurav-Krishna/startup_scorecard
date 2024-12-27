# Libraries

import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime







# creating connection to db



def create_connection(db):
    pass





# getting data from db
def get_data():
    pass



# DATA = df


# selecting apt features from db


FEATURES = ['Last Funding Amount (in USD)' 'Number of Funding Rounds'
 'Last Equity Funding Amount (in USD)' 'Total Funding Amount (in USD)'
 'Number of Articles' 'Total Products Active' 'Number of Investors'
 'Number of Founders' 'Trend Score (30 Days)' 'Trend Score (90 Days)'
 'Monthly Rank Growth' 'Global Traffic Rank' 'emp__10001+'
 'emp__1001-5000' 'emp__101-250' 'emp__11-50' 'emp__251-500'
 'emp__5001-10000' 'emp__501-1000' 'emp__51-100' 'Age_of_startup'
 'time_since_last_funding' 'acquired' 'Debt Financing' 'Post-IPO Equity'
 'Pre-Seed' 'Private Equity' 'Seed' 'Series A' 'Series B' 'Series C'
 'Series F' 'Venture - Series Unknown']


def data_cleaning(df):

    num_col = df.sele


df = pd.read_csv("Now.csv")


def feature_selection(df):

    
    col_to_drop = ['Founded Date Precision','Closed Date Precision','Company Type', 'Investor Type','Last Funding Amount', 'Last Funding Amount Currency',
               'Last Equity Funding Amount', 'Last Equity Funding Amount Currency','Total Equity Funding Amount', 'Total Equity Funding Amount Currency',
               'Total Funding Amount','Total Funding Amount Currency','Diversity Spotlight','Number of Sub-Orgs', 'Stage', 'Most Recent Valuation Range',
               'Date of Most Recent Valuation','Acquired by URL', 'Transaction Name','Number of Exits','Exit Date', 'Exit Date Precision',
               'Description', 'Website', 'Twitter', 'Facebook','Contact Email', 'Phone Number', 'Full Description','Closed Date'
               ]
    



def clean_data(df):
    df = df.drop_duplicates(subset=['Organization Name'])

def transform_data(df):
    try:
        df['Monthly Rank Growth'] = (df['MOnthly Rank Growth'].str.replace(',','')
                                 .str.rstrip('%').astype(float)/100)
        df['']
    

    # df[]
    

    except KeyError as k:
        print("Unable to transform feature {k}")

    finally:
        df = df.drop_duplicates(subset=['Organization Name'])


def scale_features():

    pass

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection parameters
DB_CONFIG = {
    "dbname": "your_database_name",
    "user": "your_user_name",
    "password": "your_password",
    "host": "your_host",
    "port": 5432  # Default port
}

def fetch_data(query):
    """
    Fetch data from the PostgreSQL database.
    Args:
        query (str): SQL query to execute.
    Returns:
        pd.DataFrame: Data fetched as a Pandas DataFrame.
    """
    try:
        logger.info("Connecting to the database...")
        conn = psycopg2.connect(**DB_CONFIG)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            logger.info("Executing query...")
            cur.execute(query)
            data = cur.fetchall()
            logger.info("Data fetched successfully.")
        conn.close()
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        raise

def perform_eda(data):
    """
    Perform basic exploratory data analysis (EDA).
    Args:
        data (pd.DataFrame): Data to analyze.
    Returns:
        None
    """
    logger.info("Performing EDA...")
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Data types:\n{data.dtypes}")
    logger.info(f"Missing values:\n{data.isnull().sum()}")
    logger.info(f"Data summary:\n{data.describe()}")

def preprocess_data(data, target_column):
    """
    Preprocess data for model consumption.
    Args:
        data (pd.DataFrame): Raw data.
        target_column (str): Name of the target column.
    Returns:
        tuple: (X, y) where X is the feature set and y is the target variable.
    """
    logger.info("Preprocessing data...")
    

    data = data.dropna()
    
    # Separate features and target
    y = data[target_column]
    X = data.drop(columns=[target_column])
    
    # Standardize numeric features
    numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    X[numeric_columns] = scaler.fit_transform(X[numeric_columns])
    
    logger.info("Data preprocessing complete.")
    return X, y

if __name__ == "__main__":
    
    QUERY = "SELECT * FROM your_table_name"  
    TARGET_COLUMN = "target_column_name"    
    
    # Fetch data
    data = fetch_data(QUERY)
    
    # Perform EDA
    perform_eda(data)
    
    # Preprocess data
    X, y = preprocess_data(data, TARGET_COLUMN)
    
    logger.info("Data ready for model prediction.")
