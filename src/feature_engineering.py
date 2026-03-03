import pandas as pd 
import os # has makerdirs method
from sklearn.feature_extraction.text import TfidfVectorizer  
import logging #module
import yaml


### Implementing Logging Module

# Ensure the "logs" directory exists
log_dir = "logs" # a directory into our project where we will be storing all our logs.
os.makedirs(log_dir , exist_ok= True) #exist_ok checks if there is any logs dir already present or not , if present then use that only . 

# logging configuration
logger = logging.getLogger('feature_engineering') # creating our logger object with name : data_ingestion 
logger.setLevel('DEBUG') # i.e give us all the levels of errors if any .

# we want the messages to show in terminal console
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG') # here also we have given the level of DEBUG to the console handler 

# since we are saving all our logs into our log folder into our project directory.so we need a file handler also and we need to set its level also , inorder incase we need segregation of the levels of informations . 
log_file_path = os.path.join(log_dir , 'feature_engineering.log') # where do we want to save the logs in the dir . 
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG') # defined the level

# logging formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s -%(message)s') # in which string format do we want to see the infomation in the console(terminal) . name  : data_ingestion (name of the object ) .
console_handler.setFormatter(formatter) # console_handler and file_handler needs to print the message in the same format . 
file_handler.setFormatter(formatter)

if not logger.handlers: # industry best practice to check if not added then add 
    logger.addHandler(console_handler) # now first we created the handler and now putting both the handlers inside the logger object 
    logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


# loading the data from the interim file of the data folder , i.e from the same last place where our data was last saved 
def load_data(file_path: str) -> pd.DataFrame:
    """ load data from a csv file """
    try:
        df = pd.read_csv(file_path)  # loading the data with the help of the url
        df.fillna('', inplace= True)
        logger.debug('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:  # CSV syntax issues
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading data from %s:', e)
        raise

# now data preporcessing 

def apply_tfidf(train_data : pd.DataFrame , test_data : pd.DataFrame , max_features : int ) ->tuple:
    """ Apply Tfidf to the data """
    try:
        vectorizer = TfidfVectorizer(max_features = max_features) 


        X_train = train_data['text'].values
        y_train = train_data['target'].values
        X_test = test_data['text'].values
        y_test = test_data['target'].values

        X_train_how = vectorizer.fit_transform(X_train)
        X_test_how = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_how.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_how.toarray())
        test_df['label'] = y_test

        logger.debug('Tfidf applied and data transformed')
        return train_df , test_df
    except Exception as e:
        logger.error('Error during bag of words transformations: %s' , e)
        raise


def save_data(df: pd.DataFrame , file_path : str ) ->None:
    """ Save the dataframe to a CSV file ."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok= True)
        df.to_csv(file_path ,index = False)
        logger.debug('data saved to %s',file_path)
    except Exception as e:
        logger.error('Unexpected error occured while saving the data: %s' ,e)
        raise

def main():
    try:
        # feature_engineering
        params = load_params(params_path='params.yaml')
        max_features = params['feature_engineering']['max_features']

        # max_features = 50 # each and every unique word becomes a column  , earlier we had only two : one test and other target

        # train_data = load_data('D:\MLOps\Day5 yt_mlops_Complete_MLpipeline\data\interim\train_processed.csv') # back slashes (\) for file path won't work here
        # test_data = load_data('D:\MLOps\Day5 yt_mlops_Complete_MLpipeline\data\interim\test_processed.csv')
# use this(BEST — Cross Platform) in place of the above : 
        train_data = load_data(os.path.join("data", "interim", "train_processed.csv"))
        test_data = load_data(os.path.join("data", "interim", "test_processed.csv"))
# or this one:
# train_data = load_data(r'D:\MLOps\Day5 yt_mlops_Complete_MLpipeline\data\interim\train_processed.csv') i.e raw string
# or this :(Forward Slashes)
# train_data = load_data('D:/MLOps/Day5 yt_mlops_Complete_MLpipeline/data/interim/train_processed.csv')

        train_df , test_df =  apply_tfidf(train_data , test_data, max_features)

        save_data(train_df , os.path.join("./data","processed" ,"train_tfidf.csv")) # now we will make a third folder inside the data with the name processed
        save_data(test_df, os.path.join("./data","processed","test_tfidf.csv"))

    except Exception as e:
        logger.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()

# 🚀 Big Picture

# Your pipeline stages are now:

# data/raw → data/interim → data/processed