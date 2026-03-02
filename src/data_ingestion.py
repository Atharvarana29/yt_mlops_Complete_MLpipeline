import pandas as pd 
import os # has makerdirs method
from sklearn.model_selection import train_test_split
import logging #module


### Implementing Logging Module

# Ensure the "logs" directory exists
log_dir = "logs" # a directory into our project where we will be storing all our logs.
os.makedirs(log_dir , exist_ok= True) #exist_ok checks if there is any logs dir already present or not , if present then use that only . 

# logging configuration
logger = logging.getLogger('data_ingestion') # creating our logger object with name : data_ingestion 
logger.setLevel('DEBUG') # i.e give us all the levels of errors if any .

#  These are levels in the logger :
# LEVELS : 
# P1 : DEBUG : when we want very basic level informations , i.e we are reading line by line . it is the least critical
# P2 : INFO : some information achieved , some process completed , more critical
# P3 : WARNING : some where our code is giving us some warning about reaching limits , but our code does not stops due to it 
# P4 : ERROR : somewhere our code did not worked literally
# P5 : CRITICAL : occurs in production level like docker file not working , sever did not work .


# now suppose if we have set the level to DEBUG then we will be able to see all the logs error that comes at diffent level below it , i.e the errors which are more sensative than that level . 

# we want the messages to show in terminal console
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG') # here also we have given the level of DEBUG to the console handler 

# since we are saving all our logs into our log folder into our project directory.so we need a file handler also and we need to set its level also , inorder incase we need segregation of the levels of informations . 
log_file_path = os.path.join(log_dir , 'data_ingestion.log') # where do we want to save the logs in the dir . 
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG') # defined the level

# logging formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s -%(message)s') # in which string format do we want to see the infomation in the console(terminal) . name  : data_ingestion (name of the object ) .
console_handler.setFormatter(formatter) # console_handler and file_handler needs to print the message in the same format . 
file_handler.setFormatter(formatter)

if not logger.handlers: # industry best practice to check if not added then add 
    logger.addHandler(console_handler) # now first we created the handler and now putting both the handlers inside the logger object 
    logger.addHandler(file_handler)


# loading the data 
def load_data(data_url: str) -> pd.DataFrame:
    """ load data from a csv file """
    try:
        df = pd.read_csv(data_url)  # loading the data with the help of the url
        logger.debug('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:  # CSV syntax issues
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading data from %s: %s', data_url, e)
        raise

# now data preporcessing 

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """ Preprocess the data """
    try:
        # drop unwanted columns; ensure exact names match 
        df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True , errors='ignore') # If any of those columns don’t exist → it will throw KeyError.
        df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
        logger.debug('Data preprocessing completed')
        return df
    except KeyError as e:
        logger.error('Missing column in the DataFrame: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise

def save_data(train_data : pd.DataFrame , test_data: pd.DataFrame , data_path : str) ->None:
    """ Save the train and test datasets ."""
    try:
        raw_data_path = os.path.join(data_path ,'raw')
        os.makedirs(raw_data_path , exist_ok= True)
        train_data.to_csv(os.path.join(raw_data_path , "train.csv"),index = False)
        test_data.to_csv(os.path.join(raw_data_path , "test.csv"),index= False)
        logger.debug('Train and test data saved to %s',raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occured while saving the data: %s' ,e)
        raise

def main():
    try:
        test_size = 0.2  # proportion of data reserved for testing
        # use raw string or forward slashes to avoid escape issues
        data_path = r'D:\MLOps\Day5 yt_mlops_Complete_MLpipeline\experiments\spam.csv'
        df = load_data(data_url=data_path)
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=2)
        save_data(train_data, test_data, data_path='./data')
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
