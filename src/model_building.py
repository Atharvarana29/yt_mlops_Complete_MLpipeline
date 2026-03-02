import os # has makerdirs method
import numpy as np
import pandas as pd
import pickle 
import logging #module
from sklearn.ensemble import RandomForestClassifier # since we had got highest accuracy in rf when doing the train test 

### Implementing Logging Module

# Ensure the "logs" directory exists
log_dir = "logs" # a directory into our project where we will be storing all our logs.
os.makedirs(log_dir , exist_ok= True) #exist_ok checks if there is any logs dir already present or not , if present then use that only . 

# logging configuration
logger = logging.getLogger('model_building') # creating our logger object with name : data_ingestion 
logger.setLevel('DEBUG') # i.e give us all the levels of errors if any .

# we want the messages to show in terminal console
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG') # here also we have given the level of DEBUG to the console handler 

# since we are saving all our logs into our log folder into our project directory.so we need a file handler also and we need to set its level also , inorder incase we need segregation of the levels of informations . 
log_file_path = os.path.join(log_dir , 'model_building.log') # where do we want to save the logs in the dir . 
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG') # defined the level

# logging formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s -%(message)s') # in which string format do we want to see the infomation in the console(terminal) . name  : data_ingestion (name of the object ) .
console_handler.setFormatter(formatter) # console_handler and file_handler needs to print the message in the same format . 
file_handler.setFormatter(formatter)

if not logger.handlers: # industry best practice to check if not added then add 
    logger.addHandler(console_handler) # now first we created the handler and now putting both the handlers inside the logger object 
    logger.addHandler(file_handler)


# this time we will be loading the data from the processed  file of the data folder , i.e like earlier components from the same last place where our data was last saved 
def load_data(file_path: str) -> pd.DataFrame:
    """ load data from a csv file.
     :param file_path : path to the CSV file
     :return : loaded Dataframe
    """
    try:
        df = pd.read_csv(file_path)  # loading the data with the help of the url
        logger.debug('Data loaded from %s with shape %s', file_path , df.shape)
        return df
    except pd.errors.ParserError as e:  # CSV syntax issues
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except FileNotFoundError as e:
        logger.error('File not found: %s' , e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading data from %s:', e)
        raise

# training the model

def train_model(X_train: np.ndarray , y_train: np.ndarray , params:dict)->RandomForestClassifier:
     # this coding style is known as type hinding , x,y train is an ndarray , params it is taking in the form of dict and what it will return is a rfclassifier , 
    # this is a very good coding practice 
    """Train the RandomForest model./
    :param x_train : Training features
    :param y_train : Training labels
    :param_params : dictionary of hyperparameters 
    :return : Trained RandomForestClassifier"""

    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("The number of samples in x_train and y_train must be the same")
    
        logger.debug('Initializing RandomForest model with parameters: %s',params)
        clf = RandomForestClassifier(n_estimators= params['n_estimators'],random_state=params['random_state'])

        logger.debug(f"Model training started with {X_train.shape[0]} samples")
        clf.fit(X_train ,y_train)
        logger.debug('Model training completed')

        return clf
    
    except ValueError as e :
        logger.error('valueError during model training: %s' ,e)
        raise
    except Exception as e :
        logger.error('Error during model training %s' , e)
        raise

# and what our model will return we will save it 
def save_model(model, file_path : str ) ->None:
    """ Save the trained model in a file .

    :param model : Trained model object
    :param_file_path : path to save the model
    
    """
    try:
        # ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok= True)

        with open(file_path , 'wb') as file :
            pickle.dump(model,file)
        logger.debug('Model saved to %s',file_path)
    except FileNotFoundError as e:
        logger.error('File path not found: %s' , e)
        raise
    except Exception as e:
        logger.error('Unexpected error occured while saving the data: %s' ,e)
        raise
    
def main():
    try:
        params = {'n_estimators':25 , 'random_state':2}
        train_data = load_data('./data/processed/train_tfidf.csv')
        # Instead of hardcoding: the above file path use below :
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# data_path = os.path.join(BASE_DIR, 'data', 'processed', 'train_tfidf.csv')
# This prevents path issues when running from different directories.
# Industry pipelines always use absolute path resolution.
        X_train = train_data.iloc[: , : -1].values
        y_train = train_data.iloc[:,-1].values

        clf = train_model(X_train , y_train , params)

        model_save_path = 'models/model.pkl'
        save_model(clf ,model_save_path)


    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()

