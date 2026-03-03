import os # has makerdirs method
import numpy as np
import pandas as pd
import pickle 
import json
from sklearn.metrics import accuracy_score , precision_score , recall_score , roc_auc_score
import logging #module
from dvclive import Live # it tracks metrics , parameters , plots , training , progress and works with DVC experiments
import yaml



### Implementing Logging Module

# Ensure the "logs" directory exists
log_dir = "logs" # a directory into our project where we will be storing all our logs.
os.makedirs(log_dir , exist_ok= True) #exist_ok checks if there is any logs dir already present or not , if present then use that only . 

# logging configuration
logger = logging.getLogger('model_evaluation') # creating our logger object with name : data_ingestion 
logger.setLevel('DEBUG') # i.e give us all the levels of errors if any .

# we want the messages to show in terminal console
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG') # here also we have given the level of DEBUG to the console handler 

# since we are saving all our logs into our log folder into our project directory.so we need a file handler also and we need to set its level also , inorder incase we need segregation of the levels of informations . 
log_file_path = os.path.join(log_dir , 'model_evaluation.log') # where do we want to save the logs in the dir . 
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

# this time we will be loading the data from the processed  file of the data folder , i.e like earlier components from the same last place where our data was last saved 
def load_model(file_path: str):
    """ load the trained model from a file """
    try:
        with open(file_path , "rb") as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', file_path)
        return model
    except FileNotFoundError :
        logger.error('File not found :%s' , file_path)
        raise
    except Exception as e :
        logger.error('Unexpected error occured while loading the model : %s',e)
        raise

# now we need to load our file_path == test_tfidf data to test the model accuracy 

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

# now evaluating the model 
def evaluate_model(clf ,X_test:np.ndarray, y_test : np.ndarray)->dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:,-1]
        
        accuracy = accuracy_score(y_test , y_pred)
        precision = precision_score(y_test , y_pred)
        recall = recall_score(y_test , y_pred)
        auc = roc_auc_score(y_test , y_pred_proba)

        metric_dict = {
            "accuracy:" : accuracy ,
            "precision:" : precision ,
            "recall:" : recall,
            "auc:": auc
        }

        logger.debug('Model evaluation metrics calculated')
        return metric_dict
    except Exception as e:
        logger.error("Error during model evaluation %s" , e)
        raise

# and what metrics our model will return we will save it 
def save_metrics(metrics : dict , file_path : str ) ->None:
    """ Save the evaluation metrics into a JSON file ."""
    try:
        # ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok= True)

        with open(file_path , 'w') as file :
            json.dump(metrics,file , indent= 4)
        logger.debug('metrics saved to %s',file_path)
    except Exception as e:
        logger.error('Unexpected error occured while saving the metrics: %s' ,e)
        raise
    
def main():
    try:
        # initiating load params
        params = load_params(params_path='params.yaml')

        clf = load_model('./models/model.pkl') # loading the model
        test_data = load_data('./data/processed/test_tfidf.csv') #loading the test data
        
        X_test = test_data.iloc[:,:-1].values # loading X_test and y_test 
        y_test = test_data.iloc[:,-1].values

# gives options to log lot of things
        with Live(save_dvc_exp=True) as live: # this params : save_dvc_exp=True : jitne baar bhi hum experiments run karenge ye sabka results record karta jayega
            live.log_metric('accuracy', accuracy_score(y_test, y_test))
            live.log_metric('precision', precision_score(y_test, y_test))
            live.log_metric('recall', recall_score(y_test, y_test))

            live.log_params(params) # logs the params used also

        metrics = evaluate_model(clf , X_test , y_test) # calling the evaluate model function

        save_metrics(metrics , 'reports/metrics.json')
    except Exception as e:
        logger.error('Failed to complete the model evaluation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()

