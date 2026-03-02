import os # has makerdirs method
import logging #module
import pandas as pd 
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')

### Implementing Logging Module

# Ensure the "logs" directory exists
log_dir = "logs" # a directory into our project where we will be storing all our logs.
os.makedirs(log_dir , exist_ok= True) #exist_ok checks if there is any logs dir already present or not , if present then use that only . 

# logging configuration
logger = logging.getLogger('data_Pre_processing') # creating our logger object with name : Pre_processing
logger.setLevel('DEBUG') # i.e give us all the levels of errors if any .

# we want the messages to show in terminal console
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG') # here also we have given the level of DEBUG to the console handler 

# since we are saving all our logs into our log folder into our project directory.so we need a file handler also and we need to set its level also , inorder incase we need segregation of the levels of informations . 
log_file_path = os.path.join(log_dir , 'Pre_processing.log') # where do we want to save the logs in the dir . 
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG') # defined the level

# logging formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s -%(message)s') # in which string format do we want to see the infomation in the console(terminal) . name  : data_ingestion (name of the object ) .
console_handler.setFormatter(formatter) # console_handler and file_handler needs to print the message in the same format . 
file_handler.setFormatter(formatter)

if not logger.handlers: # industry best practice to check if not added then add 
    logger.addHandler(console_handler) # now first we created the handler and now putting both the handlers inside the logger object 
    logger.addHandler(file_handler)


def transform_text(text):
    #### Docstring
    """ Transforms the input text by converting it to lowercase , tokeninzing , removing stopwords and punctuations and stemming"""
    ps = PorterStemmer() # created object
    text = text.lower() # convert to lower case
    text = nltk.word_tokenize(text) # tokenize the text
    text = [word for word in text if word.isalnum()] # remove non_alphanumeric tokens
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation] # remove stopwords and punctuations 
    text = [ps.stem(word) for word in text] # stem the words 
    return " ".join(text) # Join the tokens back into a single string 


# now data preporcessing 

def preprocess_df(df , text_column = 'text' , target_column = 'target'):
    """ Preprocess the Dataframe by encoding the target column , removing duplicates , and transforming the text column"""
    try:
        logger.debug('Starting preprocessing for DataFrame')
        # Encode the target column
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('Target column encoded')
        
        # Remove duplicate rows
        df = df.drop_duplicates(keep = 'first')
        logger.debug('Target column encoded')

        # Remove duplicate rows
        df = df.drop_duplicates(keep = 'first')
        logger.debug('Duplicates removed')

        # Apply text transformation to the specified text column
        df.loc[:,text_column] = df[text_column].apply(transform_text)
        logger.debug('Text column transformed')
        return df
    
    except KeyError as e:
        logger.error('Missing column in the DataFrame: %s', e)
        raise
    except Exception as e:
        logger.error('Error during text normalization: %s', e)
        raise

def main(text_column = 'text' , target_column = 'target'):
    """Main function to load raw data , preprocess it , and save the processed data"""
    try:
    #    fetch data from the data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded properly') #basically currently we are doing debug level par logger lagana

        # Transform the data
        train_processed_data = preprocess_df(train_data , text_column , target_column)
        test_processed_data = preprocess_df(test_data , text_column , target_column)

        # store the data inside data/processed 
        data_path = os.path.join("./data" ,"interim")#creating an interim named subfolder where our preprocessed data will be 
        os.makedirs(data_path , exist_ok=True)
        
        train_processed_data.to_csv(os.path.join(data_path , "train_processed.csv"), index = False)
        test_processed_data.to_csv(os.path.join(data_path , "test_processed.csv"),index =False)

        logger.debug('Processed data saved to %s',data_path)

    except FileNotFoundError as e:
        logger.error('File not found: %s' , e)
    except pd.errors.EmptyDataError as e :
        logger.error('No data %s' , e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process:%s' ,e)
        print(f'Error : {e}')

if __name__ == '__main__':
    main()
