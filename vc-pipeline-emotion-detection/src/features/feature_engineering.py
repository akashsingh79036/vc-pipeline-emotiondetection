import numpy as np
import pandas as pd
import os 
from sklearn.feature_extraction.text import CountVectorizer
import yaml
import logging


#logging  configuration
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('feature_engineering_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters loaded successfully from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error while loading parameters from %s: %s', params_path, e)
        raise
    except Exception as e:
        logger.error('Some error occurred while loading parameters from %s: %s', params_path, e)
        raise


def apply_bow(train_data:pd.DataFrame,test_data:  pd.DataFrame,max_features:int) -> tuple:
    try:
        #Apply Bag of words(countvectorizer)
        vectorizer=CountVectorizer(max_features=max_features)

        #Apply Bow
        X_train=train_data['content'].values
        y_train=train_data['sentiment'].values

        X_test=test_data['content'].values
        y_test=test_data['sentiment'].values

        #fit the vetorizer on the training data and transform it
        X_train_bow=vectorizer.fit_transform(X_train)
        #transform the test data using the same vectorizer
        X_test_bow=vectorizer.transform(X_test)

        train_df=pd.DataFrame(X_train_bow.toarray())

        train_df['label']=y_train

        test_df=pd.DataFrame(X_test_bow.toarray())

        test_df['label']=y_test

        logger.debug('Bag of words applied and data transformed')
        return train_df,test_df
    except Exception as e:
        logger.error('Error during TF-IDF application and bow transforation: %s' , e)
        raise

def load_data(file_path:str) -> pd.DataFrame:
    try:
        df=pd.read_csv(file_path)
        df.fillna('',inplace=True)
        logger.debug('data loaded successfully from %s',file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file from %s: %s',file_path,e)
        raise
    except Exception as e:
        logger.error('Error loading data from %s: %s',file_path,e)
        raise

def main(): 
    try:
        params=load_params("params.yaml")
        max_features=params['feature_engineering']['max_features']
        #store the data inside data/processed
        #fetch the data from data/interim
        train_data=load_data('./data/interim/train_processed.csv')
        test_data=load_data('./data/interim/test_processed.csv')
        train_df,test_df=apply_bow(train_data,test_data,max_features=max_features)
        save_data(train_df,os.path.join("data","processed","train_bow.csv"))
        save_data(test_df,os.path.join("data","processed","test_bow.csv"))
    except Exception as e:
        logger.error('Feature engineering failed: %s',e)
        print(f"Error: {e}")


def save_data(df:pd.DataFrame,file_path:str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        df.to_csv(file_path,index=False)
        logger.debug('data saved successfully at %s',file_path)
    except Exception as e:
        logger.error('Error saving data to %s: %s',file_path,e)
        raise
        
if __name__=="__main__":
    main()