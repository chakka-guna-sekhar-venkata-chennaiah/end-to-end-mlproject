import pandas as pd
import numpy as np 
from source.loggers import logging
from source.exception import custom_exception
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
import sys
from source.utlis import save_obj

import os

@dataclass
class transformation_config:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class transformation:
    def __init__(self):
        self.transformation_config=transformation_config()

    def get_transformer_object(self):
        try:
            num_features=['sepal_length','sepal_width','petal_length','petal_width']
            num_pipeline=Pipeline(
                steps=[
                    ('scaling',StandardScaler())
                ]
            )
            logging.info('Numerical Encoding is completed')

            preprocessor=ColumnTransformer(
                
                    [('num_pipeline',num_pipeline,num_features)]
                    
            )

            return preprocessor
        except Exception as e:
            raise custom_exception(e,sys)
    

    def initiate_transformer_object(self,train_path,test_path):

        try:
                
            train_data=pd.read_csv(train_path)
            test_data=pd.read_csv(test_path)

            logging.info('Reading the training as testing data')
            preprocessor_obj=self.get_transformer_object()

            target_column='species'

            input_feature_train_df=train_data.drop(columns=[target_column],axis=1)
            target_deature_train_df=train_data[target_column]

            input_feature_test_df=test_data.drop(columns=[target_column],axis=1)
            target_deature_test_df=test_data[target_column]

            logging.info(
                f'Successfully applied the preprocessing object on training and testing datasets'
                )

            input_feature_train_array=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_array=preprocessor_obj.transform(input_feature_test_df)

            train_array=np.c_[
                input_feature_train_array,np.array(target_deature_train_df)
            ]

            test_array=np.c_[
                input_feature_test_array,np.array(target_deature_test_df)
            ]

           

            logging.info('Saved the pickle object')
            
            save_obj(

            file_path=self.transformation_config.preprocessor_obj_file_path,
            obj=preprocessor_obj
            )

            return (
                train_array,
                test_array,
                self.transformation_config.preprocessor_obj_file_path
                )
        except Exception as e:
            raise custom_exception(e,sys)

