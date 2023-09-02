import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exceptions import CustomException
from src.logger import logging
from src.utils import save_object

import os

# Esta classe define um caminho padrão para se encontrar on dados serealizados
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        Realiza a transformação dos conjuntos de dados
        
        '''
        try:

            #Variaveis númericas do dataset
            numerical_columns = ["writing_score", "reading_score"]

            #Variaveis categorivas do modelo
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Os códigos a seguir cria pipelines para tranformar/tratar os dados. Similar ao Tidymodel
            num_pipeline= Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_enconder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            # Informção para o logging, bom para acompnhar os processo
            logging.info('Colunas categoricas codificadas.')
            logging.info('Colunas numericas padronizadas e normalizadas.')


            # Função responsavel por executar os pipelinis
            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns),
                    ('cat_pipeline',cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('Leitura completa dos dados de entrada')

            logging.info('Retirada dos objetos processados')

            preprocessing_obj=self.get_data_transformer_object()

            target_column='math_score'
            numerical_columns=['writing_score','rading_score']

            input_feature_train_df=train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df=train_df[target_column]

            input_feature_test_df=test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df=test_df[target_column]

            logging.info(
                f"Aplicando o processamento dos dados dos conjuntos de dados de treino e teste"
            )

            # Transformação das colunas de feature/variveis independentes
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            #Concatenção das colunas alvos com a coluna das features transformadas  
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info(f'Salvo o processamento do objeto')

            #Salva os objetos gerados em uma formato serializado, usando a biblioteca dill
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        



        except Exception as e:
            raise CustomException(e,sys)
