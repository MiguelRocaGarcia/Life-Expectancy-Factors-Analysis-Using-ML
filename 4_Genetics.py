#!/usr/bin/python
# -*- coding: iso-8859-1 -*-
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
import tensorflow as tf
from keras import backend as K
import requests

df = pd.read_csv('https://raw.githubusercontent.com/MiguelRocaGarcia/DatasetTFG/master/PreprocessedDataset.csv', sep=',')

df_filled = pd.read_csv('https://raw.githubusercontent.com/MiguelRocaGarcia/DatasetTFG/master/FilledDataset.csv',sep=',')
scalerLE = StandardScaler()
scalerLE.fit(df_filled['Life Expectancy'].values.reshape(-1,1))

#Aplicamos las transformaciones de estandarización y Yeo-Johnson a todo el conjunto de datos para obtener el scaler_total y power_YJ
df_transformed = df_filled.copy()
df_transformed.drop(columns=['Country'], inplace=True)
df_transformed.insert(loc=1,column='Female',value=df_transformed['Gender'].apply(lambda gender: 0 if gender == 'Male' else 1))
df_transformed.insert(loc=2,column='Male',value=df_transformed['Gender'].apply(lambda gender: 0 if gender == 'Female' else 1))
df_transformed.drop(columns=['Gender'], inplace=True)
featuresScale = ['Life Expectancy', '% Death Cardiovascular','Tobacco Prevalence','Road Traffic Deaths','% Injury Deaths',
            'Government Expenditure Education', 'Government Expenditure Health', 'Diet Composition Cereals And Grains',
            'Diet Composition Fruit And Vegetables', 'Diet Composition Oils And Fats', 'Diet Calories Plant Protein',
            'Diet Calories Carbohydrates']
scaler_total = StandardScaler()
df_transformed[featuresScale] = scaler_total.fit_transform(df_transformed[featuresScale])
columnsYJ = list(df_transformed.columns)
columnsYJ.remove('Male')
columnsYJ.remove('Female')
columnsYJ.remove('Life Expectancy')
power_YJ = PowerTransformer(method='yeo-johnson', standardize=True)
power_YJ.fit(df_transformed[columnsYJ])

def r2_score_metric(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

r = requests.get('https://github.com/MiguelRocaGarcia/DatasetTFG/raw/master/NN_Model.h5', allow_redirects=True)
open('NN_Model.h5', 'wb').write(r.content)
NN_model = tf.keras.models.load_model('NN_Model.h5', compile=False)
NN_model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['mae', 'mse', r2_score_metric])

def predictLifeExpectancy(d):
  #Aplicamos las transformaciones
  data = d.copy()
  data.drop(columns=['Country'], inplace=True)
  data.insert(loc=1,column='Female',value=data['Gender'].apply(lambda gender: 0 if gender == 'Male' else 1))
  data.insert(loc=2,column='Male',value=data['Gender'].apply(lambda gender: 0 if gender == 'Female' else 1))
  data.drop(columns=['Gender'], inplace=True)
  data[featuresScale] = scaler_total.transform(data[featuresScale])
  data[columnsYJ] = power_YJ.transform(data[columnsYJ])
  data.drop(columns=['Life Expectancy'], inplace=True)
  #Calculamos la esperanza de vida mediante la Red de Neuronas
  life_expectancy = NN_model.predict(data)
  return scalerLE.inverse_transform(life_expectancy).flatten()


# Obtener los datos de un país en un año y un género:
COUNTRY = 'Spain' #Nombre del país en inglés
YEAR = 2010 # Rango dentro del intervalo [1990, 2019]
GENDER = 'Both sexes' #Posibles valores: Male, Female y Both sexes

TARGET_FIT = 1.0
ALPHABET_MIN = -5.0
ALPHABET_MAX = 5.0
N_FEATURES = df.shape[1] - 1
 


def phenotype (chromosome):
    return 'Resultado'


def fitness (chromosome):
    score = 0
    return score


parameters = { 'alphabet':[ALPHABET_MIN, ALPHABET_MAX], 'type':'floating', 'elitism':False, 'norm':True, 'chromsize':N_FEATURES, 'pmut':0.1, 'pcross':0.5, 'target':TARGET_FIT }


print('Compila')