#!/usr/bin/python
# -*- coding: iso-8859-1 -*-


import pandas as pd
import numpy as np
import tensorflow as tf
from keras import backend as K
import pickle

df = pd.read_csv('https://raw.githubusercontent.com/MiguelRocaGarcia/DatasetTFG/master/PreprocessedDataset.csv',
                 sep=',')
df_filled = pd.read_csv('https://raw.githubusercontent.com/MiguelRocaGarcia/DatasetTFG/master/FilledDataset.csv',
                        sep=',')

# Obtener los datos de un país en un año y un género:
COUNTRY = 'Lithuania'  # Nombre del país en inglés
YEAR = 2005  # Rango dentro del intervalo [1990, 2019]
GENDER = 'Male'  # Posibles valores: Male, Female y Both sexes

MARGIN_INPUT = 5.0
W_INPUT = 0.1
W_OUTPUT = 1 - W_INPUT

TARGET_FIT = 100.0
ALPHABET_MIN = -5.0
ALPHABET_MAX = 5.0
N_FEATURES = df.shape[1] - 1

# Creamos las listas de las features que se van a transformar
columnsYJ = list(df.columns)
columnsYJ.remove('Male')
columnsYJ.remove('Female')
columnsYJ.remove('Life Expectancy')

featuresScale = ['Life Expectancy', '% Death Cardiovascular', 'Tobacco Prevalence', 'Road Traffic Deaths',
                 '% Injury Deaths',
                 'Government Expenditure Education', 'Government Expenditure Health',
                 'Diet Composition Cereals And Grains',
                 'Diet Composition Fruit And Vegetables', 'Diet Composition Oils And Fats',
                 'Diet Calories Plant Protein',
                 'Diet Calories Carbohydrates']

# Cargamos los scalers para hacer y deshacer las transformaciones
scalerLE = pickle.load(open('Data/Scalers/LE_Scaler.pkl', 'rb'))
scaler_total = pickle.load(open('Data/Scalers/Normal_Features_Scaler.pkl', 'rb'))
power_YJ = pickle.load(open('Data/Scalers/Yeo-Johnson_Scaler.pkl', 'rb'))

# Función para poder calcular el error R2 al entrenar la red de neuronas
def r2_score_metric(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return (1 - SS_res/(SS_tot + K.epsilon()) )

#Importar Red de Neuronas
NN_model = tf.keras.models.load_model('Data/Models/NN_Model.h5', compile=False)
NN_model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['mae', 'mse', r2_score_metric])

data = df_filled[(df_filled['Country'] == COUNTRY) & (df_filled['Year'] == YEAR) & (df_filled['Gender'] == GENDER)]

# Aplicamos las transformaciones
data_norm = data.copy()
data_norm.drop(columns=['Country'], inplace=True)
data_norm.insert(loc=1, column='Female', value=data_norm['Gender'].apply(lambda gender: 0 if gender == 'Male' else 1))
data_norm.insert(loc=2, column='Male', value=data_norm['Gender'].apply(lambda gender: 0 if gender == 'Female' else 1))
data_norm.drop(columns=['Gender'], inplace=True)
data_norm[featuresScale] = scaler_total.transform(data_norm[featuresScale])
data_norm[columnsYJ] = power_YJ.transform(data_norm[columnsYJ])
ORIGINAL_LE = data_norm['Life Expectancy'].values[0]
data_norm.drop(columns=['Life Expectancy'], inplace=True)
data_norm = data_norm.values[0]


def phenotype(chromosome):
    diff_input_no_mutable = np.sum(abs(data_norm[:3] - np.array(chromosome)[:3]))
    diff_input_mutable = np.sum(abs(data_norm[3:] - np.array(chromosome)[3:]))
    diff_le = NN_model.predict(np.array(chromosome).reshape(1, -1))[0][0] - ORIGINAL_LE
    return f'Diferencia de features no mutables: {diff_input_no_mutable}. \nDiferencia de features mutables: {diff_input_mutable}. \nDiferencia esperanza de vida: {diff_le}.'


def fitness(chromosome):
    #Obtenemos la similaridad entre el cromosoma y el caso elegido
    diff_input_no_mutable = np.sum(abs(data_norm[:3] - np.array(chromosome)[:3]))*100 # Penalizamos más el que se modifiquen el año y el género
    diff_input_mutable = np.sum(abs(data_norm[3:] - np.array(chromosome)[3:]))
    diff_input = diff_input_no_mutable + diff_input_mutable
    similarity_input = 1 / (diff_input + 1)

    # No beneficiamos el valor de la esperanza de vida hasta que el cromosoma se parece lo suficiente al caso elegido(margen de cambio)
    if(diff_input < MARGIN_INPUT):
        similarity_input = 1.0
        #Obtenemos el valor de la diferencia de esperanza de vida del cromosoma y la objetivo
        le_predicted = NN_model.predict(np.array(chromosome).reshape(1, -1))[0][0]
        diff_prediction = (le_predicted - ORIGINAL_LE + 1)**3
    else:
        diff_prediction = 0

    #Se pondera la importancia de la similaridad con el caso elegido y la diferencia con la esperanza de vida objetivo
    score = W_INPUT * similarity_input + W_OUTPUT * diff_prediction

    return score


parameters = {'alphabet': [ALPHABET_MIN, ALPHABET_MAX], 'type': 'floating', 'elitism': True, 'norm': True,
              'chromsize': N_FEATURES, 'pmut': 0.75, 'pcross': 0.7, 'target': TARGET_FIT}
