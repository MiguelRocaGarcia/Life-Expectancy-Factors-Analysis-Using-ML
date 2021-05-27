import logging

import azure.functions as func

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
import tensorflow as tf
from keras import backend as K
import requests


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    
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

    country = req.params.get('country')

    YEAR = 2010 # Rango dentro del intervalo [1990, 2019]
    GENDER = 'Both sexes' #Posibles valores: Male, Female y Both sexes

    data = df_filled[(df_filled['Country'] == COUNTRY)&(df_filled['Year'] == YEAR)&(df_filled['Gender'] == GENDER)]

    #Aplicamos las transformaciones
    data_norm = data.copy()
    data_norm.drop(columns=['Country'], inplace=True)
    data_norm.insert(loc=1,column='Female',value=data_norm['Gender'].apply(lambda gender: 0 if gender == 'Male' else 1))
    data_norm.insert(loc=2,column='Male',value=data_norm['Gender'].apply(lambda gender: 0 if gender == 'Female' else 1))
    data_norm.drop(columns=['Gender'], inplace=True)
    data_norm[featuresScale] = scaler_total.transform(data_norm[featuresScale])
    data_norm[columnsYJ] = power_YJ.transform(data_norm[columnsYJ])
    LE_VALUE = data_norm['Life Expectancy'].values[0]
    data_norm.drop(columns=['Life Expectancy'], inplace=True)
    data_norm = data_norm.values

    # le_predicted = NN_model.predict(np.array(chromosome).reshape(1, -1))[0][0]


    name = req.params.get('name')
    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get('name')

    if name:
        return func.HttpResponse(f"Hello, {name}. This HTTP triggered function executed successfully.")
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )
