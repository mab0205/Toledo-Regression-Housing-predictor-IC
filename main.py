# %%
import numpy as np 
import pandas as pd 
from IPython.display import HTML
import base64
import matplotlib.pyplot as plt
import seaborn as sns

# %%
from Results.results_metrics import * 
from Models.GradientBoostingRegressor  import *
from Models.LinearRegression import *
from Models.SVM import * 
from Models.XGBRegressor import *
from statics.matplot_imgs import *

# %% [markdown]
# ### Funcao cria link para imprimir dataframe

# %%
def create_download_link(df, title = "Download CSV file", filename = "data_atualizada.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

# %%
column_names = ['Tipo', 'Bairro', 'Dormitorios', 'Banheiros', 'Garagens', 'A. Terreno M²', 'A. Construída M²', 'Alarme', 'ArCondicionado', 'Armarios', 'BoxBlindex', 'CercaEletrica', 'Churrasqueira', 'Cozinha', 'Escritorio', 'HomeTheater', 'Iluminacao', 'Lavabo', 'Piscina', 'Porcelanato', 'PortaoEletronico', 'SalaEstar', 'SalaJantar', 'SemiMobiliado', 'Varanda', 'Precio']

data = pd.read_csv('C:/Users/gaboh/UTFPR/GitHub/Toledo-Regression-Housing-predictor-IC/data_17-08-2023 - Hoja 2.csv', header=None, delimiter=',', names=column_names)
data = data.iloc[1:]  # Drop the first row
print(data.head(5))
print(np.shape(data))

# %%
# Reemplazar "M²" por cadena vacía en las columnas numéricas
numeric_columns = ['Dormitorios', 'Banheiros', 'Garagens', 'A. Terreno M²', 'A. Construída M²', 'Precio']
data[numeric_columns] = data[numeric_columns].replace('M²', '', regex=True)

# Convertir las características a valores numéricos, reemplazando valores inválidos por 0
numeric_features = ['Dormitorios', 'Banheiros', 'Garagens', 'A. Terreno M²', 'A. Construída M²', 'Alarme', 'ArCondicionado',
                    'Armarios', 'BoxBlindex', 'CercaEletrica', 'Churrasqueira', 'Cozinha', 'Escritorio',
                    'HomeTheater', 'Iluminacao', 'Lavabo', 'Piscina', 'Porcelanato', 'PortaoEletronico',
                    'SalaEstar', 'SalaJantar', 'SemiMobiliado', 'Varanda', 'Precio']

data[numeric_columns] = data[numeric_columns].replace(' Alqueires Paulista', '', regex=True)
data[numeric_columns] = data[numeric_columns].replace('Consulte-nos', '', regex=True)


# %% [markdown]
# # Tratamento de Nan's 
# * Subtituimos os 0 da area para poder aproximar um valor 

# %%
# Replace NaNs with a new value
data['A. Terreno M²'].replace('0', np.nan,  inplace=True)
data['A. Construída M²'].replace('0', np.nan, inplace=True)

# Reemplazar espacios vacíos por NaN
data.replace('', np.nan, inplace=True)
# Convertir las columnas a float64
data[numeric_features] = data[numeric_features].astype('float64')

#quantidade de  Nans
data.isna().sum()

# %% [markdown]
# # # **Encoder** 
# * Para categorical features cria um valor numerico para nome dentro do conjunto
# *  label encoding, we might confuse our model into thinking that a column has data with some kind of order or hierarchy, when we clearly don’t have it. To avoid this, we ‘OneHotEncode’ that column.

# %%
from sklearn.preprocessing import OneHotEncoder

categorical_columns = ['Tipo', 'Bairro']

encoder = OneHotEncoder(sparse=False)

# Ajustar y transformar las columnas categóricas
encoded_categorical_data = encoder.fit_transform(data[categorical_columns])
feature_labels = encoder.get_feature_names_out(input_features=categorical_columns)

encoded_df = pd.DataFrame(encoded_categorical_data, columns=feature_labels)
data.drop(columns = categorical_columns, inplace=True)

# Reset index data
data.reset_index(drop=True, inplace=True)

# concatenar 2 datasets 
for column in encoded_df.columns:
    data[column] = encoded_df[column]

print(np.shape(data))
print(data.isna().sum())

# %% [markdown]
# # TRAIN TESTE
# * Dividimos o preco entre as amostras que nao tem valores vazios, pegamos a proporcao 85-15

# %%
from sklearn.model_selection import train_test_split

data_aux = data.dropna()
print(np.shape(data_aux))
train_data, test_data = train_test_split(data_aux, test_size=0.40, random_state=23)
print(np.shape(test_data))

# %%
# Filtrar data eliminando las filas con índices presentes en test_data
X_train_data = data.drop(index = test_data.index)

# Imprimir las formas de los DataFrames resultantes
print( np.shape(X_train_data))
print( np.shape(test_data))

# %% [markdown]
# ## Imputer (prencher espaços NULL)
# #### **Vamos a importar IterativeImputer de forma multivariado para conseguir prencheer os dados faltantes usando regresoes e auto-ajustes **

# %%
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

column_names = ['Dormitorios', 'Banheiros', 'Garagens', 'A. Terreno M²', 'A. Construída M²' ]
# initial_strategy{‘mean’, ‘median’, ‘most_frequent’, ‘constant’}, default=’mean’
# imputation_order{‘ascending’, ‘descending’, ‘roman’, ‘arabic’, ‘random’}, default=’ascending’

imputer = IterativeImputer(
    initial_strategy='most_frequent',
    estimator=RandomForestRegressor(),
    imputation_order='ascending'
)
X_train_data[column_names] = imputer.fit_transform(X_train_data[column_names])

# Crear un DataFrame con las columnas imputadas
aux = pd.DataFrame(X_train_data[column_names], columns=column_names)

# Eliminar las columnas originales de Tipo y Bairro del DataFrame original
X_train_data = X_train_data.drop(columns=column_names)

# Concatenar el DataFrame original con las características imputadas en 'aux'
data_filled_nan_content = X_train_data.join(aux)

#resultado
print(data_filled_nan_content.isna().sum())
print(np.shape(data_filled_nan_content))
print("Número de valores negativos:", (data_filled_nan_content < 0).sum().sum())

# %% [markdown]
# ****
# ## **Descricao general do DataFrame**
# * Verficacao do tipo de variaveis, quantidade de Null e descricao de cada feature ( mean , min, % , etc)

# %%
print("Descripción general del DataFrame:")
print(data_filled_nan_content.info())
print(data_filled_nan_content[column_names].describe())

# %% [markdown]
# # **Queremos analizar a quantidade de features e a corelacao entre elas**
# * Podemos observar que Dormitorios Banheiros Garagens sao altamente relacionados

# %%
# Filtrar solo las columnas numéricas
numeric_columns = ['Dormitorios', 'Banheiros', 'Garagens', 'A. Terreno M²', 'A. Construída M²', 'Alarme', 'ArCondicionado',
                    'Armarios', 'BoxBlindex', 'CercaEletrica', 'Churrasqueira', 'Cozinha', 'Escritorio',
                    'HomeTheater', 'Iluminacao', 'Lavabo', 'Piscina', 'Porcelanato', 'PortaoEletronico',
                    'SalaEstar', 'SalaJantar', 'SemiMobiliado', 'Varanda']

# Crear un subconjunto con las columnas numéricas
numeric_data = data_filled_nan_content[numeric_columns]

coMatrix(numeric_data)

# %%
# plot stats para features
plotFeatures(data_filled_nan_content, numeric_columns)

# %% [markdown]
# # Imprimir csv atualizado 

# %%
def create_download_link(df, title = "Download CSV file", filename = "data_atualizada.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

df_formatted = data_filled_nan_content.round(2)
create_download_link(df_formatted)

# %% [markdown]
# ## Outliers
# * Vamos tentar detetar e drop de outliers das principais features do nosso dataset 

# %%
# Box plot para identificar outliers 
boxplot(data_filled_nan_content)

# %% [markdown]
# - A partir dos graficos, definimos constantes como criterios para defirmos outliers

# %%
outlier_criteria = {
    'Banheiros': 6,    
    'Garagens': 5, 
}

# Filtrar el DataFrame para eliminar filas con outliers
for column, threshold in outlier_criteria.items():
    data_ready = data_filled_nan_content[data_filled_nan_content[column] <= threshold]

print(np.shape(data_ready)) 

# %% [markdown]
# # Normizando os dados

# %%
from sklearn.preprocessing import MinMaxScaler

# Obtener una lista de los nombres de las columnas en tus datos
column_names = data_ready.columns.tolist()

# Realizar la normalización min-max en train_data_ready
scaler = MinMaxScaler()
train_data_ready = data_ready.copy()
train_data_ready[column_names] = scaler.fit_transform(train_data_ready[column_names])

# Realizar la normalización min-max en test_data_ready
test_data_ready = test_data.copy()
test_data_ready[column_names] = scaler.transform(test_data_ready[column_names])

print(np.shape(train_data_ready))
print(np.shape(test_data_ready))
#boxplot(train_data_ready)


# %%
print(train_data_ready['Precio'].describe())
print(test_data_ready['Precio'].describe())

# %% [markdown]
# # Predicting
# ### Regressão linear
# ### SVM (Support Vector Machines)
# ### Gradient Boosting

# %% [markdown]
# > Output = Precios
# * extraimos o preco para usar-lo como target

# %%
#extraimos a coluna precio do dataset de treinamento 
train_target_precio = train_data_ready['Precio'] 
train_data_ready = train_data_ready.drop('Precio', axis=1) 

#extraimos a coluna precio do dataset de teste 
test_target_precio = test_data_ready['Precio'] 
test_data = test_data_ready.drop('Precio', axis=1) 


# %% [markdown]
# ****
# # Implementação Modelos

# %%
# x_test -> test_data -> df 15%
# y_test -> test_target_precio -> variavel com a coluna precio 
x_test = test_data.to_numpy()
y_test = test_target_precio.to_numpy()

# Convertir DataFrames en arrays NumPy
# x_train -> train_data_ready -> df 85%
# y_train -> train_target_precio -> variavel com a coluna precio 
x_train = train_data_ready.to_numpy()
y_train = train_target_precio.to_numpy()

# %%
#EvaluationMetrics(result_linear, x_train, y_train, x_test, y_test)



