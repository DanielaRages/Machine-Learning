                #IMPORTAMOS LAS LIBRERÍAS NECESARIAS
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, precision_score
import numpy as np
from scipy.spatial import distance
from sklearn.ensemble import RandomForestRegressor

                # LECTURA DE DATOS
url = 'https://raw.githubusercontent.com/Deianira95/Machine-Learning/main/imdb_top_1000.csv'
data = pd.read_csv(url)

                # CREAR UN DATAFRAME PARA VISUALIZAR LOS DATOS
df= pd.DataFrame(data)

                # VISUALIZACIÓN
print("\nDATOS PUROS:\n")
print(data.head(10))

                # EXPLORACIÓN INICIAL DE DATOS

#Gráfico de géneros
plt.figure(figsize=(15,8))
df['Genre'].value_counts()[0:10].plot.bar()

#Gráfico de ganancias y rating
sns.relplot(data = df, x = df['Gross'], y = df['IMDB_Rating'], sizes = (20,200), alpha = .5, aspect = 2, color = '#06837f')
plt.title('Relación ganancias - rating', fontsize = 15, weight = 600, color = '#333d29')

                # PREPROCESAMIENTO

                # LIMPIEZA DE DATOS
# Verificamos que no haya celdas vacías
data.isnull().any()
# Eliminamos la columna Poster Link
data = data.drop('Poster_Link',axis=1)

# Eliminamos las filas que contienen datos NaN
data = data.dropna()

# Eliminamos la fila que contiene "passed"
data = data.drop(data[data["Certificate"]== "Passed"].index)

                # VISUALIZACIÓN
print("\nDATOS LIMPIOS:\n")
print(data)

                # CONVERSIÓN DE TIPOS DE DATOS                
# Verificamos qué tipos de datos contiene
print("\nTipos de datos que contiene:\n")
print(data.dtypes)

# Convertimos el tipo de datos de la columna Gross en float
data['Gross'] = data['Gross'].str.replace(',', '').astype('float')

# Convertimos el tipo de datos de la columna Runtime en int
data['Runtime'] = data['Runtime'].apply(lambda text: text.split()[0]).astype('int')

# Convertimos el tipo de datos Meta_score a float
data['Meta_score'] = data['Meta_score'].astype('float')

# Verificamos qué tipos de datos contiene
print("\nVerificación de tipos de datos:\n")
print(data.dtypes)

                # NORMALIZACIÓN
# Normalizamos la columna Runtime
normRuntime = (data['Runtime'] - data['Runtime'].min()) / (data['Runtime'].max() - data['Runtime'].min())

# Normalizamos la columna Gross
normGross = (data['Gross'] - data['Gross'].min()) / (data['Gross'].max() - data['Gross'].min())

# Normalizamos la columna IMDb_Rating
normIMDb_Rating = (data['IMDB_Rating'] - data['IMDB_Rating'].min()) / (data['IMDB_Rating'].max() - data['IMDB_Rating'].min())

# Normalizamos la columna No_of_votes
normNo_of_Votes= (data['No_of_Votes'] - data ['No_of_Votes'].min()) / (data['No_of_Votes'].max()-data['No_of_Votes'].min())

                # VISUALIZACIÓN
print("\nDATOS NORMALIZADOS:\n")
print(data)

data.to_csv(url, sep=",")

# RECOMENDACIÓN

def valores(x):
    lista = []
    for index, row in data.iterrows():
        chequeo = row[x]
    
        for c in chequeo:
            if c not in lista:
                lista.append(c)
    return lista
listageneros = valores('Genre')
listageneros[:]

def listaBinaria(listaAComparar, valores):
    listaBinaria = []
    
    for valor in valores:
        if valor in listaAComparar:
            listaBinaria.append(1)
        else:
            listaBinaria.append(0)
    return listaBinaria
            
data['generos_binarios'] = data['Genre'].apply(lambda x: listaBinaria(x,listageneros))
data['generos_binarios'].head()

data['Director'] = data['Director'].str.strip('[]').str.replace(' ','').str.replace("'",'')
data['Director'] = data['Director'].str.split(',')

listaDirectores = valores('Director')

data['directores_binarios'] = data['Director'].apply(lambda x: listaBinaria(x,listaDirectores))
data['directores_binarios'].head()

data['Star1'] = data['Star1'].str.strip('[]').str.replace(' ','').str.replace("'",'')
data['Star1'] = data['Star1'].str.split(',')

data['Star1']

listaActores = valores('Star1')
listaActores[:]

data['actores_binarios'] = data['Star1'].apply(lambda x: listaBinaria(x,listaActores))
data['actores_binarios'].head()



def similaridad(movieId1,movieId2):
        a = data.iloc[movieId1]
        b = data.iloc[movieId2]
    
        generoA = a['generos_binarios']
        generoB = b['generos_binarios']
        
        directorA = a['directores_binarios']
        directorB = b['directores_binarios']

        actorA = a['actores_binarios']
        actorB = b['actores_binarios']
        
        generoDistancia = distance.cosine(generoA, generoB)
        directoresDistancia = distance.cosine(directorA, directorB)
        actoresDistancia =  distance.cosine(actorA, actorB)
        distancia = generoDistancia + directoresDistancia + actoresDistancia
        return distancia
        
data['id_peliculas'] = data.index

def get_nombre(pelicula_id):
    target_df = data.loc[data['id_peliculas'] == pelicula_id]
    return target_df['Series_Title'].iloc[0]
print(get_nombre(1))

def get_pelicula_id(pelicula_nombre):
    target_df = data.loc[data['Series_Title'] == pelicula_nombre]
    return target_df['id_peliculas']


import operator
def recomendacion():
    name = input('Ingrese el nombre de una película: ')
    if (data['Series_Title'].eq(name)).any(): #se fija si la pelicula se encuentra en el dataset
        ide = get_pelicula_id(name)
        fila_pelicula = data.loc[ide] #almacena la fila de la pelicula con el ide especificado
  
        def getNeighbors(baseMovie, K):
            distances = []
        
            for index, movie in data.iterrows():
                if movie['id_peliculas'] != baseMovie['id_peliculas'].values[0]:
                    dist = similaridad(baseMovie['id_peliculas'].values[0], movie['id_peliculas'])
                    distances.append((movie['id_peliculas'], dist)) #añade a la lista la tupla del id de las peliculas mas cercanas y su distancia
        
            distances.sort(key=operator.itemgetter(1)) #ordena la tupla en base a su distancia
            neighbors = []
            for x in range(K):
                    neighbors.append(distances[x])
            return neighbors #agrega a neighbors las K peliculas mas parecidas (los vecinos)
    

        print('\n')
        K = 10
        neighbors = getNeighbors(fila_pelicula, K)
        print('Las' , K, 'Peliculas recomendadas: \n')
        for neighbor in neighbors:    
            print( data.iloc[neighbor[0]][0] + " ("+ (data.iloc[neighbor[0]][1]) + ")")
            #agarra el indice del vecino y lo busca en la fila de nombres del data y luego su año
    else:
        print("La película no se encuentra en el Dataset")
  
recomendacion()
