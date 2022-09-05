                #IMPORTAMOS LAS LIBRERÍAS NECESARIAS
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

                # LECTURA DE DATOS
url = 'https://raw.githubusercontent.com/Deianira95/Machine-Learning/main/imdb_top_1000.csv'
data = pd.read_csv(url)

                # CREAR UN DATAFRAME PARA VISUALIZAR LOS DATOS
df = pd.DataFrame(data)

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

# Convertimos el tipo de datos MEta_score a float
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

                # VISUALIZACIÓN
print("\nDATOS NORMALIZADOS:\n")
print(data)

data.to_csv(url, sep=",")
                                    
                # CLASIFICACIÓN

# Creamos la columna clasificación donde van a clasificarse las peliculas de 1 a 5 estrellas.
data.insert(15, "Clasificación", value=0, allow_duplicates=False)
print(data.dtypes)

# Reestablecemos el index para no tener errores posteriores
data.reset_index(drop=True, inplace=True)    

# Función para asignar los valores en Clasificación
index = 0
for x in data["IMDB_Rating"]:
    if x > 9.0:
        data.at[index,"Clasificación"] = 5
        index = index + 1
        
    if x == 9.0:
        data.at[index,"Clasificación"] = 4
        index = index + 1

    if x <= 8.9 and x >= 8.5:
        data.at[index,"Clasificación"] = 3
        index = index + 1
      
    if x >= 8.0 and x <= 8.4:
        data.at[index,"Clasificación"] = 2
        index = index + 1
       
    if x >= 7.6 and x <= 7.9:
        data.at[index,"Clasificación"] = 1
        index = index + 1


# Entrenamiento del modelo
X = normIMDb_Rating
y = data['Clasificación']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.7)

#Reshape
X_train = X_train.values.reshape((-1,1))
X_test = X_test.values.reshape((-1,1))
y_train = y_train.values.reshape((-1,1))
y_test = y_test.values.reshape((-1,1))

# Entrenamiento
rl = LinearRegression()
rl.fit(X_train, y_train)

# Evaluación
y_test = rl.predict(X_train)

# Verificación con MSE y R2
print("Coeficientes: ", rl.coef_)
print("Variable independiente: ", rl.intercept_)
print("Mean Squared Error: %.2f" % mean_squared_error(y_train, y_test))
print("Variación de puntaje: %.2f" % r2_score(y_train, y_test))

# Visualizamos
sns.relplot(x = data['IMDB_Rating'], y = data['Clasificación'], sizes = (20,200), alpha = .5, aspect = 2, color = '#06837f')

plt.plot(y_train, color="blue", label="Valores originales de clasificación")
plt.plot(y_test, color="red", label="Predicción de clasificación")
plt.title("Valores originales vs predicción - Clasificación")
plt.xlabel("IMDB_Rating")
plt.ylabel("Clasificación")
plt.legend()
plt.show()