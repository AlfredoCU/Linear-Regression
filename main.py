#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 00:20:15 2020

@author: alfredocu
"""

# Bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model # Técnicas clasicas.

# Regreción lineal.

# Esperanza de vida en México.
data = pd.read_csv('countries.csv')
data_mex = data[data['country'] == 'Mexico']
# data_mex.plot.scatter(x='year', y='lifeExp')


# Casteamos los datos.
x = np.asanyarray(data_mex[['year']])
y = np.asanyarray(data_mex[['lifeExp']])


# Algoritmo de la biblioteca sklearn.
model = linear_model.LinearRegression()
model.fit(x, y)
ypred = model.predict(x)
plt.scatter(x, y)
plt.plot(x, ypred, '--r')

# x son los datos.
# y son los valores deseados.
# yprev son las predicciones para esos mismos valores (y)


# Pododemos predecir la esperanza de vida de los los años que no se encuentran en la lista.
print(model.predict([[1955], [1987],[2020]]))

# El margen de error
# y - prev
# Pata quitar valores negativos.
# abs(y - yprev) # En la téoria de error debemos de pensar como valores positivos del 0 hasta el infinito.

# La media del error.
# np.mean(np.abs(y - ypred))

# O utilizar esta funsión.
# from sklearn.metrics import mean_absolute_error
# print(mean_absolute_error(y, ypred))

# O mean_squared_error Se eleba al cuadrado esa diferencia. Usado en algoritmos evolutivos.

# Este es el método a utilizar cuando se trabaja con regreciones.
from sklearn.metrics import r2_score # Medida de desempeño. 0 - 1 va bien, negativo va mal.
print(r2_score(y, ypred)) # Más cerca del 1 mejor.


# Los datos estan correlacionados, pero que no sean causales. crece x, y.
# Puede que funcione bien en 2020 o 2050, pero en 3050, puede que no.
print(model.predict([[3050]]))

# Los datos deben de ser correlacionadas y causales para evitar este error del 3050.

# Guardamos la imagen en un formato .esp para una buena calidad.
plt.savefig('Regreción_Lineal.eps', format='eps')