import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as kr
import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

raw_data = pd.read_csv("https://raw.githubusercontent.com/vareladev/diplomado-ciencia-de-datos/main/modulo8/proyecto/dataset/pacientes_covid_mx_2022.csv ")
pred_data = pd.read_csv("https://raw.githubusercontent.com/vareladev/diplomado-ciencia-de-datos/main/modulo8/proyecto/dataset/pacientes_covid_mx_2022_predict.csv")

SUPERVISED_VARIABLE ="MUERTO"
X = np.array(raw_data.drop(SUPERVISED_VARIABLE, axis=1))
y = np.array(raw_data[[SUPERVISED_VARIABLE]])

x_train, x_test, y_train, y_test = train_test_split (X, y, test_size=0.25, random_state=42)

# Creación del modelo

# Compilación del modelo

# Etapa de aprendizaje con los datos de entrenamiento (x_train, y_train)

# Etapa de evaluación con los datos en prueba (x_test, y_test)

# Etapa de predicción con los datos de predicción (pred_data)
# predictions = model.predict(pred_data)
# print (predictions)