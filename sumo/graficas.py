#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

# Cargar los datos desde el archivo CSV
file_path = 'tripinfo2.csv'
df = pd.read_csv(file_path, delimiter='\t')

# Gráfico de Velocidad vs. Tiempo
plt.figure(figsize=(10, 6))
plt.scatter(df['tripinfo_depart'], df['tripinfo_arrivalSpeed'], alpha=0.5)
plt.title('Velocidad vs. Tiempo')
plt.xlabel('Tiempo de Salida')
plt.ylabel('Velocidad de Llegada')
plt.grid(True)
plt.show()

# Histograma de Retraso en la Salida
plt.figure(figsize=(10, 6))
plt.hist(df['tripinfo_departDelay'], bins=20, edgecolor='black')
plt.title('Histograma de Retraso en la Salida')
plt.xlabel('Retraso en la Salida')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()

# Gráfico de Longitud de la Ruta vs. Tiempo
plt.figure(figsize=(10, 6))
plt.scatter(df['tripinfo_depart'], df['tripinfo_routeLength'], alpha=0.5)
plt.title('Longitud de la Ruta vs. Tiempo')
plt.xlabel('Tiempo de Salida')
plt.ylabel('Longitud de la Ruta')
plt.grid(True)
plt.show()

# Gráfico de Tiempo de Espera vs. Número de Esperas
plt.figure(figsize=(10, 6))
plt.scatter(df['tripinfo_waitingCount'], df['tripinfo_waitingTime'], alpha=0.5)
plt.title('Tiempo de Espera vs. Número de Esperas')
plt.xlabel('Número de Esperas')
plt.ylabel('Tiempo de Espera')
plt.grid(True)
plt.show()

# Gráfico de Pérdida de Tiempo vs. Reruteo
plt.figure(figsize=(10, 6))
plt.scatter(df['tripinfo_rerouteNo'], df['tripinfo_timeLoss'], alpha=0.5)
plt.title('Pérdida de Tiempo vs. Reruteo')
plt.xlabel('Número de Reruteos')
plt.ylabel('Pérdida de Tiempo')
plt.grid(True)
plt.show()

# Gráfico de Velocidad Media por Tipo de Vehículo
plt.figure(figsize=(10, 6))
df_grouped = df.groupby('tripinfo_vType')['tripinfo_speedFactor'].mean().sort_values()
df_grouped.plot(kind='barh', color='skyblue')
plt.title('Velocidad Media por Tipo de Vehículo')
plt.xlabel('Velocidad Media')
plt.ylabel('Tipo de Vehículo')
plt.grid(True)
plt.show()
