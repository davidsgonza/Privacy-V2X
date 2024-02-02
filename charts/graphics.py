

import pandas as pd
import matplotlib.pyplot as plt

# Cargar el archivo CSV con punto y coma como delimitador
archivo_csv = 'tripinfo2.csv'
data = pd.read_csv(archivo_csv, delimiter=';')

# Calcular las distancias y tiempos promedio
distancias_promedio = data['tripinfo_routeLength'].mean()
tiempo_promedio = data['tripinfo_duration'].mean()
velocidad_promedio_m_s = data['tripinfo_speedFactor'].mean()
velocidad_promedio_k = velocidad_promedio_m_s * 3.6

print(f'Distancia Promedio: {distancias_promedio} metros')
print(f'Tiempo Promedio: {tiempo_promedio} segundos')
print(f'Velocidad Promedio: {velocidad_promedio_k}')


#Gráfica de barras para la distancia y el tiempo promedio
fig, ax = plt.subplots()
ax.bar(['Distancia Promedio', 'Tiempo Promedio'], [distancias_promedio, tiempo_promedio])
ax.set_ylabel('Valor Promedio')
ax.set_title('Distancia y Tiempo Promedio')
plt.show()



# Gráfica de barras para la velocidad promedio
fig, ax = plt.subplots()
ax.bar(['Velocidad Promedio'], [velocidad_promedio_k])
ax.set_ylabel('Valor Promedio')
ax.set_title('Velocidad Promedio')
plt.show()


# Gráfico Longitud de Ruta 
plt.figure(figsize=(12, 6))
plt.bar(data.index, data['tripinfo_routeLength'], color='skyblue')
plt.title('Longitud de Ruta para cada Vehículo')
plt.xlabel('Índice del Vehículo')
plt.ylabel('Longitud de Ruta')
plt.grid(axis='y')
plt.show()


# Gráfico tiempo Total
plt.figure(figsize=(12, 6))
plt.bar(data.index, data['tripinfo_duration'], color='green')
plt.title('Tiempo Total de Viaje para cada Vehículo')
plt.xlabel('Índice del Vehículo')
plt.ylabel('Tiempo Total de Viaje')
plt.grid(axis='y')
plt.show()


# Gráfico de Velocidad Promedio 
plt.figure(figsize=(12, 6))
plt.bar(data.index, data['tripinfo_speedFactor'], color='orange')
plt.title('Velocidad Promedio para cada Vehículo')
plt.xlabel('Índice del Vehículo')
plt.ylabel('Velocidad Promedio')
plt.grid(axis='y')
plt.show()

