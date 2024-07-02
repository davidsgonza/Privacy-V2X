#!/bin/bash

# Número de clientes a ejecutar
NUM_CLIENTS=3

# Ejecutar cada cliente en un proceso separado
for ((i=0; i<=$NUM_CLIENTS; i++)); do
    python client.py --client_id $i &
done

# Esperar a que todos los procesos terminen
wait


