#!/bin/bash

# Número de clientes a ejecutar
NUM_CLIENTS=3

# Obtener el número de núcleos de CPU disponibles
NUM_CORES=$(nproc)
echo "Número de núcleos de CPU disponibles: $NUM_CORES"

# Número de núcleos a asignar por cliente (ajustar según sea necesario)
CORES_PER_CLIENT=4

# Ejecutar cada cliente en un proceso separado
for ((i=0; i<=$NUM_CLIENTS; i++)); do
    start_core=$(( (i * CORES_PER_CLIENT) % NUM_CORES ))
    end_core=$(( start_core + CORES_PER_CLIENT - 1 ))

    # Asegurarse de que el rango de núcleos no exceda el número disponible
    if (( end_core >= NUM_CORES )); then
        end_core=$(( NUM_CORES - 1 ))
    fi

    echo "Ejecutando cliente $i en núcleos $start_core-$end_core"
    taskset -c $start_core-$end_core python client.py --client_id $i &
done

# Esperar a que todos los procesos terminen
wait
