import traci
import traci.constants as tc
import time

# Iniciar la conexión con SUMO
traci.start(["/home/esteban/Escritorio/SUMO/sumo-git/bin/sumo", "-c", "osm.sumocfg"])

# Archivo de salida
archivo_salida = "resultados_simulacion.txt"
while traci.simulation.getMinExpectedNumber() > 0:
    for veh_id in traci.vehicle.getIDList():
        position = traci.vehicle.getPosition(veh_id)
        CO2 = traci.vehicle.getCO2Emission(veh_id)
        print(f"Vehículo {veh_id}; Nivel CO2: {CO2} mg/s")
    traci.simulationStep()
# Obtener información de la simulación al finalizar
traci.close()
