import traci
import csv
import time

traci.close()

# Iniciar la conexión
traci.start(["/home/david/sumo-git/bin/sumo", "-c", "osm.sumocfg"])

csv_file_path = 'datos_simulacion.csv'
with open(csv_file_path, 'w', newline='') as csvfile:
    
    csv_writer = csv.writer(csvfile)

    # Cabecera del CSV
    csv_writer.writerow(['Time', 'VehicleID', 'Speed'])

    # Ejecutar la simulación 
    while traci.simulation.getMinExpectedNumber() > 0:
        current_time = traci.simulation.getCurrentTime() / 1000  # Convertir a segundos

        for veh_id in traci.vehicle.getIDList():
            speed = traci.vehicle.getSpeed(veh_id)            
            print(f"Tiempo: {current_time}, ID del Vehículo: {veh_id}, Velocidad: {speed}")

        
            csv_writer.writerow([current_time, veh_id, speed])

        traci.simulationStep()

traci.close()
