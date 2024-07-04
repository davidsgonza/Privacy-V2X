import flwr as fl
from flwr.server.strategy import FaultTolerantFedAvg
import numpy as np
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, FitRes
import pandas as pd
from datetime import datetime

# Función de agregación para las métricas de ajuste (fit)
def fit_metrics_aggregation_fn(metrics):
    aggregated_metrics = {}
    for metric in metrics[0][1].keys():
        aggregated_metrics[metric] = np.mean([m[1][metric] for m in metrics])
    return aggregated_metrics

# Función de agregación para las métricas de evaluación (evaluate)
def evaluate_metrics_aggregation_fn(metrics):
    aggregated_metrics = {}
    for metric in metrics[0][1].keys():
        aggregated_metrics[metric] = np.mean([m[1][metric] for m in metrics])
    return aggregated_metrics

class RLServerStrategy(FaultTolerantFedAvg):
    def __init__(self):
        super().__init__(fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
                         evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn)

    def configure_fit(self, server_round, parameters, client_manager):
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        if failures:
            print(f"Aggregation failed for {len(failures)} clients.")
        
        # Convertir los parámetros a arrays de NumPy
        values_aggregated = [
            parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results
        ]
        
        all_q_tables, all_states, all_actions = [], [], []
    
        for q_tables, states, actions in values_aggregated:
            all_q_tables.append(q_tables)
            all_states.append(states)
            all_actions.append(actions)
    
        if not all_q_tables or not all_states or not all_actions:
            raise ValueError("One of the aggregated lists is empty. Check the client results.")
    
        aggregated_q_tables = self.concatenate_q_tables(all_q_tables)
        aggregated_states = self.concatenate_states(all_states)
        aggregated_actions = self.concatenate_actions(all_actions)
    
        # Verificando que los datos agregados estén estructurados correctamente
        assert aggregated_q_tables.ndim > 0, "Aggregated Q-tables should not be scalar"
        assert aggregated_states.ndim > 0, "Aggregated states should not be scalar"
        assert aggregated_actions.ndim > 0, "Aggregated actions should not be scalar"
    
        # Convertir datos agregados a un DataFrame y guardar como CSV
        aggregated_data = {
            'Q-tables': aggregated_q_tables.tolist(),
            'States': aggregated_states.tolist(),
            'Actions': aggregated_actions.tolist()
        }
        df = pd.DataFrame.from_dict(aggregated_data, orient='index').transpose()
    
        # Convertir la columna 'Q-tables' de listas a cadenas para poder identificar duplicados
        df['Q-tables'] = df['Q-tables'].apply(lambda x: str(x))
    
        # Agrupar por 'Q-tables' y calcular la media de 'States' y 'Actions'
        df_aggregated = df.groupby('Q-tables').agg({'States': 'mean', 'Actions': 'mean'}).reset_index()
    
        # Convertir de nuevo la columna 'Q-tables' a listas
        df_aggregated['Q-tables'] = df_aggregated['Q-tables'].apply(lambda x: eval(x))
    
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df_aggregated.to_csv(f"ServerData/aggregated_results_{timestamp}.csv", index=False)
    
        # Combinar todos los parámetros en una sola lista
        aggregated_parameters = [
            np.array(df_aggregated['Q-tables'].tolist()), 
            np.array(df_aggregated['States'].tolist()), 
            np.array(df_aggregated['Actions'].tolist())
        ]
    
        # Convertir los parámetros agregados a ndarrays si es necesario
        #aggregated_parameters_ndarrays = ndarrays_to_parameters(aggregated_parameters)
    
        # Calcular métricas agregadas (ejemplo: recompensa media)
        metrics_aggregated = {"mean_reward": np.mean([fit_res.metrics["reward"] for _, fit_res in results])}
    
        return ndarrays_to_parameters(aggregated_parameters), metrics_aggregated

    def concatenate_q_tables(self, q_tables_list):
        if not q_tables_list:
            raise ValueError("No Q-tables to aggregate.")
        
        return np.concatenate(q_tables_list, axis=0)

    def concatenate_states(self, states_list):
        if not states_list:
            raise ValueError("No states to aggregate.")
        
        return np.concatenate(states_list, axis=0)

    def concatenate_actions(self, actions_list):
        if not actions_list:
            raise ValueError("No actions to aggregate.")
        
        return np.concatenate(actions_list, axis=0)

    def configure_evaluate(self, server_round, parameters, client_manager):         
        return super().configure_evaluate(server_round, parameters, client_manager)
    
    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None

        total_loss = 0.0
        total_examples = 0
        total_waiting_time = 0.0
        total_CO2_emissions = 0.0

        # Print results for debugging
        print(f"Results received in aggregate_evaluate: {results}")

        
        for client_proxy, evaluate_res in results:
            # Print individual result for debugging
            print(f"Individual result: {evaluate_res}")

            loss = evaluate_res.loss
            num_examples = evaluate_res.num_examples
            metrics = evaluate_res.metrics
            
            total_loss += loss * num_examples
            total_examples += num_examples

            waiting_time = metrics.get("average_waiting_time", 0.0)
            total_waiting_time += waiting_time * num_examples
            
            CO2_emissions = metrics.get("average_CO2_emissions", 0.0)
            total_CO2_emissions += CO2_emissions * num_examples


        if total_examples == 0:
            return None

        aggregated_loss = total_loss / total_examples
        aggregated_waiting_time = total_waiting_time / total_examples
        aggregated_CO2_emissions = total_CO2_emissions / total_examples
        aggregated_metrics_WT = {"average_waiting_time": aggregated_waiting_time}
        aggregated_metrics_CO2 = {"average_CO2_emissions": aggregated_CO2_emissions}
        aggregated_metrics = {**aggregated_metrics_WT, **aggregated_metrics_CO2}

        return aggregated_loss, aggregated_metrics

# Inicia el servidor de Flower con la estrategia personalizada
fl.server.start_server(
    server_address="localhost:8080",
    strategy=RLServerStrategy(),
    config=fl.server.ServerConfig(num_rounds=80),
)

