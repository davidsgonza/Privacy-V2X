import flwr as fl
from flwr.server.strategy import FaultTolerantFedAvg
import numpy as np
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, FitRes
import pandas as pd

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
        
        # Depuración: Imprimir tamaños y formas de los datos agregados
        # print(f"all_q_tables shapes: {[q.shape for q in all_q_tables]}")
        # print(f"all_states shapes: {[s.shape for s in all_states]}")
        # print(f"all_actions shapes: {[a.shape for a in all_actions]}")
    
        if not all_q_tables or not all_states or not all_actions:
            raise ValueError("One of the aggregated lists is empty. Check the client results.")
    
        aggregated_q_tables = self.aggregate_q_tables(all_q_tables)
        aggregated_states = self.aggregate_states(all_states)
        aggregated_actions = self.aggregate_actions(all_actions)
    
        # Depuración: Imprimir las formas de los datos después de la agregación
        # print(f"Aggregated Q-tables shape: {aggregated_q_tables.shape}")
        # print(f"Aggregated states shape: {aggregated_states.shape}")
        # print(f"Aggregated actions shape: {aggregated_actions.shape}")
    
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
        df.to_csv("aggregated_results.csv", index=False)
    
        # Combinar todos los parámetros en una sola lista
        aggregated_parameters = [aggregated_q_tables, aggregated_states, aggregated_actions]
    
        # Convertir los parámetros agregados a ndarrays si es necesario
        #aggregated_parameters_ndarrays = ndarrays_to_parameters(aggregated_parameters)
    
        # Calcular métricas agregadas (ejemplo: recompensa media)
        metrics_aggregated = {"mean_reward": np.mean([fit_res.metrics["reward"] for _, fit_res in results])}
    
        return ndarrays_to_parameters(aggregated_parameters), metrics_aggregated

    def aggregate_q_tables(self, q_tables_list):
        if not q_tables_list:
            raise ValueError("No Q-tables to aggregate.")
        
        # Obtener la forma máxima de las Q-tables
        max_shape = np.max([q.shape for q in q_tables_list], axis=0)
        
        # Rellenar las Q-tables para que todas tengan la misma forma
        padded_q_tables = [np.pad(q, ((0, max_shape[0] - q.shape[0]), (0, max_shape[1] - q.shape[1])), 'constant') for q in q_tables_list]
        
        # Depuración: Verificar las formas antes de la agregación
        #print(f"Padded Q-tables shapes: {[q.shape for q in padded_q_tables]}")
        
        return np.mean(padded_q_tables, axis=0)

    def aggregate_states(self, states_list):
        if not states_list:
            raise ValueError("No states to aggregate.")
        
        # Obtener la longitud máxima de los estados
        max_length = max(len(s) for s in states_list)
        
        # Rellenar los estados para que todos tengan la misma longitud
        padded_states = [np.pad(s, (0, max_length - len(s)), 'constant') for s in states_list]
        
        # Depuración: Verificar las formas antes de la agregación
        print(f"Padded states shapes: {[s.shape for s in padded_states]}")
        
        return np.mean(padded_states, axis=0)

    def aggregate_actions(self, actions_list):
        if not actions_list:
            raise ValueError("No actions to aggregate.")
        
        # Obtener la longitud máxima de las acciones
        max_length = max(len(a) for a in actions_list)
        
        # Rellenar las acciones para que todas tengan la misma longitud
        padded_actions = [np.pad(a, (0, max_length - len(a)), 'constant') for a in actions_list]
        
        # Depuración: Verificar las formas antes de la agregación
        print(f"Padded actions shapes: {[a.shape for a in padded_actions]}")
        
        return np.mean(padded_actions, axis=0)

    def configure_evaluate(self, server_round, parameters, client_manager):         
        return super().configure_evaluate(server_round, parameters, client_manager)
    
    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None

        total_loss = 0.0
        total_examples = 0
        total_waiting_time = 0.0

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

        if total_examples == 0:
            return None

        aggregated_loss = total_loss / total_examples
        aggregated_waiting_time = total_waiting_time / total_examples
        aggregated_metrics = {"average_waiting_time": aggregated_waiting_time}

        return aggregated_loss, aggregated_metrics

# Inicia el servidor de Flower con la estrategia personalizada
fl.server.start_server(
    server_address="localhost:8080",
    strategy=RLServerStrategy(),
    config=fl.server.ServerConfig(num_rounds=10),
)
