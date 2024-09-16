import flwr as fl
from flwr.server.strategy import FaultTolerantFedAvg
import numpy as np
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, FitRes
import pandas as pd
from datetime import datetime
from flwr.server.history import History

# Function to aggregate fit metrics (e.g., average rewards)
def fit_metrics_aggregation_fn(metrics):
    aggregated_metrics = {}
    # Aggregate each metric across all clients
    for metric in metrics[0][1].keys():
        aggregated_metrics[metric] = np.mean([m[1][metric] for m in metrics])
    return aggregated_metrics

# Function to aggregate evaluation metrics
def evaluate_metrics_aggregation_fn(metrics):
    aggregated_metrics = {}
    # Aggregate each metric across all clients
    for metric in metrics[0][1].keys():
        aggregated_metrics[metric] = np.mean([m[1][metric] for m in metrics])
    return aggregated_metrics

class RLServerStrategy(FaultTolerantFedAvg):
    def __init__(self):
        # Initialize the strategy with the custom aggregation functions
        super().__init__(fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
                         evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn)
        self.history = History()  # Initialize the history for tracking metrics and losses

    def configure_fit(self, server_round, parameters, client_manager):
        # Configure fit for the server round, using the parent class's method
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        # Print a message if there are failures during aggregation
        if failures:
            print(f"Aggregation failed for {len(failures)} clients.")
        
        # Convert the client parameters to NumPy arrays
        values_aggregated = [
            parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results
        ]
        
        # Initialize lists to store aggregated Q-tables, states, and actions
        all_q_tables, all_states, all_actions = [], [], []
    
        for q_tables, states, actions in values_aggregated:
            all_q_tables.append(q_tables)
            all_states.append(states)
            all_actions.append(actions)
    
        if not all_q_tables or not all_states or not all_actions:
            raise ValueError("One of the aggregated lists is empty. Check the client results.")
    
        # Concatenate the aggregated data
        aggregated_q_tables = self.concatenate_q_tables(all_q_tables)
        aggregated_states = self.concatenate_states(all_states)
        aggregated_actions = self.concatenate_actions(all_actions)
    
        # Verify that the aggregated data is structured correctly
        assert aggregated_q_tables.ndim > 0, "Aggregated Q-tables should not be scalar"
        assert aggregated_states.ndim > 0, "Aggregated states should not be scalar"
        assert aggregated_actions.ndim > 0, "Aggregated actions should not be scalar"
    
        # Convert aggregated data to a DataFrame and save as CSV (currently commented out)
        aggregated_data = {
            'Q-tables': aggregated_q_tables.tolist(),
            'States': aggregated_states.tolist(),
            'Actions': aggregated_actions.tolist()
        }
        df = pd.DataFrame.from_dict(aggregated_data, orient='index').transpose()
    
        # Convert 'Q-tables' column from lists to strings for deduplication
        df['Q-tables'] = df['Q-tables'].apply(lambda x: str(x))
    
        # Group by 'Q-tables' and calculate the mean for 'States' and 'Actions'
        df_aggregated = df.groupby('Q-tables').agg({'States': 'mean', 'Actions': 'mean'}).reset_index()
    
        # Convert 'Q-tables' column back from strings to lists
        df_aggregated['Q-tables'] = df_aggregated['Q-tables'].apply(lambda x: eval(x))
    
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Save the aggregated DataFrame to CSV (currently commented out)
        # df_aggregated.to_csv(f"ServerData/aggregated_results_{timestamp}.csv", index=False)
    
        # Combine all parameters into a single list
        aggregated_parameters = [
            np.array(df_aggregated['Q-tables'].tolist()), 
            np.array(df_aggregated['States'].tolist()), 
            np.array(df_aggregated['Actions'].tolist())
        ]
    
        # Convert aggregated parameters back to the format expected by the framework
        aggregated_parameters_ndarrays = ndarrays_to_parameters(aggregated_parameters)
    
        # Calculate aggregated metrics (e.g., mean reward)
        metrics_aggregated = {"mean_reward": np.mean([fit_res.metrics["reward"] for _, fit_res in results])}
        
        # Add losses and metrics to history
        self.history.add_loss_distributed(server_round, metrics_aggregated["mean_reward"])
        self.history.add_metrics_distributed_fit(server_round, metrics_aggregated)

        return aggregated_parameters_ndarrays, metrics_aggregated

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
    
        # Add losses and metrics to history
        self.history.add_loss_centralized(server_round, aggregated_loss)
        self.history.add_metrics_centralized(server_round, aggregated_metrics)
    
        return aggregated_loss, aggregated_metrics
    
    def save_history(self, file_path: str) -> None:
        """Save the history to a single CSV file."""
        # Convert history data to DataFrames
        df_losses_distributed = pd.DataFrame(self.history.losses_distributed, columns=['Round', 'Distributed Loss'])
        df_losses_centralized = pd.DataFrame(self.history.losses_centralized, columns=['Round', 'Centralized Loss'])
        
        # Convert metrics dictionaries to DataFrames and add Round column
        df_metrics_distributed_fit = pd.DataFrame([(round_, key, value) for key, values in self.history.metrics_distributed_fit.items() for round_, value in values], columns=['Round', 'Metric', 'Value']).pivot(index='Round', columns='Metric', values='Value').reset_index()
        df_metrics_distributed = pd.DataFrame([(round_, key, value) for key, values in self.history.metrics_distributed.items() for round_, value in values], columns=['Round', 'Metric', 'Value']).pivot(index='Round', columns='Metric', values='Value').reset_index()
        df_metrics_centralized = pd.DataFrame([(round_, key, value) for key, values in self.history.metrics_centralized.items() for round_, value in values], columns=['Round', 'Metric', 'Value']).pivot(index='Round', columns='Metric', values='Value').reset_index()
        
        # Merge all DataFrames on the 'Round' column
        df_combined = df_losses_distributed.merge(df_losses_centralized, on='Round', how='outer')
        df_combined = df_combined.merge(df_metrics_distributed_fit, on='Round', how='outer')
        df_combined = df_combined.merge(df_metrics_distributed, on='Round', how='outer')
        df_combined = df_combined.merge(df_metrics_centralized, on='Round', how='outer')
        
        # Save to a single CSV file
        df_combined.to_csv(file_path, index=False)
    
    # Create the FedAvg strategy
    strategy = RLServerStrategy()
    
    # Start Flower server with the custom strategy
    fl.server.start_server(
        server_address="localhost:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=2),
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Define the file path for saving the history
    file_path = f"ServerHistory/history_{timestamp}.csv"
    
    # Save the history after training
    strategy.save_history(file_path)
    
    strategy.save_history(file_path)

