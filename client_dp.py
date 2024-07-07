import os
import sys
import flwr as fl
import numpy as np
import time
import argparse
import matplotlib
import matplotlib.pyplot as plt
import csv
import pandas as pd
from datetime import datetime
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, Status, Code, EvaluateRes
from typing import Dict, Union
from flwr.client.mod.localdp_mod import LocalDpMod

# Configurar matplotlib para usar el backend 'Agg'
matplotlib.use('Agg')

from sumo_rl import SumoEnvironment
from sumo_rl.exploration import EpsilonGreedy
from sumo_rl.agents import QLAgent as BaseQLAgent

os.environ['SUMO_HOME'] = '/usr/bin/sumo/'

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

# Constantes
USE_GUI = False
NUM_SECONDS = 3500
MIN_GREEN = 9
DELTA_TIME = 5
REWARD_FN = "diff-waiting-time"

class QLAgent(BaseQLAgent):
    def get_q_table(self):
        return {state: np.array(actions) for state, actions in self.q_table.items()}

    def set_q_table(self, q_table):
        self.q_table = {state: np.array(actions) for state, actions in q_table.items()}

class RLClient(fl.client.NumPyClient):
    def __init__(self, client_id, clipping_norm, sensitivity, epsilon, delta):
        self.client_id = client_id
        self.alpha = 0.1
        self.gamma = 0.99
        self.decay = 1
        self.runs = 1
        self.episodes = 1
        self.env = None
        self.ql_agents = {}
        self.evaluate_rewards = []
        self.fit_rewards = []
        self.dict_sizes = {}
        self.filled_q_tables = {}
        self.enable_logging = False
        # Create an instance of LocalDpMod
        self.local_dp_obj = LocalDpMod(clipping_norm, sensitivity, epsilon, delta)
        self.initialize_environment()
        self.initialize_dict_sizes()
        self.key = f"s{self.client_id + 1}"

    def initialize_environment(self):
        while self.env is None:
            try:
                self.env = SumoEnvironment(
                    net_file=f'nets/cliente{self.client_id}.net.xml',
                    route_file=f'nets/cliente{self.client_id}.rou.xml',
                    use_gui=USE_GUI,
                    num_seconds=NUM_SECONDS,
                    min_green=MIN_GREEN,
                    delta_time=DELTA_TIME,
                    reward_fn=REWARD_FN,
                )
            except Exception as e:
                print(f"Error initializing SUMO environment: {e}")
                time.sleep(1)

    def initialize_dict_sizes(self):
        if self.env:
            initial_states = self.env.reset()
            for ts in self.env.ts_ids:
                encoded_state = self.env.encode(initial_states[ts], ts)
                self.dict_sizes[ts] = len(encoded_state)

    def dividir_diccionario_en_arrays(self, diccionario):
        diccionarios = []
        estados = []
        acciones = []

        max_length = self.get_max_key_length(diccionario)

        for agente in diccionario.values():
            for key, value in agente.items():
                padded_key = self.pad_list(key, max_length)
                diccionarios.append(padded_key)
                estados.append(value[0])
                acciones.append(value[1])

        diccionarios = np.array(diccionarios)
        estados = np.array(estados)
        acciones = np.array(acciones)
        return diccionarios, estados, acciones

    def pad_list(self, lst, target_length, padding_value=0):
        return list(lst) + [padding_value] * (target_length - len(lst))

    def get_max_key_length(self, diccionario):
        max_length = 0
        for agente in diccionario.values():
            for key in agente.keys():
                if len(key) > max_length:
                    max_length = len(key)
        return max_length

    def get_parameters(self, config):
        matrix = self.q_tables_to_matrix(self.filled_q_tables)
        df = pd.DataFrame(matrix)
        #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #df.to_csv(f"MATRIX/matrix_client_{self.client_id}_{timestamp}.csv", index=False)
        diccionarios, estados, acciones = self.dividir_diccionario_en_arrays(self.filled_q_tables)
        return [diccionarios, estados, acciones]

    def q_tables_to_matrix(self, filled_q_tables):
        matrix = []
        max_length = 0

        for agent_id, q_table in filled_q_tables.items():
            agent_index = int(agent_id[1:])
            for state, values in q_table.items():
                row = list(state) + values + [agent_index]
                matrix.append(row)
                max_length = max(max_length, len(row))

        for i in range(len(matrix)):
            if len(matrix[i]) < max_length:
                matrix[i] += [0] * (max_length - len(matrix[i]))

        return np.array(matrix)

    def get_q_table_size(self, agent):
        return len(agent.q_table)

    def pad_q_table(self, agent, max_size, default_state_length):
        current_size = len(agent.q_table)
        additional_entries = max_size - current_size
        state_length = int(default_state_length) - 1

        if additional_entries > 0:
            for i in range(current_size, max_size):
                state = tuple([0] * state_length + [i])
                agent.q_table[state] = [0, 0]

    def fit(self, parameters, config):
        print(f"Client {self.client_id} - fit called with parameters of shape: {[len(p) for p in parameters]}")
        initial_states = self.env.reset()

        if parameters:
            saved_state = self.set_parameters(parameters)

        self.ql_agents = {
            ts: QLAgent(
                starting_state=self.env.encode(initial_states[ts], ts),
                state_space=self.env.observation_space,
                action_space=self.env.action_space,
                alpha=self.alpha,
                gamma=self.gamma,
                exploration_strategy=EpsilonGreedy(initial_epsilon=0.009, min_epsilon=0.005, decay=self.decay),
                enable_logging=self.enable_logging,
                q_table={k: np.array(v) for k, v in saved_state.get(ts).get('q_table').items()} if parameters else {},
                acc_reward=saved_state.get(ts).get('acc_reward') if parameters else 0.0,
            )
            for ts in self.env.ts_ids
        }

        if not self.dict_sizes:
            self.initialize_dict_sizes()

        state_lengths = {self.key: 8}
        for agent_id, agent in self.ql_agents.items():
            print(f"Initial Q-table")

        rewards = []
        q_table_sizes = []

        for episode in range(1, self.episodes + 1):
            if episode != 1:
                initial_states = self.env.reset()
                for ts in initial_states.keys():
                    self.ql_agents[ts].state = self.env.encode(initial_states[ts], ts)

            steps = 0
            done = {"__all__": False}
            while not done["__all__"]:
                actions = {ts: self.ql_agents[ts].act() for ts in self.ql_agents.keys()}
                try:
                    s, r, done, info = self.env.step(action=actions)
                except KeyError as e:
                    print(f"KeyError during step execution: {e}")
                    break

                for agent_id in s.keys():
                    encoded_state = self.env.encode(s[agent_id], agent_id)
                    self.ql_agents[agent_id].learn(next_state=encoded_state, reward=r[agent_id])

                rewards.append(sum(r.values()))
                steps += 1

                self.fit_rewards.append({'Steps': steps, 'Total Reward': sum(r.values()), **r})

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.env.save_csv(f"WaitTime/FIT_{episode}_client_{self.client_id}_{timestamp}.csv", episode)

            episode_q_table_sizes = {ts: self.get_q_table_size(self.ql_agents[ts]) for ts in self.ql_agents.keys()}
            q_table_sizes.append(episode_q_table_sizes)

        max_q_table_size = max(max(sizes.values()) for sizes in q_table_sizes)
        print(max_q_table_size)

        self.filled_q_tables = {}
        self.pad_q_table(agent, max_q_table_size, state_lengths[self.key])
        self.filled_q_tables[self.key] = agent.q_table

        diccionarios, estados, acciones = self.dividir_diccionario_en_arrays(self.filled_q_tables)
        self.save_arrays_to_csv(diccionarios, estados, acciones)
        num_examples = len(rewards)
        metrics = {"reward": np.mean(rewards)}

        return [diccionarios, estados, acciones], num_examples, metrics

    def convert_parameters_to_saved_state(self, parameters):
        q_tables, states, acciones = parameters

        saved_state = {}

        agents = ['s1', 's2', 's3', 's4']
        for agent in agents:
            saved_state[agent] = {
                'q_table': {},
                'acc_reward': 0.0
            }

        for i in range(len(q_tables)):
            key = tuple(q_tables[i, :])
            value = [states[i], acciones[i]]
            for agent in agents:
                saved_state[agent]['q_table'][key] = value

        return saved_state

    def set_parameters(self, parameters):
        if not parameters or all(len(p) == 0 for p in parameters):
            print(f"No se reciben parámetros, devolviendo valor por defecto")
            saved_state = {
                self.key: {
                    'q_table': {(0, 0.0, 0, 0, 0, 0, 0, 0): np.array([0, 0])},
                    'acc_reward': 0.0
                }
            }
        else:
            saved_state = self.convert_parameters_to_saved_state(parameters)

        return saved_state
    
    
    def save_arrays_to_csv(self, diccionarios, estados, acciones):
        aggregated_data = {
            'Q-tables': diccionarios.tolist(),
            'States': estados.tolist(),
            'Actions': acciones.tolist()
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"arrays_client_{self.client_id}_{timestamp}"
        df = pd.DataFrame.from_dict(aggregated_data, orient='index').transpose()
        df.to_csv(f"ClientsArrays/{filename}.csv", index=False)
        
    
    def save_rewards_to_csv(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        evaluate_filename = f'Rewards/evaluate_rewards_client_{self.client_id}_{timestamp}.csv'
        fit_filename = f'Rewards/fit_rewards_client_{self.client_id}_{timestamp}.csv'

        evaluate_rewards_df = pd.DataFrame(self.evaluate_rewards)
        evaluate_rewards_df.to_csv(evaluate_filename, index=False)

        fit_rewards_df = pd.DataFrame(self.fit_rewards)
        fit_rewards_df.to_csv(fit_filename, index=False)

    def evaluate(self, parameters, config):
        print(f"Client {self.client_id} - evaluate called with parameters of shape: {[len(p) for p in parameters]}")
        parameters_original = parameters
        saved_state = self.set_parameters(parameters_original)

        rewards = []
        q_table_sizes = []

        self.env = SumoEnvironment(
            net_file=f'nets/mapa0.net.xml',
            route_file=f'nets/mapa0.rou.xml',
            use_gui=USE_GUI,
            num_seconds=NUM_SECONDS,
            min_green=MIN_GREEN,
            delta_time=DELTA_TIME,
            reward_fn=REWARD_FN,
        )

        runs = 1
        for run in range(1, runs + 1):
            initial_states = self.env.reset()

            self.ql_agents = {
                ts: QLAgent(
                    starting_state=self.env.encode(initial_states[ts], ts),
                    state_space=self.env.observation_space,
                    action_space=self.env.action_space,
                    alpha=self.alpha,
                    gamma=self.gamma,
                    exploration_strategy=EpsilonGreedy(initial_epsilon=0.9, min_epsilon=0.1, decay=0.99),
                    enable_logging=self.enable_logging,
                    q_table={k: np.array(v) for k, v in saved_state.get(ts).get('q_table').items()},
                    acc_reward=saved_state.get(ts).get('acc_reward'),
                )
                for ts in self.env.ts_ids
            }

            for episode in range(1, self.episodes + 1):
                if episode != 1:
                    initial_states = self.env.reset()
                    for ts in initial_states.keys():
                        self.ql_agents[ts].state = self.env.encode(initial_states[ts], ts)

                steps = 0
                done = {"__all__": False}
                while not done["__all__"]:
                    actions = {ts: self.ql_agents[ts].act() for ts in self.ql_agents.keys()}
                    try:
                        s, r, done, info = self.env.step(action=actions)
                    except KeyError as e:
                        print(f"KeyError during step execution: {e}")
                        break

                    for agent_id in s.keys():
                        encoded_state = self.env.encode(s[agent_id], agent_id)
                        self.ql_agents[agent_id].learn(next_state=encoded_state, reward=r[agent_id])

                    rewards.append(sum(r.values()))
                    steps += 1

                    self.evaluate_rewards.append({'Steps': steps, 'Total Reward': sum(r.values()), **r})
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_filename = f"WaitTime/EVAL_{episode}_client_{self.client_id}_{timestamp}.csv"
                self.env.save_csv(csv_filename, episode)

        eval_df = pd.read_csv(csv_filename)
        avg_waiting_time = eval_df['system_total_waiting_time'].mean()
        avg_CO2_emissions = eval_df['system_total_C02_emissions'].mean()
        
        num_examples = len(rewards)
        loss = 0.0
        Scalar = Union[int, float]
        # Creamos los diccionarios con valores de tipo Scalar
        metrics_WT: Dict[str, Scalar] = {"average_waiting_time": float(avg_waiting_time)}
        metrics_CO2: Dict[str, Scalar] = {"average_CO2_emissions": float(avg_CO2_emissions)}
        
        # Combinamos los diccionarios
        combined_metrics: Dict[str, Scalar] = {**metrics_WT, **metrics_CO2}
        
        return loss, num_examples, combined_metrics

if __name__ == "__main__":
    if 'spyder' in sys.modules:
        client_id = 1
        clipping_norm = 1.0
        sensitivity = 1.0
        epsilon = 1.0
        delta = 1e-5
    else:
        parser = argparse.ArgumentParser(description="Federated Learning SUMO Client")
        parser.add_argument("--client_id", type=int, required=True, help="Client ID")
        parser.add_argument("--clipping_norm", type=float, required=True, help="Clipping Norm")
        parser.add_argument("--sensitivity", type=float, required=True, help="Sensitivity")
        parser.add_argument("--epsilon", type=float, required=True, help="Epsilon")
        parser.add_argument("--delta", type=float, required=True, help="Delta")
        args = parser.parse_args()
        client_id = args.client_id
        clipping_norm = args.clipping_norm
        sensitivity = args.sensitivity
        epsilon = args.epsilon
        delta = args.delta

    client = RLClient(client_id, clipping_norm, sensitivity, epsilon, delta)
    #fl.common.logger.configure(identifier=f"myFlowerExperiment_{client_id}", filename=f"log_{client_id}.txt")

    # Start the client
    fl.client.start_client(
        server_address="localhost:8080",
        client=client
    )
    
    client.save_rewards_to_csv()

