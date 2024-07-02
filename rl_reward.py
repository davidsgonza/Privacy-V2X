import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import csv
from datetime import datetime
os.environ['SUMO_HOME'] = '/home/david/sumo-git/'

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy


def scale_steps_to_seconds(steps, delta_time):
    return [step * delta_time for step in steps]

def calculate_interval_averages(data, interval_size):
    return [np.mean(data[i:i + interval_size]) for i in range(0, len(data), interval_size)]

def calculate_interval_steps(steps, interval_size):
    return [steps[i] for i in range(0, len(steps), interval_size)]


if __name__ == "__main__":
    alpha = 0.1
    gamma = 0.99
    decay = 1
    runs = 1
    episodes = 10
    num_seconds = 86400
    delta_time = 5
    interval_size = 50  # Tamaño del intervalo para promediar los pasos
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env = SumoEnvironment(
        net_file="/home/david/Sumo/sumo-rl/nets/cuenca/mapa0.net.xml",
        route_file="/home/david/Sumo/sumo-rl/nets/cuenca/mapa0.rou.xml",
        #sumocfg = "/home/david/Sumo/sumo-rl/nets/cuenca/cuenca.sumocfg",
        use_gui=False,
        num_seconds=num_seconds,
        min_green=10,
        max_green=40,
        delta_time=delta_time,
        #reward_fn= "co2",
        reward_fn= "diff-waiting-time",
    )

    for run in range(1, runs + 1):
        initial_states = env.reset()
      
        ql_agents = {
            ts: QLAgent(
                starting_state=env.encode(initial_states[ts], ts),
                state_space=env.observation_space,
                action_space=env.action_space,
                alpha=alpha,
                gamma=gamma,
                exploration_strategy=EpsilonGreedy(initial_epsilon=0.05, min_epsilon=0.005, decay=decay),
            )
            for ts in env.ts_ids
        }

        for episode in range(1, episodes + 1):
            all_rewards = []
            if episode != 1:
                initial_states = env.reset()
                for ts in initial_states.keys():
                    ql_agents[ts].state = env.encode(initial_states[ts], ts)

            infos = []
            done = {"__all__": False}
            step = 0
            episode_rewards = 0
            while env.sim_step < num_seconds:
                actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

                s, r, done, info = env.step(action=actions)

                for agent_id in s.keys():
                    ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])
                    episode_rewards += r[agent_id]

                step += 1
                all_rewards.append((step, episode_rewards))
                
            # Save rewards to CSV for each episode
            with open(f'rewards_output_episode_{episode}_{timestamp}.csv', 'w', newline='') as csvfile:
                fieldnames = ['Step', 'Accumulated Rewards']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for step, reward in all_rewards:
                    writer.writerow({'Step': step, 'Accumulated Rewards': reward})

            env.save_csv(f"outputs/cuenca/pr_run{run}_{timestamp}.csv", episode)

    env.close()
    
    # Consolidate rewards for plotting
    all_rewards = []
    for episode in range(1, episodes + 1):
        with open(f'rewards_output_episode_{episode}_{timestamp}.csv', 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                all_rewards.append((int(row['Step']), float(row['Accumulated Rewards'])))

    # Plotting rewards
    steps, rewards = zip(*all_rewards)
    steps_in_seconds = scale_steps_to_seconds(steps, delta_time)
    
    plt.figure(figsize=(10, 5))
    plt.plot(steps_in_seconds, rewards, label="Accumulated Rewards")
    plt.xlabel("Simulation Time")
    plt.ylabel("Accumulated Rewards")
    plt.title("Accumulated Rewards")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calculate value loss (absolute difference in accumulated rewards)
    value_loss = [abs(rewards[i] - rewards[i - 1]) for i in range(1, len(rewards))]
    steps_in_seconds_loss = steps_in_seconds[1:]  # Align with value_loss

    # Apply interval averaging to value loss
    averaged_value_loss = calculate_interval_averages(value_loss, interval_size)
    averaged_steps_in_seconds = calculate_interval_steps(steps_in_seconds_loss, interval_size)

    # Plotting averaged value loss
    plt.figure(figsize=(10, 5))
    plt.plot(averaged_steps_in_seconds, averaged_value_loss, label="Averaged Value Loss", color='green')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Value Loss")
    plt.title("Value Loss")
    plt.legend()
    plt.grid(False)
    plt.show()
