import os
import sys
import json
import logging
import numpy as np
import torch
import torch.optim as optim
import pandas as pd
import airsim

from stable_baselines3.common.vec_env import DummyVecEnv
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Configure project root and import paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Custom imports from your Drone project structure
from source.envs.drone_controller import DroneController
from source.envs.airsim_env import AirSimEnv
from source.models.ppo.ppo_agent import PPOAgent
from source.utilities.custom_logger import CustomLogger
from source.utilities.data_processing import DataProcessor
from source.utilities.visualization import DataVisualizer

def configure_logger(name, log_dir='./logs'):
    logger = CustomLogger(name, log_dir=log_dir)
    return logger

def load_config(config_path):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config

def read_tensorboard_logs(log_dir, scalar_name):
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    scalar_values = event_acc.Scalars(scalar_name)
    return [(scalar.step, scalar.value) for scalar in scalar_values]

def visualize_results(config, logger, data_processor, data_visualizer):
    tb_log_dir = config['logging']['tensorboard_log_dir']
    scalar_name = 'Training/Episode Reward'
    steps, values = read_tensorboard_logs(tb_log_dir, scalar_name)
    df = pd.DataFrame({'steps': steps, 'values': values})
    
    if df.empty:
        logger.warning("DataFrame is empty. Skipping visualization.")
    else:
        try:
            cleaned_df = data_processor.clean_data(df)
            transformed_df = data_processor.transform_data(cleaned_df, {"values": lambda x: x * 2})

            data_visualizer.plot_time_series(transformed_df, x='steps', y='values', title='Transformed Episode Rewards')
            data_visualizer.plot_histogram(transformed_df, column='values', title='Values Distribution')
            data_visualizer.plot_correlation_matrix(transformed_df, title='Correlation Matrix')
            data_visualizer.plot_scatter(transformed_df, x='steps', y='values', title='Scatter Plot')
        except Exception as e:
            logger.error(f"Visualization error: {e}")

def train_agents(ppo_agent, env, config, logger):
    total_timesteps = config["num_timesteps"]
    ppo_save_path = config["logging"]["model_save_path"] + "/ppo_trained_model.pth"
    episode_rewards, episode_lengths = ppo_agent.train(env, total_timesteps, ppo_save_path)
    logger.info("PPO training completed and model saved.")
    return episode_rewards, episode_lengths

def main():
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'learning', 'ppo_config.json')
    config = load_config(config_path)

    logger = configure_logger("AirSimEnvLogger", config['logging']['log_dir'])
    writer = SummaryWriter(log_dir=config['logging']['tensorboard_log_dir'])

    # Initialize AirSim client
    client = airsim.MultirotorClient()
    client.confirmConnection()

    # Initialize the environment
    env = AirSimEnv(
        state_dim=config['environment']['state_dim'],
        action_dim=config['environment']['action_dim'],
        config=config,
        logger=logger,
        tensorboard_log_dir=config['logging']['tensorboard_log_dir'],
        log_enabled=True
    )
    
    # Set the client manually after initialization
    if hasattr(env, 'set_client'):
        env.set_client(client)
    elif hasattr(env, 'client'):
        env.client = client

    env = DummyVecEnv([lambda: env])

    # Create data_processor and data_visualizer instances
    data_processor = DataProcessor(logger=logger)
    data_visualizer = DataVisualizer(logger=logger)

    # Initialize the PPO agent
    ppo_agent = PPOAgent(config, logger=logger, drone_controller=env.envs[0].drone_controller)

    try:
        # Train the agent
        episode_rewards, episode_lengths = train_agents(ppo_agent, env, config, logger)

        # Evaluate the trained agent
        eval_reward = ppo_agent.evaluate(env, num_episodes=5)
        logger.info(f"Evaluation average reward: {eval_reward}")

        # Visualize results
        visualize_results(config, logger, data_processor, data_visualizer)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        env.close()
        writer.close()
        logger.info("Training completed and models saved.")

if __name__ == '__main__':
    main()
