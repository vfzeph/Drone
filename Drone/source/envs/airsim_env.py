import json
import airsim
import numpy as np
import time
import os
import gym
from gym import spaces
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import random
from threading import Thread, Lock
from collections import deque
import torch.nn.functional as F

from source.models.nn.common_layers import ICM
from source.models.nn.policy_network import AdvancedPolicyNetwork
from source.models.nn.critic_network import AdvancedCriticNetwork
from source.models.nn.Predictive_model import PredictiveModel
from source.envs.drone_controller import DroneController

class CustomLogger:
    def __init__(self, name, log_dir=None):
        self.name = name
        self.log_dir = log_dir
        self._setup_logger()

    def _setup_logger(self):
        import logging
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(self.log_dir, f'{self.name}.log'))
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)

def randomize_environment(client):
    weather_params = [
        airsim.WeatherParameter.Rain,
        airsim.WeatherParameter.Enabled
    ]
    weather = random.choice(weather_params)
    client.simSetWeatherParameter(weather, random.uniform(0, 1))

class AirSimEnv(gym.Env):
    class State:
        INITIALIZING = 'initializing'
        RESETTING = 'resetting'
        TAKING_OFF = 'taking_off'
        RUNNING = 'running'
        DONE = 'done'

    def __init__(self, state_dim, action_dim, config, target_position=np.array([0, 0, -10]), action_frequency=1.0, log_enabled=False,
                 exploration_strategy="epsilon_greedy", epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.1,
                 temperature=1.0, ucb_c=2.0, logger=None, tensorboard_log_dir=None):
        super(AirSimEnv, self).__init__()
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.drone_controller = DroneController(self.client)
        self.state_dim = state_dim
        self.max_episode_steps = config.get('max_episode_steps', 1000)
        self.current_step = 0
        self.action_dim = action_dim
        self.logger = logger if logger else CustomLogger("AirSimEnvLogger")
        self.logger.info(f"Initialized with state_dim: {self.state_dim}, action_dim: {self.action_dim}")
        self.target_position = target_position
        self.action_frequency = action_frequency
        self.start_time = None
        self.max_episode_duration = 120
        self.exploration_area = {
            "x_min": -1000, "x_max": 1000,
            "y_min": -1000, "y_max": 1000,
            "z_min": -100, "z_max": 100
        }
        
        # Calculate the total observation size
        self.image_height = config['icm']['image_height']
        self.image_width = config['icm']['image_width']
        self.image_channels = config['icm']['image_channels']
        self.image_size = self.image_height * self.image_width * self.image_channels
        self.total_obs_size = self.state_dim + self.image_size
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.total_obs_size,), dtype=np.float32)
        
        self.log_enabled = log_enabled
        self.exploration_strategy = exploration_strategy
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.temperature = temperature
        self.ucb_c = ucb_c
        self.total_steps = 0
        self.prev_action = np.zeros(self.action_dim)
        self.writer = SummaryWriter(log_dir=tensorboard_log_dir) if tensorboard_log_dir else None
        self.prev_state = None
        self.state = None
        self.next_state = None
        self.current_image = None

        self.logger.info("Initializing AirSimEnv")
        self.logger.info(f"Configuration: {json.dumps(config, indent=2)}")

        # Initialize device
        device_config = config['ppo']['device']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device_config == 'auto' else torch.device(device_config)

        self._initialize_icm(config)
        self._initialize_policy_network(config)
        self._initialize_critic_network(config)
        self._initialize_predictive_model(config)

        # Randomize environment
        self.randomize_environment()

        # Initialize state normalization parameters
        self.state_means = np.zeros(self.state_dim)
        self.state_stddevs = np.ones(self.state_dim)
        self.state_buffer = deque(maxlen=1000)
        self.state_buffer_lock = Lock()

        # Start background thread for state normalization
        self.normalization_thread = Thread(target=self._update_state_normalization)
        self.normalization_thread.start()

        # Initialize FSM state
        self.current_fsm_state = self.State.INITIALIZING

        # Initialize reward weights
        self.reward_weights = {
            'distance': 1.0,
            'velocity': 0.5,
            'collision': 1.0,
            'height': 0.5,
            'movement': 0.1,
            'smoothness': 0.1,
            'curiosity': 0.5,
            'exploration': 0.2
        }
        
    def _initialize_icm(self, config):
        try:
            if 'icm' in config:
                self.icm = ICM(config['icm']).to(self.device)
                self.icm_optimizer = optim.Adam(self.icm.parameters(), lr=config['learning_rate'])
            else:
                self.icm = None
                self.icm_optimizer = None
        except Exception as e:
            self.logger.error(f"Error initializing ICM: {e}")
            raise

    def _initialize_policy_network(self, config):
        try:
            continuous = config['ppo'].get('continuous', True)
            
            policy_config = {
                'image_channels': config['icm']['image_channels'],
                'image_height': config['icm']['image_height'],
                'image_width': config['icm']['image_width'],
                'cnn': config['icm']['cnn'],
                'use_batch_norm': config['policy_network'].get('use_batch_norm', True),
                'use_dropout': config['policy_network'].get('use_dropout', True),
                'dropout_rate': config['policy_network'].get('dropout_rate', 0.2),
                'use_attention': config['policy_network'].get('use_attention', True),
                'num_action_heads': config['policy_network'].get('num_action_heads', 1),
            }
            
            self.policy_network = AdvancedPolicyNetwork(
                self.state_dim, 
                self.action_dim, 
                continuous=continuous,
                hidden_sizes=config['policy_network']['hidden_layers'],
                config=policy_config
            ).to(self.device)
        except KeyError as e:
            self.logger.error(f"Missing key in configuration for policy network: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error initializing policy network: {e}")
            raise

    def _initialize_critic_network(self, config):
        try:
            critic_config = {
                'image_channels': config['icm']['image_channels'],
                'image_height': config['icm']['image_height'],
                'image_width': config['icm']['image_width'],
                'cnn': config['icm']['cnn'],
                'use_batch_norm': config['critic_network'].get('use_batch_norm', True),
                'use_dropout': config['critic_network'].get('use_dropout', True),
                'dropout_rate': config['critic_network'].get('dropout_rate', 0.2),
                'use_attention': config['critic_network'].get('use_attention', True),
            }

            self.critic_network = AdvancedCriticNetwork(
                self.state_dim,
                hidden_sizes=config['critic_network']['hidden_layers'],
                config=critic_config
            ).to(self.device)
        except KeyError as e:
            self.logger.error(f"Missing key in configuration for critic network: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error initializing critic network: {e}")
            raise

    def _initialize_predictive_model(self, config):
        try:
            cnn_config = config['icm']['cnn']
            cnn_config['image_channels'] = config['icm']['image_channels']
            cnn_config['image_height'] = config['icm']['image_height']
            cnn_config['image_width'] = config['icm']['image_width']

            self.predictive_model = PredictiveModel(
                self.state_dim + self.action_dim,
                self.state_dim,
                config['predictive_model']['hidden_layers'],
                cnn_config
            ).to(self.device)
            self.predictive_model_optimizer = optim.Adam(
                self.predictive_model.parameters(),
                lr=config['predictive_model']['learning_rate']
            )
        except KeyError as e:
            self.logger.error(f"Missing key in configuration for predictive model: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error initializing predictive model: {e}")
            raise

    def log(self, message):
        if self.log_enabled:
            if self.logger:
                self.logger.info(message)
            else:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

    def randomize_environment(self):
        randomize_environment(self.client)

    def reset(self):
        try:
            self.current_fsm_state = self.State.RESETTING
            self._reset_env()
            self.current_fsm_state = self.State.TAKING_OFF
            self.drone_controller.takeoff()
            self.current_fsm_state = self.State.RUNNING
            self.start_time = time.time()
            self.prev_action = np.zeros(self.action_dim)
            self.total_steps = 0
            self.current_step = 0  # Reset current step counter
            self.prev_state = None
            self.log("Environment reset and takeoff completed.")
            self.state = self._get_state()
            self.current_image = self._get_image()
            with self.state_buffer_lock:
                self.state_buffer.append(self.state)

            observation = {
                'state': self.state,
                'visual': self.current_image
            }

            self.logger.info(f"Reset completed. Observation keys: {observation.keys()}")
            self.logger.info(f"State shape: {observation['state'].shape}, Visual shape: {observation['visual'].shape}")

            return observation

        except Exception as e:
            self.logger.error(f"Error in reset function: {str(e)}")
            raise

    def _reset_env(self):
        self.drone_controller.reset()

    def predict_next_state(self, state, action, image):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(self.device)
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            next_state = self.predictive_model(state_tensor, action_tensor, image_tensor).cpu().numpy()

        return next_state
    
    def _train_predictive_model(self, state, action, next_state, image):
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        action_tensor = torch.tensor(action, dtype=torch.float32).to(self.device)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)

        predicted_next_state = self.predictive_model(state_tensor, action_tensor, image_tensor)
        loss = F.mse_loss(predicted_next_state, next_state_tensor)

        self.predictive_model_optimizer.zero_grad()
        loss.backward()
        self.predictive_model_optimizer.step()

        return loss.item()

    def step(self, action):
        if self.current_fsm_state != self.State.RUNNING:
            self.logger.error("Step called while not in RUNNING state.")
            return {'state': np.zeros(self.state_dim), 'visual': np.zeros((self.image_height, self.image_width, self.image_channels))}, 0, True, {"error": "Not in RUNNING state"}

        try:
            # Ensure action is within bounds
            action = np.clip(action, self.action_space.low, self.action_space.high)

            duration = 0.5 / self.action_frequency
            action = self._smooth_action(action)
            vx, vy, vz = action

            self.drone_controller.move_by_velocity_z(vx, vy, -vz, duration)
            time.sleep(duration)

            new_state = self._get_state()
            new_image = self._get_image()

            reward = self._compute_reward(self.state, action, self.current_image, new_state)
            done = self._check_done(new_state)

            self.log(f"Action: {action}, Velocity: ({vx}, {vy}, {vz}), Duration: {duration}, Reward: {reward}, Done: {done}")

            with self.state_buffer_lock:
                self.state_buffer.append(new_state)

            if self.writer:
                self.writer.add_scalar('Reward', reward, self.total_steps)
                self.writer.add_scalar('Epsilon', self.epsilon, self.total_steps)
                self.writer.flush()

            self.total_steps += 1
            self.current_step += 1
            self.prev_state = self.state
            self.state = new_state
            self.current_image = new_image

            observation = {
                'state': self.state,
                'visual': self.current_image
            }

            if done:
                self.current_fsm_state = self.State.DONE

            self.logger.info(f"Step completed. Observation keys: {observation.keys()}")
            self.logger.info(f"State shape: {observation['state'].shape}, Visual shape: {observation['visual'].shape}")

            # Train predictive model
            predictive_loss = self._train_predictive_model(self.prev_state, action, new_state, self.current_image)
            self.logger.info(f"Predictive model loss: {predictive_loss}")

            return observation, reward, done, {"terminated": done, "truncated": False}

        except Exception as e:
            self.logger.error(f"Error in step function: {str(e)}")
            self.logger.error(f"Action shape: {action.shape}, Current state shape: {self.state.shape}, "
                            f"Current image shape: {self.current_image.shape}")
            self.current_fsm_state = self.State.DONE
            return {'state': np.zeros(self.state_dim), 'visual': np.zeros((self.image_height, self.image_width, self.image_channels))}, 0, True, {"error": str(e)}
    
    def _smooth_action(self, action):
        action = np.tanh(action)  # Apply non-linear transformation (hyperbolic tangent)
        action = self.prev_action + 0.5 * (action - self.prev_action)
        self.prev_action = action
        return action

    def _get_state(self):
        position = self.drone_controller.get_position()
        velocity = self.drone_controller.get_velocity()
        orientation = self.drone_controller.get_orientation()
        angular_velocity = self.drone_controller.get_angular_velocity()

        # Check if position and velocity are objects with x_val, y_val, z_val attributes
        if hasattr(position, 'x_val'):
            position = np.array([position.x_val, position.y_val, position.z_val])
        if hasattr(velocity, 'x_val'):
            velocity = np.array([velocity.x_val, velocity.y_val, velocity.z_val])

        # Ensure orientation is a numpy array
        orientation = np.array(orientation)

        # Angular velocity is already a numpy array, so we don't need to modify it

        state = np.concatenate([
            position,
            velocity,
            orientation,
            angular_velocity
        ])

        # Ensure state has the correct shape
        state = np.pad(state, (0, max(0, self.state_dim - len(state))))[:self.state_dim]
        
        return state
        
    def _get_image(self):
        responses = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        response = responses[0]
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(self.image_height, self.image_width, self.image_channels)
        return img_rgb.astype(np.float32) / 255.0

    def _normalize_state(self, state):
        if state.shape != self.state_means.shape:
            raise ValueError(f"State shape {state.shape} does not match normalization parameters shape {self.state_means.shape}")
        return (state - self.state_means) / (self.state_stddevs + 1e-8)

    def _update_state_normalization(self):
        while True:
            with self.state_buffer_lock:
                if len(self.state_buffer) >= 1000:
                    states = np.array(self.state_buffer)
                    self.state_means = np.mean(states, axis=0)
                    self.state_stddevs = np.std(states, axis=0)
                    self.state_buffer.clear()

    def _compute_reward(self, state, action, current_image, next_state):
        distance_reward = self._compute_distance_reward(next_state)
        velocity_reward = self._compute_velocity_reward(next_state)
        collision_penalty = self._compute_collision_penalty()
        height_penalty = self._compute_height_penalty(next_state)
        movement_penalty = self._compute_movement_penalty(action)
        smoothness_penalty = self._compute_smoothness_penalty(action)
        curiosity_reward = self._compute_curiosity_reward(state, action, current_image, next_state)
        exploration_bonus = self._compute_exploration_bonus(next_state)

        total_reward = (
            self.reward_weights['distance'] * distance_reward +
            self.reward_weights['velocity'] * velocity_reward +
            self.reward_weights['collision'] * collision_penalty +
            self.reward_weights['height'] * height_penalty +
            self.reward_weights['movement'] * movement_penalty +
            self.reward_weights['smoothness'] * smoothness_penalty +
            self.reward_weights['curiosity'] * curiosity_reward +
            self.reward_weights['exploration'] * exploration_bonus
        )

        self.logger.info(f"Reward breakdown - Distance: {distance_reward}, Velocity: {velocity_reward}, "
                        f"Collision: {collision_penalty}, Height: {height_penalty}, Movement: {movement_penalty}, "
                        f"Smoothness: {smoothness_penalty}, Curiosity: {curiosity_reward}, "
                        f"Exploration: {exploration_bonus}, Total: {total_reward}")

        return total_reward
    
    def _compute_distance_reward(self, state):
        position = state[:3]
        distance_to_target = np.linalg.norm(self.target_position - position)
        return -distance_to_target

    def _compute_velocity_reward(self, state):
        velocity = state[3:6]
        return -np.linalg.norm(velocity)  # Penalize high velocities

    def _compute_collision_penalty(self):
        collision_info = self.drone_controller.get_collision_info()
        return -50 if collision_info.has_collided else 0

    def _compute_height_penalty(self, state):
        current_height = state[2]
        height_target = self.target_position[2]
        height_tolerance = 1.0
        height_penalty = 1.0
        return -height_penalty if abs(current_height - height_target) > height_tolerance else 0

    def _compute_movement_penalty(self, action):
        movement_penalty = 0.1
        return -movement_penalty * np.linalg.norm(action)

    def _compute_smoothness_penalty(self, action):
        smoothness_penalty = 0.1
        return -smoothness_penalty * np.linalg.norm(action - self.prev_action)

    def _compute_curiosity_reward(self, state, action, current_image, next_state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Ensure images are in the correct shape [batch, channels, height, width]
        current_image_tensor = torch.tensor(current_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
        next_image_tensor = torch.tensor(next_state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)

        curiosity_reward = self.icm.intrinsic_reward(state_tensor, next_state_tensor, action_tensor, current_image_tensor, next_image_tensor)
        return curiosity_reward.item()
    
    def _compute_exploration_bonus(self, state):
        exploration_bonus = 0.1 * np.linalg.norm(state - self.prev_state) if self.prev_state is not None else 0
        self.prev_state = state
        return exploration_bonus

    def _check_done(self, state):
        position = state[:3]
        distance_to_target = np.linalg.norm(self.target_position - position)
        collision_info = self.drone_controller.get_collision_info()

        if distance_to_target < 1.0:
            self.logger.info("Target reached.")
            return True
        if collision_info.has_collided:
            self.logger.info("Collision detected.")
            return True
        if self.current_step >= self.max_episode_steps:
            self.logger.info("Maximum episode steps reached.")
            return True
        return False
    
    def get_goal(self):
        return self.target_position
    
    def close(self):
        self.drone_controller.land()
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        self.client.reset()
        if self.writer:
            self.writer.close()
        if self.logger:
            self.logger.info("Environment closed.")

if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'learning', 'ppo_config.json')
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    logger = CustomLogger("AirSimEnvLogger", log_dir="./logs")
    env = AirSimEnv(state_dim=config['policy_network']['input_size'], 
                    action_dim=config['policy_network']['output_size'], 
                    config=config, 
                    logger=logger, 
                    tensorboard_log_dir="./logs/tensorboard_logs", 
                    log_enabled=True)
    try:
        observation = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = env.action_space.sample()  # Random action for testing
            observation, reward, done, _ = env.step(action)
            episode_reward += reward
            if done:
                logger.info(f"Episode finished. Total reward: {episode_reward}")
                observation = env.reset()
                episode_reward = 0
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
    finally:
        env.close()