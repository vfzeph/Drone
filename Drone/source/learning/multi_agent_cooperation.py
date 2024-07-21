import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple

class MultiAgentCooperation:
    def __init__(self, num_agents, state_dim, action_dim, hidden_layers):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agents = [self._create_agent(state_dim, action_dim, hidden_layers) for _ in range(num_agents)]
        self.memory = [deque(maxlen=1000) for _ in range(num_agents)]
        self.experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

    def _create_agent(self, state_dim, action_dim, hidden_layers):
        return {
            'policy': self._build_network(state_dim, action_dim, hidden_layers),
            'critic': self._build_network(state_dim, 1, hidden_layers),
            'optimizer_policy': optim.Adam(self._build_network(state_dim, action_dim, hidden_layers).parameters(), lr=1e-3),
            'optimizer_critic': optim.Adam(self._build_network(state_dim, 1, hidden_layers).parameters(), lr=1e-3),
        }

    def _build_network(self, input_dim, output_dim, hidden_layers):
        layers = []
        current_size = input_dim
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())
            current_size = hidden_size
        layers.append(nn.Linear(current_size, output_dim))
        return nn.Sequential(*layers)

    def select_actions(self, states):
        actions = []
        for i in range(self.num_agents):
            state_tensor = torch.FloatTensor(states[i]).unsqueeze(0)
            policy = self.agents[i]['policy']
            action_prob = policy(state_tensor)
            action = torch.argmax(action_prob, dim=1).item()
            actions.append(action)
        return actions

    def store_experience(self, agent_idx, state, action, reward, next_state, done):
        self.memory[agent_idx].append(self.experience(state, action, reward, next_state, done))

    def update_agents(self, gamma=0.99):
        for agent_idx in range(self.num_agents):
            if len(self.memory[agent_idx]) < 64:
                continue
            batch = self._sample_experience(agent_idx, 64)
            self._update_agent(agent_idx, batch, gamma)

    def _sample_experience(self, agent_idx, batch_size):
        experiences = random.sample(self.memory[agent_idx], batch_size)
        batch = self.experience(*zip(*experiences))
        states = torch.FloatTensor(batch.state)
        actions = torch.LongTensor(batch.action)
        rewards = torch.FloatTensor(batch.reward)
        next_states = torch.FloatTensor(batch.next_state)
        dones = torch.FloatTensor(batch.done)
        return states, actions, rewards, next_states, dones

    def _update_agent(self, agent_idx, batch, gamma):
        states, actions, rewards, next_states, dones = batch

        # Update critic
        critic = self.agents[agent_idx]['critic']
        optimizer_critic = self.agents[agent_idx]['optimizer_critic']
        target_values = rewards + gamma * critic(next_states) * (1 - dones)
        expected_values = critic(states)
        loss_critic = nn.MSELoss()(expected_values, target_values.detach())

        optimizer_critic.zero_grad()
        loss_critic.backward()
        optimizer_critic.step()

        # Update policy
        policy = self.agents[agent_idx]['policy']
        optimizer_policy = self.agents[agent_idx]['optimizer_policy']
        action_probs = policy(states)
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))
        advantages = target_values - expected_values
        loss_policy = -(log_probs * advantages.detach()).mean()

        optimizer_policy.zero_grad()
        loss_policy.backward()
        optimizer_policy.step()
