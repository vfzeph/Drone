import torch
import numpy as np
import logging

# Configure enhanced structured logging
def configure_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Adjusted level to DEBUG to capture all logs during development
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger

logger = configure_logger()

def compute_gae(next_value, rewards, dones, values, gamma, lam):
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Args:
        next_value (torch.Tensor): Value estimate of the next state.
        rewards (torch.Tensor): Tensor of rewards.
        dones (torch.Tensor): Tensor indicating whether an episode has ended.
        values (torch.Tensor): Tensor of value estimates for each state.
        gamma (float): Discount factor.
        lam (float): GAE lambda parameter.

    Returns:
        torch.Tensor: Returns and advantages as tensors.
    """
    logger.debug("Computing GAE.")
    gae = 0
    returns = []
    advantages = []
    with torch.no_grad():
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * (1 - dones[step]) * next_value - values[step]
            gae = delta + gamma * lam * (1 - dones[step]) * gae
            next_value = values[step]
            returns.insert(0, gae + values[step])
            advantages.insert(0, gae)

    logger.debug(f"GAE returns: {returns}")
    logger.debug(f"GAE advantages: {advantages}")
    return torch.tensor(returns, dtype=torch.float32), torch.tensor(advantages, dtype=torch.float32)

def normalize(x, eps=1e-8):
    """
    Normalize the input tensor to have zero mean and unit variance.

    Args:
        x (torch.Tensor): Input tensor.
        eps (float): Small epsilon value to prevent division by zero in standard deviation.

    Returns:
        torch.Tensor: Normalized tensor.
    """
    logger.debug("Normalizing tensor.")
    mean = x.mean()
    std = x.std().clamp(min=eps)
    normalized_x = (x - mean) / std
    logger.debug(f"Tensor normalized: mean={mean.item()}, std={std.item()}")
    return normalized_x

def to_tensor(np_array, device='cpu', dtype=torch.float32):
    """
    Convert a numpy array to a PyTorch tensor, efficiently transferring it to the specified device.

    Parameters:
    - np_array (numpy.ndarray): Array to convert.
    - device (str): The device to which the tensor should be transferred.
    - dtype (torch.dtype): The desired data type of the tensor.

    Returns:
    - torch.Tensor: The resulting tensor on the specified device.
    """
    logger.debug("Converting numpy array to tensor.")
    tensor = torch.tensor(np_array, device=device, dtype=dtype)
    logger.debug(f"Tensor created on {device} with data: {tensor}")
    return tensor

def reward_normalize(rewards):
    """
    Normalize rewards to improve training stability.

    Parameters:
    - rewards (torch.Tensor): Input rewards to normalize.

    Returns:
    - torch.Tensor: Normalized rewards.
    """
    logger.debug("Normalizing rewards.")
    mean = rewards.mean()
    std = rewards.std().clamp(min=1e-8)
    normalized_rewards = (rewards - mean) / std
    logger.debug(f"Rewards normalized: mean={mean.item()}, std={std.item()}")
    return normalized_rewards

def discounted_rewards(rewards, gamma):
    """
    Compute the discounted rewards for a sequence of rewards.

    Parameters:
    - rewards (torch.Tensor): Sequence of rewards.
    - gamma (float): Discount factor for future rewards.

    Returns:
    - torch.Tensor: Discounted rewards.
    """
    logger.debug("Computing discounted rewards.")
    discounted = torch.zeros_like(rewards)
    running_sum = 0
    for t in reversed(range(len(rewards))):
        running_sum = rewards[t] + gamma * running_sum
        discounted[t] = running_sum
    logger.debug(f"Discounted rewards: {discounted}")
    return discounted

# Example test cases to ensure the functions work as expected
if __name__ == "__main__":
    # Example tensors
    next_values = torch.tensor([1.0, 2.0, 3.0])
    rewards = torch.tensor([1.0, 0.5, 0.2])
    masks = torch.tensor([1.0, 0.0, 1.0])
    values = torch.tensor([[1.0], [2.0], [3.0]])
    gamma = 0.99
    tau = 0.95

    # Compute GAE
    returns, gae = compute_gae(next_values, rewards, masks, values, gamma, tau)
    logger.info(f"Computed returns: {returns}")
    logger.info(f"Computed GAE: {gae}")

    # Normalize example tensor
    example_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
    normalized_tensor = normalize(example_tensor)
    logger.info(f"Normalized tensor: {normalized_tensor}")

    # Convert numpy array to tensor
    np_array = np.array([1.0, 2.0, 3.0])
    tensor = to_tensor(np_array, device='cpu')
    logger.info(f"Converted tensor: {tensor}")

    # Normalize rewards
    rewards_tensor = torch.tensor([1.0, 2.0, 3.0])
    normalized_rewards = reward_normalize(rewards_tensor)
    logger.info(f"Normalized rewards: {normalized_rewards}")

    # Compute discounted rewards
    discounted = discounted_rewards(rewards_tensor, gamma)
    logger.info(f"Discounted rewards: {discounted}")
