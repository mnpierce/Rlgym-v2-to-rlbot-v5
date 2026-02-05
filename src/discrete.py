"""
File: discrete_policy.py
Author: Matthew Allen

Description:
    An implementation of a feed-forward neural network which parametrizes a discrete distribution over a space of actions.
"""


from torch.distributions import Categorical
import torch.nn as nn
import torch
import numpy as np


class DiscreteFF(nn.Module):
    def __init__(self, input_shape, n_actions, layer_sizes, device, add_layer_norm=False, activation=nn.ReLU):
        super().__init__()
        self.device = device

        assert len(layer_sizes) != 0, "AT LEAST ONE LAYER MUST BE SPECIFIED TO BUILD THE NEURAL NETWORK!"
        
        layers = []
        prev_size = input_shape
        for size in layer_sizes:
            layers.append(nn.Linear(prev_size, size))
            if add_layer_norm:
                layers.append(nn.LayerNorm(size))
            layers.append(activation())
            prev_size = size

        layers.append(nn.Linear(prev_size, n_actions))
        layers.append(nn.Softmax(dim=-1))
        self.model = nn.Sequential(*layers).to(self.device)

        self.n_actions = n_actions

    def get_output(self, obs):
        t = type(obs)
        if t != torch.Tensor:
            if t != np.array:
                obs = np.asarray(obs)
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        return self.model(obs)

    def get_action(self, obs, deterministic=False):
        """
        Function to get an action and the log of its probability from the policy given an observation.
        :param obs: Observation to act on.
        :param deterministic: Whether the action should be chosen deterministically.
        :return: Chosen action and its logprob.
        """

        probs = self.get_output(obs)
        probs = probs.view(-1, self.n_actions)
        probs = torch.clamp(probs, min=1e-11, max=1)

        if deterministic:
            return probs.cpu().numpy().argmax(), 0

        action = torch.multinomial(probs, 1, True)
        log_prob = torch.log(probs).gather(-1, action)

        return action.flatten().cpu(), log_prob.flatten().cpu()
