from typing import Any, Dict, List, Optional, Type

import gym
import torch as th
from torch import nn

from stable_baselines3.common.policies import BasePolicy, register_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, NatureCNN, create_mlp
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.preprocessing import get_action_dim


class Controller(BasePolicy):
    """
    Controller network for WorldModel
    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        features_extractor: nn.Module,
        features_dim: int,
        latent_dim: int,
        hidden_dim: int,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        super(Controller, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_extractor = features_extractor
        self.features_dim = features_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.normalize_images = normalize_images
        action_dim = get_action_dim(self.action_space)
        self.fc = nn.Linear(self.latent_dim + self.hidden_dim, action_dim)

    def forward(self, *state: th.Tensor) -> th.Tensor:
        cat_in = th.cat(state, dim=1)
        return self.fc(cat_in)

    def _predict(self, *state: th.Tensor, deterministic: bool = True) -> th.Tensor:
        prob = self.forward(state)
        # Greedy action
        action = prob.argmax(dim=1).reshape(-1)
        return action

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data
