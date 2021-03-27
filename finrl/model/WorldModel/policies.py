from typing import Any, Dict, List, Optional, Type

import gym
import torch as th
from torch import nn

from stable_baselines3.common.policies import BasePolicy, register_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, NatureCNN, create_mlp
from stable_baselines3.common.type_aliases import Schedule

from finrl.model.WorldModel.vae import VAE
from finrl.model.WorldModel.mdnrnn import MDNRNN
from finrl.model.WorldModel.mdnrnn import MDNRNNCell
from finrl.model.WorldModel.controller import Controller

LATENT, HIDDEN = 32, 256

class WMPolicy(BasePolicy):
    """
    Policy class with Q-Value Net and target net for DQN
    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(WMPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

        if net_arch is None:
            if features_extractor_class == FlattenExtractor:
                net_arch = [64, 64]
            else:
                net_arch = []

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.normalize_images = normalize_images

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": self.net_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }

        self.vae, self.mdnrnn, self.mdnrnncell, self.controller = None, None, None, None
        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the network and the optimizer.
        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        self.vae = self.make_vae(LATENT)
        self.vae.optimizer = self.optimizer_class(self.vae.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        self.mdnrnn, self.mdnrnncell = self.make_mdnrnn(LATENT, HIDDEN)
        self.mdnrnn.optimizer = self.optimizer_class(self.mdnrnn.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        self.controller = self.make_controller(LATENT, HIDDEN)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def make_vae(self, latent_dim) -> VAE:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        net_args.update(dict(latent_dim=latent_dim))
        return VAE(**net_args).to(self.device)

    def make_mdnrnn(self, latent_dim, hidden_dim) -> MDNRNN:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        net_args.update(dict(latent_dim=latent_dim, hidden_dim=hidden_dim, n_gaussian=5))
        return MDNRNN(**net_args).to(self.device), MDNRNNCell(**net_args).to(self.device)
    
    def make_controller(self, latent_dim, hidden_dim) -> Controller:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        net_args.update(dict(latent_dim=latent_dim, hidden_dim=hidden_dim))
        return Controller(**net_args).to(self.device)

    def forward(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        return self.controller._predict(obs, deterministic=deterministic)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_args["net_arch"],
                activation_fn=self.net_args["activation_fn"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data


MlpPolicy = WMPolicy


class CnnPolicy(WMPolicy):
    """
    Policy class for DQN when using images as input.
    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(CnnPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )


register_policy("MlpPolicy", MlpPolicy)
register_policy("CnnPolicy", CnnPolicy)