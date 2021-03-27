from typing import Any, Dict, List, Optional, Type

import gym
import torch as th
from torch import nn

from stable_baselines3.common.policies import BasePolicy, register_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, NatureCNN, create_mlp
from stable_baselines3.common.type_aliases import Schedule

def loss_function(recon_x, x, mu, logsigma):
    delta = 1e-7
    BCE = torch.mean(x * torch.log(recon_x + delta) + (1 - x) * torch.log(1 - recon_x + delta))
    # BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    
    KLD = -0.5 * th.sum(1 + 2 * logsigma + delta - mu.pow(2) - (2 * logsigma).exp())
    return BCE + KLD

class VAE(BasePolicy):
    """
    Variational AutoEncoder for WorldModel
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
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        super(VAE, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [64]
        else:
            net_arch = [64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_extractor = features_extractor
        self.features_dim = features_dim
        self.latent_dim = latent_dim
        self.normalize_images = normalize_images
        encoder = create_mlp(self.features_dim, 64, self.net_arch, self.activation_fn)
        self.encoder = nn.Sequential(*encoder)
        self.fc_mu = nn.Linear(64, self.latent_dim)
        self.fc_logsigma = nn.Linear(64, self.latent_dim)
        self.net_arch = [64, 64]
        decoder = create_mlp(self.latent_dim, self.features_dim, self.net_arch, self.activation_fn, True)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        Predict the q-values.
        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        x = self.encoder(self.extract_features(obs))
        mu = self.fc_mu(x)
        logsigma = self.fc_logsigma(x)
        sigma = logsigma.exp()
        eps = th.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)
        recon_x = self.decoder(z)

        return recon_x, mu, logsigma
    
    def _predict(self, obs: th.Tensor) -> th.Tensor:
        return self.forward(obs)

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
