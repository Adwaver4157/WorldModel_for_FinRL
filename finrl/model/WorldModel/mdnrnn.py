from typing import Any, Dict, List, Optional, Type

import gym
import torch as th
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.normal import Normal

from stable_baselines3.common.policies import BasePolicy, register_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, NatureCNN, create_mlp
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.preprocessing import get_action_dim

def gmm_loss(batch, mus, sigmas, logpi, reduce=True): # pylint: disable=too-many-arguments
    """ Computes the gmm loss.

    Compute minus the log probability of batch under the GMM model described
    by mus, sigmas, pi. Precisely, with bs1, bs2, ... the sizes of the batch
    dimensions (several batch dimension are useful when you have both a batch
    axis and a time step axis), gs the number of mixtures and fs the number of
    features.

    :args batch: (bs1, bs2, *, fs) torch tensor
    :args mus: (bs1, bs2, *, gs, fs) torch tensor
    :args sigmas: (bs1, bs2, *, gs, fs) torch tensor
    :args logpi: (bs1, bs2, *, gs) torch tensor
    :args reduce: if not reduce, the mean in the following formula is ommited

    :returns:
    loss(batch) = - mean_{i1=0..bs1, i2=0..bs2, ...} log(
        sum_{k=1..gs} pi[i1, i2, ..., k] * N(
            batch[i1, i2, ..., :] | mus[i1, i2, ..., k, :], sigmas[i1, i2, ..., k, :]))

    NOTE: The loss is not reduced along the feature dimension (i.e. it should scale ~linearily
    with fs).
    """
    batch = batch.unsqueeze(-2)
    normal_dist = Normal(mus, sigmas)
    g_log_probs = normal_dist.log_prob(batch)
    g_log_probs = logpi + th.sum(g_log_probs, dim=-1)
    max_log_probs = th.max(g_log_probs, dim=-1, keepdim=True)[0]
    g_log_probs = g_log_probs - max_log_probs

    g_probs = th.exp(g_log_probs)
    probs = th.sum(g_probs, dim=-1)

    log_prob = max_log_probs.squeeze() + th.log(probs)
    if reduce:
        return - th.mean(log_prob)
    return - log_prob

class _MDNRNNBase(BasePolicy):
    """
    MDNRNN for WorldModel
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
        n_gaussian:int = 5,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        super(_MDNRNNBase, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        self.activation_fn = activation_fn
        self.features_extractor = features_extractor
        self.features_dim = features_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_gaussian = n_gaussian
        self.normalize_images = normalize_images
        self.gmm_linear = nn.Linear(
            hidden_dim, (2 * latent_dim + 1) * n_gaussian + 2)

    def forward(self, *inputs):
        pass

    def _predict(self, obs: th.Tensor) -> th.Tensor:
        return self.forward(obs)

class MDNRNN(_MDNRNNBase):
    """ MDNRNN model for multi steps forward """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        features_extractor: nn.Module,
        features_dim: int,
        latent_dim: int,
        hidden_dim: int,
        n_gaussian:int = 5,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        super(MDNRNN, self).__init__(
            observation_space,
            action_space,
            features_extractor,
            features_dim,
            latent_dim,
            hidden_dim,
            n_gaussian,
            net_arch,
            activation_fn,
            normalize_images
        )
        action_dim = get_action_dim(self.action_space)
        self.rnn = nn.LSTM(self.latent_dim + action_dim, self.hidden_dim)

    def forward(self, actions, latents): # pylint: disable=arguments-differ
        """ MULTI STEPS forward.

        :args actions: (SEQ_LEN, BSIZE, ASIZE) torch tensor
        :args latents: (SEQ_LEN, BSIZE, LSIZE) torch tensor

        :returns: mu_nlat, sig_nlat, pi_nlat, rs, ds, parameters of the GMM
        prediction for the next latent, gaussian prediction of the reward and
        logit prediction of terminality.
            - mu_nlat: (SEQ_LEN, BSIZE, N_GAUSS, LSIZE) torch tensor
            - sigma_nlat: (SEQ_LEN, BSIZE, N_GAUSS, LSIZE) torch tensor
            - logpi_nlat: (SEQ_LEN, BSIZE, N_GAUSS) torch tensor
            - rs: (SEQ_LEN, BSIZE) torch tensor
            - ds: (SEQ_LEN, BSIZE) torch tensor
        """
        seq_len, bs = actions.size(0), actions.size(1)

        ins = th.cat([actions, latents], dim=-1)
        outs, _ = self.rnn(ins)
        gmm_outs = self.gmm_linear(outs)

        stride = self.n_gaussian * self.latent_dim

        mus = gmm_outs[:, :, :stride]
        mus = mus.view(seq_len, bs, self.n_gaussian, self.latent_dim)

        sigmas = gmm_outs[:, :, stride:2 * stride]
        sigmas = sigmas.view(seq_len, bs, self.n_gaussian, self.latent_dim)
        sigmas = th.exp(sigmas)

        pi = gmm_outs[:, :, 2 * stride: 2 * stride + self.n_gaussian]
        pi = pi.view(seq_len, bs, self.n_gaussian)
        logpi = f.log_softmax(pi, dim=-1)

        rs = gmm_outs[:, :, -2]

        ds = gmm_outs[:, :, -1]

        return mus, sigmas, logpi, rs, ds

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

class MDNRNNCell(_MDNRNNBase):
    """ MDNRNN model for multi steps forward """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        features_extractor: nn.Module,
        features_dim: int,
        latent_dim: int,
        hidden_dim: int,
        n_gaussian:int = 5,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        super(MDNRNNCell, self).__init__(
            observation_space,
            action_space,
            features_extractor,
            features_dim,
            latent_dim,
            hidden_dim,
            n_gaussian,
            net_arch,
            activation_fn,
            normalize_images
        )
        action_dim = get_action_dim(self.action_space)
        self.rnn = nn.LSTMCell(self.latent_dim + action_dim, self.hidden_dim)
    
    def forward(self, action, latent, hidden): # pylint: disable=arguments-differ
        """ ONE STEP forward.
        :args actions: (BSIZE, ASIZE) torch tensor
        :args latents: (BSIZE, LSIZE) torch tensor
        :args hidden: (BSIZE, RSIZE) torch tensor
        :returns: mu_nlat, sig_nlat, pi_nlat, r, d, next_hidden, parameters of
        the GMM prediction for the next latent, gaussian prediction of the
        reward, logit prediction of terminality and next hidden state.
            - mu_nlat: (BSIZE, N_GAUSS, LSIZE) torch tensor
            - sigma_nlat: (BSIZE, N_GAUSS, LSIZE) torch tensor
            - logpi_nlat: (BSIZE, N_GAUSS) torch tensor
            - rs: (BSIZE) torch tensor
            - ds: (BSIZE) torch tensor
        """
        in_al = th.cat([action, latent], dim=1)

        next_hidden = self.rnn(in_al, hidden)
        out_rnn = next_hidden[0]

        out_full = self.gmm_linear(out_rnn)

        stride = self.n_gaussian * self.latent_dim

        mus = out_full[:, :stride]
        mus = mus.view(-1, self.n_gaussian, self.latent_dim)

        sigmas = out_full[:, stride:2 * stride]
        sigmas = sigmas.view(-1, self.n_gaussian, self.latent_dim)
        sigmas = th.exp(sigmas)

        pi = out_full[:, 2 * stride:2 * stride + self.n_gaussian]
        pi = pi.view(-1, self.n_gaussian)
        logpi = f.log_softmax(pi, dim=-1)

        r = out_full[:, -2]

        d = out_full[:, -1]

        return mus, sigmas, logpi, r, d, next_hidden
