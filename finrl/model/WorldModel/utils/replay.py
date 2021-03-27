import warnings
from abc import ABC, abstractmethod
from typing import Dict, Generator, Optional, Union
try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

import numpy as np
import torch as th
from gym import spaces

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize

class ReplayBufferAD(ReplayBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.
    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
    ):
        super(ReplayBufferAD, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        assert n_envs == 1, "Replay buffer only support single environment for now"

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        self.optimize_memory_usage = optimize_memory_usage
        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)
        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.next_observations = None
        else:
            self.next_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            total_memory_usage = self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            if self.next_observations is not None:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(self, obs: np.ndarray, next_obs: np.ndarray, action: np.ndarray, reward: np.ndarray, done: np.ndarray) -> None:
        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()
        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()
        else:
            self.next_observations[self.pos] = np.array(next_obs).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def seq_sample(self, seq, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_seq_samples(seq, batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, 0, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, 0, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, 0, :], env),
            self.actions[batch_inds, 0, :],
            next_obs,
            self.dones[batch_inds],
            self._normalize_reward(self.rewards[batch_inds], env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def _get_seq_samples(self, seq, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        if self.optimize_memory_usage:
            index = (batch_inds + 1) % self.buffer_size
            for i in range(len(batch_inds)):
              if i == 0:
                ndata = np.expand_dims(self.observations[index[i]:index[i]+seq, 0, :], 0)
              else:
                ndata = np.append(ndata, np.expand_dims(self.observations[index[i]:index[i]+seq, 0, :], 0), axis=0)
            next_obs = self._normalize_obs(ndata, env)
        else:
            for i in range(len(batch_inds)):
              if i == 0:
                ndata = np.expand_dims(self.next_observations[batch_inds[i]:batch_inds[i]+seq, 0, :], 0)
              else:
                ndata = np.append(ndata, np.expand_dims(self.next_observations[batch_inds[i]:batch_inds[i]+seq, 0, :], 0), axis=0)
            next_obs = self._normalize_obs(ndata, env)
        for i in range(len(batch_inds)):
          if i == 0:
            odata = np.expand_dims(self.observations[batch_inds[i]:batch_inds[i]+seq, 0, :], 0)
            adata = np.expand_dims(self.actions[batch_inds[i]:batch_inds[i]+seq, 0, :], 0)
            ddata = np.expand_dims(self.dones[batch_inds[i]:batch_inds[i]+seq], 0)
            rdata = np.expand_dims(self.rewards[batch_inds[i]:batch_inds[i]+seq], 0)
          else:
            odata = np.append(odata, np.expand_dims(self.observations[batch_inds[i]:batch_inds[i]+seq, 0, :], 0), axis=0)
            adata = np.append(adata, np.expand_dims(self.actions[batch_inds[i]:batch_inds[i]+seq, 0, :], 0), axis=0)
            ddata = np.append(ddata, np.expand_dims(self.dones[batch_inds[i]:batch_inds[i]+seq], 0), axis=0)
            rdata = np.append(rdata, np.expand_dims(self.rewards[batch_inds[i]:batch_inds[i]+seq], 0), axis=0)
        data = (
            self._normalize_obs(odata, env),
            adata,
            next_obs,
            ddata,
            self._normalize_reward(rdata, env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))