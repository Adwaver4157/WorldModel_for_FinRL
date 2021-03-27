from typing import Any, Dict, List, Optional, Tuple, Type, Union
from time import sleep
import gym
import cma
import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common import logger
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_linear_fn, is_vectorized_observation, polyak_update

from torch.multiprocessing import Process, Queue

from finrl.model.WorldModel.utils.learning import EarlyStopping
from finrl.model.WorldModel.utils.learning import ReduceLROnPlateau
from finrl.model.WorldModel.utils.replay import ReplayBufferAD
from finrl.model.WorldModel.utils.misc import RolloutGenerator
from finrl.model.WorldModel.utils.misc import load_parameters
from finrl.model.WorldModel.utils.misc import flatten_parameters
from finrl.model.WorldModel.policies import WMPolicy
from finrl.model.WorldModel.mdnrnn import gmm_loss

LATENT, HIDDEN = 32, 256
BSIZE = 16
SEQ_LEN = 32

class WorldModel(OffPolicyAlgorithm):
    """
    Soft Actor-Critic (SAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup), from the softlearning repo
    (https://github.com/rail-berkeley/softlearning/)
    and from Stable Baselines (https://github.com/hill-a/stable-baselines)
    Paper: https://arxiv.org/abs/1801.01290
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html
    Note: we use double q target and not value target as discussed
    in https://github.com/hill-a/stable-baselines/issues/270
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
    :param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[WMPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1000000,
        learning_starts: int = 50000,
        batch_size: Optional[int] = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super(WorldModel, self).__init__(
            policy,
            env,
            WMPolicy,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=None,  # No action noise
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Box,),
        )

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.target_update_interval = target_update_interval
        self.max_grad_norm = max_grad_norm
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0
        # VAE and MDNRNN will be defined in `_setup_model()`
        self.vae, self.mdnrnn, self.mdnrnncell, self.controller = None, None, None, None
        self.hidden = None

        if _init_setup_model:
            self._setup_model()

    def loss_function(self, recon_x, x, mu, logsigma):
        delta = 1e-7
        BCE = th.mean(x * th.log(recon_x + delta) + (1 - x) * th.log(1 - recon_x + delta))
        # BCE = F.mse_loss(recon_x, x, size_average=False)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

        KLD = -0.5 * th.sum(1 + 2 * logsigma + delta - mu.pow(2) - (2 * logsigma).exp())
        return BCE + KLD

    def to_latent(self, obs, next_obs):
        """ Transform observations to latent space.

        :args obs: 5D torch tensor (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)
        :args next_obs: 5D torch tensor (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)

        :returns: (latent_obs, latent_next_obs)
            - latent_obs: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
            - next_latent_obs: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
        """
        with th.no_grad():
            print(obs.size())
            """
            obs, next_obs = [
                f.upsample(x.view(-1, 1, 7), size=7,
                           mode='bilinear', align_corners=True)
                for x in (obs, next_obs)]
            print(obs.size())
            """

            (obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma) = [
                self.vae(x)[1:] for x in (obs, next_obs)]

            latent_obs, latent_next_obs = [
                (x_mu + x_logsigma.exp() * th.randn_like(x_mu)).view(BSIZE, SEQ_LEN, LATENT)
                for x_mu, x_logsigma in
                [(obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma)]]
        return latent_obs, latent_next_obs

    def get_loss(self, latent_obs, action, reward, terminal,
                 latent_next_obs, include_reward: bool):
        """ Compute losses.

        The loss that is computed is:
        (GMMLoss(latent_next_obs, GMMPredicted) + MSE(reward, predicted_reward) +
             BCE(terminal, logit_terminal)) / (LSIZE + 2)
        The LSIZE + 2 factor is here to counteract the fact that the GMMLoss scales
        approximately linearily with LSIZE. All losses are averaged both on the
        batch and the sequence dimensions (the two first dimensions).

        :args latent_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor
        :args action: (BSIZE, SEQ_LEN, ASIZE) torch tensor
        :args reward: (BSIZE, SEQ_LEN) torch tensor
        :args latent_next_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor

        :returns: dictionary of losses, containing the gmm, the mse, the bce and
            the averaged loss.
        """
        latent_obs, action,\
            reward, terminal,\
            latent_next_obs = [arr.transpose(1, 0)
                               for arr in [latent_obs, action,
                                           reward, terminal,
                                           latent_next_obs]]
        mus, sigmas, logpi, rs, ds = self.mdnrnn(action, latent_obs)
        gmm = gmm_loss(latent_next_obs, mus, sigmas, logpi)
        bce = F.binary_cross_entropy_with_logits(ds, terminal)
        if include_reward:
            mse = F.mse_loss(rs, reward)
            scale = LSIZE + 2
        else:
            mse = 0
            scale = LSIZE + 1
        loss = (gmm + bce + mse) / scale
        return dict(gmm=gmm, bce=bce, mse=mse, loss=loss)
    
    def slave_routine(self, p_queue, r_queue, e_queue, p_index):
        """ Thread routine.

        Threads interact with p_queue, the parameters queue, r_queue, the result
        queue and e_queue the end queue. They pull parameters from p_queue, execute
        the corresponding rollout, then place the result in r_queue.

        Each parameter has its own unique id. Parameters are pulled as tuples
        (s_id, params) and results are pushed as (s_id, result).  The same
        parameter can appear multiple times in p_queue, displaying the same id
        each time.

        As soon as e_queue is non empty, the thread terminate.

        When multiple gpus are involved, the assigned gpu is determined by the
        process index p_index (gpu = p_index % n_gpus).

        :args p_queue: queue containing couples (s_id, parameters) to evaluate
        :args r_queue: where to place results (s_id, results)
        :args e_queue: as soon as not empty, terminate
        :args p_index: the process index
        """
        # init routine
        if th.cuda.device_count() != 0:
            gpu = p_index % th.cuda.device_count()
        else:
            gpu = 0
        device = th.device('cuda:{}'.format(gpu) if th.cuda.is_available() else 'cpu')

        with th.no_grad():
            r_gen = RolloutGenerator(
                self.vae, self.mdnrnn, self.mdnrnncell, self.controller, self.env, self.device, 1000)

            while e_queue.empty():
                if p_queue.empty():
                    sleep(.1)
                else:
                    s_id, params = p_queue.get()
                    r_queue.put((s_id, r_gen.rollout(params)))

    def evaluate(solutions, results, rollouts=100):
        """ Give current controller evaluation.

        Evaluation is minus the cumulated reward averaged over rollout runs.

        :args solutions: CMA set of solutions
        :args results: corresponding results
        :args rollouts: number of rollouts

        :returns: minus averaged cumulated reward
        """
        index_min = np.argmin(results)
        best_guess = solutions[index_min]
        restimates = []

        for s_id in range(rollouts):
            p_queue.put((s_id, best_guess))

        print("Evaluating...")
        for _ in tqdm(range(rollouts)):
            while r_queue.empty():
                sleep(.1)
            restimates.append(r_queue.get()[1])

        return best_guess, np.mean(restimates), np.std(restimates)

    def _setup_model(self) -> None:
        super(WorldModel, self)._setup_model()
        self._create_aliases()

    def _create_aliases(self) -> None:
        self.vae = self.policy.vae
        self.mdnrnn = self.policy.mdnrnn
        self.mdnrnncell = self.policy.mdnrnncell
        self.controller = self.policy.controller

    def train_vae(self, gradient_steps: int, batch_size: int = 64) -> None:
        print('TRAIN VAE')
        self._update_learning_rate(self.vae.optimizer)
        losses = []
        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            recon_batch, mu, logvar = self.vae(replay_data.observations)
            loss = self.loss_function(recon_batch, data, mu, logvar)
            losses.append(loss.item())
            # Optimize the policy
            self.vae.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.vae.parameters(), self.max_grad_norm)
            self.vae.optimizer.step()

    def train_mdnrnn(self, gradient_steps: int, batch_size: int = 64) -> None:
        print('TRAIN MDNRNN')
        self._update_learning_rate(self.vae.optimizer)
        losses = []
        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.seq_sample(SEQ_LEN, batch_size, env=self._vec_normalize_env)
            obs, action, next_obs, terminal, reward = \
                replay_data.observations, replay_data.actions, replay_data.next_observations, replay_data.dones, replay_data.rewards
            # transform obs
            latent_obs, latent_next_obs = self.to_latent(obs, next_obs)
    
            loss = get_loss(latent_obs, action, reward,
                              terminal, latent_next_obs, include_reward)
            losses.append(loss['loss'].item())
            # Optimize the policy
            self.mdnrnn.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.mdnrnn.parameters(), self.max_grad_norm)
            self.mdnrnn.optimizer.step()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default
        :param array:
        :param copy: Whether to copy or not the data
            (may be useful to avoid changing things be reference)
        :return:
        """
        if copy:
            return th.tensor(array).to(self.device)
        return th.as_tensor(array).to(self.device)

    def predict(self, obs):
        obs = map(self.to_torch, obs)
        _, latent_mu, _ = self.vae(obs)
        action = self.controller(latent_mu, self.hidden[0])
        _, _, _, _, _, self.hidden = self.mdnrnncell(action, latent_mu, self.hidden)
        return action.reshape(-1)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "run",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "OffPolicyAlgorithm":

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        callback.on_training_start(locals(), globals())

        # train vae
        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if rollout.continue_training is False:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps > 0 else rollout.episode_timesteps
                self.train_vae(batch_size=self.batch_size, gradient_steps=gradient_steps)

        # train mdnrnn
        self.replay_buffer = ReplayBufferAD(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            self.device,
            optimize_memory_usage=self.optimize_memory_usage,
        )

        total_timesteps = 30
        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if rollout.continue_training is False:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps > 0 else rollout.episode_timesteps
                self.train_mdnrnn(batch_size=self.batch_size, gradient_steps=gradient_steps)

        # train controller
        p_queue = Queue()
        r_queue = Queue()
        e_queue = Queue()
        num_workers = 16

        for p_index in range(num_workers):
            Process(target=self.slave_routine, args=(p_queue, r_queue, e_queue, p_index)).start()

        cur_best = None

        parameters = self.controller.parameters()
        es = cma.CMAEvolutionStrategy(flatten_parameters(parameters), 0.1,
                              {'popsize': 4})

        epoch = 0
        log_step = 3
        while not es.stop():
            if cur_best is not None and - cur_best > 950:
                print("Already better than target, breaking...")
                break
            
            r_list = [0] * 4  # result list
            solutions = es.ask()

            # push parameters to queue
            for s_id, s in enumerate(solutions):
                for _ in range(4):
                    p_queue.put((s_id, s))

            # retrieve results
            for _ in range(16):
                while r_queue.empty():
                    sleep(.1)
                r_s_id, r = r_queue.get()
                r_list[r_s_id] += r / 4

            es.tell(solutions, r_list)
            es.disp()

            # evaluation and saving
            if epoch % log_step == log_step - 1:
                best_params, best, std_best = self.evaluate(solutions, r_list)
                print("Current evaluation: {}".format(best))
                if not cur_best or cur_best > best:
                    cur_best = best
                    print("Saving new best with value {}+-{}...".format(-cur_best, std_best))
                    load_parameters(best_params, self.controller)
                if - best > 950:
                    print("Terminating controller training with value {}...".format(best))
                    break
                
                
            epoch += 1

        es.result_pretty()
        e_queue.put('EOP')

        callback.on_training_end()

        return self

    def _excluded_save_params(self) -> List[str]:
        return super(WorldModel, self)._excluded_save_params() + ["vae", "mdnrnn", "controller"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]
        return state_dicts, []