from dataclasses import dataclass
from typing import Optional

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from src.trainer.base import BaseTrainerParams, BaseTrainer
from src.trainer.utils import ReplayMemory, ReplayMemoryParams
from src.gym.grid import GridGym
from src.base.logger import Logger


@dataclass
class DDQNTrainerParams(BaseTrainerParams):
    # Replay Memory
    memory: ReplayMemoryParams = ReplayMemoryParams()
    rm_prefill: int = 25_000


class DDQNTrainer(BaseTrainer):
    def __init__(self, params: DDQNTrainerParams, gym: GridGym, logger: Optional[Logger]):
        super().__init__(params, gym)
        self.params = params
        self.logger = logger

        self.network = self.create_network()
        self.target_network = self.create_network()
        self.hard_update()
        if self.logger is not None:
            self.logger.save_network(self.network)

        self.replay_memory = ReplayMemory(self.params.memory)
        self.optimizer = tf.optimizers.Adam(learning_rate=self.params.learning_rate, amsgrad=self.params.amsgrad)

    def train(self):
        state, _ = self.gym.reset()
        obs = self.prepare_observations(state)
        for step in tqdm(range(self.params.training_steps)):
            action = self.get_action(obs)
            next_state, reward, terminal, truncated, info = self.gym.step(action)
            self.logger.log_step(info)
            next_obs = self.prepare_observations(next_state)
            self.store_experience(obs, action, reward, next_obs, terminal)
            self.train_step()

            if terminal:
                state, _ = self.gym.reset()
                obs = self.prepare_observations(state)
            else:
                obs = next_obs

            if step % self.params.save_period == 0:
                self.logger.save_weights(self.network)

    def get_action(self, obs):
        n = self.action_space.n
        if self.replay_memory.get_size() < self.params.rm_prefill:
            p = np.ones(n) / n
            if self.params.action_mask:
                masked_p = obs["mask"][0].numpy().astype(float)
                sum_p = np.sum(masked_p)
                if sum_p == 0:
                    p = None
                else:
                    p = masked_p / sum_p
        else:
            p = self._get_soft_action_tf(obs)[0].numpy()
        action = np.random.choice(range(n), 1, p=p)[0]
        return action

    def get_exploit_action(self, obs):
        return self._get_action_tf(obs)[0].numpy()

    @tf.function
    def _get_soft_action_tf(self, obs):
        q_values = self.network(obs)
        if self.params.action_mask:
            q_values = tf.where(obs["mask"], q_values, -np.inf)
        p = tf.math.softmax(q_values / self.params.temperature, axis=-1)
        return p

    @tf.function
    def _get_action_tf(self, obs):
        q_values = self.network(obs)
        if self.params.action_mask:
            q_values = tf.where(obs["mask"], q_values, -np.inf)

        action = tf.argmax(q_values, axis=-1)
        return action

    def hard_update(self):
        self.target_network.set_weights(self.network.get_weights())

    def store_experience(self, obs, action, reward, next_obs, terminal):
        self.replay_memory.store(
            [
                obs["global_map"].numpy()[0].astype(np.float32),
                obs["local_map"].numpy()[0].astype(np.float32),
                obs["scalars"].numpy()[0].astype(np.float32),
                np.array(action).astype(np.int32),
                np.array(reward).astype(np.float32),
                next_obs["global_map"].numpy()[0].astype(np.float32),
                next_obs["local_map"].numpy()[0].astype(np.float32),
                next_obs["scalars"].numpy()[0].astype(np.float32),
                np.array(terminal).astype(bool),
                obs["mask"].numpy()[0].astype(bool),
                next_obs["mask"].numpy()[0].astype(bool)
            ]
        )

    def train_step(self):
        if self.replay_memory.get_size() < max(self.params.rm_prefill, self.params.batch_size):
            return
        exp = self.replay_memory.sample(self.params.batch_size)

        obs = {
            "global_map": tf.convert_to_tensor(exp[0]),
            "local_map": tf.convert_to_tensor(exp[1]),
            "scalars": tf.convert_to_tensor(exp[2])
        }

        action = tf.convert_to_tensor(exp[3])
        reward = tf.convert_to_tensor(exp[4])

        next_obs = {
            "global_map": tf.convert_to_tensor(exp[5]),
            "local_map": tf.convert_to_tensor(exp[6]),
            "scalars": tf.convert_to_tensor(exp[7])
        }
        terminal = tf.convert_to_tensor(exp[8])
        mask = tf.convert_to_tensor(exp[9])
        next_mask = tf.convert_to_tensor(exp[10])

        loss, grads = self._train_step_tf(obs, action, reward, next_obs, terminal, mask, next_mask)
        self.logger.log_train({"loss": loss, "grads": grads})

    @tf.function
    def _train_step_tf(self, obs, action, reward, next_obs, terminal, mask, next_mask):
        next_q_vals = self.network(next_obs)
        if self.params.action_mask:
            next_q_vals = tf.where(next_mask, next_q_vals, -np.inf)
        next_action = tf.argmax(next_q_vals, axis=-1)
        next_q = tf.gather_nd(self.target_network(next_obs), tf.expand_dims(next_action, -1), batch_dims=1)

        target = reward + (1.0 - tf.cast(terminal, tf.float32)) * self.params.gamma * next_q

        with tf.GradientTape() as tape:
            q_values = self.network(obs)
            q = tf.gather_nd(q_values, tf.expand_dims(action, -1), batch_dims=1)
            loss = tf.reduce_mean(tf.square(q - target))

        grads = tape.gradient(loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))

        self._soft_update_tf(self.network, self.target_network, self.params.alpha)

        return loss, tf.reduce_mean([tf.reduce_mean(tf.abs(g)) for g in grads])

    def load_network(self, model_path, model_name="model", weights="weights_latest"):
        self.network = tf.keras.models.load_model(model_path + f"/{model_name}")
        self.target_network = tf.keras.models.load_model(model_path + f"/{model_name}")
        if weights is not None:
            self.load_weights(model_path, weights)

    def load_weights(self, model_path, weights="weights_latest"):
        self.network.load_weights(model_path + f"/{weights}")
        self.target_network.load_weights(model_path + f"/{weights}")
