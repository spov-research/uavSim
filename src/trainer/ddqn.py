from dataclasses import dataclass
from typing import Optional

import numpy as np
import pygame
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tqdm import tqdm

from src.trainer.base import BaseTrainer
from src.trainer.memory import ReplayMemoryFactory
from src.gym.grid import GridGym
from src.base.logger import Logger


class DDQNTrainer(BaseTrainer):
    @dataclass
    class Params(BaseTrainer.Params):
        # Replay Memory
        memory: ReplayMemoryFactory.default_param_type() = ReplayMemoryFactory.default_params()
        rm_prefill: int = 25_000
        max_inactivity_steps: int = 100
        reward_scale: float = 1.0

        cql_alpha: float = 0.0

    def __init__(self, params: Params, gym: GridGym, logger: Optional[Logger], q_model, q_target_model,
                 policy, observation_function):
        super().__init__(params, gym, observation_function)
        self.params = params
        self.logger = logger

        self.q_model = q_model
        self.target_q_model = q_target_model
        self.policy = policy

        self.train_steps = tf.Variable(initial_value=0, dtype=tf.int32)

        self.replay_memory = ReplayMemoryFactory.create(self.params.memory)

        self.learning_rate = ExponentialDecay(self.params.lr.base,
                                              decay_steps=self.params.lr.decay_steps,
                                              decay_rate=self.params.lr.decay_rate)
        self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate, amsgrad=self.params.amsgrad)
        self.gym.register_render(self.render, shape=(250, 750))

    def train(self):
        state, _ = self.gym.reset()
        obs = self.observation_function(state)
        for _ in tqdm(range(self.params.training_steps)):
            action = self.get_action(obs)
            next_state, reward, terminal, truncated, info = self.gym.step(action)
            next_obs = self.observation_function(next_state)
            self.store_experience(obs, action, reward * self.params.reward_scale, next_obs, terminal)
            self.train_step()

            self.logger.log_step(info)

            if terminal or info["timeout"]:
                self.logger.log_episode(info)
                state, _ = self.gym.reset()
                obs = self.observation_function(state)
            else:
                obs = next_obs

    def get_action(self, obs):
        if self.replay_memory.get_size() < self.params.rm_prefill:
            return self.q_model.get_random_action(obs)[0].numpy()

        return self.get_exploration_action_tf(obs)[0].numpy()

    @tf.function
    def get_exploration_action_tf(self, obs):
        nn_output = self.q_model.get_output(obs)

        return self.policy.sample(nn_output, self.train_steps)

    def store_experience(self, obs, action, reward, next_obs, terminal):
        self.replay_memory.store(
            [
                *self.observation_function.to_experience(obs),
                np.array(action).astype(np.int32),
                np.array(reward).astype(np.float32),
                *self.observation_function.to_experience(next_obs),
                np.array(terminal).astype(bool)
            ]
        )

    def train_step(self):
        if self.replay_memory.get_size() < max(self.params.rm_prefill, self.params.batch_size):
            return
        exp, w, index = self.replay_memory.sample(self.params.batch_size)

        s = self.observation_function.experience_length()

        obs = self.observation_function.from_experience_batch(exp)

        action = tf.convert_to_tensor(exp[s])
        reward = tf.convert_to_tensor(exp[s + 1])

        next_obs = self.observation_function.from_experience_batch(exp[s + 2:])
        terminal = tf.convert_to_tensor(exp[2 * s + 2])

        weight = tf.convert_to_tensor(w)

        td_error, logs = self._train_step_tf(obs, action, reward, next_obs, terminal, weight)
        self.replay_memory.update_td_error(index, td_error)
        rm_logs = self.replay_memory.get_logs()
        logs.update({"weight_min": np.min(w), "weight_max": np.max(w),
                     "weight_avg": np.mean(w)})
        logs.update(rm_logs)
        self.logger.log_train(logs)

    @tf.function
    def _train_step_tf(self, obs, action, reward, next_obs, terminal, weight):
        # Get the Q-values for the next state using the primary Q network
        next_q_vals = self.q_model.get_q_values(next_obs)
        # Select the action with the highest Q-value for the next state using the primary Q network
        next_action = tf.argmax(next_q_vals, axis=-1)
        # Get the Q-value for the next action using the target Q network
        next_q = tf.gather_nd(self.target_q_model.get_q_values(next_obs), tf.expand_dims(next_action, -1), batch_dims=1)

        # Compute the target Q-value using the Bellman equation
        target = reward + (1.0 - tf.cast(terminal, tf.float32)) * self.params.gamma * next_q

        logs = {}

        # Use a GradientTape to compute the loss and gradients
        with tf.GradientTape() as tape:
            # Get the Q-values for the current state using the primary Q network
            q_values = self.q_model.get_q_values(obs)
            # Select the Q-value for the action taken using the primary Q network
            q = tf.gather_nd(q_values, tf.expand_dims(action, -1), batch_dims=1)
            # Compute the TD error
            td_error = (target - q) * tf.cast(weight, tf.float32)
            # Compute the loss using the Huber loss function
            q_loss = tf.keras.losses.Huber()(0.0, td_error)

            if self.params.cql_alpha > 0.0:
                logsumexp = tf.reduce_logsumexp(q_values, axis=-1)
                cql_loss = tf.reduce_mean(logsumexp - q)
                loss = self.params.cql_alpha * cql_loss + 1 / 2 * q_loss
                logs["cql_loss"] = cql_loss
                logs["q_loss"] = q_loss
                logs["loss"] = loss
                logs["logsumexp"] = tf.reduce_mean(logsumexp)
            else:
                loss = q_loss
                logs["q_loss"] = loss
                logs["loss"] = loss
        logs["q"] = tf.reduce_mean(q)

        # Compute the gradients with respect to the primary Q network's trainable variables
        grads = tape.gradient(loss, self.q_model.model.trainable_variables)
        # Apply the gradients to the primary Q network
        self.optimizer.apply_gradients(zip(grads, self.q_model.model.trainable_variables))

        # Softly update the target Q network towards the primary Q network
        self._soft_update_tf(self.q_model.model, self.target_q_model.model, self.params.alpha)
        # Increment the train steps
        self.train_steps.assign_add(1)

        logs["grads"] = tf.reduce_mean([tf.reduce_mean(tf.abs(g)) for g in grads])
        logs["td_min"] = tf.reduce_min(tf.abs(td_error))
        logs["td_max"] = tf.reduce_max(tf.abs(td_error))
        logs["td_avg"] = tf.reduce_mean(tf.abs(td_error))

        # Return TD Error and logs
        return td_error, logs

    def render(self, canvas: pygame.Surface, state):
        pygame.font.init()
        font = pygame.font.SysFont('Arial', 16)
        obs = self.observation_function(state)
        nn_output = self.q_model.get_output(obs)
        q_values = nn_output["q_values"].numpy()[0]
        advantages = nn_output["advantages"].numpy()[0] if "advantages" in nn_output else None
        p = self.policy.get_probs(nn_output, step=0).numpy()

        tile_size = 50
        q_action_canvas = self.gym.draw_action_grid([f"{v:.2f}" for v in q_values], tile_size)
        p_action_canvas = self.gym.draw_action_grid([f"{v * 100:.1f}" for v in p], tile_size)
        mask_action_canvas = self.gym.draw_action_grid([f"{v:d}" for v in obs["mask"][0].numpy()], tile_size)

        canvas.blit(font.render("Mask Ground Truth", True, (0, 200, 0)), (0, 0))
        canvas.blit(mask_action_canvas, (0, 20))
        canvas.blit(font.render("Q-Values", True, (0, 200, 0)), (0, 170))
        canvas.blit(q_action_canvas, (0, 190))
        canvas.blit(font.render("pi(a|s)", True, (0, 200, 0)), (0, 510))
        canvas.blit(p_action_canvas, (0, 530))

        if advantages is not None:
            adv_action_canvas = self.gym.draw_action_grid([f"{adv:.3f}" for adv in advantages], tile_size)
            canvas.blit(font.render("Advantages", True, (0, 200, 0)), (0, 340))
            canvas.blit(adv_action_canvas, (0, 360))

        return canvas
