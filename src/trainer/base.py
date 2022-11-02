from dataclasses import dataclass

import numpy as np
from gym import spaces

from src.gym.grid import GridGym
from tensorflow.keras.layers import AvgPool2D, Input, Conv2D, Flatten, Dense, Concatenate
from tensorflow.keras import Model
import tensorflow as tf


@dataclass
class BaseTrainerParams:
    training_steps: int = 2_000_000
    batch_size: int = 128

    global_map_scaling: int = 3
    local_map_size: int = 17

    temperature: float = 0.1

    # Convolutional part config
    conv_layers: int = 2
    conv_kernel_size: int = 5
    conv_kernels: int = 16

    # Fully Connected config
    hidden_layer_size: int = 256
    hidden_layer_num: int = 3

    # Training Params
    amsgrad: bool = True
    learning_rate: float = 3e-5
    alpha: float = 0.005
    gamma: float = 0.95
    save_period: int = 10_000

    action_mask: bool = False


def polynomial_activation(layer, degree=3):
    return tf.concat([layer ** d for d in range(1, degree + 1)], axis=-1)


class BaseTrainer:
    def __init__(self, params: BaseTrainerParams, gym: GridGym):
        self.params = params
        self.params = params
        self.gym = gym

        assert isinstance(gym.action_space, spaces.Discrete)
        self.action_space: spaces.Discrete = gym.action_space
        self.observation_space = self.get_observation_space()

    def create_network(self):
        obs = self.observation_space
        global_map_input = Input(shape=obs["global_map"].shape[1:], dtype=tf.float32)
        local_map_input = Input(shape=obs["local_map"].shape[1:], dtype=tf.float32)
        scalars_input = Input(shape=obs["scalars"].shape[1:], dtype=tf.float32)

        global_map = global_map_input
        local_map = local_map_input
        conv_kernels = self.params.conv_kernels
        hidden_layer_size = self.params.hidden_layer_size
        for k in range(self.params.conv_layers):
            global_map = Conv2D(conv_kernels, self.params.conv_kernel_size, activation="relu")(global_map)
            local_map = Conv2D(conv_kernels, self.params.conv_kernel_size, activation="relu")(local_map)

        flatten_global = Flatten()(global_map)
        flatten_local = Flatten()(local_map)

        layer = Concatenate()([flatten_global, flatten_local, scalars_input])
        for k in range(self.params.hidden_layer_num):
            layer = Dense(hidden_layer_size, activation="relu")(layer)

        output = Dense(self.action_space.n, activation='linear')(layer)

        return Model(inputs={"global_map": global_map_input, "local_map": local_map_input, "scalars": scalars_input},
                     outputs=output)

    def get_observation_space(self):
        state = self.gym.observation_space.sample()
        obs = self.prepare_observations(state)
        return spaces.Dict(
            {
                "global_map": spaces.Box(low=0, high=1, shape=obs["global_map"].shape, dtype=float),
                "local_map": spaces.Box(low=0, high=1, shape=obs["local_map"].shape, dtype=float),
                "scalars": spaces.Box(low=0, high=1, shape=obs["scalars"].shape, dtype=float)
            }
        )

    def prepare_observations(self, obs):
        map_layers = tf.expand_dims(obs["map"], 0)
        budget = tf.expand_dims(obs["budget"], 0)
        mask = tf.expand_dims(obs["mask"], 0)

        global_map, local_map, scalars = self._prepare_observations_tf(map_layers, budget)
        return {"global_map": global_map, "local_map": local_map, "scalars": scalars, "mask": mask}

    @tf.function
    def _prepare_observations_tf(self, map_layers, budget):
        map_layers = tf.cast(map_layers, dtype=tf.float32)
        global_map = AvgPool2D((self.params.global_map_scaling, self.params.global_map_scaling))(map_layers)

        crop_frac = float(self.params.local_map_size) / float(map_layers.shape[1])
        local_map = tf.image.central_crop(map_layers, crop_frac)

        max_budget = self.gym.params.budget_range[1]
        scalars = budget / max_budget
        return global_map, local_map, scalars

    @staticmethod
    @tf.function
    def _soft_update_tf(network, target_network, alpha):
        weights = network.weights
        target_weights = target_network.weights
        new_weights = [w_new * alpha + w_old * (1. - alpha) for w_new, w_old
                       in zip(weights, target_weights)]
        [target_weight.assign(new_weight) for new_weight, target_weight in zip(new_weights, target_weights)]
