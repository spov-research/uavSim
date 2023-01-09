from dataclasses import dataclass

from gymnasium import spaces

from src.gym.grid import GridGym
import tensorflow as tf

from src.trainer.utils import DecayParams


def polynomial_activation(layer, degree=3):
    return tf.concat([layer ** d for d in range(1, degree + 1)], axis=-1)


class BaseTrainer:
    @dataclass
    class Params:
        training_steps: int = 2_000_000
        batch_size: int = 128

        # Training Params
        amsgrad: bool = True
        lr: DecayParams = DecayParams(3e-4, 0.1, 1_000_000)
        alpha: float = 0.005
        gamma: float = 0.95

    def __init__(self, params: Params, gym: GridGym, observation_function):
        self.params = params
        self.gym = gym
        self.observation_function = observation_function

        assert isinstance(gym.action_space, spaces.Discrete)
        self.action_space: spaces.Discrete = gym.action_space
        self.observation_space = observation_function.get_observation_space(gym.observation_space.sample())

    @staticmethod
    @tf.function
    def _soft_update_tf(network, target_network, alpha):
        weights = network.weights
        target_weights = target_network.weights
        new_weights = [w_new * alpha + w_old * (1. - alpha) for w_new, w_old
                       in zip(weights, target_weights)]
        [target_weight.assign(new_weight) for new_weight, target_weight in zip(new_weights, target_weights)]
