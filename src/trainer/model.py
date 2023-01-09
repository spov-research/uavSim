from dataclasses import dataclass
from typing import TypeVar, Tuple

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Concatenate, MaxPool2D, LayerNormalization, \
    BatchNormalization, ReLU
from tensorflow.keras.activations import swish
from tensorflow.keras import Model

from tensorflow.keras.optimizers.schedules import ExponentialDecay

from utils import Factory


class QModel:
    @dataclass
    class Params:
        action_mask: bool = True
        positional_encoding: bool = True
        dueling: bool = True

    def __init__(self, params, obs_space, act_space):
        self.params = params
        self.observation_space = obs_space
        self.action_space = act_space
        self.model = self.create_model()

    @tf.function
    def get_random_action(self, obs):
        mask = obs["mask"] if self.params.action_mask else tf.ones_like(obs["mask"])
        return tf.random.categorical(tf.where(mask, 1., -np.inf), 1)[..., 0]

    @tf.function
    def get_max_action(self, obs):
        q_vals = self.get_q_values(obs)
        return tf.argmax(q_vals, axis=1)

    @tf.function
    def get_q_values(self, obs):
        q_vals = self.model(obs)["q_values"]
        if self.params.action_mask:
            q_vals = tf.where(obs["mask"], q_vals, -np.inf)
        return q_vals

    @tf.function
    def get_advantages(self, obs):
        return self.model(obs)["advantage"]

    @tf.function
    def get_output(self, obs):
        return self.model(obs)

    def create_model(self):
        pass

    def add_head(self, layer, hidden_layer_size, mask=None):
        if self.params.dueling:
            v_layer = Dense(hidden_layer_size, activation="relu")(layer)
            v_out = Dense(1, activation='linear')(v_layer)

            adv_layer = Dense(hidden_layer_size, activation="relu")(layer)
            adv_out = Dense(self.action_space.n, activation='linear')(adv_layer)

            if self.params.action_mask:
                assert mask is not None, "Mask is needed in dueling head"
                adv_out = tf.where(mask, adv_out, -np.inf)

            adv_out = adv_out - tf.reduce_max(adv_out, axis=-1, keepdims=True)

            q_out = v_out + adv_out

            outputs = {"q_values": q_out, "values": v_out, "advantage": adv_out}

        else:

            q_layer = Dense(hidden_layer_size, activation="relu")(layer)
            q_out = Dense(self.action_space.n, activation='linear')(q_layer)

            if self.params.action_mask:
                assert mask is not None, "Mask is needed in dueling head"
                q_out = tf.where(mask, q_out, -np.inf)

            values = tf.reduce_max(q_out, axis=-1, keepdims=True)
            advantages = q_out - values

            outputs = {"q_values": q_out, "values": values, "advantage": advantages}

        return outputs

    def positional_encoding(self, features, n=10_000):
        B, X, Y, d = features.shape
        I = tf.cast(d / 4, tf.int32)

        x = tf.reshape(tf.range(X, dtype=tf.float32), (-1, 1, 1))  # [X, 1, 1]
        y = tf.reshape(tf.range(Y, dtype=tf.float32), (1, -1, 1))  # [1, Y, 1]
        i = tf.reshape(tf.range(I, dtype=tf.float32), (1, 1, -1))  # [1, 1, I]

        s_x = tf.math.sin(x / tf.pow(n, 4 * (2 * i) / d))  # [X, 1, I]
        c_x = tf.math.cos(x / tf.pow(n, 4 * (2 * i + 1) / d))  # [X, 1, I]
        sc_x = tf.stack((s_x, c_x), axis=-1)  # [X, 1, I, 2]
        sc_x = tf.reshape(sc_x, (X, 1, -1))  # [X, 1, 2I]
        sc_x = tf.repeat(sc_x, Y, axis=1)  # [X, Y, 2I]

        s_y = tf.math.sin(y / tf.pow(n, 4 * (2 * i) / d))  # [1, Y, I]
        c_y = tf.math.cos(y / tf.pow(n, 4 * (2 * i + 1) / d))  # [1, Y, I]
        sc_y = tf.stack((s_y, c_y), axis=-1)  # [1, Y, I, 2]
        sc_y = tf.reshape(sc_y, (1, Y, -1))  # [1, Y, 2I]
        sc_y = tf.repeat(sc_y, X, axis=0)  # [X, Y, 2I]

        sc = tf.concat((sc_x, sc_y), axis=-1)  # [X, Y, d]
        sc = tf.expand_dims(sc, 0)  # [1, X, Y, d]

        return features + sc

    def load_network(self, model_path, model_name="model", weights="weights_latest"):
        self.model = tf.keras.models.load_model(model_path + f"/{model_name}")
        if weights is not None:
            self.load_weights(model_path, weights)

    def load_weights(self, model_path, weights="weights_latest"):
        self.model.load_weights(model_path + f"/{weights}")


class GlobLocFlattenQModel(QModel):
    @dataclass
    class Params(QModel.Params):
        # Convolutional part config
        conv_layers: int = 2
        conv_kernel_size: int = 5
        conv_kernels: int = 16

        # Fully Connected config
        hidden_layer_size: int = 256
        hidden_layer_num: int = 3

    def __init__(self, params: Params, obs_space, act_space):
        super().__init__(params, obs_space, act_space)

    def create_model(self):
        obs = self.observation_space
        global_map_input = Input(shape=obs["global_map"].shape[1:], dtype=tf.float32)
        local_map_input = Input(shape=obs["local_map"].shape[1:], dtype=tf.float32)
        scalars_input = Input(shape=obs["scalars"].shape[1:], dtype=tf.float32)
        mask_input = Input(shape=obs["mask"].shape[1:], dtype=tf.bool)

        global_map = global_map_input
        local_map = local_map_input
        conv_kernels = self.params.conv_kernels
        hidden_layer_size = self.params.hidden_layer_size

        if self.params.positional_encoding:
            global_map = Conv2D(conv_kernels, 1, activation=None)(global_map)  # linear pixel-wise embedding
            global_map = self.positional_encoding(global_map)
            local_map = Conv2D(conv_kernels, 1, activation=None)(local_map)  # linear pixel-wise embedding
            local_map = self.positional_encoding(local_map)

        for k in range(self.params.conv_layers):
            global_map = Conv2D(conv_kernels, self.params.conv_kernel_size, activation="relu")(global_map)
            local_map = Conv2D(conv_kernels, self.params.conv_kernel_size, activation="relu")(local_map)

        flatten_global = Flatten()(global_map)
        flatten_local = Flatten()(local_map)
        layer = Concatenate()([flatten_global, flatten_local, scalars_input])

        hidden_layer_num = self.params.hidden_layer_num - 1
        for k in range(hidden_layer_num):
            layer = Dense(hidden_layer_size, activation="relu")(layer)

        outputs = self.add_head(layer, hidden_layer_size, mask_input)

        return Model(
            inputs={"global_map": global_map_input, "local_map": local_map_input, "scalars": scalars_input,
                    "mask": mask_input},
            outputs=outputs)


class GlobLocReduceQModel(QModel):
    @dataclass
    class Params(QModel.Params):
        # Convolutional part config
        conv_layers: int = 4
        conv_kernel_size: int = 3
        conv_kernels: int = 32

        # Fully Connected config
        hidden_layer_size: int = 256
        hidden_layer_num: int = 3

    def __init__(self, params: Params, obs_space, act_space):
        super().__init__(params, obs_space, act_space)

    def create_model(self):
        obs = self.observation_space
        global_map_input = Input(shape=obs["global_map"].shape[1:], dtype=tf.float32)
        local_map_input = Input(shape=obs["local_map"].shape[1:], dtype=tf.float32)
        scalars_input = Input(shape=obs["scalars"].shape[1:], dtype=tf.float32)
        mask_input = Input(shape=obs["mask"].shape[1:], dtype=tf.bool)

        global_map = global_map_input
        local_map = local_map_input
        conv_kernels = self.params.conv_kernels
        hidden_layer_size = self.params.hidden_layer_size
        kernel_size = self.params.conv_kernel_size

        if self.params.positional_encoding:
            global_map = Conv2D(conv_kernels, 1, activation=None)(global_map)  # linear pixel-wise embedding
            global_map = self.positional_encoding(global_map)
            local_map = Conv2D(conv_kernels, 1, activation=None)(local_map)  # linear pixel-wise embedding
            local_map = self.positional_encoding(local_map)

        # Feature Extraction
        for _ in range(self.params.conv_layers):
            global_map = Conv2D(conv_kernels, kernel_size, activation="relu", padding="same")(global_map)
            local_map = Conv2D(conv_kernels, kernel_size, activation="relu", padding="same")(local_map)

            conv_kernels *= 2
            global_map = Conv2D(conv_kernels, kernel_size, activation="relu", padding="same")(global_map)
            local_map = Conv2D(conv_kernels, kernel_size, activation="relu", padding="same")(local_map)
            global_map = MaxPool2D(2)(global_map)
            local_map = MaxPool2D(2)(local_map)

        # Global Feature and Scalars mixing
        global_features = tf.reduce_max(tf.reduce_max(global_map, axis=1), axis=1)
        global_features = tf.concat((global_features, scalars_input), axis=1)
        global_features = Dense(hidden_layer_size, activation="relu")(global_features)

        # Local and global Feature Mixing
        _, x, y, _ = local_map.shape
        global_features = tf.reshape(global_features, (-1, 1, 1, hidden_layer_size))
        global_features = tf.repeat(global_features, x, axis=1)
        global_features = tf.repeat(global_features, y, axis=2)
        local_map = tf.concat((local_map, global_features), axis=3)

        # Shared-MLP
        local_map = Conv2D(hidden_layer_size, 1, activation="relu")(local_map)
        local_map = Conv2D(hidden_layer_size, 1, activation="relu")(local_map)

        # Feature extraction for landing
        layer = tf.reduce_max(tf.reduce_max(local_map, axis=1), axis=1)
        hidden_layer_num = self.params.hidden_layer_num - 1
        for k in range(hidden_layer_num):
            layer = Dense(hidden_layer_size, activation="relu")(layer)

        outputs = self.add_head(layer, hidden_layer_size, mask_input)

        return Model(
            inputs={"global_map": global_map_input, "local_map": local_map_input, "scalars": scalars_input,
                    "mask": mask_input},
            outputs=outputs)


class GlobLocResNetReduceQModel(QModel):
    @dataclass
    class Params(QModel.Params):
        # Convolutional part config
        conv_layers: int = 2
        conv_kernel_size: int = 3
        conv_kernels: int = 32

        # Fully Connected config
        hidden_layer_size: int = 256
        hidden_layer_num: int = 3

    def __init__(self, params: Params, obs_space, act_space):
        super().__init__(params, obs_space, act_space)

    def res_block(self, input_layer, pooling=2, bottleneck_filters=16, output_filters=32, layer_normalization=False,
                  swish_activation=False):
        normalization_layer = LayerNormalization if layer_normalization else BatchNormalization
        skip_layer = input_layer
        if pooling > 1 or skip_layer.shape[-1] != output_filters:
            skip_layer = Conv2D(output_filters, 3, padding='same', activation=None)(skip_layer)
            skip_layer = MaxPool2D(pooling)(skip_layer)
        conv_layer = Conv2D(bottleneck_filters, 1, padding='same', activation=None)(input_layer)
        conv_layer = normalization_layer()(conv_layer)
        conv_layer = swish(conv_layer) if swish_activation else ReLU()(conv_layer)
        conv_layer = Conv2D(bottleneck_filters, 3, padding='same', activation=None)(conv_layer)
        conv_layer = normalization_layer()(conv_layer)
        conv_layer = swish(conv_layer) if swish_activation else ReLU()(conv_layer)
        conv_layer = MaxPool2D(pooling)(conv_layer)
        conv_layer = Conv2D(output_filters, 1, padding='same', activation=None)(conv_layer)
        conv_layer = normalization_layer()(conv_layer)
        conv_layer += skip_layer
        conv_layer = swish(conv_layer) if swish_activation else ReLU()(conv_layer)
        return conv_layer

    def create_model(self):
        obs = self.observation_space
        global_map_input = Input(shape=obs["global_map"].shape[1:], dtype=tf.float32)
        local_map_input = Input(shape=obs["local_map"].shape[1:], dtype=tf.float32)
        scalars_input = Input(shape=obs["scalars"].shape[1:], dtype=tf.float32)

        global_map = global_map_input
        local_map = local_map_input
        conv_kernels = self.params.conv_kernels
        hidden_layer_size = self.params.hidden_layer_size
        kernel_size = self.params.conv_kernel_size

        global_map = Conv2D(conv_kernels, 1, activation=None)(global_map)  # linear pixel-wise embedding
        local_map = Conv2D(conv_kernels, 1, activation=None)(local_map)  # linear pixel-wise embedding

        if self.params.positional_encoding:
            global_map = self.positional_encoding(global_map)
            local_map = self.positional_encoding(local_map)

        # Feature Extraction
        for _ in range(self.params.conv_layers):
            global_map = self.res_block(input_layer=global_map, pooling=2, bottleneck_filters=conv_kernels,
                                        output_filters=conv_kernels * 2, layer_normalization=True,
                                        swish_activation=True)
            local_map = self.res_block(input_layer=local_map, pooling=2, bottleneck_filters=conv_kernels,
                                       output_filters=conv_kernels * 2, layer_normalization=True,
                                       swish_activation=True)
            conv_kernels *= 2

        # Global Feature and Scalars mixing
        global_features = tf.reduce_max(tf.reduce_max(global_map, axis=1), axis=1)
        global_features = tf.concat((global_features, scalars_input), axis=1)
        global_features = Dense(hidden_layer_size, activation="relu")(global_features)

        # Local and global Feature Mixing
        _, x, y, _ = local_map.shape
        global_features = tf.reshape(global_features, (-1, 1, 1, hidden_layer_size))
        global_features = tf.repeat(global_features, x, axis=1)
        global_features = tf.repeat(global_features, y, axis=2)
        local_map = tf.concat((local_map, global_features), axis=3)

        # Shared-MLP
        local_map = Conv2D(hidden_layer_size, 1, activation="relu")(local_map)
        local_map = Conv2D(hidden_layer_size, 1, activation="relu")(local_map)

        # Feature extraction for landing
        layer = tf.reduce_max(tf.reduce_max(local_map, axis=1), axis=1)
        hidden_layer_num = self.params.hidden_layer_num
        for k in range(hidden_layer_num):
            layer = Dense(hidden_layer_size, activation="relu")(layer)

        output = Dense(self.action_space.n, activation='linear')(layer)

        return Model(
            inputs={"global_map": global_map_input, "local_map": local_map_input, "scalars": scalars_input},
            outputs={"q_values": output})


class CenteredReduceQModel(QModel):
    @dataclass
    class Params(QModel.Params):
        # Convolutional part config
        conv_layers: int = 4
        conv_kernel_size: int = 3
        conv_kernels: int = 32

        # Fully Connected config
        hidden_layer_size: int = 256
        hidden_layer_num: int = 3

    def __init__(self, params: Params, obs_space, act_space):
        super().__init__(params, obs_space, act_space)

    def create_model(self):
        obs = self.observation_space
        centered_map_input = Input(shape=obs["centered_map"].shape[1:], dtype=tf.float32)
        scalars_input = Input(shape=obs["scalars"].shape[1:], dtype=tf.float32)

        centered_map = centered_map_input
        conv_kernels = self.params.conv_kernels
        hidden_layer_size = self.params.hidden_layer_size
        kernel_size = self.params.conv_kernel_size

        # Feature Extraction
        for _ in range(self.params.conv_layers):
            centered_map = Conv2D(conv_kernels, kernel_size, activation="relu", padding="same")(centered_map)

            conv_kernels *= 2
            centered_map = Conv2D(conv_kernels, kernel_size, activation="relu", padding="same")(centered_map)
            centered_map = MaxPool2D(2)(centered_map)

        scalars_layer = scalars_input
        scalars_layer = Dense(hidden_layer_size, activation="relu")(scalars_layer)

        # Local and global Feature Mixing
        _, x, y, _ = centered_map.shape
        scalars_features = tf.reshape(scalars_layer, (-1, 1, 1, hidden_layer_size))
        scalars_features = tf.repeat(scalars_features, x, axis=1)
        scalars_features = tf.repeat(scalars_features, y, axis=2)
        centered_map = tf.concat((centered_map, scalars_features), axis=3)

        # Shared-MLP
        centered_map = Conv2D(hidden_layer_size, 1, activation="relu")(centered_map)
        centered_map = Conv2D(hidden_layer_size, 1, activation="relu")(centered_map)

        # Feature extraction for landing
        layer = tf.reduce_max(tf.reduce_max(centered_map, axis=1), axis=1)
        hidden_layer_num = self.params.hidden_layer_num
        for k in range(hidden_layer_num):
            layer = Dense(hidden_layer_size, activation="relu")(layer)

        output = Dense(self.action_space.n, activation='linear')(layer)

        return Model(
            inputs={"centered_map": centered_map_input, "scalars": scalars_input},
            outputs={"q_values": output})


class QModelFactory(Factory):

    @classmethod
    def registry(cls):
        return {
            "flatten": GlobLocFlattenQModel,
            "reduce": GlobLocReduceQModel,
            "centered": CenteredReduceQModel,
            "resnet": GlobLocResNetReduceQModel
        }

    @classmethod
    def defaults(cls) -> Tuple[str, type]:
        return "flatten", GlobLocFlattenQModel


@dataclass
class SoftmaxPolicyParams:
    temperature: float = 0.1
    decay_steps: int = 2_000_000
    decay_rate: float = 0.01
    advantage_explore: bool = False


class SoftmaxPolicy:

    def __init__(self, params: SoftmaxPolicyParams, n):
        super().__init__()
        self.params = params
        self.temperature = ExponentialDecay(self.params.temperature,
                                            decay_steps=self.params.decay_steps,
                                            decay_rate=self.params.decay_rate)
        self.n = n

    @tf.function
    def sample(self, nn_output, step):
        values = self.get_values(nn_output)
        return tf.random.categorical(values / self.temperature(step), 1)[..., 0]

    @tf.function
    def get_probs(self, nn_output, step):
        values = self.get_values(nn_output)
        return tf.math.softmax(values / self.temperature(step))

    def get_values(self, nn_output):
        return nn_output["advantages"] if self.params.advantage_explore else nn_output["q_values"]
