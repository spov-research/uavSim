import copy
from copy import deepcopy
from dataclasses import dataclass

import gym
import numpy as np
import pygame
from gym import spaces

from src.gym.grid import GridGym, GridGymParams
from src.gym.utils import load_or_create_shadowing
from seaborn import color_palette


@dataclass
class ChannelParams:
    cell_edge_snr: float = -25  # in dB
    los_path_loss_exp: float = 2.27
    nlos_path_loss_exp: float = 3.64
    uav_altitude: float = 10.0  # in m
    cell_size: float = 10.0  # in m
    los_shadowing_variance: float = 2.0
    nlos_shadowing_variance: float = 5.0


class Channel:
    def __init__(self, params: ChannelParams, map_path):
        self.params = params
        self.total_shadow_map = load_or_create_shadowing(map_path)
        self._norm_distance = np.sqrt(2) * 0.5 * self.total_shadow_map.shape[0] * self.params.cell_size
        self.los_norm_factor = 10 ** (self.params.cell_edge_snr / 10) / (
                self._norm_distance ** (-self.params.los_path_loss_exp))
        self.los_shadowing_sigma = np.sqrt(self.params.los_shadowing_variance)
        self.nlos_shadowing_sigma = np.sqrt(self.params.nlos_shadowing_variance)

    def get_max_rate(self):
        dist = self.params.uav_altitude

        snr = self.los_norm_factor * dist ** (-self.params.los_path_loss_exp)

        rate = np.log2(1 + snr)

        return rate

    def compute_rate(self, uav_pos, device_pos):
        dist = np.sqrt(
            ((device_pos[0] - uav_pos[0]) * self.params.cell_size) ** 2 +
            ((device_pos[1] - uav_pos[1]) * self.params.cell_size) ** 2 +
            self.params.uav_altitude ** 2)

        if self.total_shadow_map[int(round(device_pos[0])), int(round(device_pos[1])),
                                 int(round(uav_pos[0])), int(round(uav_pos[1]))]:
            snr = self.los_norm_factor * dist ** (
                -self.params.nlos_path_loss_exp) * 10 ** (np.random.normal(0., self.nlos_shadowing_sigma) / 10)
        else:
            snr = self.los_norm_factor * dist ** (
                -self.params.los_path_loss_exp) * 10 ** (np.random.normal(0., self.los_shadowing_sigma) / 10)

        rate = np.log2(1 + snr)

        return rate


@dataclass
class DHGymParams(GridGymParams):
    data_reward: float = 1.0
    device_count_range: (int, int) = (3, 10)
    data_range: (float, float) = (5.0, 20.0)
    comm_steps: int = 4
    channel: ChannelParams = ChannelParams()

    # Render
    show_data_rates: bool = True


class DHGym(GridGym):
    def __init__(self, params: DHGymParams):
        super().__init__(params)
        self.params = params
        self.action_to_direction.update({5: np.array([0, 0])})
        self.action_space = spaces.Discrete(6)
        self.devices = []
        self.channel = Channel(params.channel, params.map_path)
        self.map = np.concatenate((self.map, np.zeros(self.map.shape[:-1] + (1,))), axis=-1)
        self.padding_values += [0]
        self.centered_map = self.pad_centered()

        map_obs_shape = list(self.centered_map.shape)
        map_obs_shape[-1] = 4  # Target and covered are merged
        self.observation_space = spaces.Dict(
            {
                "map": spaces.Box(low=0, high=1, shape=map_obs_shape, dtype=float),
                "budget": spaces.Box(low=0, high=self.params.budget_range[1], dtype=int),
                "mask": spaces.Box(low=0, high=1, shape=(5,), dtype=bool)
            }
        )
        self.colors = np.array(color_palette(n_colors=self.params.device_count_range[1])) * 255.0

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed, options=options)

        self.devices = self.generate_devices()
        self.map[..., 3] = 0
        self.update_device_map()
        self._reset()

        if self.params.render:
            self._render_frame()

        return self._get_obs(), self._get_info()

    def _get_obs(self):
        return {
            "map": self.centered_map,
            "budget": self.budget,
            "mask": self.get_action_mask()
        }

    def generate_devices(self):
        num_devices = np.random.randint(*self.params.device_count_range)
        data = np.random.uniform(*self.params.data_range, size=num_devices)
        allowed_zone = np.logical_not(np.logical_or(self.map[..., 0], self.map[..., 2]))
        allowed_cells = np.transpose(np.where(allowed_zone))
        idx = np.random.choice(range(len(allowed_cells)), num_devices, replace=False)
        positions = [allowed_cells[i] for i in idx]

        devices = [{"position": pos, "data": d, "consumed": 0, "last_consumed": 0} for pos, d in zip(positions, data)]
        return devices

    def step(self, action):
        old_position = deepcopy(self.position)
        super()._step(action)
        cells_remaining = self.get_remaining_data()
        terminated = self.landed or self.crashed
        if not terminated:
            positions = list(
                reversed(np.linspace(self.position, old_position, num=self.params.comm_steps, endpoint=False)))
            for device in self.devices:
                device["last_consumed"] = 0
            for position in positions:
                device_done = [device["data"] <= device["consumed"] for device in self.devices]
                data_rates = [self.channel.compute_rate(position, device["position"]) if not device_done[k] else 0 for
                              k, device in enumerate(self.devices)]
                idx = np.argmax(data_rates)
                device = self.devices[idx]
                consumed = min(data_rates[idx], device["data"] - device["consumed"])
                device["consumed"] += consumed
                device["last_consumed"] += consumed
            self.update_device_map()

        reward = self._get_rewards()
        reward += (cells_remaining - self.get_remaining_data()) * self.params.data_reward
        self.episodic_reward += reward

        if self.params.render:
            self._render_frame()

        return self._get_obs(), reward, terminated, self.truncated, self._get_info()

    def get_remaining_data(self):
        return np.sum([device["data"] - device["consumed"] for device in self.devices])

    def update_device_map(self):
        for device in self.devices:
            self.update_device_map_single(device)

    def update_device_map_single(self, device):
        pos = device["position"]
        c_pos = self.get_centered_pos(pos)
        remaining = (device["data"] - device["consumed"]) / self.params.data_range[1]
        self.map[pos[0], pos[1], 3] = remaining
        self.centered_map[c_pos[0], c_pos[1], 3] = remaining

    def _draw_cells(self, canvas, map_data, pix_size):
        pix_x = np.array((pix_size, 0))
        pix_y = np.array((0, pix_size))
        for x in range(map_data.shape[0]):
            for y in range(map_data.shape[1]):
                cell = map_data[x, y]
                pos = np.array((x, y))
                pos_image = pos * pix_size
                patch = pygame.Rect(pos_image, (pix_size + 1, pix_size + 1))
                if cell[1] or cell[0]:
                    # NFZ
                    color = (255 * cell[1], 0, 255 * cell[0])
                    pygame.draw.rect(canvas, np.array(color), patch)
                if cell[2]:
                    # Obstacles
                    pygame.draw.line(canvas, (0, 0, 0), pos_image, pos_image + pix_x + pix_y, width=3)
                    pygame.draw.line(canvas, (0, 0, 0), pos_image + pix_y, pos_image + pix_x, width=3)
                if not cell[:3].any():
                    pygame.draw.rect(canvas, np.array((255, 255, 255)), patch)

        for k, device in enumerate(self.devices):
            pos_image = (device["position"] * pix_size).astype(int)
            empty_color = copy.deepcopy(canvas.get_at((pos_image + pix_size / 2).astype(int)))
            patch = pygame.Rect(pos_image + 1,
                                (pix_size - 1, int((pix_size - 1) * (
                                        (self.params.data_range[1] - device["data"]) + device["consumed"]) /
                                                   self.params.data_range[1])))
            color = self.colors[k] if device["consumed"] < device["data"] else (0, 0, 0)
            pygame.draw.circle(canvas, color, pos_image + np.array((pix_size, pix_size)) / 2,
                               radius=pix_size / 2 - 2,
                               width=0)
            pygame.draw.rect(canvas, empty_color, patch)
            pygame.draw.circle(canvas, color, pos_image + np.array((pix_size, pix_size)) / 2,
                               radius=pix_size / 2 - 2,
                               width=2)

        if self.params.show_data_rates:
            for k, device in enumerate(self.devices):
                if device["last_consumed"] > 0:
                    pos_image = (device["position"] * pix_size).astype(int)
                    pygame.draw.line(canvas, self.colors[k], pos_image + np.array((pix_size, pix_size)) / 2,
                                     self.position * pix_size + np.array((pix_size, pix_size)) / 2,
                                     width=int(np.ceil(min(device["last_consumed"], pix_size / 3))))

    def _get_info(self):
        info = super()._get_info()
        collection_ratio = 1 - self.get_remaining_data() / np.sum([device["data"] for device in self.devices])
        info.update({
            "collection_ratio": collection_ratio,
            "cral": collection_ratio * self.landed
        })
        return info
