from dataclasses import dataclass

import numpy as np
from src.Map.Shadowing import load_or_create_shadowing


@dataclass
class ChannelParams:
    cell_edge_snr: float = -25  # in dB
    los_path_loss_exp: float = 2.27
    nlos_path_loss_exp: float = 3.64
    uav_altitude: float = 10.0  # in m
    cell_size: float = 10.0  # in m
    los_shadowing_variance: float = 2.0
    nlos_shadowing_variance: float = 5.0
    map_path: str = "res/downtown.png"


class Channel:
    def __init__(self, params: ChannelParams):
        self.params = params
        self._norm_distance = None
        self.los_norm_factor = None
        self.los_shadowing_sigma = None
        self.nlos_shadowing_sigma = None
        self.total_shadow_map = load_or_create_shadowing(self.params.map_path)

    def reset(self, area_size):
        self._norm_distance = np.sqrt(2) * 0.5 * area_size * self.params.cell_size
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
