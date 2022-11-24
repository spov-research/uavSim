import gym
import numpy as np
import pygame
from gym import spaces
from dataclasses import dataclass

from skimage.draw import random_shapes
import logging

from src.gym.cpp import CPPGym, CPPGymParams
from src.gym.grid import GridGym, GridGymParams
from src.gym.utils import load_or_create_shadowing, Map, load_or_create_shortest_distance


@dataclass
class HCPPGymParams(CPPGymParams):
    target_shape: (int, int) = (17, 17)


class HCPPGym(CPPGym):
    def __init__(self, params: HCPPGymParams):
        super().__init__(params)
        self.params = params
        self._shortest_distance = []
        for path in self.map_path:
            shortest = load_or_create_shortest_distance(path).astype(float)
            shortest = np.where(shortest == -1.0, np.inf, shortest)
            self._shortest_distance.append(shortest)

        self.landing_action = int(np.prod(self.params.target_shape))
        self.action_dim = np.prod(params.target_shape) + 1
        self.action_space = spaces.Discrete(self.action_dim)
        map_obs_shape = list(self.centered_map.shape)
        map_obs_shape[-1] = 4  # Target and covered are merged
        self.observation_space = spaces.Dict(
            {
                "map": spaces.Box(low=0, high=1, shape=map_obs_shape, dtype=bool),
                "budget": spaces.Box(low=0, high=self.params.budget_range[1], dtype=int),
                "mask": spaces.Box(low=0, high=1, shape=(self.action_dim,), dtype=bool)
            }
        )

    def step(self, action):
        if action == self.landing_action:
            return CPPGym.step(self, 4)

        target_position = np.array(np.unravel_index(action, self.params.target_shape))
        target_position = target_position - np.array(self.params.target_shape) // 2 + self.position

        reward = 0
        term = False
        while not (self.position == target_position).all():
            dist_up = self.shortest_distance[
                self.position[0], self.position[1] + 1, target_position[0], target_position[1]]
            dist_down = self.shortest_distance[
                self.position[0], self.position[1] - 1, target_position[0], target_position[1]]
            dist_left = self.shortest_distance[
                self.position[0] - 1, self.position[1], target_position[0], target_position[1]]
            dist_right = self.shortest_distance[
                self.position[0] + 1, self.position[1], target_position[0], target_position[1]]

            distances = [dist_right, dist_up, dist_left, dist_down]
            action = np.argmin(distances)

            _, r, term, _, _ = CPPGym.step(self, action)
            reward += r
            if term:
                break

        return self._get_obs(), reward, term, self.truncated, self._get_info()

    def get_action_mask(self):
        actions = np.arange(self.action_dim - 1)

        target_positions = np.array(np.unravel_index(actions, self.params.target_shape)).transpose()
        target_positions = target_positions + np.expand_dims(self.position - np.array(self.params.target_shape) // 2, 0)

        x, y, _, _ = self.shortest_distance.shape
        outside = np.logical_or.reduce(np.logical_or(target_positions < 0, target_positions > (x, y)), axis=1)
        target_positions = np.clip(target_positions, 0, (x - 1, y - 1))
        total_distance = self.shortest_distance[
                             self.position[0], self.position[1], target_positions[:, 0], target_positions[:, 1]] + \
                         self.landing_map[target_positions[:, 0], target_positions[:, 1]]
        action_mask = np.logical_not(np.logical_or(outside, total_distance > self.budget))
        action_mask = np.append(action_mask, self.map[self.position[0], self.position[1], 0])  # Is in Landing zone
        return action_mask

    def get_remaining_cells(self):
        return np.sum(np.logical_and(self.centered_map[..., 3], np.logical_not(self.centered_map[..., 4])))

    def _draw_cells(self, canvas, map_data, pix_size):
        pix_x = np.array((pix_size, 0))
        pix_y = np.array((0, pix_size))
        for x in range(map_data.shape[0]):
            for y in range(map_data.shape[1]):
                cell = map_data[x, y]
                pos = np.array((x, y))
                pos_image = pos * pix_size
                dim_factor = 1.0 if cell[4] else 0.6
                patch = pygame.Rect(pos_image, (pix_size + 1, pix_size + 1))
                if cell[1] or cell[3] or cell[0]:
                    # NFZ
                    color = (255 * cell[1], 255 * cell[3], 255 * cell[0])
                    pygame.draw.rect(canvas, np.array(color) * dim_factor, patch)
                if cell[2]:
                    # Obstacles
                    pygame.draw.line(canvas, (0, 0, 0), pos_image, pos_image + pix_x + pix_y, width=3)
                    pygame.draw.line(canvas, (0, 0, 0), pos_image + pix_y, pos_image + pix_x, width=3)
                if not cell[:4].any():
                    pygame.draw.rect(canvas, np.array((255, 255, 255)) * dim_factor, patch)

    @property
    def shortest_distance(self):
        return self._shortest_distance[self.map_index]
