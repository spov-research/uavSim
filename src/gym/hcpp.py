import numpy as np
import pygame
from gymnasium import spaces
from dataclasses import dataclass

from src.gym.cpp import CPPGym
from src.gym.utils import load_or_create_shortest_distance


class HCPPGym(CPPGym):
    @dataclass
    class Params(CPPGym.Params):
        target_shape: (int, int) = (17, 17)
        l2_prio: bool = True

    def __init__(self, params: Params):
        super().__init__(params)
        self.params = params
        self._shortest_distance = []
        for path in self.map_path:
            shortest = load_or_create_shortest_distance(path).astype(float)
            shortest = np.where(shortest == -1.0, np.inf, shortest)
            self._shortest_distance.append(shortest)

        self.substep = False
        self.target_position = np.zeros((2,))
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
            self.substep = False
            return CPPGym.step(self, 4)

        target_position = np.array(np.unravel_index(action, self.params.target_shape))
        target_position = target_position - np.array(self.params.target_shape) // 2 + self.position
        self.target_position = target_position

        reward = 0
        term = False
        shortest_distance = np.pad(self.shortest_distance[:, :, target_position[0], target_position[1]],
                                   ((1, 1), (1, 1)), constant_values=np.inf)
        while not (self.position == target_position).all():
            query = self.position + 1  # Padding compensation
            dist_up = shortest_distance[query[0], query[1] + 1]
            dist_down = shortest_distance[query[0], query[1] - 1]
            dist_left = shortest_distance[query[0] - 1, query[1]]
            dist_right = shortest_distance[query[0] + 1, query[1]]

            distances = [dist_right, dist_up, dist_left, dist_down]
            actions = np.where(distances == np.min(distances))[0]
            if self.params.l2_prio:
                l2 = np.linalg.norm(
                    np.expand_dims(self.position, 0) + self._action_to_direction_np[actions] - target_position, axis=1)
                min_dist = np.argmin(l2)
            else:
                min_dist = 0
            action = actions[min_dist]

            self.substep = True
            _, r, term, _, _ = CPPGym.step(self, action)
            reward += r
            if term:
                break

        self.substep = False
        self.render()
        return self._get_obs(), reward, term, self.truncated, self._get_info()

    def get_action_mask(self):
        actions = np.arange(self.action_dim - 1)

        target_positions = np.array(np.unravel_index(actions, self.params.target_shape)).transpose()
        target_positions = target_positions + np.expand_dims(self.position - np.array(self.params.target_shape) // 2, 0)

        x, y, _, _ = self.shortest_distance.shape
        outside = np.logical_or.reduce(np.logical_or(target_positions < 0, target_positions >= (x, y)), axis=1)
        target_positions = np.clip(target_positions, 0, (x - 1, y - 1))
        total_distance = self.shortest_distance[
                             self.position[0], self.position[1], target_positions[:, 0], target_positions[:, 1]] + \
                         self.landing_map[target_positions[:, 0], target_positions[:, 1]]
        action_mask = np.logical_not(np.logical_or(outside, total_distance > self.budget))
        action_mask = np.append(action_mask, self.map[self.position[0], self.position[1], 0])  # Is in Landing zone
        action_mask[self.action_dim // 2 - 1] = False  # Disallow center
        return action_mask

    def get_remaining_cells(self):
        return np.sum(np.logical_and(self.centered_map[..., 3], np.logical_not(self.centered_map[..., 4])))

    def _draw_cells(self, canvas, map_data, pix_size):
        CPPGym._draw_cells(self, canvas, map_data, pix_size)
        if self.substep:
            pos_image = self.target_position * pix_size + pix_size / 6
            patch = pygame.Rect(pos_image, (2 * pix_size / 3, 2 * pix_size / 3))
            color = (255, 180, 0)
            pygame.draw.rect(canvas, color, patch)
        else:
            action_mask = self.get_action_mask()
            for k, mask in enumerate(action_mask[:-1]):
                pos_image = self.action_to_target(k) * pix_size + pix_size / 6
                patch = pygame.Rect(pos_image.astype(int), (2 * pix_size / 3, 2 * pix_size / 3))
                color = (0, 0, 0) if not mask else (180, 180, 180)
                pygame.draw.rect(canvas, color, patch)

    @property
    def shortest_distance(self):
        return self._shortest_distance[self.map_index]

    def action_to_target(self, action) -> np.ndarray:
        target_position = np.array(np.unravel_index(action, self.params.target_shape))
        return target_position + self.position - np.array(self.params.target_shape) // 2
