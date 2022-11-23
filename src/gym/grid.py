from dataclasses import dataclass
from typing import Optional, Union, List, Callable

import gym
import numpy as np
from gym import spaces
from gym.core import RenderFrame

import pygame

from src.gym.utils import Map, load_or_create_landing


@dataclass
class GridGymParams:
    map_path: Union[str, list[str]] = "res/manhattan32.png"
    budget_range: (int, int) = (50, 150)
    budget_std: float = 0.0
    spawn_anywhere: float = 0.0  # Probability of being spawned randomly, otherwise is spawned on slz
    safety_controller: bool = True

    # Rewards
    boundary_penalty: float = 1.0
    empty_battery_penalty: float = 150.0
    movement_penalty: float = 0.2

    # Rendering
    render: bool = False
    display_mode: str = "fixed"
    render_fps: int = 8
    window_size: int = 768


class GridGym(gym.Env):
    def __init__(self, params: GridGymParams):
        self.params = params
        self.map_index = 0
        self._map_image = []
        self._landing_map = []
        self.map = None

        self.map_path = params.map_path
        if not isinstance(self.map_path, list):
            self.map_path = [self.map_path]

        self.load_maps(self.map_path)
        self._initialize_map(map_index=0)

        self.action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
            4: np.array([0, 0])
        }
        self.center = self.shape - 1
        self.padding_values = [0, 1, 1]

        self.position = np.array((0, 0))
        self.budget = 0
        self.initial_budget = 0
        self.truncated = False
        self.crashed = False
        self.landed = False
        self.boundary_counter = 0
        self.episodic_reward = 0

        self.centered_map = self.pad_centered()

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict(
            {
                "map": spaces.Box(low=0, high=1, shape=self.centered_map.shape, dtype=bool),
                "budget": spaces.Box(low=0, high=self.params.budget_range[1], dtype=int),
                "mask": spaces.Box(low=0, high=1, shape=(5,), dtype=bool)
            }
        )

        self.window_size = [self.params.window_size] * 2
        self.render_registry = []
        self.window = None
        self.clock = None

    def register_render(self, render_func: Callable[[pygame.Surface, dict], pygame.Surface], shape: [int, int]):
        item = {
            "render_func": render_func,
            "shape": shape,
            "offset": self.window_size[0]
        }
        self.window_size[0] += shape[0]
        self.window_size[1] = max(self.window_size[1], shape[1])
        self.render_registry.append(item)

    def pad_centered(self):
        padding = np.ceil(self.shape / 2.0)
        position_offset = padding - self.position
        pad_width = np.array([padding + position_offset - 1, padding - position_offset]).transpose().astype(int)
        layers = []
        for k, layer in enumerate(np.moveaxis(self.map, 2, 0)):
            layers.append(np.pad(layer, pad_width=pad_width, mode='constant', constant_values=self.padding_values[k]))

        return np.stack(layers, axis=-1)

    def step(self, action):

        self._step(action)

        terminated = self.landed or self.crashed
        reward = self._get_rewards()
        self.episodic_reward += reward

        if self.params.render:
            self._render_frame()

        return self._get_obs(), reward, terminated, self.truncated, self._get_info()

    def _get_rewards(self):
        reward = -self.params.movement_penalty if not self.landed else 0.0
        if self.truncated:
            reward -= self.params.boundary_penalty
        if self.crashed:
            reward -= self.params.empty_battery_penalty
        return reward

    def _step(self, action):
        if action == 4:
            # Landing action
            self.landed = self.centered_map[self.center[0], self.center[1], 0]
            self.truncated = not self.landed
        else:
            motion = self.action_to_direction[action]
            idx = self.center + motion
            # Check if action is safe
            self.truncated = self.centered_map[idx[0], idx[1], 1]
            if not self.truncated:
                # Apply motion
                self.position += motion
                self.centered_map = np.roll(self.centered_map, shift=-motion, axis=[0, 1])

        self.boundary_counter += int(self.truncated)
        # Always consume battery
        self.budget -= max(1 + np.random.normal(scale=self.params.budget_std), 0.0)
        self.crashed = self.budget <= 0 and not self.landed
        if not self.params.safety_controller:
            self.crashed = self.crashed or self.truncated
        self.truncated = self.truncated or self.crashed

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self._initialize_map()

        self._reset()

        if self.params.render:
            self._render_frame()

        return self._get_obs(), self._get_info()

    def _reset(self):
        if np.random.uniform() < self.params.spawn_anywhere:
            position_mask = np.logical_not(self.map_image.nfz).astype(float)
        else:
            position_mask = self.map_image.start_land_zone.astype(float)
        self.position = np.array(np.unravel_index(np.argmax(
            np.random.uniform(0, 1, size=self.shape) * position_mask), self.shape))
        self.budget = np.round(np.random.uniform(low=self.params.budget_range[0], high=self.params.budget_range[1]))
        self.initial_budget = self.budget
        self.centered_map = self.pad_centered()
        self.truncated = False
        self.crashed = False
        self.landed = False
        self.boundary_counter = 0
        self.episodic_reward = 0

    def _get_obs(self):
        return {
            "map": self.centered_map,
            "budget": self.budget,
            "mask": self.get_action_mask()
        }

    def _get_info(self):
        return {
            "landed": self.landed,
            "crashed": self.crashed,
            "terminal": self.crashed or self.landed,
            "budget_ratio": self.budget / self.initial_budget,
            "boundary_counter": self.boundary_counter,
            "episodic_reward": self.episodic_reward
        }

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        if self.params.render:
            return self._render_frame()

    def _render_frame(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size[0], self.window_size[1]))
            pygame.font.init()
            self.font = pygame.font.SysFont('Arial', 16)

        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.params.display_mode == "centered":
            shape = np.array(self.centered_map.shape[:2])
            map_data = self.centered_map
            pix_size = self.params.window_size / shape[0]
            map_position = (self.center * pix_size).astype(int)
        else:
            shape = self.shape
            map_data = self.map
            pix_size = self.params.window_size / shape[0]
            map_position = self.position * pix_size

        canvas = pygame.Surface((self.params.window_size, self.params.window_size))
        canvas.fill((255, 255, 255))

        self._draw_cells(canvas, map_data, pix_size)

        # Position
        color = (0, 180, 0)
        if self.truncated:
            color = (255, 180, 0)
        if self.crashed:
            color = (180, 0, 0)
        if self.landed:
            color = (0, 90, 0)
        pygame.draw.circle(canvas, color, map_position + np.array((pix_size, pix_size)) / 2, radius=pix_size / 2 - 2,
                           width=0)

        text_surface = self.font.render(f"{self.budget:.1f}", False, (0, 0, 0))
        canvas.blit(text_surface, map_position)

        self.window.blit(canvas, canvas.get_rect())

        obs = self._get_obs()
        for item in self.render_registry:
            canvas = pygame.Surface(item["shape"])
            canvas = item["render_func"](canvas, obs)
            self.window.blit(canvas, dest=(item["offset"], 0))

        pygame.event.pump()
        pygame.display.update()

        self.clock.tick(self.params.render_fps)

    def _draw_cells(self, canvas, map_data, pix_size):
        pix_x = np.array((pix_size, 0))
        pix_y = np.array((0, pix_size))
        for x in range(map_data.shape[0]):
            for y in range(map_data.shape[1]):
                cell = map_data[x, y]
                pos = np.array((x, y))
                pos_image = pos * pix_size
                if cell[0]:
                    # Start landing zone
                    pygame.draw.rect(canvas, (0, 0, 255), pygame.Rect(pos_image, (pix_size + 1, pix_size + 1)))
                if cell[1]:
                    # NFZ
                    pygame.draw.rect(canvas, (255, 0, 0), pygame.Rect(pos_image, (pix_size + 1, pix_size + 1)))
                if cell[2]:
                    # Obstacles
                    pygame.draw.line(canvas, (0, 0, 0), pos_image, pos_image + pix_x + pix_y, width=3)
                    pygame.draw.line(canvas, (0, 0, 0), pos_image + pix_y, pos_image + pix_x, width=3)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None

    def get_action_mask(self):
        mask = np.ones(len(self.action_to_direction), dtype=bool)
        # Get boundary mask
        mask[4] = self.centered_map[self.center[0], self.center[1], 0]
        for a in range(4):
            target = self.center + self.action_to_direction[a]
            target_position = np.clip(self.position + self.action_to_direction[a], (0, 0),
                                      np.array(self.landing_map.shape) - 1)
            boundary = not self.centered_map[target[0], target[1], 1]
            budget = self.budget > self.landing_map[target_position[0], target_position[1]]
            mask[a] = boundary and budget

        return mask

    def get_centered_pos(self, pos):
        return pos - self.position + self.center

    def load_maps(self, map_paths):
        self._map_image = []
        self._landing_map = []
        max_size = [0, 0]
        for path in map_paths:
            m = Map.load_map(path)
            max_size = max(max_size[0], m.get_size()[0]), max(max_size[1], m.get_size()[1])
            self._map_image.append(m)
            self._landing_map.append(load_or_create_landing(path))

        for m in self._map_image:
            m.pad_to_size(max_size)
        self.shape = np.array(max_size)

    def _initialize_map(self, map_index=None):
        if map_index is None:
            map_index = np.random.randint(len(self._map_image))
        self.map_index = map_index
        self.map = np.stack((self.map_image.start_land_zone, self.map_image.nfz, self.map_image.obstacles), axis=-1)

    def initialize_map(self, map_index=None):
        self._initialize_map(map_index)

    @property
    def map_image(self):
        return self._map_image[self.map_index]

    @property
    def landing_map(self):
        return self._landing_map[self.map_index]
