import copy
import math
from dataclasses import dataclass
from typing import Optional, Union, List, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import RenderFrame

import pygame

from src.gym.utils import Map, load_or_create_landing


class GridGym(gym.Env):
    @dataclass
    class Params:
        map_path: Union[str, List[str]] = "res/manhattan32.png"
        budget_range: (int, int) = (50, 150)
        budget_std: float = 0.0
        spawn_anywhere: float = 0.0  # Probability of being spawned randomly, otherwise is spawned on slz
        safety_controller: bool = True

        # Rewards
        boundary_penalty: float = 1.0
        empty_battery_penalty: float = 150.0
        movement_penalty: float = 0.2

        # Charging
        recharge: bool = True
        start_landed: bool = True
        charge_amount: float = 2.0
        timeout_steps: int = 1000

        # Rendering
        render: bool = False
        display_mode: str = "fixed"
        render_fps: int = 8
        window_size: int = 768
        draw_trajectory: bool = False

    def __init__(self, params: Params):
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
        self.action_to_name = {
            0: "right",
            1: "down",
            2: "left",
            3: "up",
            4: "land"
        }
        if self.params.recharge:
            # Add take off action
            self.action_to_direction[5] = np.array([0, 0])
            self.action_to_name[5] = "take off"

        self._action_to_direction_np = np.array(list(self.action_to_direction.values()))
        self.center = self.shape - 1
        self.padding_values = [0, 1, 1]

        self.position = np.array((0, 0))
        self.budget = 0
        self.initial_budget = 0
        self.truncated = False
        self.crashed = False
        self.landed = False
        self.terminated = False
        self.boundary_counter = 0
        self.episodic_reward = 0
        self.steps = 0
        self.trajectory = []

        self.centered_map = self.pad_centered()

        num_actions = len(self.action_to_direction)
        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Dict(
            {"map": spaces.Box(low=0, high=1, shape=self.centered_map.shape, dtype=bool),
             "budget": spaces.Box(low=0, high=self.params.budget_range[1], dtype=int),
             "landed": spaces.Box(low=0, high=1, dtype=bool),
             "mask": spaces.Box(low=0, high=1, shape=(num_actions,), dtype=bool)})

        self.window_size = [self.params.window_size * 5 // 4, self.params.window_size]
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
        reward = self._get_rewards()
        self.episodic_reward += reward

        if self.params.render:
            self._render_frame()

        return self.get_obs(), reward, self.terminated, self.truncated, self._get_info()

    def _get_rewards(self):
        reward = -self.params.movement_penalty
        if self.truncated:
            reward -= self.params.boundary_penalty
        if self.crashed:
            reward -= self.params.empty_battery_penalty
        return reward

    def _step(self, action):
        if action == 4:
            can_land_ = self.can_land()
            if not can_land_:
                self.truncated = True
            else:
                if self.params.recharge and self.landed:
                    # Already landed and charging
                    self.budget += self.params.charge_amount + 1  # +1 for movement subtraction
                    self.budget = min(self.budget, self.params.budget_range[1] + 1)  # Constrain to max battery
                self.landed = True
        elif action == 5:
            self.landed = False
            self.truncated = False
        else:
            if self.landed:
                self.truncated = True
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
        self.steps += 1
        self.trajectory.append(copy.deepcopy(self.position))

        if self.params.recharge:
            self.terminated = (self.landed and self.task_solved()) or self.crashed
        else:
            self.terminated = self.landed or self.crashed

    def can_land(self):
        can_land = self.centered_map[self.center[0], self.center[1], 0]
        return can_land

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self._initialize_map()

        self._reset()

        if self.params.render:
            self._render_frame()

        return self.get_obs(), self._get_info()

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
        self.landed = self.params.start_landed
        self.terminated = False
        self.boundary_counter = 0
        self.episodic_reward = 0
        self.steps = 0
        self.trajectory = [copy.deepcopy(self.position)]

    def task_solved(self):
        return True

    def _get_obs(self):
        return {
            "map": self.centered_map,
            "budget": self.budget,
            "landed": self.landed,
            "mask": self.get_action_mask()
        }

    def get_obs(self):
        obs = self._get_obs()
        return obs

    def _get_info(self):
        return {
            "landed": self.landed,
            "crashed": self.crashed,
            "terminal": self.terminated,
            "budget_ratio": self.budget / self.initial_budget,
            "boundary_counter": self.boundary_counter,
            "episodic_reward": self.episodic_reward,
            "task_solved": self.task_solved(),
            "total_steps": self.steps,
            "timeout": self.steps >= self.params.timeout_steps
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

        if self.params.draw_trajectory or self.terminated:
            self.draw_trajectory(canvas, self.trajectory, pix_size)

        # Position
        self.draw_agent(canvas, map_position, pix_size)

        self.window.blit(canvas, dest=(0, 0))

        # Draw stats
        stats_canvas = self.draw_stats()
        self.window.blit(stats_canvas, dest=(self.params.window_size + 1, 0))

        obs = self.get_obs()
        for item in self.render_registry:
            canvas = pygame.Surface(item["shape"])
            canvas = item["render_func"](canvas, obs)
            self.window.blit(canvas, dest=(item["offset"], 0))

        pygame.event.pump()
        pygame.display.update()

        self.clock.tick(self.params.render_fps)

    def draw_agent(self, canvas, map_position, pix_size):
        color = (0, 180, 0)
        if self.truncated:
            color = (255, 180, 0)
        if self.crashed:
            color = (180, 0, 0)
        if self.landed:
            color = (0, 90, 0)
        pygame.draw.circle(canvas, color, map_position + np.array((pix_size, pix_size)) / 2, radius=pix_size / 2 - 2,
                           width=0)
        budget_text = f"{self.budget:.1f}" if self.params.budget_std > 0 else f"{int(self.budget)}"
        text_surface = self.font.render(budget_text, True, (0, 0, 0))
        canvas.blit(text_surface,
                    map_position + (np.array((pix_size, pix_size)) / 2 - np.array(text_surface.get_size()) / 2).astype(
                        int))

    def draw_stats(self):
        info = self._get_info()
        info_texts = [(f"{key}:", f"{value:.3f}") if isinstance(value, float) else (f"{key}:", f"{value}") for
                      key, value in
                      info.items()]
        infos = [(self.font.render(name, True, (0, 0, 0)), self.font.render(value, True, (0, 0, 0))) for name, value in
                 info_texts]
        max_height = max([text.get_size()[1] for text, _ in infos])
        height = max_height + 5
        width = self.params.window_size // 4
        stats_canvas_base = pygame.Surface((width, self.params.window_size))
        stats_canvas_base.fill((0, 0, 0))
        stats_canvas = pygame.Surface((width, height * len(info_texts)))
        stats_canvas.fill((255, 255, 255))
        for k, (name, value) in enumerate(infos):
            stats_canvas.blit(name, (1, k * height))
            stats_canvas.blit(value, (width - value.get_size()[0] - 1, k * height))
        stats_canvas_base.blit(stats_canvas, dest=(0, 0))
        return stats_canvas_base

    def draw_trajectory(self, canvas, trajectory, pix_size, num_labels=10):
        last_pos = trajectory[0]
        for pos in trajectory[1:]:
            if pos[0] != last_pos[0] or pos[1] != last_pos[1]:
                orig = (last_pos + 0.5) * pix_size
                dest = (pos + 0.5) * pix_size

                points = self.get_arrow_polygon(orig, dest)
                pygame.draw.polygon(canvas, (0, 0, 0), points)

            last_pos = pos

        steps = len(trajectory)
        step_size = np.ceil(steps // 25 / (num_labels - 1)).astype(int) * 25
        font = pygame.font.SysFont('Arial', 11)
        for k in range(0, steps, step_size):
            center = (trajectory[k] + 0.5) * pix_size
            pygame.draw.circle(canvas, (255, 255, 255), center, radius=pix_size / 2, width=0)
            pygame.draw.circle(canvas, (0, 0, 0), center, radius=pix_size / 2, width=2)

            label = font.render(str(k), True, (0, 0, 0))
            canvas.blit(label, center - np.array(label.get_size()) / 2)

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
        if self.landed:
            mask[5] = self.budget >= 2
            mask[:4] = False
            mask[4] = self.budget < self.params.budget_range[1]
        else:
            mask[5] = False
            for a in range(4):
                action = np.zeros_like(self.action_to_direction[a]) if self.landed else self.action_to_direction[a]
                target = self.center + action
                target_position = np.clip(self.position + action, (0, 0),
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

    def draw_action_grid(self, strings, tile_size):
        font = pygame.font.SysFont('Arial', 16)
        action_font = pygame.font.SysFont('Arial', 12)
        action_canvas = pygame.Surface((4 * tile_size, 3 * tile_size))
        action_canvas.fill((0, 0, 0))
        special_actions = 0
        for a, action in self.action_to_direction.items():
            if sum(action) == 0:
                offset = np.array([3, special_actions]) * tile_size
                special_actions += 1
            else:
                offset = action * tile_size + tile_size

            s = (tile_size, tile_size)
            pygame.draw.rect(action_canvas, (255, 255, 255), pygame.Rect(offset, s), width=0)
            pygame.draw.rect(action_canvas, (0, 0, 0), pygame.Rect(offset, s), width=1)
            f_val = font.render(strings[a], True, (0, 0, 0))
            action_name = action_font.render(self.action_to_name[a], True, (50, 50, 50))
            action_canvas.blit(action_name, offset + np.array([1, 0]))
            action_canvas.blit(f_val, offset + (np.array(s) / 2 - np.array(f_val.get_size()) / 2).astype(int))

        return action_canvas

    @property
    def map_image(self):
        return self._map_image[self.map_index]

    @property
    def landing_map(self):
        return self._landing_map[self.map_index]

    @staticmethod
    def get_arrow_polygon(origin, destination, head_width=8, head_length=8, shaft_width=2):
        # Calculate the unit vector in the direction of the arrow
        diff = [destination[0] - origin[0], destination[1] - origin[1]]
        arrow_length = math.sqrt(diff[0] ** 2 + diff[1] ** 2)
        unit_vec = [diff[0] / arrow_length, diff[1] / arrow_length]

        # Calculate the points on the arrowhead
        tip = destination
        base1 = [destination[0] - head_length * unit_vec[0] - head_width / 2 * unit_vec[1],
                 destination[1] - head_length * unit_vec[1] + head_width / 2 * unit_vec[0]]
        base2 = [destination[0] - head_length * unit_vec[0] + head_width / 2 * unit_vec[1],
                 destination[1] - head_length * unit_vec[1] - head_width / 2 * unit_vec[0]]

        # Calculate the points on the shaft
        shaft_base1 = [origin[0] + shaft_width / 2 * unit_vec[1], origin[1] - shaft_width / 2 * unit_vec[0]]
        shaft_base2 = [origin[0] - shaft_width / 2 * unit_vec[1], origin[1] + shaft_width / 2 * unit_vec[0]]
        shaft_tip1 = [origin[0] + (arrow_length - head_length) * unit_vec[0] + shaft_width / 2 * unit_vec[1],
                      origin[1] + (arrow_length - head_length) * unit_vec[1] - shaft_width / 2 * unit_vec[0]]
        shaft_tip2 = [origin[0] + (arrow_length - head_length) * unit_vec[0] - shaft_width / 2 * unit_vec[1],
                      origin[1] + (arrow_length - head_length) * unit_vec[1] + shaft_width / 2 * unit_vec[0]]

        # Return the points as a list of coordinates
        return [tip, base1, shaft_tip1, shaft_base1, shaft_base2, shaft_tip2, base2]
