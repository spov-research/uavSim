import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from dataclasses import dataclass

from skimage.draw import random_shapes
import logging

from src.gym.grid import GridGym
from src.gym.utils import load_or_create_shadowing, Map


class RandomTargetGenerator:
    @dataclass
    class Params:
        coverage_range: (float, float) = (0.2, 0.5)
        shape_range: (int, int) = (3, 8)

    def __init__(self, params: Params, shape):
        self.params = params
        self.shape = shape

    def generate_target(self, obstacles):

        area = np.product(self.shape)

        target = self.__generate_random_shapes_area(
            self.params.shape_range[0],
            self.params.shape_range[1],
            area * self.params.coverage_range[0],
            area * self.params.coverage_range[1]
        )

        return target & ~obstacles

    def __generate_random_shapes(self, min_shapes, max_shapes):
        img, _ = random_shapes(self.shape, max_shapes, min_shapes=min_shapes, channel_axis=None,
                               allow_overlap=True, random_seed=np.random.randint(2 ** 32 - 1))
        # Numpy random usage for random seed unifies random seed which can be set for repeatability
        attempt = np.array(img != 255, dtype=bool)
        return attempt, np.sum(attempt)

    def __generate_random_shapes_area(self, min_shapes, max_shapes, min_area, max_area, retry=100):
        for attemptno in range(retry):
            attempt, area = self.__generate_random_shapes(min_shapes, max_shapes)
            if min_area is not None and min_area > area:
                continue
            if max_area is not None and max_area < area:
                continue
            return attempt
        logging.warning("Was not able to generate shapes with given area constraint in allowed number of tries."
                        " Randomly returning next attempt.")
        attempt, area = self.__generate_random_shapes(min_shapes, max_shapes)
        logging.warning("Size is: ", area)
        return attempt

    def __generate_exclusive_shapes(self, exclusion, min_shapes, max_shapes):
        attempt, area = self.__generate_random_shapes(min_shapes, max_shapes)
        attempt = attempt & (~exclusion)
        area = np.sum(attempt)
        return attempt, area

    # Create target image and then subtract exclusion area
    def __generate_exclusive_shapes_area(self, exclusion, min_shapes, max_shapes, min_area, max_area, retry=100):
        for attemptno in range(retry):
            attempt, area = self.__generate_exclusive_shapes(exclusion, min_shapes, max_shapes)
            if min_area is not None and min_area > area:
                continue
            if max_area is not None and max_area < area:
                continue
            return attempt

        logging.warning("Was not able to generate shapes with given area constraint in allowed number of tries."
                        " Randomly returning next attempt.")
        attempt, area = self.__generate_exclusive_shapes(exclusion, min_shapes, max_shapes)
        logging.warning("Size is: ", area)
        return attempt


class SimpleSquareCamera:

    def __init__(self, camera_half_length: int, map_path: str):
        chl = camera_half_length
        total_map = Map.load_map(map_path)
        shadowing = load_or_create_shadowing(map_path)
        self.camera_half_length = chl
        self.size = np.array([2 * chl + 1] * 2)
        self.obstacles = np.pad(total_map.obstacles, ((chl, chl), (chl, chl)), mode='constant', constant_values=True)
        self.obstruction_map = np.pad(shadowing, ((0, 0), (0, 0), (chl, chl), (chl, chl)),
                                      mode='constant', constant_values=True)

    def computeView(self, position):
        x_pos, y_pos = position
        x_size, y_size = self.size

        view = np.ones(self.size, dtype=bool)
        view &= ~self.obstacles[x_pos: x_pos + x_size, y_pos: y_pos + y_size]
        view &= ~self.obstruction_map[x_pos, y_pos, x_pos: x_pos + x_size, y_pos: y_pos + y_size]
        return view


class CPPGym(GridGym):
    @dataclass
    class Params(GridGym.Params):
        cell_reward: float = 0.4
        completion_reward: float = 0.0
        camera_half_length: int = 2
        generator: RandomTargetGenerator.Params = RandomTargetGenerator.Params()
        inactivity_timeout: bool = False

    def __init__(self, params: Params):
        super().__init__(params)
        self.params = params
        self.initialize_map(0)
        self._camera = []
        for path in self.map_path:
            self._camera.append(SimpleSquareCamera(params.camera_half_length, path))
        self.padding_values += [0, 0]
        self.centered_map = self.pad_centered()
        self.generator = RandomTargetGenerator(params.generator, self.map_image.get_size())

        map_obs_shape = list(self.centered_map.shape)
        map_obs_shape[-1] = 4  # Target and covered are merged
        self.observation_space = spaces.Dict(
            {
                "map": spaces.Box(low=0, high=1, shape=map_obs_shape, dtype=bool),
                "budget": spaces.Box(low=0, high=self.params.budget_range[1], dtype=int),
                "landed": spaces.Box(low=0, high=1, dtype=bool),
                "mask": spaces.Box(low=0, high=1, shape=(len(self.action_to_direction),), dtype=bool)
            }
        )

        self.inactivity_steps = 0

    def initialize_map(self, map_index=None):
        self._initialize_map(map_index)
        target_and_covered = np.zeros(self.map.shape[:2] + (2,), dtype=bool)
        self.map = np.concatenate((self.map, target_and_covered), axis=-1)

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed, options=options)

        self.initialize_map()
        self.map[..., 3] = self.generator.generate_target(self.map_image.obstacles)
        self.map[..., 4] = False
        self.inactivity_steps = 0
        self._reset()

        if self.params.render:
            self._render_frame()

        return self.get_obs(), self._get_info()

    def _get_obs(self):
        remaining_target = np.logical_and(self.centered_map[..., 3], np.logical_not(self.centered_map[..., 4]))
        map_obs = np.concatenate((self.centered_map[..., :3], np.expand_dims(remaining_target, -1)), axis=-1)
        return {
            "map": map_obs,
            "budget": self.budget,
            "landed": self.landed,
            "mask": self.get_action_mask()
        }

    def step(self, action):

        super()._step(action)
        cells_remaining = self.get_remaining_cells()
        if not self.terminated and not self.landed:
            view = self.camera.computeView(self.position)
            chl = self.params.camera_half_length
            self.centered_map[self.center[0] - chl: self.center[0] + chl + 1,
            self.center[1] - chl: self.center[1] + chl + 1, 4] |= view
            self.map = self.centered_map[self.center[0] - self.position[0]: 2 * self.center[0] - self.position[0] + 1,
                       self.center[1] - self.position[1]: 2 * self.center[1] - self.position[1] + 1,
                       :]

        reward = self._get_rewards()
        cells_collected = cells_remaining - self.get_remaining_cells()
        reward += cells_collected * self.params.cell_reward
        if self.task_solved():
            reward += self.params.completion_reward
        self.episodic_reward += reward

        if cells_collected > 0:
            self.inactivity_steps = 0
        else:
            self.inactivity_steps += 1

        if self.params.render:
            self._render_frame()

        return self.get_obs(), reward, self.terminated, self.truncated, self._get_info()

    def get_remaining_cells(self):
        return np.sum(np.logical_and(self.centered_map[..., 3], np.logical_not(self.centered_map[..., 4])))

    def task_solved(self):
        return self.get_remaining_cells() == 0 and self.landed

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

    def _get_info(self):
        info = super()._get_info()
        collection_ratio = 1 - self.get_remaining_cells() / np.sum(self.centered_map[..., 3])
        info.update({
            "collection_ratio": collection_ratio,
            "cral": collection_ratio * self.landed
        })
        if self.params.inactivity_timeout:
            info["timeout"] = self.inactivity_steps >= self.params.timeout_steps
        if collection_ratio == 1:
            info["completion_steps"] = self.steps
        return info

    @property
    def camera(self):
        return self._camera[self.map_index]
