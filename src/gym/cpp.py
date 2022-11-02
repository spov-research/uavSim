import gym
import numpy as np
import pygame
from gym import spaces
from dataclasses import dataclass

from skimage.draw import random_shapes
import logging

from src.gym.grid import GridGym, GridGymParams
from src.gym.utils import load_or_create_shadowing, Map


@dataclass
class RandomTargetGeneratorParams:
    coverage_range: (float, float) = (0.2, 0.5)
    shape_range: (int, int) = (3, 8)


class RandomTargetGenerator:

    def __init__(self, params: RandomTargetGeneratorParams, shape):
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
        self.obstacles = np.pad(total_map.obstacles, ((chl, chl), (chl, chl)), mode='constant', constant_values=False)
        self.obstruction_map = np.pad(shadowing, ((0, 0), (0, 0), (chl, chl), (chl, chl)),
                                      mode='constant', constant_values=False)

    def computeView(self, position):
        x_pos, y_pos = position
        x_size, y_size = self.size

        view = np.ones(self.size, dtype=bool)
        view &= ~self.obstacles[x_pos: x_pos + x_size, y_pos: y_pos + y_size]
        view &= ~self.obstruction_map[x_pos, y_pos, x_pos: x_pos + x_size, y_pos: y_pos + y_size]
        return view


@dataclass
class CPPGymParams(GridGymParams):
    cell_reward: float = 0.4
    camera_half_length: int = 2
    generator: RandomTargetGeneratorParams = RandomTargetGeneratorParams()


class CPPGym(GridGym):
    def __init__(self, params: CPPGymParams):
        super().__init__(params)
        self.params = params
        self.camera = SimpleSquareCamera(params.camera_half_length, params.map_path)
        target_and_covered = np.zeros(self.map.shape[:2] + (2,), dtype=bool)
        self.map = np.concatenate((self.map, target_and_covered), axis=-1)
        self.padding_values += [0, 0]
        self.centered_map = self.pad_centered()
        self.generator = RandomTargetGenerator(params.generator, self.map_image.get_size())

        map_obs_shape = list(self.centered_map.shape)
        map_obs_shape[-1] = 4  # Target and covered are merged
        self.observation_space = spaces.Dict(
            {
                "map": spaces.Box(low=0, high=1, shape=map_obs_shape, dtype=bool),
                "budget": spaces.Box(low=0, high=self.params.budget_range[1], dtype=int),
                "mask": spaces.Box(low=0, high=1, shape=(5,), dtype=bool)
            }
        )

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed, options=options)

        self.map[..., 3] = self.generator.generate_target(self.map_image.obstacles)
        self.map[..., 4] = False
        self._reset()

        if self.params.render:
            self._render_frame()

        return self._get_obs(), self._get_info()

    def _get_obs(self):
        remaining_target = np.logical_and(self.centered_map[..., 3], np.logical_not(self.centered_map[..., 4]))
        map_obs = np.concatenate((self.centered_map[..., :3], np.expand_dims(remaining_target, -1)), axis=-1)
        return {
            "map": map_obs,
            "budget": self.budget,
            "mask": self.get_action_mask()
        }

    def step(self, action):
        super()._step(action)
        cells_remaining = self.get_remaining_cells()
        terminated = self.landed or self.crashed
        if not terminated:
            view = self.camera.computeView(self.position)
            chl = self.params.camera_half_length
            self.centered_map[self.center[0] - chl: self.center[0] + chl + 1,
            self.center[1] - chl: self.center[1] + chl + 1, 4] |= view
            self.map = self.centered_map[self.center[0] - self.position[0]: 2 * self.center[0] - self.position[0] + 1,
                       self.center[1] - self.position[1]: 2 * self.center[1] - self.position[1] + 1,
                       :]

        reward = self._get_rewards()
        reward += (cells_remaining - self.get_remaining_cells()) * self.params.cell_reward
        self.episodic_reward += reward

        if self.params.render:
            self._render_frame()

        return self._get_obs(), reward, terminated, self.truncated, self._get_info()

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

    def _get_info(self):
        info = super()._get_info()
        collection_ratio = 1 - self.get_remaining_cells() / np.sum(self.centered_map[..., 3])
        info.update({
            "collection_ratio": collection_ratio,
            "cral": collection_ratio * self.landed
        })
        return info
