import sys
from dataclasses import dataclass

import numpy as np

from src.gym.utils import load_or_create_shortest_distance, load_or_create_landing, load_or_create_shadowing

from scipy.signal import convolve2d


class GreedyHeuristic:
    @dataclass
    class Params:
        pass

    def __init__(self, gym):
        self.phases = {
            0: "greedy_collect",
            1: "charge_full"
        }
        self.phase = 1
        self.gym = gym
        map_path = self.gym.map_path
        self.distance_map = load_or_create_shortest_distance(map_path[0])
        self.landing_map = load_or_create_landing(map_path[0])
        self.shadowing_map = load_or_create_shadowing(map_path[0])
        self.max_budget = self.gym.max_budget

    def get_action(self, obs):
        position = self.gym.position
        # Charge fully
        if obs["landed"]:
            if obs["budget"] >= self.max_budget:
                self.phase = 0
                return 5  # Take off action
            return 4  # Charging action
        if self.phase == 0:  # Greedy Collect
            # Find the closest target cell from which the landing zone can be reached in time
            budget = obs["budget"]
            landing = self.landing_map
            distances = self.distance_map[position[0], position[1]]
            targets = np.logical_and(self.gym.map[..., 3], np.logical_not(self.gym.map[..., 4]))

            # Find all targets under NFZs
            p = self.gym.camera.camera_half_length
            nfz_target_idx = np.array(np.where(np.logical_and(targets, self.gym.map[..., 1]))).transpose()
            visible_map = np.pad(np.logical_not(self.shadowing_map), ((0, 0), (0, 0), (p, p), (p, p)))
            for idx in nfz_target_idx:
                visible = visible_map[idx[0], idx[1]]
                padded_idx = idx + p
                view = np.zeros_like(visible)
                view[padded_idx[0] - p:padded_idx[0] + p + 1, padded_idx[1] - p:padded_idx[1] + p + 1] = 1
                visible = np.logical_and(visible, view)
                targets = np.logical_or(targets, visible[p:-p, p:-p])

            # Get all relevant cells, reachable and target
            reachable = np.logical_and(landing + distances < budget, distances > 0)
            relevant = np.logical_and(reachable, targets)

            if not np.any(relevant):
                self.phase = 1
                return self.get_action(obs)
            np.set_printoptions(threshold=sys.maxsize)
            distances = np.where(relevant, distances, np.inf)
            distances = np.where(distances == -1, np.inf, distances)
            goal = np.unravel_index(distances.argmin(), distances.shape)
            goal_ = self.distance_map[:, :, goal[0], goal[1]]
            return self.shortest_path_action(goal_, position)

        elif self.phase == 1:
            if self.gym.map[position[0], position[1], 0]:
                return 4  # Land
            return self.shortest_path_action(self.landing_map, position)

    @staticmethod
    def shortest_path_action(distance_map, position):
        shortest_distance = np.pad(distance_map,
                                   ((1, 1), (1, 1)), constant_values=1000)
        shortest_distance = np.where(shortest_distance == -1, 1000, shortest_distance)
        query = position + 1  # Padding compensation
        dist_up = shortest_distance[query[0], query[1] + 1]
        dist_down = shortest_distance[query[0], query[1] - 1]
        dist_left = shortest_distance[query[0] - 1, query[1]]
        dist_right = shortest_distance[query[0] + 1, query[1]]

        distances = [dist_right, dist_up, dist_left, dist_down]
        actions = np.where(distances == np.min(distances))[0]
        return actions[0]
