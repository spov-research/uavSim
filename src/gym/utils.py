import os

import numpy as np
from tqdm import tqdm
from skimage import io


class Map:
    def __init__(self, map_data):
        self.start_land_zone = map_data[:, :, 2].astype(bool).transpose()
        self.nfz = map_data[:, :, 0].astype(bool).transpose()
        self.obstacles = map_data[:, :, 1].astype(bool).transpose()

    def get_starting_vector(self):
        similar = np.where(self.start_land_zone)
        return list(zip(similar[1], similar[0]))

    def get_free_space_vector(self):
        free_space = np.logical_not(
            np.logical_or(self.obstacles, self.start_land_zone))
        free_idcs = np.where(free_space)
        return list(zip(free_idcs[1], free_idcs[0]))

    def get_size(self):
        return self.start_land_zone.shape[:2]

    @staticmethod
    def load_map(path):
        if type(path) is not str:
            raise TypeError('path needs to be a string')
        data = io.imread(path, as_gray=False)
        return Map(data)


def load_image(path):
    if type(path) is not str:
        raise TypeError('path needs to be a string')
    data = io.imread(path, as_gray=True)
    return np.array(data, dtype=bool)


def save_image(path, image):
    if type(path) is not str:
        raise TypeError('path needs to be a string')
    if image.dtype == bool:
        io.imsave(path, image * np.uint8(255))
    else:
        io.imsave(path, image)


def load_target(path, obstacles=None):
    if type(path) is not str:
        raise TypeError('path needs to be a string')

    data = np.array(io.imread(path, as_gray=True), dtype=bool)
    if obstacles is not None:
        data = data & ~obstacles
    return data


def calculate_landing(map_path):
    print("Calculating landing map")
    total_map = Map.load_map(map_path)
    nfz = np.pad(total_map.nfz, pad_width=((1, 1), (1, 1)), constant_values=1)
    slz = np.pad(total_map.start_land_zone, pad_width=((1, 1), (1, 1)), constant_values=0)

    min_distance = np.where(slz, 1, np.inf)
    previous = np.zeros_like(min_distance)
    while np.not_equal(previous, min_distance).any():
        previous = min_distance
        temp = min_distance + 1
        t1 = np.roll(temp, shift=1, axis=0)
        t2 = np.roll(temp, shift=-1, axis=0)
        t3 = np.roll(temp, shift=1, axis=1)
        t4 = np.roll(temp, shift=-1, axis=1)
        t = np.stack((min_distance, t1, t2, t3, t4), axis=-1)
        min_distance = np.where(nfz, np.inf, np.min(t, axis=-1))

    min_distance = min_distance[1:-1, 1:-1]
    min_distance = np.where(min_distance == np.inf, -1, min_distance).astype(int)
    return min_distance


def load_or_create_landing(map_path):
    landing_file_name = os.path.splitext(map_path)[0] + "_landing.npy"
    if os.path.exists(landing_file_name):
        return np.load(landing_file_name)
    else:
        lm = calculate_landing(map_path)
        np.save(landing_file_name, lm)
        return lm


def bresenham(x0, y0, x1, y1, obstacles, shadow_map):
    x_dist = abs(x0 - x1)
    y_dist = -abs(y0 - y1)
    x_step = 1 if x1 > x0 else -1
    y_step = 1 if y1 > y0 else -1

    error = x_dist + y_dist

    # shadowed = False
    shadow_map[x0, y0] = False

    while x0 != x1 or y0 != y1:
        if 2 * error - y_dist > x_dist - 2 * error:
            # horizontal step
            error += y_dist
            x0 += x_step
        else:
            # vertical step
            error += x_dist
            y0 += y_step

        if obstacles[x0, y0]:
            # shadowed = True
            return

        # if shadowed:
        shadow_map[x0, y0] = False


def calculate_shadowing(map_path, save_as):
    print("Calculating shadowing maps")
    total_map = Map.load_map(map_path)
    obstacles = total_map.obstacles
    size = total_map.obstacles.shape[0]
    total = size * size

    total_shadow_map = np.ones((size, size, size, size), dtype=bool)
    with tqdm(total=total) as pbar:
        for i, j in np.ndindex(total_map.obstacles.shape):
            shadow_map = np.ones((size, size), dtype=bool)

            for x in range(size):
                bresenham(i, j, x, 0, obstacles, shadow_map)
                bresenham(i, j, x, size - 1, obstacles, shadow_map)
                bresenham(i, j, 0, x, obstacles, shadow_map)
                bresenham(i, j, size - 1, x, obstacles, shadow_map)

            total_shadow_map[i, j] = shadow_map
            pbar.update(1)

    np.save(save_as, total_shadow_map)
    return total_shadow_map


def load_or_create_shadowing(map_path):
    shadow_file_name = os.path.splitext(map_path)[0] + "_shadowing.npy"
    if os.path.exists(shadow_file_name):
        return np.load(shadow_file_name)
    else:
        return calculate_shadowing(map_path, shadow_file_name)
