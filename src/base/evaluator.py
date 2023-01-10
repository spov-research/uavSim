from dataclasses import dataclass
from typing import Tuple, List

import cv2
import numpy as np
import pygame
from tqdm import tqdm
import tensorflow as tf


class PyGameHuman:
    def __init__(self, key_action_mapping: List[Tuple[int, int]], terminate_key=pygame.K_t, kill_key=pygame.K_q):
        self.kill_key = kill_key
        self.terminate_key = terminate_key
        self.key_action_mapping = key_action_mapping

    def get_action(self, position) -> (int, bool, bool):
        while True:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    return 0, False, True
                elif event.type == pygame.KEYDOWN:
                    keys = pygame.key.get_pressed()
                    if keys[self.kill_key]:
                        return 0, False, True
                    if keys[self.terminate_key]:
                        return 0, True, False
                    else:
                        for key, action in self.key_action_mapping:
                            if keys[key]:
                                return action, False, False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = np.array(pygame.mouse.get_pos())

                    print(pos)

    def wait_key(self):
        while True:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.KEYDOWN:
                    return

    def get_action_non_blocking(self, position) -> (int, bool, bool):
        keys = pygame.key.get_pressed()
        if keys[self.kill_key]:
            return None, False, True
        if keys[self.terminate_key]:
            return None, True, False
        else:
            for key, action in self.key_action_mapping:
                if keys[key]:
                    return action, False, False
        return None, False, False


class PyGameHumanMouse:
    def __init__(self, key_action_mapping: List[Tuple[int, int]], window_size=768, cells=32, target_size=(15, 15),
                 terminate_key=pygame.K_t,
                 kill_key=pygame.K_q):
        self.kill_key = kill_key
        self.terminate_key = terminate_key
        self.key_action_mapping = key_action_mapping
        self.window_size = window_size
        self.cells = cells
        self.target_size = np.array(target_size)

    def get_action(self, position) -> (int, bool, bool):
        while True:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    return 0, False, True
                elif event.type == pygame.KEYDOWN:
                    keys = pygame.key.get_pressed()
                    if keys[self.kill_key]:
                        return 0, False, True
                    if keys[self.terminate_key]:
                        return 0, True, False
                    else:
                        for key, action in self.key_action_mapping:
                            if keys[key]:
                                return action, False, False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = np.array(pygame.mouse.get_pos())
                    action = (pos * self.cells / self.window_size).astype(int)
                    action = (action - position) + self.target_size // 2

                    if (action >= 0).all() and (action < self.target_size).all():
                        flat_action = action[0] * self.target_size[0] + action[1]
                        return flat_action, False, False

    def get_action_non_blocking(self, position) -> (int, bool, bool):
        keys = pygame.key.get_pressed()
        if keys[self.kill_key]:
            return None, False, True
        if keys[self.terminate_key]:
            return None, True, False
        pressed = pygame.mouse.get_pressed()
        if pressed[0]:
            pos = np.array(pygame.mouse.get_pos())
            action = (pos * self.cells / self.window_size).astype(int)
            action = (action - position) + self.target_size // 2

            if (action >= 0).all() and (action < self.target_size).all():
                flat_action = action[0] * self.target_size[0] + action[1]
                return flat_action, False, False
        return None, False, False


@dataclass
class EvaluatorParams:
    show_eval: bool = False
    use_softmax: bool = False


class Evaluator:

    def __init__(self, params: EvaluatorParams, trainer, gym, human=None):
        self.params = params
        self.trainer = trainer
        self.gym = gym.__class__(gym.params)
        for render in gym.render_registry:
            self.gym.register_render(render["render_func"], render["shape"])
        if self.params.show_eval:
            self.gym.params.render = True

        self.human = human
        self.mode = "human"
        self.stochastic = self.params.use_softmax
        self.recorder = None

    def evaluate_episode(self):
        state, info = self.gym.reset()
        while not info["timeout"]:
            obs = self.trainer.observation_function(state)
            if self.params.use_softmax:
                action = self.trainer.get_action(obs)
            else:
                action = self.trainer.q_model.get_max_action(obs)[0].numpy()
            state, reward, terminal, truncated, info = self.gym.step(action)
            if terminal:
                break

        if self.params.show_eval:
            self.gym.close()
        return info

    def evaluate_multiple_episodes(self, n):
        stats = []
        for _ in tqdm(range(n)):
            stats.append(self.evaluate_episode())
        return stats

    def evaluate_episode_interactive(self):
        if self.human is None:
            print("No interface configured. Cannot do interactive.")
            return
        state, info = self.gym.reset()
        while not info["timeout"]:
            obs = self.trainer.observation_function(state)
            action = self.handle_events()
            if self.recorder is not None:
                frame = pygame.surfarray.pixels3d(self.gym.window)
                frame = np.transpose(frame, (1, 0, 2))
                frame = frame[..., ::-1]
                self.recorder.write(frame)
                del frame
            if action is None:
                if self.stochastic:
                    action = self.trainer.get_action(obs)
                else:
                    action = self.trainer.q_model.get_max_action(obs)[0].numpy()
            state, reward, terminal, truncated, info = self.gym.step(action)
            if terminal:
                break
        if self.recorder is not None:
            frame = pygame.surfarray.pixels3d(self.gym.window)
            frame = np.transpose(frame, (1, 0, 2))
            frame = frame[..., ::-1]
            for _ in range(8):
                self.recorder.write(frame)
            del frame
            self.recorder.release()
            self.recorder = None
        if self.mode == "run_to_end":
            self.mode = "human"
            self.human.wait_key()
        if self.mode == "blind":
            self.mode = "human"
            self.gym.params.render = True
            self.gym.params.draw_trajectory = True
            self.gym.render()
            self.gym.params.draw_trajectory = False
            self.human.wait_key()
        return info

    def handle_events(self):
        while True:
            action = None
            key_pressed = False
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                    keys = pygame.key.get_pressed()
                    key_pressed = True
                    if keys[pygame.K_h]:
                        self.mode = "human"
                    elif keys[pygame.K_s]:
                        self.mode = "human"
                        return None
                    elif keys[pygame.K_r]:
                        self.mode = "run"
                    elif keys[pygame.K_y]:
                        self.mode = "run_to_end"
                    elif keys[pygame.K_u]:
                        self.mode = "blind"
                        self.gym.params.render = False
                    elif keys[pygame.K_t]:
                        self.stochastic = not self.stochastic
                        print("Stochastic actions") if self.stochastic else print("Greedy actions")
                    elif keys[pygame.K_o]:
                        record_to = input("Record to: [video.mp4]")
                        if not record_to:
                            record_to = "video.mp4"
                        record_fps = input("Framerate: [8]fps")
                        if not record_fps:
                            record_fps = 8
                        record_fps = int(record_fps)
                        self.recorder = cv2.VideoWriter(record_to, cv2.VideoWriter_fourcc(*'mp4v'), record_fps,
                                                        self.gym.window.get_size())

                    action, terminate, kill = self.human.get_action_non_blocking(self.gym.position)
                    if kill:
                        exit(0)
            if self.mode in ["run", "run_to_end", "blind"]:
                return None
            if self.mode == "human":
                if action is not None:
                    return action
            if self.mode == "step":
                if key_pressed:
                    return None
