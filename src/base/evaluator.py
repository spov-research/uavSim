from dataclasses import dataclass

import pygame
from tqdm import tqdm


class PyGameHuman:
    def __init__(self, key_action_mapping: list[tuple[int, int]], terminate_key=pygame.K_t, kill_key=pygame.K_q):
        self.kill_key = kill_key
        self.terminate_key = terminate_key
        self.key_action_mapping = key_action_mapping

    def get_action(self) -> (int, bool, bool):
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

    def get_action_non_blocking(self) -> (int, bool, bool):
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


@dataclass
class EvaluatorParams:
    show_eval: bool = False
    use_softmax: bool = False


class Evaluator:

    def __init__(self, params: EvaluatorParams, trainer, gym):
        self.params = params
        self.trainer = trainer
        self.gym = gym.__class__(gym.params)
        for render in gym.render_registry:
            self.gym.register_render(render["render_func"], render["shape"])
        if self.params.show_eval:
            self.gym.params.render = True

        self.human = None
        self.mode = "run"

    def evaluate_episode(self):
        state, info = self.gym.reset()
        obs = self.trainer.prepare_observations(state)
        terminal = False
        while not terminal:
            if self.params.use_softmax:
                action = self.trainer.get_action(obs)
            else:
                action = self.trainer.get_exploit_action(obs)
            state, reward, terminal, truncated, info = self.gym.step(action)
            obs = self.trainer.prepare_observations(state)

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
            self.human = PyGameHuman(key_action_mapping=[(pygame.K_RIGHT, 0),
                                                         (pygame.K_DOWN, 1),
                                                         (pygame.K_LEFT, 2),
                                                         (pygame.K_UP, 3),
                                                         (pygame.K_SPACE, 4)])
        state, info = self.gym.reset()
        obs = self.trainer.prepare_observations(state)
        terminal = False
        while not terminal:
            action = self.handle_events()
            if action is None:
                if self.params.use_softmax:
                    action = self.trainer.get_action(obs)
                else:
                    action = self.trainer.get_exploit_action(obs)
            state, reward, terminal, truncated, info = self.gym.step(action)
            obs = self.trainer.prepare_observations(state)
        return info

    def handle_events(self):
        while True:
            action = None
            key_pressed = False
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.KEYDOWN:
                    keys = pygame.key.get_pressed()
                    key_pressed = True
                    if keys[pygame.K_h]:
                        self.mode = "human"
                    elif keys[pygame.K_s]:
                        self.mode = "step"
                    elif keys[pygame.K_r]:
                        self.mode = "run"
                    action, terminate, kill = self.human.get_action_non_blocking()
                    if kill:
                        exit(0)

            if self.mode == "run":
                return None
            if self.mode == "human":
                if action is not None:
                    return action
            if self.mode == "step":
                if key_pressed:
                    return None
