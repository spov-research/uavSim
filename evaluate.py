import argparse

import pygame

from src.gym import PathPlanningGymFactory
from src.trainer.observation import ObservationFunctionFactory
from src.trainer.ddqn import DDQNTrainer
from src.base.evaluator import Evaluator, PyGameHuman
from src.trainer.model import SoftmaxPolicy, QModelFactory

from train import PathPlanningParams
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mc', default=0, help='evaluates monte carlo')
    params, args = PathPlanningParams.from_args(parser)
    log_dir = args.config.rsplit('/', maxsplit=1)[0]

    params.evaluator.show_eval = int(args.mc) == 0
    params.trainer.rm_prefill = 0

    gym = PathPlanningGymFactory.create(params.gym)
    observation_function = ObservationFunctionFactory.create(params.observation,
                                                             max_budget=gym.params.budget_range[-1])
    obs_space = observation_function.get_observation_space(gym.observation_space.sample())

    q_model = QModelFactory.create(params.q_model, obs_space=obs_space, act_space=gym.action_space)
    if args.verbose:
        q_model.model.summary()
    exploration_policy = SoftmaxPolicy(params.policy, gym.action_space.n)
    trainer = DDQNTrainer(params.trainer, gym, None, q_model, None, exploration_policy,
                          observation_function=observation_function)

    human = PyGameHuman([(pygame.K_RIGHT, 0),
                         (pygame.K_DOWN, 1),
                         (pygame.K_LEFT, 2),
                         (pygame.K_UP, 3),
                         (pygame.K_SPACE, 4)])
    evaluator = Evaluator(params.evaluator, trainer, gym, human)

    q_model.load_network(log_dir + "/models")

    if int(args.mc) > 0:
        q_model.load_weights(log_dir + "/models")
        stats = evaluator.evaluate_multiple_episodes(int(args.mc))
        df = pd.DataFrame(stats)
        print("Mean:")
        print(df.mean())
        print("Std:")
        print(df.std())
        return

    while True:
        q_model.load_weights(log_dir + "/models")
        evaluator.evaluate_episode_interactive()


if __name__ == "__main__":
    main()
