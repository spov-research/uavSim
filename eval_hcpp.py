import numpy as np
import pygame

from main_hcpp import HCPPParams
from src.gym.hcpp import HCPPGym
from src.base.evaluator import Evaluator, PyGameHumanMouse
from src.trainer.hddqn import HDDQNTrainer

if __name__ == "__main__":

    params, args = HCPPParams.from_args()
    log_dir = args.config.rsplit('/', maxsplit=1)[0]

    params.evaluator.show_eval = True
    params.evaluator.use_softmax = False

    gym = HCPPGym(params.gym)
    trainer = HDDQNTrainer(params.trainer, gym, None)
    human = PyGameHumanMouse(key_action_mapping=[(pygame.K_SPACE, int(np.prod(params.gym.target_shape).astype(int)))])
    evaluator = Evaluator(params.evaluator, trainer, gym, human=human)

    trainer.load_network(log_dir + "/models")

    while True:
        trainer.load_weights(log_dir + "/models")
        evaluator.evaluate_episode_interactive()
