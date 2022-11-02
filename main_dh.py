from dataclasses import dataclass

from src.gym.dh import DHGymParams, DHGym
from src.trainer.ddqn import DDQNTrainerParams, DDQNTrainer
from src.base.evaluator import Evaluator, EvaluatorParams
from src.base.logger import LoggerParams, Logger
from utils import AbstractParams


@dataclass
class DHParams(AbstractParams):
    trainer: DDQNTrainerParams = DDQNTrainerParams()
    gym: DHGymParams = DHGymParams()
    logger: LoggerParams = LoggerParams()
    evaluator: EvaluatorParams = EvaluatorParams()


if __name__ == "__main__":

    params, args = DHParams.from_args()
    log_dir = params.create_folders(args)

    logger = Logger(params.logger, log_dir)
    gym = DHGym(params.gym)
    trainer = DDQNTrainer(params.trainer, gym, logger)
    evaluator = Evaluator(params.evaluator, trainer, gym)
    logger.evaluator = evaluator

    trainer.train()
