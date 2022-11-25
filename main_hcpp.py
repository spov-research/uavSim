from dataclasses import dataclass

from src.gym.cpp import CPPGymParams, CPPGym
from src.gym.hcpp import HCPPGymParams, HCPPGym
from src.base.evaluator import Evaluator, EvaluatorParams
from src.base.logger import LoggerParams, Logger
from src.trainer.hddqn import HDDQNTrainer, HDDQNTrainerParams
from utils import AbstractParams


@dataclass
class HCPPParams(AbstractParams):
    trainer: HDDQNTrainerParams = HDDQNTrainerParams()
    gym: HCPPGymParams = HCPPGymParams()
    logger: LoggerParams = LoggerParams()
    evaluator: EvaluatorParams = EvaluatorParams()


if __name__ == "__main__":

    params, args = HCPPParams.from_args()
    log_dir = params.create_folders(args)

    logger = Logger(params.logger, log_dir)
    gym = HCPPGym(params.gym)
    trainer = HDDQNTrainer(params.trainer, gym, logger)
    evaluator = Evaluator(params.evaluator, trainer, gym)
    logger.evaluator = evaluator

    trainer.train()
