from dataclasses import dataclass

from src.gym import PathPlanningGymFactory
from src.trainer.observation import ObservationFunctionFactory
from src.trainer.ddqn import DDQNTrainer
from src.base.evaluator import Evaluator, EvaluatorParams
from src.base.logger import LoggerParams, Logger
from src.trainer.model import SoftmaxPolicy, SoftmaxPolicyParams, QModelFactory
from utils import AbstractParams


@dataclass
class PathPlanningParams(AbstractParams):
    trainer: DDQNTrainer.Params = DDQNTrainer.Params()
    gym: PathPlanningGymFactory.default_param_type() = PathPlanningGymFactory.default_params()
    logger: LoggerParams = LoggerParams()
    evaluator: EvaluatorParams = EvaluatorParams()
    q_model: QModelFactory.default_param_type() = QModelFactory.default_params()
    observation: ObservationFunctionFactory.default_param_type() = ObservationFunctionFactory.default_params()
    policy: SoftmaxPolicyParams = SoftmaxPolicyParams()


if __name__ == "__main__":
    params, args = PathPlanningParams.from_args()
    log_dir = params.create_folders(args)

    gym = PathPlanningGymFactory.create(params.gym)

    observation_function = ObservationFunctionFactory.create(params.observation,
                                                             max_budget=gym.params.budget_range[-1])
    obs_space = observation_function.get_observation_space(gym.observation_space.sample())

    q_model = QModelFactory.create(params.q_model, obs_space=obs_space, act_space=gym.action_space)
    target_q_model = QModelFactory.create(params.q_model, obs_space=obs_space,
                                          act_space=gym.action_space)
    target_q_model.model.set_weights(q_model.model.get_weights())  # Hard update
    if args.verbose:
        q_model.model.summary()

    exploration_policy = SoftmaxPolicy(params.policy, gym.action_space.n)
    logger = Logger(params.logger, log_dir, q_model)
    trainer = DDQNTrainer(params.trainer, gym, logger, q_model, target_q_model, exploration_policy,
                          observation_function=observation_function)
    evaluator = Evaluator(params.evaluator, trainer, gym)
    logger.evaluator = evaluator

    params.save_to(params.log_dir + "config.json")

    trainer.train()
