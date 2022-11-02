from collections import deque
from dataclasses import dataclass
import tensorflow as tf

from utils import dict_mean


@dataclass
class LoggerParams:
    loss_period: int = 100
    evaluation_period: int = 1_000
    save_period: int = 10_000


class Logger:
    def __init__(self, params: LoggerParams, log_dir):
        self.params = params
        self.log_dir = log_dir
        self.evaluator = None

        self.log_writer = tf.summary.create_file_writer(self.log_dir + '/training')

        self.train_steps = 0
        self.steps = 0

        self.train_logs = deque(maxlen=self.params.loss_period)

    def log_train(self, train_log):
        self.train_steps += 1
        self.train_logs.append(train_log)

        if self.train_steps % self.params.loss_period == 0:
            logs = dict_mean(self.train_logs)
            with self.log_writer.as_default():
                for name, value in logs.items():
                    tf.summary.scalar(f'training/{name}', value, self.train_steps)

    def log_step(self, step_info):
        self.steps += 1
        if step_info["terminal"]:
            with self.log_writer.as_default():
                for name, value in step_info.items():
                    tf.summary.scalar(f'episodic/{name}', value, self.steps)

        if self.steps % self.params.evaluation_period == 0:
            if self.evaluator is not None:
                info = self.evaluator.evaluate_episode()
                with self.log_writer.as_default():
                    for name, value in info.items():
                        tf.summary.scalar(f'evaluation/{name}', value, self.steps)

    def save_network(self, network, name="model"):
        network.save(self.log_dir + f"/models/{name}")

    def save_weights(self, network, name="weights_latest"):
        network.save_weights(self.log_dir + f"/models/{name}")
