import argparse
import os
import distutils.util
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Generic, Tuple

from dataclasses_json import DataClassJsonMixin

import tensorflow as tf

from src.trainer.utils import ParamType


def getattr_recursive(obj, s):
    if isinstance(s, list):
        split = s
    else:
        split = s.split('/')
    if isinstance(obj, dict):
        if len(split) > 1:
            return getattr_recursive(obj[split[0]], split[1:])
        else:
            return obj[split[0]]
    return getattr_recursive(getattr(obj, split[0]), split[1:]) if len(split) > 1 else getattr(obj, split[0])


def setattr_recursive(obj, s, val):
    if isinstance(s, list):
        split = s
    else:
        split = s.split('/')
    if isinstance(obj, dict):
        if len(split) > 1:
            return setattr_recursive(obj[split[0]], split[1:], val)
        else:
            obj[split[0]] = val
            return None
    return setattr_recursive(getattr(obj, split[0]), split[1:], val) if len(split) > 1 else setattr(obj, split[0], val)


def get_bool_user(message, default: bool):
    if default:
        default_string = '[Y/n]'
    else:
        default_string = '[y/N]'
    resp = input('{} {}\n'.format(message, default_string))
    try:
        if distutils.util.strtobool(resp):
            return True
        else:
            return False
    except ValueError:
        return default


def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict


class AbstractParams(DataClassJsonMixin):

    def __init__(self):
        self.log_dir = None

    def save_to(self, config_path):
        js = self.to_dict()
        with open(config_path, 'w') as f:
            json.dump(js, f, indent=4)

    @classmethod
    def read_from(cls, config_path):
        with open(config_path, 'r') as f:
            js = json.load(f)
            params = cls.from_dict(js)
            return params

    def override_params(self, overrides):
        assert (len(overrides) % 2 == 0)
        for k in range(0, len(overrides), 2):
            oldval = getattr_recursive(self, overrides[k])
            if type(oldval) == bool:
                to_val = bool(distutils.util.strtobool(overrides[k + 1]))
            else:
                to_val = type(oldval)(overrides[k + 1])
            setattr_recursive(self, overrides[k],
                              to_val)
            print("Overriding param", overrides[k], "from", oldval, "to", to_val)

        return self

    @classmethod
    def from_args(cls, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser()
        parser.add_argument('--gpu', action='store_true', help='Activates usage of GPU')
        parser.add_argument('--gpu_id', default=0, help='Activates usage of GPU')
        parser.add_argument('--id', default=None, help='Log file name')
        parser.add_argument('--load_weights', default=None, help='')
        parser.add_argument('--generate', action='store_true', help='')
        parser.add_argument('--verbose', action='store_true')
        parser.add_argument('--params', nargs='*', default=None)
        parser.add_argument('config', help='Path to config file')
        args = parser.parse_args()

        if args.generate or not os.path.isfile(args.config):
            cls().save_to(args.config)
            print(f"Saved config to {args.config}")
            exit(0)
        params = cls.read_from(args.config)
        if args.params is not None:
            params.override_params(args.params)

        if not args.gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        else:
            physical_devices = tf.config.list_physical_devices('GPU')
            try:
                gpu_used = physical_devices[int(args.gpu_id)]
                tf.config.set_visible_devices(gpu_used, 'GPU')
                tf.config.experimental.set_memory_growth(gpu_used, True)
                print('Using following GPU: ', gpu_used.name)
            except:
                print("Invalid device or cannot modify virtual devices once initialized. Not too good probably")
                exit(0)
                pass
        return params, args

    def create_folders(self, args, config_name="config.json"):
        run_id = "_" + args.id if args.id is not None else ""
        self.log_dir = "logs/" + datetime.now().strftime("%Y%m%d_%H%M%S") + run_id + "/"
        os.makedirs(self.log_dir, exist_ok=True)
        model_dir = self.log_dir + 'models/'
        os.makedirs(model_dir, exist_ok=True)

        self.save_to(self.log_dir + config_name)

        return self.log_dir


def print_nn_summary(network_path):
    model = tf.keras.models.load_model(network_path)
    model.summary()


@dataclass
class FactoryParams(Generic[ParamType]):
    type: str
    params: ParamType


class Factory:

    @classmethod
    def registry(cls):
        raise NotImplementedError()

    @staticmethod
    def resolve_recursive(default, params):
        for key, value in params.items():
            if isinstance(value, dict):
                t = type(getattr(default, key))
                Factory.resolve_recursive(t(), value)
                params[key] = t(**value)
        # return params

    @classmethod
    def create(cls, params: dict, **kwargs):
        type_id = params["type"] if "type" in params.keys() else cls.defaults()[0]
        obj_type = cls.registry()[type_id]
        p = params["params"] if "params" in params.keys() else params
        if isinstance(p, dict):
            cls.resolve_recursive(obj_type.Params(), p)
            obj_params = obj_type.Params(**p)
        else:
            obj_params = p
        if "params" in params.keys():
            params["params"] = obj_params
        return obj_type(obj_params, **kwargs)

    @classmethod
    def default_params(cls):
        return FactoryParams[cls.defaults()[1].Params](type=cls.defaults()[0], params=cls.defaults()[1].Params())

    @classmethod
    def defaults(cls) -> Tuple[str, type]:
        raise NotImplementedError()

    @classmethod
    def default_param_type(cls):
        return FactoryParams[cls.defaults()[1].Params]

    @classmethod
    def type_ids(cls):
        return list(cls.registry().keys())

    @classmethod
    def param_types(cls):
        return [obj_type.Params for obj_type in cls.registry().values()]

    @classmethod
    def obj_types(cls):
        return [obj_type for obj_type, _ in cls.registry().values()]
