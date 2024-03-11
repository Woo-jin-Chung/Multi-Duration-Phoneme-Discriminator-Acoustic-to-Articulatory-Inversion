import os
import shutil
from argparse import Namespace

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def build_env(config, config_name, path):
    t_path = os.path.join(path, config_name)
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, os.path.join(path, config_name))


def Recursive_Parse(args_dict):
    parsed_dict = {}
    for key, value in args_dict.items():
        if isinstance(value, dict):
            value = Recursive_Parse(value)
        parsed_dict[key]= value

    args = Namespace()
    args.__dict__ = parsed_dict
    return args