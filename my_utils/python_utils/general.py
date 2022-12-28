from six import iteritems
from os import makedirs
from os.path import exists
import json


def make_dir_if_not_exist(dir_path):
    if not exists(dir_path):
        makedirs(dir_path)
    return dir_path


def get_arg_string(args, include_explicit_args=False):
    arg_str = ""
    for key, val in iteritems(args):
        if isinstance(val, dict):
            new_val = json.dumps(val)
            arg_str += "--{}='{}' ".format(key, new_val)
        elif isinstance(val, str):
            arg_str += '--{}="{}" '.format(key, val)
        elif isinstance(val, (list, tuple)):
            arg_str += '--{} '.format(key)
            for v in val:
                if isinstance(v, str):
                    arg_str += '"{}" '.format(v)
                else:
                    arg_str += '{} '.format(v)
        else:
            arg_str += '--{}={} '.format(key, val)

    if include_explicit_args:
        arg_str += "--_explicit_args='{}' ".format(json.dumps(args))
    arg_str = arg_str.strip()

    return arg_str