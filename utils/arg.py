import argparse
import yaml
from easydict import EasyDict as edict
import sys


def get_config():
    def join(loader, node):
        seq = loader.construct_sequence(node)
        return ''.join([str(i) for i in seq])

    yaml.add_constructor('!join', join)

    def value_type(v):
        """Convert str to bool/int/float if possible"""
        try:
            if v.lower() == "true":
                return True
            elif v.lower() == "false":
                return False
            else:
                try:
                    return int(v)
                except ValueError:
                    try:
                        return float(v)
                    except ValueError:
                        return v
        except AttributeError:
            return v

    def set_nested_key(data, keys, v):
        """Sets value in nested dictionary"""
        key = keys.pop(0)

        if keys:
            if key not in data:
                data[key] = {}
            set_nested_key(data[key], keys, v)
        else:
            data[key] = value_type(v)

    parser = argparse.ArgumentParser(description="Override YAML values")
    parser.add_argument(
        "--config", "-c", type=str, required=True, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--load", type=str, default="", help="Force to load the weight from somewhere else"
    )
    parser.add_argument(
        "--set",
        "-s",
        type=str,
        action="append",
        nargs=2,
        metavar=("KEY", "VALUE"),
        help="New value for the key",
    )
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    if args.set is not None:
        for key_value in args.set:
            key_parts = key_value[0].split(".")
            value = key_value[1]
            set_nested_key(config, key_parts, value)

    config['load'] = args.load
    config = edict(config)

    return config, args.config, ' '.join(sys.argv)
