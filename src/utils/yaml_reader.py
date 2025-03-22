import yaml


def read_yaml(path):
    with open(path) as f:
        result = yaml.safe_load(f)
    return result
