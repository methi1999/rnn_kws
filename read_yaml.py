import yaml


# read yaml files that defines hyper-parameters and the location of data
def read_yaml(path='config.yaml'):
    with open(path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)