import yaml


# read yaml files that defines hyper-parameters and the location of data
def read_yaml(path='config.yaml'):
    with open(path, 'r') as stream:
        try:
            with open(path) as fixed_stream:

                z = {**yaml.load(stream), **yaml.load(fixed_stream)}
                return z

        except yaml.YAMLError as exc:
            return exc
