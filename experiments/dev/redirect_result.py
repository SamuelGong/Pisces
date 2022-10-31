import os
import sys
import yaml


def load_yaml_conf(yaml_file):
    with open(yaml_file, 'r') as fin:
        data = yaml.load(fin, Loader=yaml.FullLoader)
    return data

def store_yaml_conf(dictionary, yaml_file):
    with open(yaml_file, 'w') as fout:
        yaml.dump(dictionary, fout)

def main(args):
    config_path, temp_config_path, result_save_path = args
    config = load_yaml_conf(config_path)

    if 'results' in config:
        config['results']['results_dir'] = result_save_path + "/"

    store_yaml_conf(config, temp_config_path)


if __name__ == '__main__':
    main(sys.argv[1:4])

