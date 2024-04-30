import os
import yaml
import rospkg

rospack = rospkg.RosPack()
base_path = rospack.get_path('lics')
default_config_path = os.path.join(base_path, 'config', 'config.yaml')


def load_config(config_path=default_config_path):
    with open(config_path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)
