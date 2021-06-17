import os
import yaml
from pathlib import Path



def cond_mkdir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def save_config(path, config):
    """ Saves config file.

    Args:
        path (str): Path to config file.
        config (dict): Config dictionary.
    """
    cond_mkdir(os.path.dirname(path))

    config['git_head'] = _get_git_commit_head()
    with open(path, 'w') as f:
        yaml.dump(config, f)


def _get_git_commit_head():
    try:
        import subprocess
        head = subprocess.check_output("git rev-parse HEAD", stderr=subprocess.DEVNULL, shell=True)
        return head.decode('utf-8').strip()
    except:
        return ''
