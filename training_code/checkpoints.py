import os
import torch
from torch.utils import model_zoo
from urllib.parse import urlparse

import utils


class CheckpointIO:
    """ CheckpointIO class.

    It handles saving and loading checkpoints.

    Args:
        checkpoint_dir (str): path where checkpoints are saved
    """

    def __init__(self, checkpoint_dir, model, optimizer, cfg):
        self.module_dict_params = {
            f"{cfg['method']}_model": model,
            f"optimizer": optimizer,
            f"{cfg['method']}_config": cfg['model'],
        }
        self.checkpoint_dir = checkpoint_dir
        utils.cond_mkdir(checkpoint_dir)

    def save(self, filename, **kwargs):
        """ Saves the current module dictionary.

        Args:
            filename (str): name of output file
        """
        if not os.path.isabs(filename):
            filename = os.path.join(self.checkpoint_dir, filename)

        out_dict = kwargs
        for k, v in self.module_dict_params.items():
            out_dict[k] = v
            if hasattr(v, 'state_dict'):
                out_dict[k] = v.state_dict()

        torch.save(out_dict, filename)

    def load(self, filename):
        """ Loads a module dictionary from local file or url.

        Args:
            filename (str): name of saved module dictionary
        """
        # parse file
        if urlparse(filename).scheme in ('http', 'https'):
            state_dict = model_zoo.load_url(filename, progress=True)
        else:
            if not os.path.isabs(filename):
                filename = os.path.join(self.checkpoint_dir, filename)

            if not os.path.exists(filename):
                raise FileExistsError

            state_dict = torch.load(filename)

        print(f'=> Loading checkpoint from: {filename}')
        for k, v in self.module_dict_params.items():
            if hasattr(v, 'load_state_dict') and v is not None:
                v.load_state_dict(state_dict[k])

        scalars = {k: v for k, v in state_dict.items() if k not in self.module_dict_params}
        return scalars
