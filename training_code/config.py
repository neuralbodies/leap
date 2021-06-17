import os

import yaml

import datasets
import leap
import trainers


def load_config(path):
    """ Loads config file.

    Args:
        path (str): path to config file
    """
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)

    # add additional attributes
    bm_path = os.path.join(cfg['data']['bm_path'], 'neutral', 'model.npz')
    model_type, num_joints = leap.LEAPBodyModel.get_num_joints(bm_path)

    cfg['model']['num_joints'] = num_joints
    cfg['model']['model_type'] = model_type
    cfg['model']['parent_mapping'] = leap.LEAPBodyModel.get_parent_mapping(model_type)
    if cfg['method'] == 'leap_model':
        for key in ['inv_lbs_model_config', 'fwd_lbs_model_config']:
            for attr in ['num_joints', 'model_type', 'parent_mapping']:
                cfg['model'][key][attr] = cfg['model'][attr]

    return cfg


def get_model(cfg):
    """ Returns the model instance.

    Args:
        cfg (dict): config dictionary

    Returns:
        model (torch.nn.Module)
    """
    method = cfg['method']

    assert method in ['leap_model', 'inv_lbs', 'fwd_lbs'], \
        'Not supported method type'

    model = {
        'leap_model': leap.LEAPModel,
        'inv_lbs': leap.INVLBS,
        'fwd_lbs': leap.FWDLBS,
    }[method].from_cfg(cfg['model'])

    return model.to(device=cfg['device'])


def get_trainer(model, optimizer, cfg):
    """ Returns a trainer instance.

    Args:
        model (nn.Module): the model which is used
        optimizer (optimizer): pytorch optimizer
        cfg (dict): config dictionary

    Returns:
        trainer instance (BaseTrainer)
    """
    method = cfg['method']

    assert method in ['leap_model', 'inv_lbs', 'fwd_lbs'], \
        'Not supported method type'

    trainer = {
        'leap_model': trainers.LEAPModelTrainer,
        'inv_lbs': trainers.INVLBSTrainer,
        'fwd_lbs': trainers.FWDLBSTrainer,
    }[method](model, optimizer, cfg)

    return trainer


def get_dataset(mode, cfg):
    """ Returns the dataset.

    Args:
        mode (str): `train`, `val`, or 'test' dataset mode
        cfg (dict): config dictionary

    Returns:
        dataset (torch.data.utils.data.Dataset)
    """
    method = cfg['method']
    dataset_type = cfg['data']['dataset']

    assert method in ['leap_model', 'inv_lbs', 'fwd_lbs']
    assert dataset_type in ['amass']
    assert mode in ['train', 'val', 'test']

    # Create dataset
    if dataset_type == 'amass':
        dataset = {
            'leap_model': datasets.AmassLEAPOccupancyDataset,
            'inv_lbs': datasets.AmassINVLBSDataset,
            'fwd_lbs': datasets.AmassFWDLBSDataset,
        }[method]
    else:
        raise NotImplementedError(f'Not supported dataset type ({dataset_type})')

    dataset = dataset(cfg['data'], mode)

    return dataset

# def get_generator(model, cfg):
#     """ Returns a generator instance.
#
#     Args:
#         model (nn.Module): the model which is used
#         cfg (dict): config dictionary
#         device (device): pytorch device
#     """
#     assert cfg['method'] == 'leap_model'
#     generator = None  # todo impl this
#     return generator


# Datasets
