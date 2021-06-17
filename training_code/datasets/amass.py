import glob
import os.path as osp

import numpy as np
from scipy.spatial import cKDTree as KDTree
from torch.utils import data
from trimesh import Trimesh
from leap.tools.libmesh import check_mesh_contains


class AmassDataset(data.Dataset):
    """ AMASS dataset class for occupancy training. """

    def __init__(self, cfg, mode):
        """ Initialization of the the 3D shape dataset.

        Args:
            cfg (dict): dataset configuration
            mode (str): `train`, `val`, or 'test' dataset mode
        """
        # Attributes
        self.dataset_folder = cfg['dataset_folder']
        self.bm_path = cfg['bm_path']
        self.split_file = cfg[f'{mode}_split']

        # Sampling config
        sampling_config = cfg.get('sampling_config', {})
        self.points_uniform_ratio = sampling_config.get('points_uniform_ratio', 0.5)
        self.bbox_padding = sampling_config.get('bbox_padding', 0)
        self.points_padding = sampling_config.get('points_padding', 0.1)
        self.points_sigma = sampling_config.get('points_sigma', 0.01)

        self.n_points_posed = sampling_config.get('n_points_posed', 2048)
        self.n_points_can = sampling_config.get('n_points_can', 2048)

        # Get all models
        self.data = self._load_data_files()

    def _load_data_files(self):
        # load SMPL datasets
        # smpl_model, num_joints, model_type = BodyModel.load_smpl_model()
        self.faces = np.load(osp.join(self.bm_path, 'neutral', 'model.npz'))['f']
        self.sk_weights = {
            gender: np.load(osp.join(self.bm_path, gender, 'model.npz'))['weights']
            for gender in ['male', 'female', 'neutral']
        }

        # list files
        data_list = []
        with open(self.split_file, 'r') as f:
            for _sequence in f:
                sequence = _sequence.strip()  # sequence in format dataset/subject/sequence
                sequence = sequence.replace('/', osp.sep)
                points_dir = osp.join(self.dataset_folder, sequence)
                data_files = sorted(glob.glob(osp.join(points_dir, '*.npz')))
                data_list.extend(data_files)

        return data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """ Returns an item of the dataset.

        Args:
            idx (int): ID of datasets point
        """
        data_path = self.data[idx]
        np_data = np.load(data_path)

        to_ret = {
            'can_vertices': np_data['can_vertices'].astype(np.float32),
            'posed_vertices': np_data['posed_vertices'].astype(np.float32),

            'fwd_transformation': np_data['fwd_transformation'].astype(np.float32),
            'pose': np_data['pose_mat'].astype(np.float32),
            'rel_joints': np_data['rel_joints'].astype(np.float32),
        }
        self.add_data_files(np_data, to_ret)

        return to_ret

    def add_data_files(self, np_data, to_ret):
        pass

    def compute_sk_weights(self, vertices, sampled_points, gender):
        kd_tree = KDTree(vertices)
        p_idx = kd_tree.query(sampled_points)[1]
        points_sk_weights = self.sk_weights[gender][p_idx, :]

        return points_sk_weights.astype(np.float32)

    def sample_points(self, mesh, n_points, prefix='', compute_occupancy=False):
        # Get extents of model.
        bb_min = np.min(mesh.vertices, axis=0)
        bb_max = np.max(mesh.vertices, axis=0)
        total_size = (bb_max - bb_min).max()

        # Scales all dimensions equally.
        scale = total_size / (1 - self.bbox_padding)
        loc = np.array([(bb_min[0] + bb_max[0]) / 2.,
                        (bb_min[1] + bb_max[1]) / 2.,
                        (bb_min[2] + bb_max[2]) / 2.], dtype=np.float32)

        n_points_uniform = int(n_points * self.points_uniform_ratio)
        n_points_surface = n_points - n_points_uniform

        box_size = 1 + self.points_padding
        points_uniform = np.random.rand(n_points_uniform, 3)
        points_uniform = box_size * (points_uniform - 0.5)
        # Scale points in (padded) unit box back to the original space
        points_uniform *= scale
        points_uniform += loc
        # Sample points around posed-mesh surface
        n_points_surface_cloth = n_points_surface
        points_surface = mesh.sample(n_points_surface_cloth)

        points_surface = points_surface[:n_points_surface_cloth]
        points_surface += np.random.normal(scale=self.points_sigma, size=points_surface.shape)

        # Check occupancy values for sampled points
        query_points = np.vstack([points_uniform, points_surface]).astype(np.float32)

        to_ret = {
            f'{prefix}points': query_points,
            f'{prefix}loc': loc,
            f'{prefix}scale': np.asarray(scale),
        }
        if compute_occupancy:
            to_ret[f'{prefix}occ'] = check_mesh_contains(mesh, query_points).astype(np.float32)

        return to_ret


class AmassLEAPOccupancyDataset(AmassDataset):
    """ AMASS dataset class for occupancy training. """

    def __init__(self, cfg, mode):
        super().__init__(cfg, mode)

    def add_data_files(self, np_data, to_ret):
        # sample training points
        to_ret.update(self.sample_points(
            Trimesh(to_ret['posed_vertices'], self.faces),
            self.n_points_posed,
            compute_occupancy=True))

        to_ret.update(self.sample_points(
            Trimesh(to_ret['can_vertices'], self.faces),
            self.n_points_can,
            prefix='can_',
            compute_occupancy=True))

        to_ret['can_points_sk_weights'] = self.compute_sk_weights(
            to_ret['can_vertices'],
            to_ret['can_points'],
            np_data['gender'].item())


class AmassFWDLBSDataset(AmassDataset):
    """ AMASS dataset class for forward LBS training. """

    def __init__(self, cfg, mode):
        super().__init__(cfg, mode)

    def __getitem__(self, idx):
        """ Returns an item of the dataset.

        Args:
            idx (int): ID of datasets point
        """
        data_path = self.data[idx]
        np_data = np.load(data_path, allow_pickle=True)

        to_ret = {
            'can_vertices': np_data['can_vertices'].astype(np.float32),
        }

        # sample points
        to_ret.update(self.sample_points(
            Trimesh(to_ret['can_vertices'], self.faces),
            self.n_points_can,
            prefix='can_'))

        # proxy skinning weights
        to_ret['can_points_sk_weights'] = self.compute_sk_weights(
            to_ret['can_vertices'],
            to_ret['can_points'],
            np_data['gender'].item())

        return to_ret


class AmassINVLBSDataset(AmassDataset):
    """ AMASS dataset class for inverse LBS training. """

    def __init__(self, cfg, mode):
        super().__init__(cfg, mode)

    def __getitem__(self, idx):
        data_path = self.data[idx]
        np_data = np.load(data_path)

        to_ret = {
            'can_vertices': np_data['can_vertices'].astype(np.float32),
            'posed_vertices': np_data['posed_vertices'].astype(np.float32),
            'fwd_transformation': np_data['fwd_transformation'].astype(np.float32),
        }

        # sample points
        to_ret.update(self.sample_points(
            Trimesh(to_ret['posed_vertices'], self.faces), self.n_points_posed))

        # proxy skinning weights
        to_ret['points_sk_weights'] = self.compute_sk_weights(
            to_ret['posed_vertices'],
            to_ret['points'],
            np_data['gender'].item())

        return to_ret
