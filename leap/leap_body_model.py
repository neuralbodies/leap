import os.path as osp
import pickle

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh

try:
    import torchgeometry
    from skimage import measure
except Exception:
    pass

from .modules import LEAPModel
from .tools.libmise import MISE


class LEAPBodyModel(nn.Module):
    def __init__(self,
                 leap_path=None,
                 bm_path=None,
                 num_betas=16,
                 batch_size=1,
                 dtype=torch.float32,
                 device='cpu'):
        """ Interface for LEAP body model.

        Args:
            leap_path (str): Path to pretrained LEAP model.
            bm_path (str): Path to a SMPL-compatible model file.
            num_betas (int): Number of shape coefficients for SMPL.
            batch_size (int): Batch size.
            dtype (torch.dtype): Datatype of pytorch tensors
            device (str): Which device to use.
        """
        super(LEAPBodyModel, self).__init__()

        self.batch_size = batch_size
        self.num_betas = num_betas
        self.dtype = dtype
        self.device = device

        # load SMPL-based model
        smpl_dict, self.model_type = self.load_smpl_model(bm_path)
        self.model = None
        if leap_path is not None:
            self.model = LEAPModel.load_from_file(leap_path)
            self.model = self.model.to(device=device)
            self.model.eval()

        # parse loaded model dictionary
        if self.num_betas < 1:
            self.num_betas = smpl_dict['shapedirs'].shape[-1]

        weights = np.repeat(smpl_dict['weights'][np.newaxis], batch_size, axis=0)
        kintree_table = smpl_dict['kintree_table'].astype(np.int32)
        v_template = np.repeat(smpl_dict['v_template'][np.newaxis], batch_size, axis=0)
        joint_regressor = smpl_dict['J_regressor']  # V x K
        pose_dirs = smpl_dict['posedirs'].reshape([smpl_dict['posedirs'].shape[0] * 3, -1]).T  # 6890*30 x 207
        shape_dirs = smpl_dict['shapedirs'][:, :, :self.num_betas]

        self.v_template = torch.tensor(v_template, dtype=dtype)
        self.shape_dirs = torch.tensor(shape_dirs, dtype=dtype)
        self.pose_dirs = torch.tensor(pose_dirs, dtype=dtype)
        self.joint_regressor = torch.tensor(joint_regressor, dtype=dtype)
        self.kintree_table = torch.tensor(kintree_table, dtype=torch.int32)
        self.weights = torch.tensor(weights, dtype=dtype)

        if self.model_type == 'smplx':
            begin_shape_id = 300 if smpl_dict['shapedirs'].shape[-1] > 300 else 10
            num_expressions = 10
            expr_dirs = smpl_dict['shapedirs'][:, :, begin_shape_id:(begin_shape_id + num_expressions)]
            expr_dirs = torch.tensor(expr_dirs, dtype=dtype)
            self.shape_dirs = torch.cat([self.shape_dirs, expr_dirs])

        # init controllable parameters
        self.betas = None
        self.root_loc = None
        self.root_orient = None
        self.pose_body = None
        self.pose_hand = None
        self.pose_jaw = None
        self.pose_eye = None

        # intermediate representations
        self.pose_rot_mat = None
        self.posed_vert = None
        self.can_vert = None
        self.rel_joints = None
        self.fwd_transformation = None

        if self.device != 'cpu':
            self.to_device(self.device)

    def to_device(self, device):
        """ Move pytorch tensor variables to a device.

        Args:
            device (str): PyTorch device.
        """
        self.device = device

        for attr in self.__dict__.keys():
            var = self.__getattribute__(attr)
            if torch.is_tensor(var):
                self.__setattr__(attr, var.to(device=self.device))

    def set_grad_param(self, require_grad=True):
        """ Set require_grad of pytorch tensors.

        Args:
            require_grad (bool):
        """
        for attr in self.__dict__.keys():
            var = self.__getattribute__(attr)
            if torch.is_tensor(var):
                var.require_grad = require_grad

    def set_parameters(self,
                       betas=None,
                       pose_body=None,
                       pose_hand=None,
                       pose_jaw=None,
                       pose_eye=None,
                       expression=None):
        """ Set controllable parameters.

        Args:
            betas (torch.tensor): SMPL shape coefficients (B x betas len).
            pose_body (torch.tensor): Body pose parameters (B x body joints * 3).
            pose_hand (torch.tensor): Hand pose parameters (B x hand joints * 3).
            pose_jaw (torch.tensor): Jaw pose parameters (compatible with SMPL+X) (B x jaw joints * 3).
            pose_eye (torch.tensor): Eye pose parameters (compatible with SMPL+X) (B x eye joints * 3).
            expression (torch.tensor): Expression coefficients (compatible with SMPL+X) (B x expr len).
        """
        if betas is None:
            betas = torch.tensor(np.zeros((self.batch_size, self.num_betas)), dtype=self.dtype)
        else:
            betas = betas.view(self.batch_size, self.num_betas)

        if pose_body is None and self.model_type in ['smpl', 'smplh', 'smplx']:
            pose_body = torch.tensor(np.zeros((self.batch_size, 63)), dtype=self.dtype)
        else:
            pose_body = pose_body.view(self.batch_size, 63)

        # pose_hand
        if pose_hand is None:
            if self.model_type in ['smpl']:
                pose_hand = torch.tensor(np.zeros((self.batch_size, 1 * 3 * 2)), dtype=self.dtype)
            elif self.model_type in ['smplh', 'smplx']:
                pose_hand = torch.tensor(np.zeros((self.batch_size, 15 * 3 * 2)), dtype=self.dtype)
            elif self.model_type in ['mano']:
                pose_hand = torch.tensor(np.zeros((self.batch_size, 15 * 3)), dtype=self.dtype)
        else:
            pose_hand = pose_hand.view(self.batch_size, -1)

        # face poses
        if self.model_type == 'smplx':
            if pose_jaw is None:
                pose_jaw = torch.tensor(np.zeros((self.batch_size, 1 * 3)), dtype=self.dtype)
            else:
                pose_jaw = pose_jaw.view(self.batch_size, 1*3)

            if pose_eye is None:
                pose_eye = torch.tensor(np.zeros((self.batch_size, 2 * 3)), dtype=self.dtype)
            else:
                pose_eye = pose_eye.view(self.batch_size, 2*3)

            if expression is None:
                expression = torch.tensor(np.zeros((self.batch_size, self.num_expressions)), dtype=self.dtype)
            else:
                expression = expression.view(self.batch_size, self.num_expressions)

            betas = torch.cat([betas, expression], dim=-1)

        self.root_loc = torch.tensor(np.zeros((self.batch_size, 1*3)), dtype=self.dtype, device=self.device)
        self.root_orient = torch.tensor(np.zeros((self.batch_size, 1*3)), dtype=self.dtype, device=self.device)

        self.betas = betas
        self.pose_body = pose_body
        self.pose_hand = pose_hand
        self.pose_jaw = pose_jaw
        self.pose_eye = pose_eye

    def _get_full_pose(self):
        """ Concatenates joints.

        Returns:
            full_pose (torch.tensor): Full pose (B, num_joints*3)
        """
        full_pose = [self.root_orient]
        if self.model_type in ['smplh', 'smpl']:
            full_pose.extend([self.pose_body, self.pose_hand])
        elif self.model_type == 'smplx':
            full_pose.extend([self.pose_body, self.pose_jaw, self.pose_eye, self.pose_hand])
        elif self.model_type in ['mano']:
            full_pose.extend([self.pose_hand])
        else:
            raise Exception('Unsupported model type.')

        full_pose = torch.cat(full_pose, dim=1)
        return full_pose

    def forward(self, points):
        """ Checks whether given query points are located inside of a human body.

        Args:
            points (torch.tensor): Query points (B x T x 3)

        Returns:
            occupancy values (torch.tensor): (B x T)
        """
        self.forward_parametric_model()
        occupancy = self._query_occupancy(points)
        return occupancy

    def _query_occupancy(self, points, canonical_points=False):
        if not canonical_points:
            # project query points to the canonical space
            point_weights, can_points = \
                self.model.inv_lbs(points, self.can_vert, self.posed_vert, self.fwd_transformation)
            fwd_point_weights = self.model.fwd_lbs(can_points, self.can_vert)
            cycle_distance = torch.sum((point_weights - fwd_point_weights).abs(), dim=-1, keepdim=True)
        else:  # if points are directly sampled in the canonical space
            can_points = points
            point_weights = self.model.fwd_lbs(can_points, self.can_vert)
            cycle_distance = torch.zeros_like(point_weights[..., :1])

        # occupancy check
        occupancy = self.model.leap_occupancy_decoder(
            can_points=can_points, point_weights=point_weights, cycle_distance=cycle_distance,
            can_vert=self.can_vert,
            rot_mats=self.pose_rot_mat, rel_joints=self.rel_joints,
            root_loc=self.root_loc, fwd_transformation=self.fwd_transformation)
        return occupancy

    def forward_parametric_model(self):
        B = self.pose_body.shape[0]

        # pose to rot matrices
        full_pose = self._get_full_pose()
        full_pose = full_pose.view(B, -1, 3)

        self.pose_rot_mat = torchgeometry.angle_axis_to_rotation_matrix(full_pose.view(-1, 3))[:, :3, :3]
        self.pose_rot_mat = self.pose_rot_mat.view(B, -1, 3, 3)

        # Compute identity-dependent correctives
        identity_offsets = torch.einsum('bl,mkl->bmk', self.betas, self.shape_dirs)

        # Compute pose-dependent correctives
        _pose_feature = self.pose_rot_mat[:, 1:, :, :] - torch.eye(3, dtype=self.dtype, device=self.device)  # (NxKx3x3)
        pose_offsets = torch.matmul(
            _pose_feature.view(B, -1),
            self.pose_dirs
        ).view(B, -1, 3)  # (N x P) x (P, V * 3) -> N x V x 3

        self.can_vert = self.v_template + identity_offsets + pose_offsets

        # Regress joint locations
        self.can_joint_loc = torch.einsum('bik,ji->bjk', self.v_template + identity_offsets, self.joint_regressor)

        # Skinning
        self.fwd_transformation, self.rel_joints = self.batch_rigid_transform(self.pose_rot_mat, self.can_joint_loc)
        self.posed_vert = self.lbs_skinning(self.fwd_transformation, self.can_vert)

    def batch_rigid_transform(self, rot_mats, joints):
        """ Rigid transformations over joints

        Args:
            rot_mats (torch.tensor): Rotation matrices (BxNx3x3).
            joints (torch.tensor): Joint locations (BxNx3).

        Returns:
            posed_joints (torch.tensor): The locations of the joints after applying transformations (BxNx3).
            rel_transforms (torch.tensor): Relative wrt root joint rigid transformations (BxNx4x4).
        """
        B, K = rot_mats.shape[0], joints.shape[1]

        parents = self.kintree_table[0].long()

        joints = torch.unsqueeze(joints, dim=-1)

        rel_joints = joints.clone()
        rel_joints[:, 1:] -= joints[:, parents[1:]]

        transforms_mat = torch.cat([
            F.pad(rot_mats.reshape(-1, 3, 3), [0, 0, 0, 1]),
            F.pad(rel_joints.reshape(-1, 3, 1), [0, 0, 0, 1], value=1)
        ], dim=2).reshape(-1, joints.shape[1], 4, 4)

        transform_chain = [transforms_mat[:, 0]]
        for i in range(1, parents.shape[0]):
            curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[:, i])
            transform_chain.append(curr_res)

        transforms = torch.stack(transform_chain, dim=1)
        joints_hom = torch.cat([
            joints,
            torch.zeros([B, K, 1, 1], dtype=self.dtype, device=self.device)
        ], dim=2)
        init_bone = F.pad(torch.matmul(transforms, joints_hom), [3, 0, 0, 0, 0, 0, 0, 0])
        rel_transforms = transforms - init_bone

        return rel_transforms, rel_joints

    @staticmethod
    def load_smpl_model(bm_path):
        assert osp.exists(bm_path), f'File does not exist: {bm_path}'

        # load smpl parameters
        ext = osp.splitext(bm_path)[-1]
        if ext == '.npz':
            smpl_model = np.load(bm_path, allow_pickle=True)
        elif ext == 'pkl':
            with open(bm_path, 'rb') as smpl_file:
                smpl_model = pickle.load(smpl_file, encoding='latin1')
        else:
            raise ValueError(f'Invalid file type: {ext}')

        num_joints = smpl_model['posedirs'].shape[2] // 3
        model_type = {69: 'smpl', 153: 'smplh', 162: 'smplx', 45: 'mano'}[num_joints]

        return smpl_model, model_type

    @staticmethod
    def get_num_joints(bm_path):
        model_type = LEAPBodyModel.load_smpl_model(bm_path)[1]

        num_joints = {
            'smpl': 24,
            'smplh': 52,
            'smplx': 55,
            'mano': 16,
        }[model_type]

        return model_type, num_joints

    @staticmethod
    def get_parent_mapping(model_type):
        smplh_mappings = [
            -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 22, 23, 20, 25,
            26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 21, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50
        ]

        if model_type == 'smpl':
            smpl_mappings = smplh_mappings[:22] + [smplh_mappings[25]] + [smplh_mappings[40]]
            return smpl_mappings
        elif model_type == 'smplh':
            return smplh_mappings
        else:
            raise NotImplementedError

    def parse_state_dict(self, state_dict):
        """ Parse state_dict of model and return scalars.

        Args:
            state_dict (dict): State dict of model
        """

        for k, v in self.module_dict.items():
            if k in state_dict:
                v.load_state_dict(state_dict[k])
            else:
                print(f'Warning: Could not find {k} in checkpoint!')
        scalars = {k: v for k, v in state_dict.items() if k not in self.module_dict}
        return scalars

    def lbs_skinning(self, fwd_transformation, can_vert):
        """ Conversion of canonical vertices to posed vertices via linear blend skinning.

        Args:
            fwd_transformation (torch.tensor): Forward rigid transformation tensor (B x K x 4 x 4).
            can_vert (torch.tensor): Canonical vertices (B x V x 3).

        Returns:
            posed_vert (torch.tensor): Posed vertices (B x V x 3).
        """
        B = fwd_transformation.shape[0]

        _fwd_lbs_trans = torch.matmul(
            self.weights,
            fwd_transformation.view(B, -1, 16)
        ).view(B, -1, 4, 4)

        _vert_hom = torch.cat([
            can_vert,
            torch.ones([B, can_vert.shape[1], 1], dtype=self.dtype, device=self.device)
        ], dim=2)
        posed_vert = torch.matmul(_fwd_lbs_trans, torch.unsqueeze(_vert_hom, dim=-1))[:, :, :3, 0]
        return posed_vert

    @torch.no_grad()
    def _extract_mesh(self, vertices, resolution0, upsampling_steps, canonical_points=False):
        """ Runs marching cubes to extract mesh for the occupancy representation. """
        device = self.device

        # compute scale and loc
        bb_min = np.min(vertices, axis=0)
        bb_max = np.max(vertices, axis=0)
        loc = np.array([
            (bb_min[0] + bb_max[0]) / 2,
            (bb_min[1] + bb_max[1]) / 2,
            (bb_min[2] + bb_max[2]) / 2
        ])
        scale = (bb_max - bb_min).max()

        scale = torch.FloatTensor([scale]).to(device=device)
        loc = torch.from_numpy(loc).to(device=device)

        # create MISE
        threshold = 0.5
        padding = 0.1
        box_size = 1 + padding
        mesh_extractor = MISE(resolution0, upsampling_steps, threshold)

        # sample initial points
        points = mesh_extractor.query()
        while points.shape[0] != 0:
            sampled_points = torch.FloatTensor(points).to(device=device)  # Query points
            sampled_points = sampled_points / mesh_extractor.resolution  # Normalize to bounding box
            sampled_points = box_size * (sampled_points - 0.5)
            sampled_points *= scale
            sampled_points += loc

            # check occupancy for sampled points
            p_split = torch.split(sampled_points, 50000)  # to prevent OOM
            occ_hats = []
            for pi in p_split:
                pi = pi.unsqueeze(0).to(device=device)
                occ_hats.append(self._query_occupancy(pi, canonical_points).cpu().squeeze(0))
            values = torch.cat(occ_hats, dim=0).numpy().astype(np.float64)

            # sample points again
            mesh_extractor.update(points, values)
            points = mesh_extractor.query()

        occ_hat = mesh_extractor.to_dense()
        # Some short hands
        n_x, n_y, n_z = occ_hat.shape
        box_size = 1 + padding
        # Make sure that mesh is watertight
        occ_hat_padded = np.pad(occ_hat, 1, 'constant', constant_values=-1e6)
        vertices, faces, _, _ = measure.marching_cubes(occ_hat_padded, level=threshold)

        vertices -= 1  # Undo padding
        # Normalize to bounding box
        vertices /= np.array([n_x - 1, n_y - 1, n_z - 1])
        vertices = box_size * (vertices - 0.5)
        vertices = vertices * scale.item()
        vertices = vertices + loc.view(1, 3).detach().cpu().numpy()

        # Create mesh
        mesh = trimesh.Trimesh(vertices, faces, process=False)
        return mesh

    @torch.no_grad()
    def extract_canonical_mesh(self, resolution0=32, upsampling_steps=3):
        self.model.eval()
        self.forward_parametric_model()

        mesh = self._extract_mesh(self.can_vert.squeeze(0).detach().cpu().numpy(),
                                  resolution0,
                                  upsampling_steps,
                                  canonical_points=True)
        return mesh

    @torch.no_grad()
    def extract_posed_mesh(self, resolution0=32, upsampling_steps=3):
        self.model.eval()
        self.forward_parametric_model()

        mesh = self._extract_mesh(self.posed_vert.squeeze(0).detach().cpu().numpy(),
                                  resolution0,
                                  upsampling_steps,
                                  canonical_points=False)
        return mesh
