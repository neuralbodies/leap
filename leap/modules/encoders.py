from abc import ABCMeta, ABC
from copy import deepcopy
from urllib.parse import urlparse
from os import path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo

from .layers import CBatchNorm1d, ResnetPointnet, BoneMLP, CResnetBlockConv1d


class BaseModule(nn.Module, metaclass=ABCMeta):
    @classmethod
    def load(cls, config, state_dict=None):
        model = cls.from_cfg(config)  # create model
        if model is not None and state_dict is not None:
            model.load_state_dict(state_dict)  # load weights

        return model

    @classmethod
    def from_cfg(cls, config):
        raise NotImplementedError

    @staticmethod
    def parse_pytorch_file(file_path):
        # parse file
        if urlparse(file_path).scheme in ('http', 'https'):
            state_dict = model_zoo.load_url(file_path, progress=True)
        else:
            assert osp.exists(file_path), f'File does not exist: {file_path}'
            print('=> Loading checkpoint from local file...')
            state_dict = torch.load(file_path, map_location=torch.device('cpu'))
        return state_dict


class LBSNet(BaseModule, ABC):
    def __init__(self, num_joints, hidden_size, pn_dim):
        super().__init__()

        self.dim = 3
        self.pn_dim = pn_dim
        self.c_dim = self.get_c_dim()
        self.num_joints = num_joints
        self.hidden_size = hidden_size

        # create network
        self.point_encoder = ResnetPointnet(hidden_dim=hidden_size, out_dim=self.pn_dim)

        self.fc_p = nn.Conv1d(self.dim, self.hidden_size, (1,))

        self.block0 = CResnetBlockConv1d(self.c_dim, self.hidden_size)
        self.block1 = CResnetBlockConv1d(self.c_dim, self.hidden_size)
        self.block2 = CResnetBlockConv1d(self.c_dim, self.hidden_size)

        self.bn = CBatchNorm1d(self.c_dim, self.hidden_size)

        self.fc_out = nn.Conv1d(self.hidden_size, self.num_joints, (1,))

        self.act = F.relu

    def get_c_dim(self):
        raise NotImplementedError

    def _forward(self, points, cond_code):
        B, T, _ = points.shape
        p = points.transpose(1, 2)

        net = self.fc_p(p)  # B x hidden_dim x T

        lbs_code = cond_code.unsqueeze(1).repeat(1, points.shape[1], 1)  # B x T x c_dim

        c = lbs_code.transpose(1, 2)
        net = self.block0(net, c)
        net = self.block1(net, c)
        net = self.block2(net, c)

        out = self.fc_out(self.act(self.bn(net, c)))  # B x K x T
        out = out.transpose(1, 2)  # B x T x K

        point_weights = torch.softmax(out, dim=-1)
        return point_weights


class ONet(BaseModule):
    def __init__(self, num_joints, point_feature_len, hidden_size):
        super().__init__()

        self.num_joints = num_joints
        self.point_feature_len = point_feature_len
        self.c_dim = point_feature_len + 1  # + 1 for the cycle-distance feature

        self.fc_p = nn.Conv1d(3, hidden_size, (1,))
        self.fc_0 = nn.Conv1d(hidden_size, hidden_size, (1,))
        self.fc_1 = nn.Conv1d(hidden_size, hidden_size, (1,))
        self.fc_2 = nn.Conv1d(hidden_size, hidden_size, (1,))
        self.fc_3 = nn.Conv1d(hidden_size, hidden_size, (1,))
        self.fc_4 = nn.Conv1d(hidden_size, hidden_size, (1,))

        self.bn_0 = CBatchNorm1d(self.c_dim, hidden_size)
        self.bn_1 = CBatchNorm1d(self.c_dim, hidden_size)
        self.bn_2 = CBatchNorm1d(self.c_dim, hidden_size)
        self.bn_3 = CBatchNorm1d(self.c_dim, hidden_size)
        self.bn_4 = CBatchNorm1d(self.c_dim, hidden_size)
        self.bn_5 = CBatchNorm1d(self.c_dim, hidden_size)

        self.fc_out = nn.Conv1d(hidden_size, 1, (1,))

        self.act = F.relu

    def forward(self, can_points, local_point_feature, cycle_distance):
        can_points = can_points.transpose(1, 2)
        local_cond_code = torch.cat((local_point_feature, cycle_distance), dim=-1)
        local_cond_code = local_cond_code.transpose(1, 2)

        net = self.fc_p(can_points)
        net = self.act(self.bn_0(net, local_cond_code))
        net = self.fc_0(net)
        net = self.act(self.bn_1(net, local_cond_code))
        net = self.fc_1(net)
        net = self.act(self.bn_2(net, local_cond_code))
        net = self.fc_2(net)
        net = self.act(self.bn_3(net, local_cond_code))
        net = self.fc_3(net)
        net = self.act(self.bn_4(net, local_cond_code))
        net = self.fc_4(net)
        net = self.act(self.bn_5(net, local_cond_code))
        out = self.fc_out(net)
        out = out.squeeze(1)

        return out

    @classmethod
    def from_cfg(cls, config):
        return cls(
            num_joints=config['num_joints'],
            point_feature_len=config['point_feature_len'],
            hidden_size=config['hidden_size'],
        )


class ShapeEncoder(nn.Module):
    def __init__(self, out_dim, hidden_size):
        super().__init__()

        self.out_dim = out_dim
        self.hidden_dim = hidden_size
        self.point_encoder = ResnetPointnet(out_dim, hidden_size)

    def get_out_dim(self):
        return self.out_dim

    @classmethod
    def from_cfg(cls, config):
        return cls(
            out_dim=config['out_dim'],
            hidden_size=config['hidden_size'],
        )

    def forward(self, can_vertices):
        """
        Args:
            can_vertices: B x N x 3
        Returns:

        """

        return self.point_encoder(can_vertices)


class StructureEncoder(nn.Module):
    def __init__(self, local_feature_size, parent_mapping):
        super().__init__()

        self.bone_dim = 12  # 3x3 for pose and 1x3 for joint loc
        self.input_dim = self.bone_dim + 1  # +1 for bone length
        self.parent_mapping = parent_mapping
        self.num_joints = len(parent_mapping)
        self.out_dim = self.num_joints * local_feature_size

        self.proj_bone_prior = nn.Linear(self.num_joints * self.bone_dim, local_feature_size)
        self.net = nn.ModuleList([
            BoneMLP(self.input_dim, local_feature_size) for _ in range(self.num_joints)
        ])

    def get_out_dim(self):
        return self.out_dim

    @classmethod
    def from_cfg(cls, config):
        return cls(
            local_feature_size=config['local_feature_size'],
            parent_mapping=config['parent_mapping']
        )

    def forward(self, pose, rel_joints):
        """

        Args:
            pose: B x num_joints x 3 x 3
            rel_joints: B x num_joints x 3
        """
        B, K = rel_joints.shape[0], rel_joints.shape[1]
        bone_lengths = torch.norm(rel_joints.squeeze(-1), dim=-1).view(B, K, 1)  # B x num_joints x 1

        bone_features = torch.cat((pose.contiguous().view(B, K, -1),
                                   rel_joints.contiguous().view(B, K, -1)), dim=-1)

        root_bone_prior = self.proj_bone_prior(bone_features.contiguous().view(B, -1))  # B, bottleneck

        # fwd pass through the bone encoder
        features = [None] * self.num_joints
        bone_transforms = torch.cat((bone_features, bone_lengths), dim=-1)

        for i, mlp in enumerate(self.net):
            parent = self.parent_mapping[i]
            if parent == -1:
                features[i] = mlp(bone_transforms[:, i, :], root_bone_prior)
            else:
                features[i] = mlp(bone_transforms[:, i, :], features[parent])

        features = torch.cat(features, dim=-1)  # B x f_len
        return features


class PoseEncoder(BaseModule):
    def __init__(self, num_joints, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_joints = num_joints

    def get_out_dim(self):
        if self.cfg is None:
            return 0

        return self.num_joints * 3

    @classmethod
    def from_cfg(cls, config):
        num_joints = 0 if config is None else config['num_joints']

        return cls(num_joints, config)

    @staticmethod
    def forward(trans, fwd_transformation):
        """

        Args:
            trans: B x 3
            fwd_transformation (optional): B x K x 4 x 4

        Returns:
        """
        B, K = fwd_transformation.shape[0], fwd_transformation.shape[1]
        trans = torch.cat([
            trans,
            torch.ones(B, 1, device=trans.device)
        ], dim=-1).unsqueeze(1).repeat(1, K, 1)
        fwd_transformation = torch.inverse(fwd_transformation).view(-1, 4, 4)

        root_proj_embedding = torch.matmul(
            fwd_transformation,
            trans.view(-1, 4, 1)
        ).view(B, K, 4)[:, :, :3].contiguous().view(B, -1)

        return root_proj_embedding


class LocalFeatureEncoder(BaseModule):
    def __init__(self, num_joints, z_dim, point_feature_len):
        super().__init__()

        self.num_joints = num_joints
        self.z_dim = z_dim
        self.point_feature_len = point_feature_len

        self.net = nn.Conv1d(
            in_channels=self.num_joints * self.z_dim,
            out_channels=self.num_joints * self.point_feature_len,
            kernel_size=(1,),
            groups=self.num_joints
        )

    def get_out_dim(self):
        return self.point_feature_len

    @classmethod
    def from_cfg(cls, config):
        return cls(
            num_joints=config['num_joints'],
            z_dim=config['z_dim'],
            point_feature_len=config['point_feature_len']
        )

    def forward(self, shape_code, structure_code, pose_code, lbs_weights):
        """
            skinning_weights: B x T x K
        """
        B, T, K = lbs_weights.shape
        assert K == self.num_joints

        # compute global feature vector
        global_feature = []
        if shape_code is not None:
            global_feature.append(shape_code)
        if structure_code is not None:
            global_feature.append(structure_code)
        if pose_code is not None:
            global_feature.append(pose_code)

        global_feature = torch.cat(global_feature, dim=-1)  # B x -1

        # compute per-point local feature vector
        global_feature = global_feature.unsqueeze(1).repeat(1, K, 1)  # B x K x -1
        global_feature = global_feature.view(B, -1, 1)  # B x -1 x 1

        local_feature = self.net(global_feature).view(B, K, -1)

        # weighted average based on the skinning net
        local_feature = local_feature.unsqueeze(1).repeat(1, T, 1, 1)  # B x T x K x F
        local_feature = local_feature.view(B * T, K, -1)  # B*T x K x F
        lbs_weights = lbs_weights.view(B*T, 1, K)  # B*T x 1 x K
        local_feature = torch.bmm(lbs_weights, local_feature).view(B, T, -1)  # B x T x F

        return local_feature


class LEAPOccupancyDecoder(BaseModule):
    def __init__(self,
                 shape_encoder: ShapeEncoder,
                 structure_encoder: StructureEncoder,
                 pose_encoder: PoseEncoder,
                 local_feature_encoder: LocalFeatureEncoder,
                 onet: ONet):
        super().__init__()

        self.shape_encoder = shape_encoder
        self.structure_encoder = structure_encoder
        self.pose_encoder = pose_encoder

        self.local_feature_encoder = local_feature_encoder
        self.onet = onet

    @classmethod
    def from_cfg(cls, config):
        shape_encoder = ShapeEncoder.from_cfg(deepcopy(config['shape_encoder']))

        structure_encoder_config = deepcopy(config['structure_encoder'])
        structure_encoder_config['parent_mapping'] = config['parent_mapping']
        structure_encoder = StructureEncoder.from_cfg(structure_encoder_config)

        pose_encoder = PoseEncoder.from_cfg(deepcopy(config['pose_encoder']))

        local_feature_encoder_config = deepcopy(config['local_feature_encoder'])
        local_feature_encoder_config['num_joints'] = config['num_joints']
        z_dim = pose_encoder.get_out_dim() + shape_encoder.get_out_dim() + structure_encoder.get_out_dim()
        local_feature_encoder_config['z_dim'] = z_dim
        local_feature_encoder = LocalFeatureEncoder.from_cfg(local_feature_encoder_config)

        onet_config = deepcopy(config['onet'])
        onet_config['num_joints'] = config['num_joints']
        onet_config['point_feature_len'] = local_feature_encoder.get_out_dim()
        onet = ONet.from_cfg(onet_config)

        return cls(shape_encoder, structure_encoder, pose_encoder, local_feature_encoder, onet)

    @classmethod
    def load_from_file(cls, file_path):
        state_dict = cls.parse_pytorch_file(file_path)
        config = state_dict['leap_occupancy_decoder_config']
        model_state_dict = state_dict['leap_occupancy_decoder_weights']
        return cls.load(config, model_state_dict)

    def forward(self,
                can_points, point_weights, cycle_distance,
                can_vert=None,  # for shape code
                rot_mats=None, rel_joints=None,  # for structure code
                root_loc=None, fwd_transformation=None):  # for pose code
        shape_code, pose_code, structure_code = None, None, None

        if self.shape_encoder.get_out_dim() > 0:
            shape_code = self.shape_encoder(can_vert)

        if self.structure_encoder.get_out_dim() > 0:
            structure_code = self.structure_encoder(rot_mats, rel_joints)

        if self.pose_encoder.get_out_dim() > 0:
            pose_code = self.pose_encoder(root_loc, fwd_transformation)

        local_feature = self.local_feature_encoder(shape_code, structure_code, pose_code, point_weights)

        occupancy = self.onet(can_points, local_feature, cycle_distance)

        return occupancy
