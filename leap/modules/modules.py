import torch
import torch.nn as nn

from .encoders import LBSNet, LEAPOccupancyDecoder, BaseModule


class INVLBS(LBSNet):
    def __init__(self, num_joints, hidden_size, pn_dim, fwd_trans_cond_dim):
        self.fwd_trans_cond_dim = fwd_trans_cond_dim
        super().__init__(num_joints, hidden_size, pn_dim)

        self.fc_fwd = nn.Sequential(
            nn.Linear(self.num_joints * 12, 100), nn.ReLU(),
            nn.Linear(100, 100), nn.ReLU(),
            nn.Linear(100, self.fwd_trans_cond_dim),
        )

    def get_c_dim(self):
        return self.pn_dim * 2 + self.fwd_trans_cond_dim

    @classmethod
    def load_from_file(cls, file_path):
        state_dict = cls.parse_pytorch_file(file_path)
        config = state_dict['inv_lbs_config']
        model_state_dict = state_dict['inv_lbs_model']
        return cls.load(config, model_state_dict)

    @classmethod
    def from_cfg(cls, config):
        model = cls(
            num_joints=config['num_joints'],
            hidden_size=config['hidden_size'],
            pn_dim=config['pn_dim'],
            fwd_trans_cond_dim=config['fwd_trans_cond_dim'])

        return model

    def forward(self, points, can_vertices, posed_vertices, fwd_transformation, compute_can_points=True):
        """
        Args:
            points: B x T x 3
            can_vertices: B x N x 3
            posed_vertices: B x N x 3
            fwd_transformation (torch.tensor): Forward transformation tensor. (B x K x 4 x 4)
            compute_can_points (bool): Whether to return estimated canonical points.

        Returns:
            if compute_can_points is True tuple of:
                skinning weights (torch.Tensor): B x T x K
                canonical points (torch.Tensor): B x T x 3
            otherwise:
                skinning weights (torch.Tensor): B x T x K
        """
        B, K = fwd_transformation.shape[:2]

        can_code = self.point_encoder(can_vertices)
        posed_code = self.point_encoder(posed_vertices)
        fwd_trans_code = self.fc_fwd(fwd_transformation[..., :3, :].reshape(B, -1))

        lbs_code = torch.cat((can_code, posed_code, fwd_trans_code), dim=-1)
        point_weights = self._forward(points, lbs_code)

        if compute_can_points:
            can_points = self.posed2can_points(points, point_weights, fwd_transformation)
            ret_interface = point_weights, can_points  # B x T x K
        else:
            ret_interface = point_weights

        return ret_interface

    @staticmethod
    def posed2can_points(points, point_weights, fwd_transformation):
        """
        Args:
            points: B x T x 3
            point_weights: B x T x K
            fwd_transformation: B x K x 4 x 4

        Returns:
            canonical points: B x T x 3
        """
        B, T, K = point_weights.shape
        point_weights = point_weights.view(B * T, 1, K)  # B*T x 1 x K

        fwd_transformation = fwd_transformation.unsqueeze(1).repeat(1, T, 1, 1, 1)  # B X K x 4 x 4 -> B x T x K x 4 x 4
        fwd_transformation = fwd_transformation.view(B * T, K, -1)  # B*T x K x 16
        back_trans = torch.bmm(point_weights, fwd_transformation).view(B * T, 4, 4)
        back_trans = torch.inverse(back_trans)

        points = torch.cat([points, torch.ones(B, T, 1, device=points.device)], dim=-1).view(B * T, 4, 1)
        can_points = torch.bmm(back_trans, points)[:, :3, 0].view(B, T, 3)

        return can_points


class FWDLBS(LBSNet):
    def __init__(self, num_joints, hidden_size, pn_dim):
        super().__init__(num_joints, hidden_size, pn_dim)

    def get_c_dim(self):
        return self.pn_dim

    @classmethod
    def load_from_file(cls, file_path):
        state_dict = cls.parse_pytorch_file(file_path)
        config = state_dict['fwd_lbs_config']
        model_state_dict = state_dict['fwd_lbs_model']
        return cls.load(config, model_state_dict)

    @classmethod
    def from_cfg(cls, config):
        model = cls(
            num_joints=config['num_joints'],
            hidden_size=config['hidden_size'],
            pn_dim=config['pn_dim'])

        return model

    def forward(self, points, can_vertices):
        """
        Args:
            points: B x T x 3
            can_vertices: B x N x 3
        Returns:

        """
        vert_code = self.point_encoder(can_vertices)  # B x pn_dim
        point_weights = self._forward(points, vert_code)
        return point_weights  # B x T x K


class LEAPModel(BaseModule):
    def __init__(self,
                 inv_lbs: INVLBS,
                 fwd_lbs: FWDLBS,
                 leap_occupancy_decoder: LEAPOccupancyDecoder):
        super(LEAPModel, self).__init__()

        # NN modules
        self.inv_lbs = inv_lbs
        self.fwd_lbs = fwd_lbs
        self.leap_occupancy_decoder = leap_occupancy_decoder

    @classmethod
    def from_cfg(cls, config):
        leap_model = cls(
            inv_lbs=INVLBS.load_from_file(config['inv_lbs_model_path']),
            fwd_lbs=FWDLBS.load_from_file(config['fwd_lbs_model_path']),
            leap_occupancy_decoder=LEAPOccupancyDecoder.from_cfg(config))

        return leap_model

    @classmethod
    def load_from_file(cls, file_path):
        state_dict = cls.parse_pytorch_file(file_path)
        config = state_dict['leap_model_config']
        model_state_dict = state_dict['leap_model_model']

        leap_model = cls(
            inv_lbs=INVLBS.from_cfg(config['inv_lbs_model_config']),
            fwd_lbs=FWDLBS.from_cfg(config['fwd_lbs_model_config']),
            leap_occupancy_decoder=LEAPOccupancyDecoder.from_cfg(config))

        leap_model.load_state_dict(model_state_dict)
        return leap_model

    def to(self, **kwargs):
        self.inv_lbs = self.inv_lbs.to(**kwargs)
        self.fwd_lbs = self.fwd_lbs.to(**kwargs)
        self.leap_occupancy_decoder = self.leap_occupancy_decoder.to(**kwargs)
        return self

    def eval(self):
        self.inv_lbs.eval()
        self.fwd_lbs.eval()
        self.leap_occupancy_decoder.eval()
