from collections import defaultdict

import numpy as np
import torch
import tqdm

from leap.modules import LEAPModel, INVLBS, FWDLBS


class BaseTrainer:
    """ Base trainers class.

    Args:
        model (torch.nn.Module): Occupancy Network model
        optimizer (torch.optim.Optimizer): pytorch optimizer object
        cfg (dict): configuration
    """

    def __init__(self, model, optimizer, cfg):
        self.model = model
        self.optimizer = optimizer
        self.device = cfg['device']

    def evaluate(self, val_loader):
        """ Performs an evaluation.
        Args:
            val_loader (torch.DataLoader): pytorch dataloader
        """
        eval_list = defaultdict(list)

        for data in tqdm.tqdm(val_loader):
            eval_step_dict = self.eval_step(data)

            for k, v in eval_step_dict.items():
                eval_list[k].append(v)

        eval_dict = {k: np.mean(v) for k, v in eval_list.items()}
        return eval_dict

    def _train_mode(self):
        self.model.train()

    def train_step(self, data):
        self._train_mode()
        self.optimizer.zero_grad()
        loss_dict = self.compute_loss(data)
        loss_dict['total_loss'].backward()
        self.optimizer.step()
        return {k: v.item() for k, v in loss_dict.items()}

    @torch.no_grad()
    def eval_step(self, data):
        """ Performs an evaluation step.

        Args:
            data (dict): datasets dictionary
        """
        self.model.eval()
        eval_loss_dict = self.compute_eval_loss(data)
        return {k: v.item() for k, v in eval_loss_dict.items()}

    def compute_loss(self, *kwargs):
        """ Computes the training loss.

        Args:
            kwargs (dict): datasets dictionary
        """
        raise NotImplementedError

    @torch.no_grad()
    def compute_eval_loss(self, data):
        """ Computes the validation loss.

        Args:
            data (dict): datasets dictionary
        """
        return self.compute_loss(data)


class FWDLBSTrainer(BaseTrainer):

    def __init__(self, model: FWDLBS, optimizer: torch.optim.Optimizer, cfg: dict):
        super().__init__(model, optimizer, cfg)

    def compute_loss(self, data):
        sk_weights = self.model(
            data['can_points'].to(device=self.device),
            data['can_vertices'].to(device=self.device))
        gt_sk_weights = data['can_points_sk_weights'].to(device=self.device)

        loss_dict = {
            'sk_loss': (sk_weights - gt_sk_weights).abs().sum(-1).mean(-1).mean()
        }
        loss_dict['total_loss'] = loss_dict['sk_loss']
        return loss_dict


class INVLBSTrainer(BaseTrainer):
    def __init__(self, model: INVLBS, optimizer: torch.optim.Optimizer, cfg: dict):
        super().__init__(model, optimizer, cfg)

    def compute_loss(self, data):
        sk_weights = self.model(
            points=data['points'].to(device=self.device),
            can_vertices=data['can_vertices'].to(device=self.device),
            posed_vertices=data['posed_vertices'].to(device=self.device),
            fwd_transformation=data['fwd_transformation'].to(device=self.device),
            compute_can_points=False)
        gt_sk_weights = data['points_sk_weights'].to(device=self.device)

        loss_dict = {
            'sk_loss': (sk_weights - gt_sk_weights).abs().sum(-1).mean(-1).mean()
        }
        loss_dict['total_loss'] = loss_dict['sk_loss']
        return loss_dict


class LEAPModelTrainer(BaseTrainer):
    def __init__(self, model: LEAPModel, optimizer: torch.optim.Optimizer, cfg: dict):
        super().__init__(model, optimizer, cfg)

        self._eval_lbs_mode()

    def _eval_lbs_mode(self):
        self.model.inv_lbs.require_grad = False
        self.model.inv_lbs.eval()

        self.model.fwd_lbs.require_grad = False
        self.model.fwd_lbs.eval()

    def _train_mode(self):
        self.model.train()
        self._eval_lbs_mode()

    @torch.no_grad()
    def compute_eval_loss(self, data):
        # Only evaluate uniformly sampled points
        n_points = data['points'].shape[1] // 2

        gt_occupancy = data['occ'][:, :n_points].to(device=self.device)
        can_vert = data['can_vertices'].to(device=self.device)
        point_weights, can_points = self.model.inv_lbs(
            points=data['points'][:, :n_points, :].to(device=self.device),
            can_vertices=can_vert,
            posed_vertices=data['posed_vertices'].to(device=self.device),
            fwd_transformation=data['fwd_transformation'].to(device=self.device))

        fwd_point_weights = self.model.fwd_lbs(can_points, can_vert)
        cycle_distance = torch.sum((point_weights - fwd_point_weights).abs(), dim=-1, keepdim=True)

        occupancy = torch.sigmoid(self.model.leap_occupancy_decoder(
            can_points=can_points, point_weights=point_weights, cycle_distance=cycle_distance, can_vert=can_vert,
            rot_mats=data['pose'].to(device=self.device), rel_joints=data['rel_joints'].to(device=self.device)))

        return {
            'iou': self.compute_iou(occupancy >= 0.5, gt_occupancy >= 0.5).mean()
        }

    def compute_loss(self, data):
        gt_occupancy = data['occ'].to(device=self.device)
        can_vert = data['can_vertices'].to(device=self.device)
        with torch.no_grad():
            point_weights, can_points = self.model.inv_lbs(
                data['points'].to(device=self.device),
                can_vert,
                data['posed_vertices'].to(device=self.device),
                data['fwd_transformation'].to(device=self.device))
            fwd_point_weights = self.model.fwd_lbs(can_points, can_vert)
            cycle_distance = torch.sum((point_weights - fwd_point_weights).abs(), dim=-1, keepdim=True)

            # handle points directly sampled in the canonical space
            if 'can_points' in data:
                can_points = torch.cat((
                    can_points,
                    data['can_points'].to(device=self.device)
                ), dim=1)

                point_weights = torch.cat((
                    point_weights,
                    data['can_points_sk_weights'].to(device=self.device)
                ), dim=1)

                cycle_distance = torch.cat((
                    cycle_distance,
                    torch.zeros((*data['can_points'].shape[:-1], cycle_distance.shape[-1]),
                                device=self.device, dtype=cycle_distance.dtype)
                ), dim=1)

                gt_occupancy = torch.cat((
                    gt_occupancy,
                    data['can_occ'].to(device=self.device)
                ), dim=1)

        occupancy = torch.sigmoid(self.model.leap_occupancy_decoder(
            can_points=can_points, point_weights=point_weights, cycle_distance=cycle_distance, can_vert=can_vert,
            rot_mats=data['pose'].to(device=self.device), rel_joints=data['rel_joints'].to(device=self.device)))

        loss_dict = {
            'occ_loss': ((occupancy - gt_occupancy) ** 2).sum(-1).mean(),
        }
        loss_dict['total_loss'] = loss_dict['occ_loss']
        return loss_dict

    @staticmethod
    def compute_iou(occ1, occ2):
        """ Computes the Intersection over Union (IoU) value for two sets of
        occupancy values.

        Args:
            occ1 (tensor): first set of occupancy values
            occ2 (tensor): second set of occupancy values
        """
        # Also works for 1-dimensional data
        if len(occ1.shape) >= 2:
            occ1 = occ1.reshape(occ1.shape[0], -1)
        if len(occ2.shape) >= 2:
            occ2 = occ2.reshape(occ2.shape[0], -1)

        # Convert to boolean values
        occ1 = (occ1 >= 0.5)
        occ2 = (occ2 >= 0.5)

        # Compute IOU
        area_union = (occ1 | occ2).float().sum(axis=-1)
        area_intersect = (occ1 & occ2).float().sum(axis=-1)

        iou = (area_intersect / area_union)

        return iou
