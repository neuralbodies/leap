import os
import argparse

import pyrender
import torch
import trimesh
import numpy as np

from leap import LEAPBodyModel


def sample_points(mesh_vert, n_points=15000):
    vert = mesh_vert.detach().cpu().numpy().reshape((-1, 3))
    bb_min, bb_max = np.min(vert, axis=0), np.max(vert, axis=0)
    loc = np.array([(bb_min[0] + bb_max[0]) / 2, (bb_min[1] + bb_max[1]) / 2, (bb_min[2] + bb_max[2]) / 2])
    scale = (bb_max - bb_min).max()

    points_uniform = np.random.rand(n_points, 3)
    points_uniform = 1.1 * (points_uniform - 0.5)
    points_uniform *= scale
    np_points = points_uniform + np.expand_dims(loc, axis=0)

    tensor_points = torch.from_numpy(np_points).to(device=mesh_vert.device, dtype=mesh_vert.dtype)
    tensor_points = tensor_points.unsqueeze(0)
    return np_points, tensor_points


def vis_create_pc(pts, color=(0.0, 1.0, 0.0), radius=0.005):
    if torch.is_tensor(pts):
        pts = pts.cpu().numpy()

    tfs = np.tile(np.eye(4), (pts.shape[0], 1, 1))
    tfs[:, :3, 3] = pts
    sm_in = trimesh.creation.uv_sphere(radius=radius)
    sm_in.visual.vertex_colors = color

    return pyrender.Mesh.from_trimesh(sm_in, poses=tfs)


def main(leap_path, smpl_param_file, bm_path, device):
    # load SMPL parameters
    smpl_body = torch.load(smpl_param_file, map_location=torch.device('cpu'))
    smpl_body = {key: val.to(device=device) if torch.is_tensor(val) else val for key, val in smpl_body.items()}

    # load LEAP
    leap_model = LEAPBodyModel(leap_path,
                               bm_path=os.path.join(bm_path, smpl_body['gender'], 'model.npz'),
                               num_betas=smpl_body['betas'].shape[1],
                               batch_size=smpl_body['betas'].shape[0],
                               device=device)
    leap_model.set_parameters(betas=smpl_body['betas'],
                              pose_body=smpl_body['pose_body'],
                              pose_hand=smpl_body['pose_hand'])
    leap_model.forward_parametric_model()

    # uniform points
    np_query_points, tensor_query_points = sample_points(leap_model.posed_vert)
    occupancy = leap_model(tensor_query_points) < 0.5
    inside_points = (occupancy < 0.5).squeeze().detach().cpu().numpy()  # 0.5 is threshold

    posed_mesh = leap_model.extract_posed_mesh()
    # can_mesh = leap_model.extract_canonical_mesh()  # mesh in canonical pose

    # visualize
    scene = pyrender.Scene(ambient_light=[.1, 0.1, 0.1], bg_color=[1.0, 1.0, 1.0])
    scene.add(pyrender.Mesh.from_trimesh(posed_mesh))
    scene.add(vis_create_pc(np_query_points[inside_points], color=(1., 0., 0.)))  # red - inside points
    scene.add(vis_create_pc(np_query_points[~inside_points], color=(0., 1., 0.)))  # blue - outside points
    pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize LEAP mesh and query points.')
    parser.add_argument('--leap_path', type=str,
                        help='Path to a pretrained LEAP model.')

    parser.add_argument('--bm_path', type=str,
                        help='Path to the SMPL+H body model.')

    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda',
                        help='Device (cuda or cpu).')

    args = parser.parse_args()
    main(args.leap_path,
         './sample_smph_body.pt',
         args.bm_path,
         args.device)
