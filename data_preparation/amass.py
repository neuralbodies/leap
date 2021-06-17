import argparse
from glob import glob
from os.path import basename, join, splitext
from pathlib import Path

import numpy as np
import torch

from leap.leap_body_model import LEAPBodyModel


@torch.no_grad()
def main(args):
    for subset in args.subsets.split(','):
        subset_dir = join(args.src_dataset_path, subset)
        subjects = [basename(s_dir) for s_dir in sorted(glob(join(subset_dir, '*')))]

        for subject in subjects:
            subject_dir = join(subset_dir, subject)
            shape_data = np.load(join(subject_dir, 'shape.npz'))
            gender = shape_data['gender'].item()
            bm_path = join(args.bm_dir_path, gender, 'model.npz')

            sequences = [basename(sn) for sn in glob(join(subject_dir, '*.npz')) if not sn.endswith('shape.npz')]
            for sequence in sequences:
                sequence_path = join(subject_dir, sequence)
                sequence_name = splitext(sequence)[0]
                data = np.load(sequence_path, allow_pickle=True)

                b_size = data['poses'].shape[0]
                leap_body_model = LEAPBodyModel(bm_path=bm_path, num_betas=data['betas'].shape[0], batch_size=b_size)

                leap_body_model.set_parameters(
                    betas=torch.Tensor(data['betas']).unsqueeze(0).repeat(b_size, 1),
                    pose_body=torch.Tensor(data['poses'][:, 3:66]),  # 21 joints
                    pose_hand=torch.Tensor(data['poses'][:, 66:]))
                leap_body_model.forward_parametric_model()

                to_save = {
                    'can_vertices': leap_body_model.can_vert,
                    'posed_vertices': leap_body_model.posed_vert,
                    'pose_mat': leap_body_model.pose_rot_mat,
                    'rel_joints': leap_body_model.rel_joints,
                    'fwd_transformation': leap_body_model.fwd_transformation,
                    'frame_name': [f'{sequence_name}_{f_idx:06d}' for f_idx in range(b_size)],
                    'gender': [gender] * b_size
                }
                dir_path = join(args.dst_dataset_path, subset, subject, sequence_name)
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                print(f'Saving:\t{dir_path}')
                for b_ind in range(b_size):
                    with open(join(dir_path, f'{b_ind:06d}.npz'), 'wb') as file:
                        np.savez(file, **{key: to_np(val[b_ind]) for key, val in to_save.items()})


def to_np(variable):
    if torch.is_tensor(variable):
        variable = variable.detach().cpu().numpy()

    return variable


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Preprocess AMASS dataset.')
    parser.add_argument('--src_dataset_path', type=str, required=True,
                        help='Path to AMASS dataset.')
    parser.add_argument('--dst_dataset_path', type=str, required=True,
                        help='Directory path to store preprocessed dataset.')
    parser.add_argument('--subsets', type=str, metavar='LIST', required=True,
                        help='Subsets of AMASS to use, separated by comma.')
    parser.add_argument('--bm_dir_path', type=str, required=True,
                        help='Path to body model')

    main(parser.parse_args())
