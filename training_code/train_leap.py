import os
import argparse
import logging

import numpy as np
import torch
import torch.optim as optim
import tensorboardX

import config
import checkpoints
import utils


def main(cfg, num_workers):
    # Shortened
    out_dir = cfg['training']['out_dir']
    batch_size = cfg['training']['batch_size']
    backup_every = cfg['training']['backup_every']
    utils.save_config(os.path.join(out_dir, 'config.yml'), cfg)

    model_selection_metric = cfg['training']['model_selection_metric']
    model_selection_sign = 1 if cfg['training']['model_selection_mode'] == 'maximize' else -1

    # Output directory
    utils.cond_mkdir(out_dir)

    # Dataset
    train_dataset = config.get_dataset('train', cfg)
    val_dataset = config.get_dataset('val', cfg)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False)

    # Model
    model = config.get_model(cfg)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    trainer = config.get_trainer(model, optimizer, cfg)

    # Print model
    print(model)
    logger = logging.getLogger(__name__)
    logger.info(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}')

    # load pretrained model
    tb_logger = tensorboardX.SummaryWriter(os.path.join(out_dir, 'logs'))
    ckp = checkpoints.CheckpointIO(out_dir, model, optimizer, cfg)
    try:
        load_dict = ckp.load('model_best.pt')
        logger.info('Model loaded')
    except FileExistsError:
        logger.info('Model NOT loaded')
        load_dict = dict()

    epoch_it = load_dict.get('epoch_it', -1)
    it = load_dict.get('it', -1)
    metric_val_best = load_dict.get('loss_val_best', -model_selection_sign * np.inf)

    logger.info(f'Current best validation metric ({model_selection_metric}): {metric_val_best:.6f}')

    # Shortened
    print_every = cfg['training']['print_every']
    validate_every = cfg['training']['validate_every']
    max_iterations = cfg['training']['max_iterations']
    max_epochs = cfg['training']['max_epochs']

    while True:
        epoch_it += 1

        for batch in train_loader:
            it += 1
            loss_dict = trainer.train_step(batch)
            loss = loss_dict['total_loss']
            for k, v in loss_dict.items():
                tb_logger.add_scalar(f'train/{k}', v, it)

            # Print output
            if print_every > 0 and (it % print_every) == 0:
                logger.info(f'[Epoch {epoch_it:02d}] it={it:03d}, loss={loss:.8f}')

            # Backup if necessary
            if backup_every > 0 and (it % backup_every) == 0:
                logger.info('Backup checkpoint')
                ckp.save(f'model_{it:d}.pt', epoch_it=epoch_it, it=it, loss_val_best=metric_val_best)

            # Run validation
            if validate_every > 0 and (it % validate_every) == 0:
                eval_dict = trainer.evaluate(val_loader)
                print('eval_dict=\n', eval_dict)
                metric_val = eval_dict[model_selection_metric]
                logger.info(f'Validation metric ({model_selection_metric}): {metric_val:.8f}')

                for k, v in eval_dict.items():
                    tb_logger.add_scalar(f'val/{k}', v, it)

                if model_selection_sign * (metric_val - metric_val_best) > 0:
                    metric_val_best = metric_val
                    logger.info(f'New best model (loss {metric_val_best:.8f}')
                    ckp.save('model_best.pt', epoch_it=epoch_it, it=it, loss_val_best=metric_val_best)

            if (0 < max_iterations <= it) or (0 < max_epochs <= epoch_it):
                logger.info(f'Maximum iteration/epochs ({epoch_it}/{it}) reached. Exiting.')
                ckp.save(f'model_{it:d}.pt', epoch_it=epoch_it, it=it, loss_val_best=metric_val_best)
                exit(3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config',
        type=str,
        help='Path to a config file.')
    parser.add_argument(
        '--num_workers',
        type=int,
        default=6,
        help='Number of workers for datasets loaders.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(cfg=config.load_config(args.config),
         num_workers=args.num_workers)
