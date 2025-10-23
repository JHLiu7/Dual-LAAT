import argparse
import logging
import warnings
from datetime import datetime
from collections import defaultdict
import os, torch

from lightning.pytorch import seed_everything

from src.data_modules import prepare_dataloaders
from src.utils import update_config, get_trainer, ICD_TARGET_FREQURNCIES
from src.model import LitModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

warnings.filterwarnings("ignore", category=RuntimeWarning)


# get slurm job id from environment variable
def get_slurm_job_id():
    return os.environ.get('SLURM_JOB_ID', 'no_id')

def parse_args():
    parser = argparse.ArgumentParser(description="Configuration parser")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the config file')
    parser.add_argument('--overrides', nargs='+', help='Overrides for the config file in the format key=value')

    # add checkpoint argument
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to the model checkpoint')

    args = parser.parse_args()
    return update_config(args.config, args.overrides), args

def log_time_taken(start_time, end_time):
    time_taken = end_time - start_time
    days = time_taken.days
    hours, remainder = divmod(time_taken.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if days > 0:
        logging.info(f"Time taken: {days} days, {hours} hours, {minutes} minutes")
    else:
        logging.info(f"Time taken: {hours} hours, {minutes} minutes")


def main():
    cfg, args = parse_args()
    icd_type = cfg.icd_type


    if cfg.seed is not None:
        logging.info(f"Setting random seed to {cfg.seed}")
        seed_everything(cfg.seed)

    start_time = datetime.now()

    # Get trainer
    trainer = get_trainer(cfg)

    # Get data loaders
    train_loader, val_loaders, test_loaders, num_train_steps = prepare_dataloaders(cfg)

    # Load model
    cfg.num_train_steps = num_train_steps

    if args.checkpoint:
        logging.info(f"Loading model from checkpoint: {args.checkpoint}")
        model = LitModel.load_from_checkpoint(args.checkpoint, cfg=cfg)
    else:
        model = LitModel(cfg)

    # Train
    trainer.fit(model, train_loader, val_loaders)

    # Test
    icd_types = icd_type.split('+') if '+' in icd_type else [icd_type]
    for icd_type in icd_types:
        for icd_target_frequency in ICD_TARGET_FREQURNCIES:
            test_loader = test_loaders[f'{icd_type}_{icd_target_frequency}']
            logging.info(f"Testing model on {icd_type} ({icd_target_frequency})")
            trainer.test(model, test_loader)

    # Collect outputs
    def _parse_step_outputs(step_outputs):
        logits = torch.cat([x['logits'] for x in step_outputs])
        labels = torch.cat([x['y'] for x in step_outputs])
        idx = [i for x in step_outputs for i in x['idx']] if 'idx' in step_outputs[0] else None
        icd_type = [x['icd_type'] for x in step_outputs]
        assert len(set(icd_type)) == 1, f"All outputs should have the same ICD type {str(icd_type)}"
        icd_type = icd_type[0]
        logits = logits.cpu().detach()
        labels = labels.cpu().detach()
        return logits, labels, idx, icd_type
    
    val_outputs, test_outputs = defaultdict(dict), defaultdict(dict)
    for icd_type in icd_types:
        for icd_target_frequency in ICD_TARGET_FREQURNCIES:
            val_loader = val_loaders[f'{icd_type}_{icd_target_frequency}']
            test_loader = test_loaders[f'{icd_type}_{icd_target_frequency}']
            val_outputs[icd_type][icd_target_frequency] = _parse_step_outputs(trainer.predict(model, val_loader))
            test_outputs[icd_type][icd_target_frequency] = _parse_step_outputs(trainer.predict(model, test_loader))

    # Save outputs and model
    slurm_job_id = get_slurm_job_id()
    output_dir = f"{cfg.output_dir}/{cfg.model_type}/slurm-{slurm_job_id}"
    os.makedirs(output_dir, exist_ok=True)
    torch.save(val_outputs, os.path.join(output_dir, "val_outputs.pt"))
    torch.save(test_outputs, os.path.join(output_dir, "test_outputs.pt"))

    trainer.save_checkpoint(f"{output_dir}/final_model.pt")

    logging.info(f"Outputs saved to {output_dir}")
    logging.info(f"Model checkpoint saved to {output_dir}/final_model.pt")

    # Print time taken, in hours and minutes
    end_time = datetime.now()
    log_time_taken(start_time, end_time)


if __name__ == '__main__':
    main()
