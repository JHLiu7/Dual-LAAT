import os
import yaml

from datetime import datetime
from types import SimpleNamespace

import lightning as L
import logging
import torch
import numpy as np
import pandas as pd
from collections import OrderedDict

from src.metrics_caml import all_metrics
import src.metrics_edin as metrics

ICD_TARGET_FREQURNCIES = ['frequent', 'full', 'rare']


def _load_data_df_and_codes(
    icd_type,
    data_file_path,
    icd_target_frequency,
    code_description_file_path, 
):
    id_column = '_id'
    text_column = 'text'

    if icd_target_frequency == 'frequent':
        target_column = 'target'
    elif icd_target_frequency == 'full':
        target_column = 'full_target'
    elif icd_target_frequency == 'rare':
        target_column = 'rare_target'
    else:
        raise ValueError(f"Invalid icd_target_frequency: {icd_target_frequency}")

    logging.info(f"Preparing dataset for {icd_type} ({icd_target_frequency})...")

    label2description = pd.read_pickle(code_description_file_path)[
        icd_type.split('_')[1]
    ]

    data_df = pd.read_feather(data_file_path)

    if icd_type == 'iii_icd9':
        # drop labels not in label2description
        data_df['full_target'] = data_df['full_target'].apply(lambda x: [l for l in x if l in label2description])
        data_df['target'] = data_df['target'].apply(lambda x: [l for l in x if l in label2description])
        data_df['rare_target'] = data_df['rare_target'].apply(lambda x: [l for l in x if l in label2description])
        data_df = data_df.dropna(subset=['full_target'])
        logging.info(f'ICD-9: Dropped {data_df["full_target"].isna().sum()} rows with NaN in full_target')

    
    label2id_all = {label: i for i, label in enumerate(sorted(data_df['full_target'].explode().dropna().unique()))}
    label2id = {label: i for i, label in enumerate(sorted(data_df[target_column].explode().dropna().unique()))}


    if icd_target_frequency == 'rare':
        # drop empty rows (no rare targets)
        data_df = data_df[data_df['rare_target'].apply(lambda x: len(x) > 0)]
        # labels not in label2description already dropped

    return data_df, (id_column, text_column, target_column), (label2id, label2id_all, label2description)


def _split_data(data_df, split_file_path, id_column='_id', debug=False):
    # logging.info(f"Loading split ids from {split_file_path}...")
    train_ids, val_ids, test_ids = _load_split_ids(split_file_path, debug)

    train_df = data_df[data_df[id_column].isin(train_ids)]
    val_df = data_df[data_df[id_column].isin(val_ids)]
    test_df = data_df[data_df[id_column].isin(test_ids)]

    return train_df, val_df, test_df


def _get_file_paths(data_dir, icd_type, data_w2v_dir=None, alternative_data_folder=None):
    code_description_file_path = os.path.join(data_dir, 'processed_data/code_descriptions.pkl')
    if alternative_data_folder is not None:
        data_file_path = os.path.join(data_dir, alternative_data_folder, f'mimic{icd_type}.feather')
    else:
        data_file_path = os.path.join(data_dir, f'processed_data/mimic{icd_type}.feather')

    if data_w2v_dir is None:
        data_w2v_dir = os.path.join(data_dir, 'w2v')
    token2id_file_path = os.path.join(data_w2v_dir, 'token2id.pkl')

    if icd_type == 'iii_icd9':
        split_file_path = os.path.join(data_dir, 'splits/mimiciii_clean_splits.feather')
    else:
        split_file_path = os.path.join(data_dir, f'splits/mimic{icd_type}_split.feather')
    
    return code_description_file_path, data_file_path, token2id_file_path, split_file_path


def _load_split_ids(split_file_path, debug=False):
    split_df = pd.read_feather(split_file_path)
    train_ids = split_df[split_df['split'] == 'train']['_id'].tolist()
    val_ids = split_df[split_df['split'] == 'val']['_id'].tolist()
    test_ids = split_df[split_df['split'] == 'test']['_id'].tolist()
    if debug:
        N = 500
        train_ids = train_ids[:N]
        val_ids = val_ids[:N]
        test_ids = test_ids[:N]
    return train_ids, val_ids, test_ids

def _get_codes_for_sampling(scope_for_negative_samples, label2description, label2id, label2id_all):
    if scope_for_negative_samples == 'full':
        codes_for_sampling = list(label2id_all.keys())
    elif scope_for_negative_samples == 'frequent':
        codes_for_sampling = list(label2id.keys())
    elif scope_for_negative_samples == 'terminology':
        codes_for_sampling = list(label2description.keys())
    else:
        raise ValueError(f"Invalid scope for negative samples: {scope_for_negative_samples}")
    
    code_descriptions_for_sampling = [label2description[code] for code in codes_for_sampling]
    return codes_for_sampling, code_descriptions_for_sampling




yaml_string = """
- name: F1Score
  configs: 
    average: "micro"
    threshold: 0.5
- name: F1Score
  configs: 
    average: "macro"
    threshold: 0.5
- name: Precision_K
  configs: 
    k: 5
- name: Precision_K
  configs: 
    k: 8
- name: Precision_K
  configs: 
    k: 15
- name: MeanAveragePrecision
  configs: {}
- name: PrecisionAtRecall
  configs: {}
"""

def get_metric_collection(num_classes):
    config = yaml.safe_load(yaml_string.strip())

    metric_list = []
    for metric in config:
        metric_class = getattr(metrics, metric['name'])
        metric_func = metric_class(number_of_classes=num_classes, **metric['configs'])
        metric_list.append(metric_func)
    return metrics.MetricCollection(metric_list)


class Evaluator:
    def __init__(self, mode, main_score='f1_micro'):
        """
        Args:
            mode (str): 'basic' for basic metrics based on CAML paper, 'edin' for EDIN version of metrics, 'basic_and_edin' for both
            main_score (str): The main score to monitor, default is 'f1_micro'
        """
        self.main_score = main_score
        self.mode = mode

        self.apply_sigmoid = True  # assume logits need sigmoid by default

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor):
        if self.mode == 'edin':
            scores = self.evaluate_edin(logits, targets)
        elif self.mode == 'basic':
            scores = self.evaluate_basic(logits, targets)
        elif self.mode == 'basic_and_edin':
            scores = self.evaluate_basic_and_edin(logits, targets)
        elif self.mode == 'final':
            scores = self.evaluate_final(logits, targets)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        score_line = "\t".join([
            f"{k}:{v:.3f}" for k, v in scores.items()
        ])
        return scores, score_line
    
    def prepare_inputs(self, logits: torch.Tensor, targets: torch.Tensor, apply_sigmoid=None):
        apply_sigmoid = self.apply_sigmoid if apply_sigmoid is None else apply_sigmoid
        assert logits.shape[0] == targets.shape[0], f"Logits batch size {logits.shape[0]} and targets batch size {targets.shape[0]} do not match."
        if apply_sigmoid:
            logits = logits.sigmoid()
        logits = logits.cpu()
        targets = targets.cpu()
        return logits, targets

    def evaluate_basic(self, logits: torch.Tensor, targets: torch.Tensor):
        logits, targets = self.prepare_inputs(logits, targets)
        logits = logits.numpy()
        targets = targets.numpy()

        yhat_raw = np.nan_to_num(logits)
        yhat = (yhat_raw > 0.5).astype(int)
        scores = all_metrics(yhat=yhat, y=targets, yhat_raw=yhat_raw, k=[8, 15])

        scores['prec_at_08'] = scores.pop('prec_at_8')
        scores = {m: scores[m] for m in ['auc_micro', 'auc_macro','f1_micro', 'f1_macro', 'prec_at_08', 'prec_at_15']}
        return scores 
    
    def evaluate_edin(self, logits: torch.Tensor, targets: torch.Tensor, threshold=None):
        logits, targets = self.prepare_inputs(logits, targets)

        scorer = get_metric_collection(num_classes=logits.shape[1])
        scorer.reset()
        if threshold is not None:
            scorer.set_threshold(threshold)
            logging.info(f"Using threshold: {threshold}")
        scorer.update({'logits': logits, 'targets': targets})
        scores = scorer.compute()
        scores = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in scores.items()}
        return scores
    
    def evaluate_basic_and_edin(self, logits: torch.Tensor, targets: torch.Tensor):
        scores = self.evaluate_basic(logits, targets)
        scores_edin = self.evaluate_edin(logits, targets)

        scores = OrderedDict(scores)  # Ensure order is maintained

        # merge edin scores with existing scores
        for key, value in scores_edin.items():
            scores[f'edin_{key}'] = value
        return scores
    
    def evaluate_final(self, logits: torch.Tensor, targets: torch.Tensor):
        scores_auc = self.evaluate_basic(logits, targets)
        scores_edin = self.evaluate_edin(logits, targets)

        scores = OrderedDict(scores_edin)  

        # only add AUC scores from basic
        for key in ['auc_micro', 'auc_macro']:
            scores[key] = scores_auc[key]
        return scores

        
    

def f1_score_db_tuning(logits, targets, average="micro", type="single"):
    if average not in ["micro", "macro"]:
        raise ValueError("Average must be either 'micro' or 'macro'")
    dbs = torch.linspace(0, 1, 100)
    tp = torch.zeros((len(dbs), targets.shape[1]))
    fp = torch.zeros((len(dbs), targets.shape[1]))
    fn = torch.zeros((len(dbs), targets.shape[1]))
    for idx, db in enumerate(dbs):
        predictions = (logits > db).long()
        tp[idx] = torch.sum((predictions) * (targets), dim=0)
        fp[idx] = torch.sum(predictions * (1 - targets), dim=0)
        fn[idx] = torch.sum((1 - predictions) * targets, dim=0)
    if average == "micro":
        f1_scores = tp.sum(1) / (tp.sum(1) + 0.5 * (fp.sum(1) + fn.sum(1)) + 1e-10)
    else:
        f1_scores = torch.mean(tp / (tp + 0.5 * (fp + fn) + 1e-10), dim=1)
    if type == "single":
        best_f1 = f1_scores.max()
        best_db = dbs[f1_scores.argmax()]
        print(f"Best F1: {best_f1:.4f} at DB: {best_db:.4f}")
        return best_f1, best_db
    if type == "per_class":
        best_f1 = f1_scores.max(1)
        best_db = dbs[f1_scores.argmax(0)]
        print(f"Best F1: {best_f1} at DB: {best_db}")
        return best_f1, best_db
    

def as_namespace(obj):
    """Recursively convert dicts (and lists of dicts) into a SimpleNamespace."""
    if isinstance(obj, dict):
        ns = SimpleNamespace()
        for k, v in obj.items():
            setattr(ns, k, as_namespace(v))
        return ns
    elif isinstance(obj, list):
        return [as_namespace(item) for item in obj]
    else:
        return obj

def update_config(config_path="configs/config.yaml", overrides=None):

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if overrides:
        for override in overrides:
            keys = override.split('=')
            if len(keys) != 2:
                print(f"Invalid override format: {override}. Expected key=value.")
                continue
            key, value = keys
            keys = key.split('.')
            d = config
            for k in keys[:-1]:
                if k not in d:
                    d[k] = {}
                d = d[k]
            # Attempt to cast value to appropriate type
            try:
                # Handle boolean values
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                else:
                    value = eval(value)
            except:
                pass  # Keep as string if casting fails
            d[keys[-1]] = value

    return as_namespace(config)


def get_trainer(cfg):

    output_dir = os.path.join(cfg.output_dir, cfg.model_type, cfg.icd_type, datetime.now().strftime("%b%d-%H:%M:%S-%f"))
    if cfg.debug:
        output_dir = os.path.join(cfg.output_dir, cfg.model_type, 'debug', datetime.now().strftime("%b%d-%H:%M:%S-%f"))
    os.makedirs(output_dir, exist_ok=True)


    logger = L.pytorch.loggers.TensorBoardLogger(
        save_dir=output_dir,
        default_hp_metric=False,
        name='logs',
        version=''
    )

    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=output_dir,
        filename='ckpt/{epoch}-{train_loss:.5f}',
        monitor='train_loss',
        mode='min',
        save_top_k=1,
        save_last=True
    )

    if hasattr(cfg, 'patience'):
        logging.info(f"Using early stopping with patience: {cfg.patience}")
        icd_type = 'iv_icd10'  
        target_freq = 'frequent'

        score = f'hp/{icd_type}_{target_freq}'
        
        early_stopping_callback = L.pytorch.callbacks.EarlyStopping(
            monitor=score,
            patience=cfg.patience,
            mode='max'
        )
        checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
            dirpath=output_dir,
            filename='ckpt/{epoch}-{score:.5f}',
            monitor=score,
            mode='max',
            save_top_k=cfg.patience,
            save_weights_only=True,
            save_last=True
        )
        callbacks = [checkpoint_callback, early_stopping_callback]
    else:
        checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
            dirpath=output_dir,
            filename='ckpt/{epoch}-{train_loss:.5f}',
            monitor='train_loss',
            mode='min',
            save_top_k=1,
            save_last=True
        )
        callbacks = [checkpoint_callback]


    trainer = L.Trainer(
        logger=logger,
        # callbacks=[checkpoint_callback],
        callbacks=callbacks,
        max_epochs=cfg.max_epochs if not cfg.debug else 2,
        # max_epochs=max_epochs,
        max_steps=cfg.max_steps,
        accelerator="gpu",
        devices=cfg.gpus,
        precision=cfg.precision,
        accumulate_grad_batches=cfg.accumulate_grad_batches,

        # gradient_clip_val=cfg.gradient_clip_val,
        # log_every_n_steps=cfg.log_every_n_steps,

        enable_progress_bar=not cfg.silent,

        # progress_bar_refresh_rate=cfg.progress_bar_refresh_rate,
        # num_sanity_val_steps=cfg.num_sanity_val_steps,
        # resume_from_checkpoint=cfg.resume_from_checkpoint
    )

    return trainer


