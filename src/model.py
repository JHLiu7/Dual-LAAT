import os
import torch
import logging
import numpy as np
import lightning as L

from collections import defaultdict
from typing import Dict, Optional, List
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import get_scheduler

from src.model_layers import DualRNN, DualCNN
from src.utils import ICD_TARGET_FREQURNCIES, Evaluator


class LitModel(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        if cfg.model_type == 'caml':
            data_w2v_dir = cfg.data_w2v_dir if hasattr(cfg, 'data_w2v_dir') and cfg.data_w2v_dir is not None else os.path.join(cfg.data_dir, 'w2v')
            word_embeddings = np.load(os.path.join(data_w2v_dir, 'vectors.npy'))
            self.model = DualCNN(
                word_embeddings=word_embeddings,
                kernel_size=cfg.kernel_size,
                num_filters=cfg.num_filters,
                dropout=cfg.dropout,
                num_mha_heads=cfg.num_mha_heads,
                projection_dim=cfg.projection_dim,
                dual_attention=cfg.dual_attention,
                dual_attention_lambda=cfg.dual_attention_lambda,
            )
            self.save_hyperparameters({
                'model_type': cfg.model_type,
                'dual_attention': cfg.dual_attention,
                'dual_attention_lambda': cfg.dual_attention_lambda,
                'kernel_size': cfg.kernel_size,
                'num_filters': cfg.num_filters,
                'dropout': cfg.dropout,
                'num_mha_heads': cfg.num_mha_heads,
                'projection_dim': cfg.projection_dim,
                'lr': cfg.lr,
                'weight_decay': cfg.weight_decay,
                'num_warmup_steps': cfg.num_warmup_steps,
                'num_train_steps': cfg.num_train_steps,
                'scheduler_type': cfg.scheduler_type,
                'batch_size': cfg.batch_size * cfg.accumulate_grad_batches,
                'icd_type': cfg.icd_type,
                'icd_target_frequency': cfg.icd_target_frequency,
                'negative_sampling_strategy': cfg.negative_sampling_strategy,
                'label_space': cfg.label_space if cfg.label_space is not None else 0,
                'scope_for_negative_samples': cfg.scope_for_negative_samples,
                'max_input_length': cfg.max_input_length,
            })
        elif cfg.model_type == 'laat':
            data_w2v_dir = cfg.data_w2v_dir if hasattr(cfg, 'data_w2v_dir') and cfg.data_w2v_dir is not None else os.path.join(cfg.data_dir, 'w2v')
            word_embeddings = np.load(os.path.join(data_w2v_dir, 'vectors.npy'))
            self.model = DualRNN(
                word_embeddings=word_embeddings,
                rnn_hidden_size=cfg.rnn_hidden_size,
                rnn_num_layers=cfg.rnn_num_layers,
                dropout=cfg.dropout,
                num_mha_heads=cfg.num_mha_heads,
                projection_dim=cfg.projection_dim,
                rnn_type=cfg.rnn_type
            )
            self.save_hyperparameters({
                'model_type': cfg.model_type,
                'rnn_hidden_size': cfg.rnn_hidden_size,
                'rnn_num_layers': cfg.rnn_num_layers,
                'rnn_type': cfg.rnn_type,
                'dropout': cfg.dropout,
                'num_mha_heads': cfg.num_mha_heads,
                'projection_dim': cfg.projection_dim,
                'lr': cfg.lr,
                'weight_decay': cfg.weight_decay,
                'num_warmup_steps': cfg.num_warmup_steps,
                'num_train_steps': cfg.num_train_steps,
                'scheduler_type': cfg.scheduler_type,
                'batch_size': cfg.batch_size * cfg.accumulate_grad_batches,
                'icd_type': cfg.icd_type,
                'icd_target_frequency': cfg.icd_target_frequency,
                'negative_sampling_strategy': cfg.negative_sampling_strategy,
                'label_space': cfg.label_space if cfg.label_space is not None else 0,
                'scope_for_negative_samples': cfg.scope_for_negative_samples,
                'max_input_length': cfg.max_input_length,
            })
        else:
            raise NotImplementedError(f"Model type {cfg.model_type} not implemented.")

        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.evaluator = Evaluator('basic_and_edin')
        self.main_metric = 'f1_micro'
        self.train_icd_target_frequency = cfg.icd_target_frequency

        self.validation_step_output_dict = defaultdict(list)
        self.test_step_output = []


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay
        )
        if self.cfg.num_warmup_steps is not None:
            num_warmup_steps = self.cfg.num_warmup_steps
            num_training_steps = self.cfg.num_train_steps
            scheduler = get_scheduler(
                self.cfg.scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        else:
            return optimizer


    def forward(self, x_text, x_code):
        logits = self.model(x_text, x_code)
        return logits
    
    def step(self, batch, batch_idx):
        x_text, x_code, y, _, icd_type = batch
        logits = self(x_text, x_code)
        loss = self.loss_fn(logits, y.float())
        return loss, logits, y, icd_type
    
    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self.step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log('my_loss', loss, on_step=True, prog_bar=False, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        loss, logits, y, icd_type = self.step(batch, batch_idx)
        self.log('val_loss', loss)
        if dataloader_idx is None:
            dataloader_idx = 0
        self.validation_step_output_dict[dataloader_idx].append({
            'logits': logits,
            'y': y,
            'icd_type': icd_type
        })
        return loss
    
    def test_step(self, batch, batch_idx, dataloader_idx=None):
        loss, logits, y, icd_type = self.step(batch, batch_idx)
        self.test_step_output.append({
            'logits': logits,
            'y': y,
            'icd_type': icd_type
        })
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        _, _, _, idx, _ = batch
        _, logits, y, icd_type = self.step(batch, batch_idx)
        return {
            'logits': logits,
            'y': y,
            'idx': idx,
            'icd_type': icd_type,
        }

    def _evaluate(self, step_outputs):
        y_hats = torch.cat([x['logits'] for x in step_outputs])
        ys = torch.cat([x['y'] for x in step_outputs])
        y_hats = y_hats.cpu().detach()
        ys = ys.cpu().detach()

        icd_types = [x['icd_type'] for x in step_outputs]
        assert len(set(icd_types)) == 1
        icd_type = icd_types[0]

        score_dict, score_line = self.evaluator(y_hats, ys)

        num_dict = {}
        num_dict['num_samples'] = ys.shape[0]
        num_dict['num_codes'] = ys.shape[1]
        num_dict['num_positives_ratio'] = ys.sum() / ys.numel()
        num_dict['num_positives'] = ys.sum()

        return score_dict, num_dict, score_line, icd_type
    
    def on_test_epoch_end(self):
        score_dict, num_dict, score_line, icd_type = self._evaluate(self.test_step_output)
        score_dict = {f'test_{k}': v for k, v in score_dict.items()}
        num_dict = {f'test_{k}': v for k, v in num_dict.items()}

        self.log_dict({**score_dict, **num_dict})
        self.test_step_output = []

    def on_train_start(self):
        hp_metrics = {}
        for icd_type in ['iv_icd10', 'iv_icd9', 'iii_icd9']:
            for target_freq in ICD_TARGET_FREQURNCIES:
                hp_metrics[f'hp/{icd_type}_{target_freq}'] = 0
        self.logger.log_hyperparams(self.hparams, hp_metrics)

    def on_validation_epoch_end(self):
        for dataloader_idx, step_outputs in self.validation_step_output_dict.items():
            score_dict, num_dict, score_line, icd_type = self._evaluate(step_outputs)

            target_freq = ICD_TARGET_FREQURNCIES[dataloader_idx % 3]

            main_score = score_dict[self.main_metric]
            self.log(f'hp/{icd_type}_{target_freq}', main_score)

            score_dict = {f'val_{k}': v for k, v in score_dict.items()}
            self.log_dict(score_dict)

            logging.info(f"Validation {icd_type} ({target_freq}, {num_dict['num_codes']} codes):\t{score_line}")
            
        self.validation_step_output_dict = defaultdict(list)
        

