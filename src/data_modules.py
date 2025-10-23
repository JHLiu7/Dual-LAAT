import json
import os
import torch
import copy
import logging 
import pandas as pd
import numpy as np
import datasets
import lightning as L
import torch
import pickle
import json
import random
import logging
from torch.utils.data import Sampler, Dataset, DataLoader
from typing import List, Set, Optional
from collections import defaultdict
from dataclasses import dataclass


from src.utils import ICD_TARGET_FREQURNCIES, _get_file_paths, _load_data_df_and_codes, _split_data, _get_codes_for_sampling



class MultiLabelTextDataset(Dataset):
    def __init__(self, texts, labels, ids, icd_type):
        self.texts = texts
        self.x_len = len(self.texts)
        self.labels = labels
        self.ids = ids
        self.icd_type = icd_type
        assert self.x_len == len(labels), f"Number of texts ({self.x_len}) != number of labels ({len(labels)})"

    def __len__(self):
        return self.x_len
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx], self.ids[idx], self.icd_type


@dataclass
class W2vTextDataModule(L.LightningDataModule):
    data_dir: str
    data_w2v_dir: str
    icd_type: str
    icd_target_frequency: str
    
    negative_sampling_strategy: str = 'null'
    label_space: Optional[int] = None
    scope_for_negative_samples: Optional[str] = None

    batch_size: int = 32
    max_input_length: int = 4000

    alternative_data_folder: Optional[str] = None
    debug: bool = False
        
    def prepare(self):
        self.code_description_file_path, self.data_file_path, self.token2id_file_path, self.split_file_path = \
            _get_file_paths(self.data_dir, self.icd_type, self.data_w2v_dir, alternative_data_folder=self.alternative_data_folder)

        self.label2embeddings = None

        if self.negative_sampling_strategy != 'null':
            assert self.label_space is not None, "Label space must be provided for 'sample' training regime."
            assert self.label_space % 2 == 0, "Label space must be an even number."
            self.train_collate_fn = collate_fn_with_label_sampling
            self.inf_collate_fn = collate_fn_without_label_sampling
            self.target_space = self.label_space 

            # Load label embeddings
            if self.negative_sampling_strategy == 'embeddings':
                label2embeddings_file_path = os.path.join(self.data_dir, f'coding_data/code_embeddings.pkl')
                icd_type = self.icd_type.split('_')[1]
                label2embeddings = pd.read_pickle(label2embeddings_file_path)[icd_type]
            else:
                self.label2embeddings = None

        else:
            self.train_collate_fn = collate_fn_without_label_sampling
            self.inf_collate_fn = collate_fn_without_label_sampling
            self.target_space = None

        data_df, (self.id_column, self.text_column, self.target_column), (self.label2id, label2id_all, label2description) = _load_data_df_and_codes(
            self.icd_type, self.data_file_path, self.icd_target_frequency, self.code_description_file_path
        )

        self.train_df, self.val_df, self.test_df = _split_data(data_df, self.split_file_path, self.id_column, self.debug)

        # get code descriptions
        label2description = {k: v.lower() for k, v in label2description.items()}
        all_codes, all_code_descriptions = _get_codes_for_sampling(
            self.scope_for_negative_samples, label2description, self.label2id, label2id_all
        )

        self.token2id = pd.read_pickle(self.token2id_file_path)
        self.pad_token_id = self.token2id['<PAD>']

        max_desc_len = max([len(desc.split()) for desc in all_code_descriptions])
        self.label2description = {code: self.encode_text(desc, max_length=max_desc_len) 
            for code, desc in zip(all_codes, all_code_descriptions)}
        
        if self.negative_sampling_strategy == 'embeddings' and label2embeddings is not None:
            # Convert label2embeddings to tensor
            self.label2embeddings = {code: label2embeddings[code] for code in all_codes}
            logging.info(f"Prepared {len(self.label2embeddings)} label embeddings for {self.icd_type} with scope '{self.scope_for_negative_samples}'.")


    def load_dataset(self, df):
        texts = df[self.text_column].apply(self.encode_text).tolist()
        labels = df[self.target_column].tolist()
        ids = df[self.id_column].tolist()
        return MultiLabelTextDataset(texts, labels, ids, self.icd_type)

    def encode_text(self, text, max_length=None):
        if max_length is None:
            max_length = self.max_input_length
        ids = [self.token2id[t] if t in self.token2id else self.token2id['<UNK>'] for t in text.split()]
        padded = np.full((max_length), self.token2id['<PAD>'], dtype=int)
        min_len = min(len(ids), max_length)
        padded[:min_len] = ids[:min_len]
        return padded
    
    def setup(self, stage=None):
        self.prepare()

        if stage == 'train' or stage is None:
            self.train_dataset = self.load_dataset(self.train_df)
            logging.info(f"Train dataset size: {len(self.train_dataset)}")
        if stage == 'evaluate' or stage is None:
            self.val_dataset = self.load_dataset(self.val_df)
            self.test_dataset = self.load_dataset(self.test_df)
            logging.info(f"Test dataset size: {len(self.test_dataset)}")
            logging.info(f"Val dataset size: {len(self.val_dataset)}")
        if stage == 'debug':
            self.train_dataset = self.load_dataset(self.train_df.head(100))
            self.val_dataset = self.load_dataset(self.val_df.head(100))
            self.test_dataset = self.load_dataset(self.test_df.head(100))

    def train_dataloader(self, batch_size=None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, 
                collate_fn=lambda batch: self.train_collate_fn(
                    batch=batch, label2id=self.label2id, label2desc=self.label2description, target_space=self.target_space, label2embeddings=self.label2embeddings
                ))
    
    def val_dataloader(self, batch_size=None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False,
                collate_fn=lambda batch: self.inf_collate_fn(
                    batch=batch, label2id=self.label2id, label2desc=self.label2description
                ))
    
    def test_dataloader(self, batch_size=None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False,
                collate_fn=lambda batch: self.inf_collate_fn(
                    batch=batch, label2id=self.label2id, label2desc=self.label2description
                ))




def collate_fn_without_label_sampling(batch, label2id: dict, label2desc: dict, **kwargs):
    texts, labels, ids, icd_type = zip(*batch)
    texts = torch.tensor(np.array(texts))
    target_matrix = onehot_encode_target(labels, label2id)
    codes = torch.tensor(np.array([label2desc[l] for l in label2id.keys()]))
    icd_type = list(set(icd_type))
    assert len(icd_type) == 1
    icd_type = icd_type[0]
    return texts, codes, target_matrix, ids, icd_type

def collate_fn_with_label_sampling(batch, label2desc: dict, target_space: int, label2embeddings: Optional[dict] = None, **kwargs):
    texts, labels, ids, icd_type = zip(*batch)
    texts = torch.tensor(np.array(texts))

    all_labels = list(label2desc.keys())
    pos_labels = list(set([l for t in labels for l in t if l in all_labels]))

    _, batch_labels = sample_negatives(pos_labels, all_labels, target_space, label2embeddings)
    target_matrix = onehot_encode_target(labels, batch_labels=batch_labels)
    codes = torch.tensor(np.array([label2desc[l] for l in batch_labels]))

    icd_type = list(set(icd_type))
    assert len(icd_type) == 1
    icd_type = icd_type[0]
    return texts, codes, target_matrix, ids, icd_type





class ICDSampler(Sampler):
    def __init__(
        self,
        dataset,
        batch_size,
        shuffle_within: bool = False,
        drop_last: bool = True,
    ):
        self.batch_size    = batch_size
        self.shuffle_within= shuffle_within
        self.drop_last     = drop_last

        # group indices by id
        self.id_to_idxs = {}
        for idx in range(len(dataset)):
            data = dataset[idx]
            if isinstance(data, dict):
                icd_type = data['icd_type']
            else:
                icd_type = data[-1]  # last element is icd_type
            self.id_to_idxs.setdefault(icd_type, []).append(idx)
        self.ids = list(self.id_to_idxs.keys())

    def __iter__(self):
        # for each id, optionally shuffle its indices, then chunk
        batches_per_id = {}
        for gid, idxs in self.id_to_idxs.items():
            idxs = idxs[:]  # copy
            if self.shuffle_within:
                random.shuffle(idxs)

            chunks = [
                idxs[i : i + self.batch_size]
                for i in range(0, len(idxs), self.batch_size)
            ]
            if self.drop_last:
                chunks = [c for c in chunks if len(c) == self.batch_size]
            batches_per_id[gid] = chunks

        # flatten and shuffle all batches
        all_batches = [b for lst in batches_per_id.values() for b in lst]
        random.shuffle(all_batches)
        for batch in all_batches:
            yield batch

    def __len__(self):
        # total number of batches across all ids
        total = 0
        for idxs in self.id_to_idxs.values():
            cnt = len(idxs) // self.batch_size
            if not self.drop_last and (len(idxs) % self.batch_size):
                cnt += 1
            total += cnt
        return total





def sample_negatives(pos_labels, all_labels, target_space, repeated_labels=None, label2embeddings=None):
    # Filter negative candidates
    neg_candidates = [l for l in all_labels if l not in pos_labels]

    if repeated_labels is not None:
        num_neg_old = len(neg_candidates)
        neg_candidates = [l for l in neg_candidates if l not in repeated_labels]
        num_neg_new = len(neg_candidates)
        # if num_neg_new < num_neg_old:
        #     logging.info(f"Filtered negative candidates: {num_neg_old} -> {num_neg_new}")

    num_negatives = target_space - len(pos_labels)

    # Ensure we only take the first `num_negatives` unique indices
    if num_negatives > len(neg_candidates):
        logging.warning(f"Requested {num_negatives} negatives, but only {len(neg_candidates)} available. Adjusting to available count.")
        num_negatives = len(neg_candidates)
    
    if label2embeddings is None:
        # Sample random negatives
        try:
            neg_samples = random.sample(neg_candidates, num_negatives)
        except ValueError:
            print(f"Not enough negative candidates. Available: {len(neg_candidates)}, Required: {num_negatives}")
            print("target space:", target_space)
            print("num positives:", len(pos_labels))
    else:
        # Sample negatives based on embedding similarity
        pos_embeddings = torch.stack([label2embeddings[l] for l in pos_labels])
        neg_embeddings = torch.stack([label2embeddings[l] for l in neg_candidates])
        
        # Compute cosine similarity between positive and negative embeddings
        similarity_matrix = torch.mm(pos_embeddings, neg_embeddings.t())

        # Get the top indices for each positive embedding
        top_matrix = similarity_matrix.argsort(dim=1, descending=True)
        # logging.info(f"Top matrix shape: {top_matrix.shape}")

        final_neg_indices = set()
        for i in range(top_matrix.shape[1]):
            # Get the top indices at each position
            # This will give us the indices of the top negative candidates for each positive label
            top_indices_per_rank = top_matrix[:, i]

            # Convert to a set to ensure uniqueness
            unique_indices = set(top_indices_per_rank.tolist())
            final_neg_indices.update(unique_indices)

            if len(final_neg_indices) >= num_negatives:
                break

        final_neg_indices = list(final_neg_indices)[:num_negatives]

        neg_samples = [neg_candidates[idx] for idx in final_neg_indices]


    batch_labels = pos_labels + neg_samples
    assert len(batch_labels) == target_space

    return neg_samples, batch_labels


def onehot_encode_target(labels, label2id=None, batch_labels=None):
    # torch tensor for batching

    if label2id is not None:
        assert batch_labels is None, "Either provide label2id or batch_labels"
        target_matrix = torch.zeros((len(labels), len(label2id)), dtype=torch.float)
        for i, label in enumerate(labels):
            for l in label:
                target_matrix[i, label2id[l]] = 1.0
    else:
        assert batch_labels is not None, "Either provide label2id or batch_labels"
        target_matrix = torch.zeros((len(labels), len(batch_labels)), dtype=torch.float)
        for i, label in enumerate(labels):
            for l in label:
                target_matrix[i, batch_labels.index(l)] = 1.0
    return target_matrix






@dataclass
class ICDCollatorWithSampling:
    target_space: int
    icdtype2label2desc: dict
    # label2desc_iv_icd10: dict
    # label2desc_iv_icd9: dict
    # label2desc_iii_icd9: Optional[dict] = None
    # label2embeddings_iv_icd10: Optional[dict] = None
    # label2embeddings_iv_icd9: Optional[dict] = None
    # label2embeddings_iii_icd9: Optional[dict] = None
    icdtype2label2embeddings: Optional[dict] = None

    codes_with_same_descriptions: Optional[dict] = None
    pad_token_id: Optional[int] = None

    def __post_init__(self):
        self.collate_fn = self.w2v_collate_fn_with_label_sampling        

        if self.codes_with_same_descriptions is not None:
            self.code_to_repetition = defaultdict(dict)

            for key in self.codes_with_same_descriptions.keys():
                same_desc_codes = self.codes_with_same_descriptions[key]
                key = key.replace('mimic', '')

                for _, codes in same_desc_codes.items():
                    assert len(codes) == 2
                    self.code_to_repetition[key][codes[0]] = codes[1]
                    self.code_to_repetition[key][codes[1]] = codes[0]

            assert 'iv_icd10' in self.code_to_repetition
            assert 'iv_icd9' in self.code_to_repetition
            assert 'iii_icd9' in self.code_to_repetition


    def __call__(self, batch, *args, **kwargs):
        return self.collate_fn(batch)
    
    def _get_batch_labels(self, labels, target_space, label2desc, label2embeddings, repeated_code_dict:dict=None):
        if target_space == 0:
            return []
        all_labels = list(label2desc.keys())
        pos_labels = list(set([l for t in labels for l in t if l in all_labels]))
        if pos_labels == []:
            return []
        
        if repeated_code_dict is not None:
            # codes with same descriptions: we don't want them as negatives
            repeated_labels = [repeated_code_dict[code] for code in pos_labels if code in repeated_code_dict]
        else:
            repeated_labels = None

        _, batch_labels = sample_negatives(pos_labels, all_labels, target_space, repeated_labels=repeated_labels, label2embeddings=label2embeddings)
        return batch_labels

    def w2v_collate_fn_with_label_sampling(self, batch, **kwargs):
        texts, labels, ids, icd_types = zip(*batch)

        icd_types = list(set(icd_types))
        assert len(icd_types) == 1, f"Expected single ICD type, got {icd_types}"
        icd_type = icd_types[0]

        repeated_code_dict = self.code_to_repetition.get(icd_type, None)

        batch_labels = self._get_batch_labels(
            labels, self.target_space, 
            self.icdtype2label2desc[icd_type], 
            self.icdtype2label2embeddings[icd_type],
            repeated_code_dict=repeated_code_dict
        )
        target_matrix = onehot_encode_target(labels, batch_labels=batch_labels)

        texts = torch.tensor(np.array(texts))

        codes = [self.icdtype2label2desc[icd_type][l] for l in batch_labels]
        codes = torch.nn.utils.rnn.pad_sequence(codes, batch_first=True, padding_value=self.pad_token_id)

        return texts, codes, target_matrix, ids, icd_type




def _get_val_and_test_loaders(cfg, icd_type):
    # a single icd type
    val_loaders, test_loaders = {}, {}

    cfg_icd = copy.deepcopy(cfg)
    cfg_icd.icd_type = icd_type

    for icd_target_frequency in ICD_TARGET_FREQURNCIES:
        cfg_icd_freq = copy.deepcopy(cfg_icd)
        cfg_icd_freq.icd_target_frequency = icd_target_frequency
        cfg_icd_freq.negative_sampling_strategy = 'null'  # no negative sampling for evaluation
        dm = get_datamodule(cfg_icd_freq)
        dm.setup('evaluate')

        val_loader = dm.val_dataloader()
        test_loader = dm.test_dataloader()

        val_loaders[f'{icd_type}_{icd_target_frequency}'] = val_loader
        test_loaders[f'{icd_type}_{icd_target_frequency}'] = test_loader
    return val_loaders, test_loaders


def _get_train_module(cfg, icd_type, alternative_data_folder=None):
    # a single icd type
    cfg_icd = copy.deepcopy(cfg)
    cfg_icd.icd_type = icd_type
    dm = get_datamodule(cfg_icd, alternative_data_folder=alternative_data_folder)
    dm.setup('train')
    return dm


def prepare_dataloaders(cfg):
    """
    Prepares the data loaders for training, validation, and testing.

    Allow loading multiple ICD types by specifying them in the `icd_type` attribute of the config, separated by '+'.
    Allow loading data from an alternative folder by specifying the `alternative_data_folder` attribute of the config.
    E.g., `cfg.icd_type = 'iv_icd10+iv_icd9'` and `cfg.alternative_data_folder = 'augment_data_rewritten'`
    
    By default, the alternative data is used for augmentation of the original training data.
    For alternative data folder, specify `only` to mean training on synthetic data only.
    
    """

    icd_type = cfg.icd_type
    val_loader_dict = {}
    test_loader_dict = {}

    # if '+' not in icd_type:
    #     dm = _get_train_module(cfg, icd_type)

    #     train_loader = dm.train_dataloader()

    #     val_loaders, test_loaders = _get_val_and_test_loaders(cfg, icd_type)
    #     val_loader_dict.update(val_loaders)
    #     test_loader_dict.update(test_loaders)
    # else:


    ## Prepare multiple ICD types
    

    alternative_data_folders = []
    alternative_icd_types = []
    use_alternative_only = False
    if hasattr(cfg, 'alternative_data_folder') and cfg.alternative_data_folder is not None:
        # check only
        if 'only:' in cfg.alternative_data_folder:
            # only use synthetic data for training
            folder = cfg.alternative_data_folder.replace('only:', '')
            assert folder != '', "Please provide a valid alternative data folder name after 'only:'."
        else:
            folder = cfg.alternative_data_folder

        # check if multiple folders
        if '+' in folder:
            alternative_data_folders = folder.split('+')
        else:
            alternative_data_folders = [folder]

        logging.info(f"Using alternative data folder(s): {alternative_data_folders}")
        if 'only:' in cfg.alternative_data_folder:
            use_alternative_only = True
            logging.info(f"******* Using alternative data only for training. *******")

    if hasattr(cfg, 'alternative_icd_type') and cfg.alternative_icd_type is not None:
        alt_icd_type = cfg.alternative_icd_type
        assert alt_icd_type != '', "Please provide a valid alternative icd_type."
        if '+' in alt_icd_type:
            alternative_icd_types = alt_icd_type.split('+')
        else:
            alternative_icd_types = [alt_icd_type]


    if '+' not in icd_type:
        icd_types = [icd_type]
    else:
        # Multi ICD type
        icd_types = icd_type.split('+')

    # Load training data loaders
    train_data_modules = {}

    if not use_alternative_only:
        # always include original icd types
        for icd_type in icd_types:
            dm = _get_train_module(cfg, icd_type)
            train_data_modules[icd_type] = dm

    if len(alternative_data_folders) > 0 and len(alternative_icd_types) > 0:
        for icd_type in alternative_icd_types:
            for i, alternative_data_folder in enumerate(alternative_data_folders):

                dm = _get_train_module(cfg, icd_type, alternative_data_folder=alternative_data_folder)
                key_name = f"{icd_type}" if use_alternative_only and i==0 else f"{icd_type}_altset{i+1}"
                train_data_modules[key_name] = dm
                logging.info(f"Loaded alternative data for ICD type {icd_type} from folder '{alternative_data_folder}' with {len(dm.train_dataset)} training samples.")


    # Concatenate all training datasets
    concat_train_dataset = torch.utils.data.ConcatDataset([
        dm.train_dataset for dm in train_data_modules.values()
    ])


    # Load evaluation data loaders
    for icd_type in icd_types:
        val_loaders, test_loaders = _get_val_and_test_loaders(cfg, icd_type)
        val_loader_dict.update(val_loaders)
        test_loader_dict.update(test_loaders)
        
    
    # Prepare data functions

    codes_with_same_descriptions = json.load(open(os.path.join(cfg.data_dir, 'coding_data/same_desc_codes.json'), 'r'))
    icdtype2label2desc = {
        key: {k: torch.tensor(v) for k, v in dm.label2description.items()} 
        for key, dm in train_data_modules.items()
    }
    icdtype2label2embeddings = {
        key: dm.label2embeddings for key, dm in train_data_modules.items()
    }

    collator = ICDCollatorWithSampling(
        target_space=cfg.label_space,
        icdtype2label2desc=icdtype2label2desc,
        icdtype2label2embeddings=icdtype2label2embeddings,
        # label2desc_iv_icd10=train_data_modules['iv_icd10'].label2description,
        # label2desc_iv_icd9=train_data_modules['iv_icd9'].label2description,
        # label2desc_iii_icd9=train_data_modules['iii_icd9'].label2description if 'iii_icd9' in train_data_modules else None,

        # label2embeddings_iv_icd10=train_data_modules['iv_icd10'].label2embeddings,
        # label2embeddings_iv_icd9=train_data_modules['iv_icd9'].label2embeddings,
        # label2embeddings_iii_icd9=train_data_modules['iii_icd9'].label2embeddings if 'iii_icd9' in train_data_modules else None,

        codes_with_same_descriptions=codes_with_same_descriptions,
        pad_token_id=list(train_data_modules.values())[0].pad_token_id
    )

    train_loader = torch.utils.data.DataLoader(
        concat_train_dataset, #batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collator, num_workers=cfg.num_workers,
        batch_sampler=ICDSampler(
            dataset=concat_train_dataset,
            batch_size=cfg.batch_size,
            shuffle_within=True, drop_last=False
        )
    )

    num_train_steps = _get_train_steps(cfg, train_loader)

    epochs = (num_train_steps // (len(train_loader) // cfg.accumulate_grad_batches))

    logging.info(f"##### Number of training samples: {len(train_loader.dataset)}")
    logging.info(f"##### Batch size: {cfg.batch_size * cfg.accumulate_grad_batches}")
    logging.info(f"##### Target space: {cfg.label_space}")
    logging.info(f"##### Number of batches per epoch: {len(train_loader) // cfg.accumulate_grad_batches}")
    logging.info(f"##### Accumulate grad batches: {cfg.accumulate_grad_batches}")
    logging.info(f"##### Number of epochs: {epochs}")
    logging.info(f"##### Number of max steps: {cfg.max_steps}")
    logging.info(f"##### Number of training steps: {num_train_steps}")

    return train_loader, val_loader_dict, test_loader_dict, num_train_steps


def _get_train_steps(config, train_loader):
    train_loader_len = len(train_loader)
    if config.max_steps > 0:
        return config.max_steps
    else:
        return config.max_epochs * (train_loader_len // config.accumulate_grad_batches)


def get_datamodule(cfg, alternative_data_folder=None):
    dm = W2vTextDataModule(
        data_dir=cfg.data_dir,
        data_w2v_dir=cfg.data_w2v_dir,
        icd_type=cfg.icd_type,
        icd_target_frequency=cfg.icd_target_frequency,
        negative_sampling_strategy=cfg.negative_sampling_strategy,
        scope_for_negative_samples=cfg.scope_for_negative_samples,
        label_space=cfg.label_space,
        batch_size=cfg.batch_size,
        max_input_length=cfg.max_input_length,
        debug=cfg.debug,
        alternative_data_folder=alternative_data_folder
    )
    return dm
