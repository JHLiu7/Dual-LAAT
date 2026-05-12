# Data layout

This folder ships only the resources that are safe to redistribute: the
official train/val/test splits and the public code-description / code-scope
JSONs used at inference. The MIMIC datasets themselves are credentialed and
must be obtained from PhysioNet by the user.

## What's already in the repo

```
data/
├── code_descriptions.json     # icd9/icd10 code → official long-title
├── code_scopes.json           # per-dataset, per-scope code→id maps
├── splits/
│   ├── mimiciii_icd9_split.feather   # _id, split ∈ {train, val, test}
│   ├── mimiciv_icd9_split.feather
│   └── mimiciv_icd10_split.feather
├── prepare_mimiciii.py        # MIMIC-III → processed feather
├── prepare_mimiciv.py         # MIMIC-IV → processed feather
├── train_w2v.py               # train word2vec on processed notes
└── utils.py                   # preprocessing helpers (also exports DOWNLOAD_DIR_* constants)
```

## What you need to produce locally

The training and full-evaluation pipelines expect three artifact folders that
this repo does **not** ship. You produce them once, locally, after obtaining
MIMIC-III and MIMIC-IV access from PhysioNet:

```
data/
├── coding_data/                       # produced by prepare_mimic{iii,iv}.py
│   ├── mimiciii_icd9.feather
│   ├── mimiciv_icd9.feather
│   ├── mimiciv_icd10.feather
│   └── code_descriptions.pkl
└── w2v/                               # produced by train_w2v.py
    ├── token2id.pkl
    └── vectors.npy
```

### Step 1 — preprocess the raw data

Edit the four download-directory constants at the top of
`data/utils.py` to point at your local MIMIC dumps:

```python
DOWNLOAD_DIRECTORY_MIMICIII      = "/path/to/physionet.org/files/mimiciii/1.4"
DOWNLOAD_DIRECTORY_MIMICIV       = "/path/to/physionet.org/files/mimiciv/2.2"
DOWNLOAD_DIRECTORY_MIMICIV_NOTE  = "/path/to/physionet.org/files/mimic-iv-note/2.2"
DATA_DIRECTORY_PROCESSED         = "/path/to/Dual-LAAT/data/coding_data"
```

Then run, from the repo root:

```bash
python data/prepare_mimiciii.py
python data/prepare_mimiciv.py
```

Both scripts write feather files into `DATA_DIRECTORY_PROCESSED` plus a
`code_descriptions.pkl`.

### Step 2 — train the word2vec embeddings

```bash
cd data && python train_w2v.py
```

Adjust the `data_dir`, `split_dir`, and `output_dir` constants at the top of
`train_w2v.py` first (the defaults assume a local working directory). The
output is the `token2id.pkl` + `vectors.npy` pair that the training pipeline
loads from `data/w2v/`.



## Split file naming

All splits are named `mimic{icd_type}_split.feather` (singular), with
`{icd_type}` ∈ {`iii_icd9`, `iv_icd9`, `iv_icd10`}. The schema is two columns
(`_id`, `split`) where `split ∈ {train, val, test}`. The training pipeline
also accepts the legacy `mimiciii_clean_splits.feather` name as a fallback.
