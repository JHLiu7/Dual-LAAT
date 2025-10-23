# DualLAAT-ICD

High-Performing and Version-Agnostic ICD Classification with Training Data Enhancement


## Overview

DualLAAT is a deep learning model designed for multi-label classification of ICD (International Classification of Diseases) codes from clinical text. The model is ICD-version-agnostic and can handle both ICD-9 and ICD-10. We provide model checkpoints trained on the three MIMIC ICD datasets to support future research on automated and assisted ICD coding.

<!-- ### Key Features

- **Dual attention mechanism** for both clinical notes and ICD code descriptions
- **Flexible encoder architecture** supporting both RNN (LSTM/GRU) and CNN encoders
- **Multi-head attention** for capturing complex text-code relationships
- **Support for multiple datasets**: MIMIC-III (ICD-9) and MIMIC-IV (ICD-9/ICD-10)
- **Pretrained model checkpoints** for quick deployment -->


## Evaluation Results

The evaluation is based on the SIGIR 2023 paper [Automated Medical Coding on MIMIC-III and MIMIC-IV: A Critical Review and Replicability Study](https://dl.acm.org/doi/10.1145/3539618.3591918). [[GitHub](https://github.com/JoakimEdin/medical-coding-reproducibility/tree/main)] Notice the baseline PLM-ICD is trained for each different MIMIC dataset, whereas our DualLAAT models handle the three datasets together.

### MIMIC-IV ICD-10

| Model | AUC Micro | AUC Macro | F1 Micro | F1 Macro | P@8 | P@15 | P@R | MAP |
|---|---|---|---|---|---|---|---|---|
| PLM-ICD (Previous SoTA) | 99.2±0.0 | 96.6±0.2 | 58.5±0.7 | 21.1±2.3 | 69.9±0.6 | 55.0±0.6 | 57.9±0.8 | 61.9±0.9 |
| DualLAAT-cnn | 99.3±0.0 | 97.4±0.0 | 57.1±0.1 | 28.2±0.3 | 69.0±0.1 | 54.1±0.0 | 57.1±0.0 | 61.2±0.0 |
| DualLAAT | 99.4±0.0 | 97.6±0.0 | 58.4±0.1 | 32.1±0.1 | 70.3±0.0 | 55.3±0.0 | 58.5±0.1 | 62.9±0.0 |
| DualLAAT (trained on frequent codes) | 99.4±0.0 | 97.4±0.1 | 59.0±0.1 | 30.8±0.1 | 70.9±0.1 | 55.7±0.1 | 59.0±0.1 | 63.5±0.1 |


### MIMIC-IV ICD-9

| Model | AUC Micro | AUC Macro | F1 Micro | F1 Macro | P@8 | P@15 | P@R | MAP |
|---|---|---|---|---|---|---|---|---|
| PLM-ICD (Previous SoTA) | 99.4±0.0 | 97.2±0.2 | 62.6±0.3 | 29.8±1.0 | 70.0±0.2 | 53.5±0.2 | 62.7±0.3 | 68.0±0.3  |
| DualLAAT-cnn | 99.4±0.0 | 97.5±0.0 | 61.3±0.1 | 29.1±0.2 | 69.7±0.0 | 53.2±0.0 | 62.3±0.0 | 67.6±0.0 |
| DualLAAT | 99.5±0.0 | 97.7±0.0 | 63.5±0.1 | 34.9±0.1 | 71.1±0.1 | 54.4±0.0 | 63.9±0.0 | 69.5±0.0 |
| DualLAAT (trained on frequent codes) | 99.5±0.0 | 97.5±0.1 | 63.4±0.1 | 34.9±0.2 | 70.9±0.1 | 54.3±0.0 | 63.7±0.1 | 69.2±0.1 |


### MIMIC-III

| Model | AUC Micro | AUC Macro | F1 Micro | F1 Macro | P@8 | P@15 | P@R | MAP |
|---|---|---|---|---|---|---|---|---|
| PLM-ICD (Previous SoTA) | 98.9±0.0 | 95.9±0.1 | 59.6±0.2 | 26.6±0.8 | 72.1±0.2 | 56.5±0.1 | 60.1±0.1 | 64.6±0.2  |
| DualLAAT-cnn | 99.1±0.0 | 96.5±0.1 | 58.4±0.1 | 27.2±0.1 | 72.8±0.0 | 57.0±0.0 | 60.7±0.1 | 65.5±0.1 |
| DualLAAT | 99.2±0.0 | 96.9±0.1 | 61.8±0.1 | 33.1±0.2 | 75.1±0.1 | 58.9±0.1 | 62.9±0.1 | 68.2±0.1 |
| DualLAAT (trained on frequent codes) | 99.2±0.0 | 96.8±0.0 | 61.6±0.1 | 32.7±0.5 | 75.0±0.1 | 58.9±0.1 | 63.0±0.0 | 68.2±0.1 |


## Installation

### Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DualLAAT-ICD.git
cd DualLAAT-ICD
```

2. Install dependencies for inference:
```bash
pip install torch
```

3. (Optional) Install dependencies to reproduce experiments; only needed for training:
```bash
pip install -r requirements.txt
```

## Quick Start

### Using Pretrained Models

Load and use a pretrained DualLAAT model for inference:

```python
from duallaat import DualLAAT

# Load pretrained model
model = DualLAAT.from_pretrained('checkpoints/DualLAAT-ckpt-1')

# Define candidate ICD-10 codes
candidate_codes = {
    "I10": "Essential (primary) hypertension",
    "E11.9": "Type 2 diabetes mellitus without complications",
    "J44.9": "Chronic obstructive pulmonary disease, unspecified",
    "F41.1": "Generalized anxiety disorder",
    "M54.5": "Low back pain"
}

# ============================================
# Sample Clinical Note
# Ground truth codes: I10, M54.5
# ============================================
clinical_note = """
Patient is a 62-year-old male presenting for routine follow-up. He has a 
history of essential hypertension, currently managed with lisinopril 10mg 
daily with good control. Blood pressure today is 128/82. Patient also reports 
chronic low back pain that has been present for the past 6 months, worse with 
prolonged sitting. Pain is managed with NSAIDs and physical therapy. No 
radiating symptoms or neurological deficits noted on examination.
"""

# Make predictions
results = model.predict(
    notes_to_code=clinical_note,
    codes_to_consider=list(candidate_codes.values()),
)

# Get predicted probabilities (after sigmoid)
predictions = results['probabilities']
print(predictions)
```

## Data Preparation

1. Prepare your MIMIC dataset:
```bash
cd data
python prepare_mimiciii.py  # For MIMIC-III
# or
python prepare_mimiciv.py   # For MIMIC-IV
```

2. (Optional) Train word embeddings:
```bash
python data/train_w2v.py
```

## Inference and Evaluation

```bash
python evaluate_duallaat.py \
    --model_path checkpoints/DualLAAT-ckpt-1 \
    --dataset mimiciv_icd10 \
    --code_scope frequent \
    --split test \
    --batch_size 32 \
    --output_file eval_results.json
```

## Training

### Basic Training

Train a DualLAAT model using a configuration file:

```bash
python train.py --config configs/laat.yaml
```

### Custom Training

Override configuration parameters:

```bash
python train.py \
    --config configs/laat.yaml \
    --overrides \
        batch_size=16 \
        max_epochs=30 \
        lr=0.0005 \
        icd_type=iv_icd10
```

<!-- ### Key Configuration Options

- `icd_type`: Dataset variant (`iv_icd10`, `iv_icd9`, `iii_icd9`)
- `icd_target_frequency`: Code scope (`frequent`, `full`)
- `encoder_type`: Encoder architecture (`rnn`, `cnn`)
- `rnn_type`: RNN variant (`lstm`, `gru`)
- `max_input_length`: Maximum sequence length (default: 4000)
- `batch_size`: Training batch size
- `lr`: Learning rate
- `dropout`: Dropout rate
 -->

<!-- ## Citation

If you use this code in your research, please cite:

```bibtex
@article{duallaat2024,
  title={DualLAAT: Dual Label-Aware Attention for Automated ICD Coding},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
``` -->

## Acknowledgments

This work builds upon several important contributions to the automated ICD coding research community:

- **MIMIC Datasets**: We gratefully acknowledge the MIT-LCP team for providing the [MIMIC-III](https://physionet.org/content/mimiciii/) and [MIMIC-IV](https://physionet.org/content/mimiciv/) datasets, which are essential resources for clinical NLP research.

- **Reproducibility Study**: Our evaluation framework is based on the SIGIR 2023 paper by Edin et al., ["Automated Medical Coding on MIMIC-III and MIMIC-IV: A Critical Review and Replicability Study"](https://dl.acm.org/doi/10.1145/3539618.3591918), which established standardized benchmarks for comparing ICD coding models.

- **Prior Work**: This research was inspired by label-attention approaches in multi-label classification, particularly [CAML](https://aclanthology.org/N18-1100/) and [LAAT](https://www.ijcai.org/proceedings/2020/461) architectures that pioneered the use of label attention for medical coding.


We thank the broader clinical NLP and machine learning communities for their ongoing efforts to improve automated medical coding systems.