import argparse
import logging
import os
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

from duallaat import DualLAAT
from src.utils import Evaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_mimic_codes_to_consider(dataset, code_scope, resource_dir='./'):
    """Load MIMIC codes and descriptions."""
    code_scopes = json.load(open(os.path.join(resource_dir, 'code_scopes.json'), 'r'))
    code_descriptions = json.load(open(os.path.join(resource_dir, 'code_descriptions.json'), 'r'))
    
    code2id = code_scopes[f'dataset:{dataset}'][f'code_scope:{code_scope}']
    code2desc = code_descriptions[dataset.split('_')[-1]]
    
    code_candidates = [code2desc[code] for code in code2id]
    return code_candidates, code2id


def load_test_data(dataset, data_dir='./data', split='test'):
    """Load test data for evaluation."""
    logger.info(f"Loading {split} data for {dataset}")
    
    # Load main dataset
    # df = pd.read_feather(os.path.join(data_dir, 'coding_data', f'{dataset}.feather'))
    df = pd.read_feather(os.path.join(data_dir,  f'{dataset}.feather'))
    
    # Load splits
    split_df = pd.read_feather(os.path.join(data_dir, 'splits', f'{dataset}_split.feather'))
    split_ids = split_df[split_df['split'] == split]['_id'].tolist()
    
    # Filter data
    test_df = df[df['_id'].isin(split_ids)].reset_index(drop=True)
    
    logger.info(f"Loaded {len(test_df)} {split} samples")
    return test_df


def onehot_encode_target(labels, code2id):
    """Convert target labels to one-hot encoding."""
    target_matrix = torch.zeros((len(labels), len(code2id)), dtype=torch.float)
    for i, label in enumerate(labels):
        for l in label:
            if l in code2id:
                target_matrix[i, code2id[l]] = 1.0
    return target_matrix


def evaluate_model(
    model_path: str,
    dataset: str = 'mimiciv_icd10',
    code_scope: str = 'frequent',
    data_dir: str = './data',
    split: str = 'test',
    batch_size: int = 32,
    max_samples: int = None,
    output_file: str = None
):
    """
    Evaluate DualLAAT model on test data.
    
    Args:
        model_path: Path to saved DualLAAT model
        dataset: Dataset name (e.g., 'mimiciv_icd10')
        code_scope: Code scope (e.g., 'frequent')
        data_dir: Path to data directory
        split: Data split to evaluate ('test', 'val')
        batch_size: Batch size for inference
        max_samples: Maximum number of samples to evaluate (for testing)
        output_file: Path to save detailed results
    """
    
    logger.info(f"Starting evaluation of model: {model_path}")
    
    # Load model
    logger.info("Loading DualLAAT model...")
    model = DualLAAT.from_pretrained(model_path)
    model.eval()
    
    
    # Load codes and data
    logger.info(f"Loading codes for {dataset}_{code_scope}")
    codes_to_consider, code2id = load_mimic_codes_to_consider(dataset, code_scope)
    
    logger.info(f"Loading {split} data...")
    test_df = load_test_data(dataset, data_dir, split)
    
    # Limit samples if specified
    if max_samples is not None:
        test_df = test_df.head(max_samples)
        logger.info(f"Limited evaluation to {len(test_df)} samples")
    

    notes = test_df['RAW_text'].tolist()
    targets = test_df['target'].tolist()
    
    # Convert targets to one-hot encoding
    y_true = onehot_encode_target(targets, code2id)
    
    # Run inference in batches
    logger.info(f"Running inference on {len(notes)} samples...")
        
    try:
        # Get predictions
        results = model.predict(
            notes_to_code=notes,
            codes_to_consider=codes_to_consider,
            batch_size=batch_size,
        )
        
        y_pred_logits = results['logits']
        
    except Exception as e:
        logger.error(f"Error processing batch : {e}")
        raise e
    
    
    logger.info(f"Prediction shape: {y_pred_logits.shape}")
    logger.info(f"Target shape: {y_true.shape}")
    
    # Run evaluation
    logger.info("Computing evaluation metrics...")
    evaluator = Evaluator('final')
    metrics, line = evaluator(y_pred_logits, y_true)
    
    # Print results
    print("\n" + "="*50)
    print(f"EVALUATION RESULTS - {dataset}_{code_scope} ({split})")
    print("="*50)
    print(f"Dataset: {dataset}")
    print(f"Code scope: {code_scope}")
    print(f"Split: {split}")
    print(f"Number of samples: {len(notes)}")
    print(f"Number of codes: {len(codes_to_consider)}")
    print(f"Model path: {model_path}")
    print("\nMetrics:")
    for metric_name, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {metric_name}: {value:.4f}")
        else:
            print(f"  {metric_name}: {value}")
    print("="*50)
    
    # Save detailed results if requested
    if output_file:
        logger.info(f"Saving detailed results to {output_file}")
        
        results_data = {
            'evaluation_info': {
                'model_path': model_path,
                'dataset': dataset,
                'code_scope': code_scope,
                'split': split,
                'num_samples': len(notes),
                'num_codes': len(codes_to_consider),
                'batch_size': batch_size,
                'timestamp': str(pd.Timestamp.now())
            },
            'metrics': metrics,
        }
        
        # Save as JSON
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
    
    return metrics, y_pred_logits, y_true


def main():
    parser = argparse.ArgumentParser(description="Evaluate DualLAAT model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to saved DualLAAT model directory")
    parser.add_argument("--dataset", type=str, default="mimiciv_icd10",
                        help="Dataset name")
    parser.add_argument("--code_scope", type=str, default="frequent",
                        help="Code scope (frequent, common, all)")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Path to data directory")
    parser.add_argument("--split", type=str, default="test",
                        help="Data split to evaluate (test, val)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to evaluate")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to save detailed results (JSON)")
    
    args = parser.parse_args()
    
    # Evaluate single model
    evaluate_model(
        model_path=args.model_path,
        dataset=args.dataset,
        code_scope=args.code_scope,
        data_dir=args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        output_file=args.output_file
    )


if __name__ == "__main__":
    main()


