#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)

from sklearn.metrics import (f1_score, precision_score, recall_score, confusion_matrix,
                            precision_recall_curve, auc, roc_auc_score)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    f1 = f1_score(labels, preds, average='binary')
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    
    # Calculate false positives (predicted 1 but actually 0)
    cm = confusion_matrix(labels, preds)
    # Handle case where confusion matrix might not be 2x2
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    elif cm.size == 1:
        # Only one class predicted/actual
        if labels[0] == 0:
            tn, fp, fn, tp = cm[0, 0], 0, 0, 0
        else:
            tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
    else:
        # Handle other edge cases
        tn = fp = fn = tp = 0
        if len(cm) == 2:
            tn, fp = cm[0, 0], cm[0, 1] if cm.shape[1] > 1 else 0
            fn, tp = cm[1, 0] if cm.shape[0] > 1 else 0, cm[1, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
    
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Calculate PR-AUC
    # Get probabilities for class 1
    import torch
    probs = torch.softmax(torch.tensor(pred.predictions), dim=1).numpy()
    prob_class_1 = probs[:, 1]
    
    try:
        precision_curve, recall_curve, thresholds_pr = precision_recall_curve(labels, prob_class_1)
        pr_auc = auc(recall_curve, precision_curve)
        # Also calculate ROC-AUC for completeness
        roc_auc = roc_auc_score(labels, prob_class_1)
    except Exception as e:
        pr_auc = 0.0
        roc_auc = 0.0
    
    return {
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'false_positives': fp,
        'false_positive_rate': false_positive_rate,
        'true_negatives': tn,
        'true_positives': tp,
        'false_negatives': fn,
        'pr_auc': pr_auc,
        'roc_auc': roc_auc
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='Path to CSV dataset')
    parser.add_argument('--tokenizer_dir', default='avichr/heBERT', help='Path to tokenizer dir (pretrained tokenizer) or model name')
    parser.add_argument('--output_dir', default='./trained_model', help='Where to save trained model')
    parser.add_argument('--model_name', default='avichr/heBERT', help='Base model name to fine-tune')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_len', type=int, default=256)
    args = parser.parse_args()

    # Read CSV
    df = pd.read_csv(args.csv)
    if 'text' not in df.columns:
        raise ValueError('CSV must contain a `text` column')
    if 'Tag' not in df.columns:
        raise ValueError('CSV must contain a `Tag` column')

    # Ensure labels are integers 0/1
    df['label'] = df['Tag'].fillna(0).astype(int)
    ds = Dataset.from_pandas(df[['text','label']])

    # Split into train and test sets
    ds = ds.shuffle(seed=42)
    split = ds.train_test_split(test_size=0.2, seed=42)  # 20% test set
    train_ds = split['train']
    eval_ds = split['test']
    
    print(f'Train set size: {len(train_ds)}')
    print(f'Test set size: {len(eval_ds)}')
    print(f'Label distribution in train set:')
    train_labels = [ex['label'] for ex in train_ds]
    print(f'  Class 0: {train_labels.count(0)}, Class 1: {train_labels.count(1)}')
    print(f'Label distribution in test set:')
    test_labels = [ex['label'] for ex in eval_ds]
    print(f'  Class 0: {test_labels.count(0)}, Class 1: {test_labels.count(1)}')
    print()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    def tokenize_fn(example):
        return tokenizer(example['text'], truncation=True, padding='max_length', max_length=args.max_len)

    train_ds = train_ds.map(tokenize_fn, batched=True)
    eval_ds = eval_ds.map(tokenize_fn, batched=True)

    train_ds = train_ds.remove_columns(['text'])
    eval_ds = eval_ds.remove_columns(['text'])
    train_ds.set_format('torch')
    eval_ds.set_format('torch')

    num_labels = len(df['label'].unique())

    # Load base model
    try:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)
    except Exception as e:
        print('Could not load base model from hub, initializing from config. Error:', e)
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(args.model_name, num_labels=num_labels)
        model = AutoModelForSequenceClassification.from_config(cfg)

    # Use a minimal TrainingArguments signature for compatibility with older transformers
    # Save multiple checkpoints to select best model based on false positive rate
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=50,
        eval_strategy='epoch',  # Evaluate at end of each epoch
        save_strategy='epoch',  # Save checkpoint at end of each epoch
        save_total_limit=args.epochs,  # Keep all epoch checkpoints
        load_best_model_at_end=False,  # We'll select manually based on FP rate
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Find best threshold using test set
    print('\n' + '='*60)
    print('Finding optimal threshold based on PR-AUC and F1 score...')
    print('='*60)
    
    # Get predictions on test set
    import torch
    test_predictions = trainer.predict(eval_ds)
    test_labels = test_predictions.label_ids
    test_probs = torch.softmax(torch.tensor(test_predictions.predictions), dim=1).numpy()
    test_prob_class_1 = test_probs[:, 1]
    
    # Calculate PR-AUC
    precision_curve, recall_curve, thresholds_pr = precision_recall_curve(test_labels, test_prob_class_1)
    pr_auc = auc(recall_curve, precision_curve)
    
    print(f'\nPR-AUC: {pr_auc:.4f}')
    
    # Find best threshold (maximize F1 score)
    best_threshold = 0.5
    best_f1 = 0.0
    best_metrics = {}
    
    # Test thresholds from 0.1 to 0.9
    threshold_candidates = np.arange(0.1, 1.0, 0.05)
    
    print('\nEvaluating thresholds:')
    print('Threshold | F1     | Precision | Recall  | FP Rate | FP')
    print('-' * 60)
    
    for threshold in threshold_candidates:
        preds = (test_prob_class_1 >= threshold).astype(int)
        f1 = f1_score(test_labels, preds, average='binary')
        prec = precision_score(test_labels, preds, zero_division=0)
        rec = recall_score(test_labels, preds, zero_division=0)
        cm = confusion_matrix(test_labels, preds)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        print(f'{threshold:8.2f} | {f1:6.4f} | {prec:9.4f} | {rec:7.4f} | {fp_rate:7.4f} | {fp:2d}')
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                'threshold': threshold,
                'f1': f1,
                'precision': prec,
                'recall': rec,
                'false_positives': fp,
                'false_positive_rate': fp_rate,
                'true_positives': tp,
                'false_negatives': fn
            }
    
    print('\n' + '='*60)
    print(f'Best threshold: {best_threshold:.4f}')
    print(f'  F1 Score: {best_metrics["f1"]:.4f}')
    print(f'  Precision: {best_metrics["precision"]:.4f}')
    print(f'  Recall: {best_metrics["recall"]:.4f}')
    print(f'  False Positive Rate: {best_metrics["false_positive_rate"]:.4f}')
    print(f'  False Positives: {best_metrics["false_positives"]}')
    print(f'  PR-AUC: {pr_auc:.4f}')
    print('='*60)
    
    # Save best threshold to a file
    with open(os.path.join(args.output_dir, 'best_threshold.txt'), 'w') as f:
        f.write(f'{best_threshold:.4f}\n')
        f.write(f'PR-AUC: {pr_auc:.4f}\n')
        f.write(f'F1: {best_metrics["f1"]:.4f}\n')
        f.write(f'Precision: {best_metrics["precision"]:.4f}\n')
        f.write(f'Recall: {best_metrics["recall"]:.4f}\n')
        f.write(f'False Positive Rate: {best_metrics["false_positive_rate"]:.4f}\n')
    
    print(f'\nSaved best threshold to {os.path.join(args.output_dir, "best_threshold.txt")}')

    # Evaluate all checkpoints and select model with lowest false positive rate
    print('\n' + '='*60)
    print('Evaluating all checkpoints to find best model (lowest false positive rate)...')
    print('='*60)
    
    checkpoint_dirs = []
    if os.path.exists(args.output_dir):
        for item in os.listdir(args.output_dir):
            checkpoint_path = os.path.join(args.output_dir, item)
            if os.path.isdir(checkpoint_path) and item.startswith('checkpoint-'):
                checkpoint_dirs.append((int(item.split('-')[1]), checkpoint_path))
    
    checkpoint_dirs.sort(key=lambda x: x[0])  # Sort by checkpoint number
    
    best_fp_rate = float('inf')
    best_checkpoint = None
    best_metrics = None
    all_checkpoint_metrics = []
    
    # Evaluate final model first
    final_metrics = trainer.evaluate(eval_ds)
    all_checkpoint_metrics.append(('final', final_metrics))
    print(f'\nFinal model metrics:')
    print(f'  False Positives: {final_metrics.get("eval_false_positives", "N/A")}')
    print(f'  False Positive Rate: {final_metrics.get("eval_false_positive_rate", "N/A"):.4f}')
    print(f'  F1: {final_metrics.get("eval_f1", "N/A"):.4f}')
    print(f'  Precision: {final_metrics.get("eval_precision", "N/A"):.4f}')
    print(f'  Recall: {final_metrics.get("eval_recall", "N/A"):.4f}')
    
    if final_metrics.get('eval_false_positive_rate', float('inf')) < best_fp_rate:
        best_fp_rate = final_metrics.get('eval_false_positive_rate', float('inf'))
        best_checkpoint = args.output_dir
        best_metrics = final_metrics
    
    # Evaluate each checkpoint
    for checkpoint_num, checkpoint_path in checkpoint_dirs:
        try:
            checkpoint_model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, num_labels=num_labels)
            checkpoint_trainer = Trainer(
                model=checkpoint_model,
                args=training_args,
                eval_dataset=eval_ds,
                compute_metrics=compute_metrics,
            )
            checkpoint_metrics = checkpoint_trainer.evaluate(eval_ds)
            all_checkpoint_metrics.append((f'checkpoint-{checkpoint_num}', checkpoint_metrics))
            
            fp_rate = checkpoint_metrics.get('eval_false_positive_rate', float('inf'))
            print(f'\nCheckpoint {checkpoint_num} metrics:')
            print(f'  False Positives: {checkpoint_metrics.get("eval_false_positives", "N/A")}')
            print(f'  False Positive Rate: {fp_rate:.4f}')
            print(f'  F1: {checkpoint_metrics.get("eval_f1", "N/A"):.4f}')
            print(f'  Precision: {checkpoint_metrics.get("eval_precision", "N/A"):.4f}')
            print(f'  Recall: {checkpoint_metrics.get("eval_recall", "N/A"):.4f}')
            
            if fp_rate < best_fp_rate:
                best_fp_rate = fp_rate
                best_checkpoint = checkpoint_path
                best_metrics = checkpoint_metrics
        except Exception as e:
            print(f'Warning: Could not evaluate checkpoint {checkpoint_num}: {e}')
    
    # Select and save best model
    print('\n' + '='*60)
    print(f'Best model: {best_checkpoint}')
    print(f'  False Positive Rate: {best_fp_rate:.4f}')
    print(f'  False Positives: {best_metrics.get("eval_false_positives", "N/A")}')
    print(f'  F1: {best_metrics.get("eval_f1", "N/A"):.4f}')
    print(f'  Precision: {best_metrics.get("eval_precision", "N/A"):.4f}')
    print(f'  Recall: {best_metrics.get("eval_recall", "N/A"):.4f}')
    print('='*60)
    
    # Load best model and save to final output directory
    if best_checkpoint != args.output_dir:
        print(f'\nLoading best model from {best_checkpoint}...')
        best_model = AutoModelForSequenceClassification.from_pretrained(best_checkpoint, num_labels=num_labels)
        # Use the original tokenizer (checkpoints don't always save tokenizer properly)
        # Save best model to output_dir
        os.makedirs(args.output_dir, exist_ok=True)
        best_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f'Saved best model to {args.output_dir}')
    else:
        # Final model is already the best, just ensure tokenizer is saved
        os.makedirs(args.output_dir, exist_ok=True)
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f'Final model is the best. Saved to {args.output_dir}')


if __name__ == '__main__':
    main()
