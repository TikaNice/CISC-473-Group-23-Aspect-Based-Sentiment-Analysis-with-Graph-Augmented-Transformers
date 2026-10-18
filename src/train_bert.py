"""Minimal aspect-conditioned BERT baseline starter. Review and validate all choices before reporting.

Example: python -m src.train_bert --domain restaurant --smoke
         python -m src.train_bert --domain restaurant --epochs 3 --batch-size 8

Uses group-disjoint validation split, 3-class polarity labels, and test gold only
for FINAL evaluation (never for checkpoint selection). No reported results are fabricated.
"""
from src.result_paths import result_path, checkpoint_dir

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

LABEL_TO_ID = {'negative': 0, 'neutral': 1, 'positive': 2}
MODEL = 'bert-base-uncased'
DATA_SPLIT_SEED = 2026


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    valid = [r for r in rows if r.get('label') in LABEL_TO_ID and r.get('text') and r.get('span')]
    print(f'{path}: kept {len(valid)}/{len(rows)} three-class aspect records')
    return valid


def split_by_sentence(rows):
    groups = [x['text'].casefold().strip() for x in rows]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=DATA_SPLIT_SEED)
    a, b = next(splitter.split(rows, groups=groups))
    train = [rows[i] for i in a]
    val = [rows[i] for i in b]
    assert not ({r['text'].casefold().strip() for r in train} &
                {r['text'].casefold().strip() for r in val}), 'Data leakage: shared sentences!'
    return train, val


def choose_train_fraction(rows, fraction):
    if fraction == 1.0:
        return rows
    # Sample whole sentences instead of separating aspects from the same sentence.
    groups = sorted({r['text'].casefold().strip() for r in rows})
    rng = random.Random(DATA_SPLIT_SEED)
    rng.shuffle(groups)
    keep = set(groups[:max(1, math.ceil(len(groups) * fraction))])
    return [r for r in rows if r['text'].casefold().strip() in keep]


def make_dataset(rows, tokenizer):
    raw = Dataset.from_list([{'text': r['text'], 'aspect': r['span'], 'labels': LABEL_TO_ID[r['label']]}
                             for r in rows])
    def tokenize(batch):
        # Distinct (sentence, aspect) pairs let the same sentence have different labels.
        return tokenizer(batch['text'], batch['aspect'], padding='max_length',
                         truncation=True, max_length=128)
    encoded = raw.map(tokenize, batched=True, remove_columns=['text', 'aspect'])
    columns = [x for x in ['input_ids', 'attention_mask', 'token_type_ids', 'labels'] if x in encoded.column_names]
    encoded.set_format('torch', columns=columns)
    return encoded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', choices=['restaurant','laptop'], default='restaurant')
    parser.add_argument('--epochs', type=float, default=3)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-fraction', type=float, default=1.0)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--run-tag', default='', type=str)
    args = parser.parse_args()
    if not args.smoke and not args.run_tag and args.epochs != 3:
        args.run_tag = f'ep{args.epochs:g}'
    if not 0 < args.train_fraction <= 1:
        parser.error('--train-fraction must be in (0,1]')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.backends.mps.is_available():
        print('MPS available: training will prefer MPS if Trainer recognizes it')
    else:
        print('MPS not available; check CPU/CUDA device used by Trainer')

    train_rows, val_rows = split_by_sentence(read_jsonl(f'data/raw/{args.domain}/train.jsonl'))
    train_rows = choose_train_fraction(train_rows, args.train_fraction)
    if args.smoke:
        train_rows = train_rows[:64]
        val_rows = val_rows[:32]
    print(f'Final train={len(train_rows)}, validation={len(val_rows)}')

    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=3, id2label={v:k for k,v in LABEL_TO_ID.items()},
        label2id=LABEL_TO_ID,
    )
    train_ds = make_dataset(train_rows, tokenizer)
    val_ds = make_dataset(val_rows, tokenizer)

    suffix = f'{args.domain}_fraction{args.train_fraction:g}_seed{args.seed}' + ('_smoke' if args.smoke else '')
    if args.run_tag:
        if not args.run_tag.replace('_', '').replace('-', '').isalnum():
            raise ValueError('--run-tag must contain letters, digits, _ or -')
        suffix += '_' + args.run_tag
    output_dir = str(checkpoint_dir('bert', args.domain, args.train_fraction, args.seed, args.run_tag, args.smoke))
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1 if args.smoke else args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='macro_f1',
        greater_is_better=True,
        save_total_limit=1,
        seed=args.seed,
        data_seed=DATA_SPLIT_SEED,
        report_to='none',
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        fp16=False,
        bf16=False,
        logging_steps=10,
    )
    def metrics(pred):
        y_pred = np.asarray(pred.predictions).argmax(axis=-1)
        y_true = np.asarray(pred.label_ids)
        return {'accuracy': accuracy_score(y_true,y_pred),
                'macro_f1': f1_score(y_true, y_pred, labels=[0,1,2],
                                     average='macro', zero_division=0)}

    trainer = Trainer(model=model, args=training_args,
                      train_dataset=train_ds, eval_dataset=val_ds,
                      processing_class=tokenizer, compute_metrics=metrics)
    print('Actual Trainer device:', trainer.args.device)
    trainer.train()
    result = {'experiment': suffix, 'seed': args.seed,
              'train_size':len(train_rows), 'validation_size':len(val_rows),
              'model':MODEL, 'model_type':'BERT sentence-aspect pair, CLS',
              'device':str(trainer.args.device),
              'validation_metrics': trainer.evaluate(val_ds)}

    # Never use the official gold test to choose hyperparameters or checkpoints.
    if not args.smoke:
        test_gold = Path(f'data/processed/{args.domain}/test_gold.jsonl')
        if test_gold.is_file():
            test_rows = read_jsonl(test_gold)
            result['test_size'] = len(test_rows)
            result['final_test_metrics'] = trainer.evaluate(make_dataset(test_rows, tokenizer), metric_key_prefix='test')
        else:
            print(f'WARNING: {test_gold} not found; skipping official test metrics')

    target = result_path('bert', args.domain, args.train_fraction, args.seed, args.run_tag, args.smoke)
    target.parent.mkdir(exist_ok=True,parents=True)
    with target.open('w',encoding='utf-8') as f:
        json.dump(result,f,indent=2,default=float)
    print('Saved:',target)
    print(json.dumps(result,indent=2,default=float))

if __name__ == '__main__':
    main()
