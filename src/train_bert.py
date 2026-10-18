"""Aspect-conditioned BERT baseline (sentence-aspect pair, [CLS] classifier).

Example: python -m src.train_bert --domain restaurant --smoke
         python -m src.train_bert --domain restaurant --epochs 3 --batch-size 8

Uses group-disjoint validation split, 3-class polarity labels, and test gold only
for FINAL evaluation (never for checkpoint selection). After training, the model is
evaluated on every gold test set in --eval-domains (in-domain and cross-domain), and
per-example predictions are written to results/preds/.
"""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.model_selection import GroupShuffleSplit
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

from src.common import (LABEL_TO_ID, add_common_args, build_optimizer, compute_metrics,
                        evaluate_test_sets, experiment_suffix, save_result)

MODEL = 'bert-base-uncased'
MAX_LENGTH = 128
# Pad batches up to a multiple of this, so MPS only ever sees a few tensor shapes
# (every new shape gets its own cached kernels/buffers, which exhausted 8 GB of RAM).
PAD_MULTIPLE = 32
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
    # The same shuffle is used for every fraction, so smaller subsets are nested in larger ones.
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
        # Padding is done per batch by the collator.
        return tokenizer(batch['text'], batch['aspect'], truncation=True, max_length=MAX_LENGTH)
    encoded = raw.map(tokenize, batched=True, remove_columns=['text', 'aspect'])
    columns = [x for x in ['input_ids', 'attention_mask', 'token_type_ids', 'labels'] if x in encoded.column_names]
    encoded.set_format(columns=columns)
    return encoded


def main():
    parser = add_common_args(argparse.ArgumentParser())
    args = parser.parse_args()
    if not 0 < args.train_fraction <= 1:
        parser.error('--train-fraction must be in (0,1]')
    head_lr = args.head_lr if args.head_lr is not None else args.lr

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_rows, val_rows = split_by_sentence(read_jsonl(f'data/raw/{args.domain}/train.jsonl'))
    train_rows = choose_train_fraction(train_rows, args.train_fraction)
    if args.smoke:
        train_rows = train_rows[:64]
        val_rows = val_rows[:32]
    print(f'Final train={len(train_rows)}, validation={len(val_rows)}')

    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=3, id2label={v: k for k, v in LABEL_TO_ID.items()},
        label2id=LABEL_TO_ID,
    )
    train_ds = make_dataset(train_rows, tokenizer)
    val_ds = make_dataset(val_rows, tokenizer)

    suffix = experiment_suffix(args)
    output_dir = f'checkpoints/bert_{suffix}'
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1 if args.smoke else args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_only_model=True,
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
    trainer = Trainer(model=model, args=training_args,
                      train_dataset=train_ds, eval_dataset=val_ds,
                      processing_class=tokenizer, compute_metrics=compute_metrics,
                      data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=PAD_MULTIPLE),
                      optimizers=(build_optimizer(model, args.lr, head_lr), None))
    print('Actual Trainer device:', trainer.args.device)
    trainer.train()
    result = {'experiment': suffix, 'seed': args.seed, 'domain': args.domain,
              'train_fraction': args.train_fraction,
              'train_size': len(train_rows), 'validation_size': len(val_rows),
              'model': MODEL, 'model_type': 'BERT sentence-aspect pair, CLS',
              'lr': args.lr, 'head_lr': head_lr, 'epochs': args.epochs, 'batch_size': args.batch_size,
              'device': str(trainer.args.device),
              'validation_metrics': trainer.evaluate(val_ds)}

    # Never use the official gold test to choose hyperparameters or checkpoints.
    if not args.smoke:
        evaluate_test_sets(trainer, args, 'bert', suffix,
                           lambda rows: (make_dataset(rows, tokenizer), {}), read_jsonl, result)
    save_result(result, trainer, 'bert', suffix, output_dir, args)
    print(json.dumps({k: v for k, v in result.items() if k != 'log_history'}, indent=2, default=float))


if __name__ == '__main__':
    main()
