"""Train BERT + dependency GCN. Run: python -m src.train_gcn --domain restaurant --smoke"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import spacy
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, Trainer, TrainingArguments

from src.model_gcn import BertDependencyGCN
from src.train_bert import (
    MODEL, LABEL_TO_ID, DATA_SPLIT_SEED, read_jsonl,
    split_by_sentence, choose_train_fraction,
)

MAX_LENGTH = 128


def find_aspect(text, span, row):
    # Prefer verified XML offsets when available.
    start, end = row.get("start"), row.get("end")
    if isinstance(start, int) and isinstance(end, int):
        if text[start:end] == span:
            return start, end
        raise ValueError("Invalid provided aspect offsets")

    # SetFit ordinal: zero-based occurrence index.
    if "ordinal" not in row:
        raise ValueError("Missing aspect ordinal")

    ordinal = int(row["ordinal"])
    if ordinal < 0:
        raise ValueError("Negative aspect ordinal")

    positions = []
    pos = 0
    lowered = text.lower()
    target = span.lower()

    while True:
        found = lowered.find(target, pos)
        if found == -1:
            break
        positions.append(found)
        pos = found + len(span)

    if ordinal >= len(positions):
        raise ValueError(
            f"Invalid ordinal {ordinal}: "
            f"{span!r} has {len(positions)} matches"
        )

    start = positions[ordinal]
    return start, start + len(span)


def encode_row(row, tokenizer, nlp):
    text, span = row['text'], row['span']
    start, end = find_aspect(text, span, row)
    doc = nlp(text)
    words = [t.text for t in doc]
    # Pre-tokenized sentence + aspect preserves sentence-level word_ids.
    encoded = tokenizer(
        words, [span], is_split_into_words=True, truncation=True,
        max_length=MAX_LENGTH, padding='max_length',
    )
    ids = encoded.word_ids(batch_index=0)
    type_ids = encoded.get('token_type_ids', [0] * len(ids))
    first_ids = [wid if wid is not None and type_ids[i] == 0 else -1
                 for i, wid in enumerate(ids)]
    kept = sorted(set(wid for wid in first_ids if wid >= 0))
    lookup = {old: new for new, old in enumerate(kept)}
    mapped = [lookup.get(wid, -1) for wid in first_ids]
    n = len(kept)
    if not n:
        raise ValueError('No sentence tokens after truncation')
    A = np.eye(n, dtype=np.float32)
    for t in doc:
        if t.i in lookup and t.head.i in lookup:
            i, j = lookup[t.i], lookup[t.head.i]
            A[i, j] = A[j, i] = 1.0
    degree = A.sum(axis=1)
    scale = np.power(np.maximum(degree, 1), -0.5)
    A *= scale[:, None] * scale[None, :]
    aspect_mask = [int(doc[i].idx < end and doc[i].idx + len(doc[i]) > start)
                   for i in kept]
    if not any(aspect_mask):
        raise ValueError(f'Aspect truncated or unaligned: {span!r} in {text!r}')
    return {
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask'],
        'token_type_ids': type_ids,
        'word_ids': mapped, 'adjacency': A.tolist(),
        'aspect_mask': aspect_mask,
        'labels': LABEL_TO_ID[row['label']],
    }


def collate(features):
    b, max_words = len(features), max(len(x['aspect_mask']) for x in features)
    A = torch.zeros(b, max_words, max_words, dtype=torch.float32)
    mask = torch.zeros(b, max_words, dtype=torch.float32)
    for i, x in enumerate(features):
        n = len(x['aspect_mask'])
        A[i, :n, :n] = torch.tensor(x['adjacency'], dtype=torch.float32)
        mask[i, :n] = torch.tensor(x['aspect_mask'], dtype=torch.float32)
    return {
        'input_ids': torch.tensor([x['input_ids'] for x in features], dtype=torch.long),
        'attention_mask': torch.tensor([x['attention_mask'] for x in features], dtype=torch.long),
        'token_type_ids': torch.tensor([x['token_type_ids'] for x in features], dtype=torch.long),
        'word_ids': torch.tensor([x['word_ids'] for x in features], dtype=torch.long),
        'adjacency': A, 'aspect_mask': mask,
        'labels': torch.tensor([x['labels'] for x in features], dtype=torch.long),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--domain', choices=['restaurant', 'laptop'], default='restaurant')
    p.add_argument('--epochs', type=float, default=3)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--train-fraction', type=float, default=1.)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if not 0 < args.train_fraction <= 1:
        p.error('--train-fraction must be in (0,1]')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    tr, val = split_by_sentence(read_jsonl(f'data/raw/{args.domain}/train.jsonl'))
    tr = choose_train_fraction(tr, args.train_fraction)
    if args.smoke:
        tr, val = tr[:64], val[:32]
    print(f'Train: {len(tr)}, validation: {len(val)}')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    nlp = spacy.load('en_core_web_sm', disable=['ner'])
    def ds(rows):
        encoded = []
        skipped = []

        for row in rows:
            try:
                encoded.append(encode_row(row, tokenizer, nlp))
            except ValueError as exc:
                skipped.append({
                    "text": row["text"],
                    "span": row["span"],
                    "reason": str(exc),
                })

        print(
            f"Encoded {len(encoded)}/{len(rows)} records; "
            f"skipped {len(skipped)} ambiguous or unaligned records"
        )

        if skipped:
            audit_dir = Path("results/audit")
            audit_dir.mkdir(parents=True, exist_ok=True)
            audit_file = audit_dir / f"gcn_{args.domain}_skipped.jsonl"
            with audit_file.open("a", encoding="utf-8") as f:
                for record in skipped:
                    f.write(json.dumps(record, ensure_ascii=False) + "\\n")
            print("Skipped-record audit:", audit_file)

        if not encoded:
            raise ValueError("No valid records remain after encoding")

        return Dataset.from_list(encoded)
    train_ds, val_ds = ds(tr), ds(val)
    model = BertDependencyGCN(model_name=MODEL)
    suffix = f'{args.domain}_fraction{args.train_fraction:g}_seed{args.seed}' + ('_smoke' if args.smoke else '')
    opts = TrainingArguments(
        output_dir=f'checkpoints/gcn_{suffix}',
        num_train_epochs=1 if args.smoke else args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=2e-5, weight_decay=0.01,
        eval_strategy='epoch', save_strategy='epoch',
        load_best_model_at_end=True, metric_for_best_model='macro_f1',
        greater_is_better=True, save_total_limit=1,
        seed=args.seed, data_seed=DATA_SPLIT_SEED,
        report_to='none', dataloader_num_workers=0,
        dataloader_pin_memory=False, fp16=False, bf16=False,
        remove_unused_columns=False, logging_steps=10,
    )
    def metrics(pred):
        guesses = np.asarray(pred.predictions).argmax(-1)
        truth = np.asarray(pred.label_ids)
        return {'accuracy': accuracy_score(truth, guesses),
                'macro_f1': f1_score(truth, guesses, labels=[0,1,2], average='macro', zero_division=0)}
    trainer = Trainer(model=model, args=opts, train_dataset=train_ds,
                      eval_dataset=val_ds, data_collator=collate,
                      compute_metrics=metrics, processing_class=tokenizer)
    print('Trainer device:', trainer.args.device)
    trainer.train()
    result = {
        'experiment': suffix, 'model_type': 'BERT pair + 2-layer dependency GCN + aspect pooling',
        'device': str(trainer.args.device), 'seed': args.seed,
        'train_size': len(tr), 'validation_size': len(val),
        'validation_metrics': trainer.evaluate(val_ds),
    }
    if not args.smoke:
        gold = Path(f'data/processed/{args.domain}/test_gold.jsonl')
        if gold.is_file():
            test_rows = read_jsonl(gold)
            result['test_size'] = len(test_rows)
            result['final_test_metrics'] = trainer.evaluate(ds(test_rows), metric_key_prefix='test')
        else:
            print(f'WARNING: missing gold data {gold}; skipping test')
    out = Path(f'results/gcn_{suffix}.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=float), encoding='utf-8')
    print('Saved:', out)
    print(json.dumps(result, indent=2, default=float))

if __name__ == '__main__':
    main()
