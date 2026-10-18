"""Train BERT + dependency GCN. Run: python -m src.train_gcn --domain restaurant --smoke

Training/validation records whose aspect cannot be located are skipped (and logged to
results/audit/). Test records are NEVER skipped, so BERT and BERT+GCN are scored on the
same examples: if the aspect cannot be aligned, the record is kept with an empty aspect
mask (the classifier then relies on [CLS] alone) and counted in `n_aspect_unaligned`.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import spacy
import torch
from datasets import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments

from src.common import (LABEL_TO_ID, add_common_args, build_optimizer, compute_metrics,
                        evaluate_test_sets, experiment_suffix, save_result)
from src.model_gcn import BertDependencyGCN
from src.train_bert import (
    MODEL, MAX_LENGTH, PAD_MULTIPLE, DATA_SPLIT_SEED, read_jsonl,
    split_by_sentence, choose_train_fraction,
)


def find_aspect(text, span, row):
    # Prefer verified XML offsets when available; fall back to the ordinal if they don't match.
    start, end = row.get("start"), row.get("end")
    if isinstance(start, int) and isinstance(end, int) and start >= 0:
        if text[start:end] == span:
            return start, end

    # Zero-based occurrence index (SetFit rows, or XML rows with mismatched offsets).
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


def encode_row(row, doc, tokenizer, strict=True):
    """Encode one record. With strict=False, alignment failures give an empty aspect mask
    instead of raising; the returned flag says whether the aspect was aligned."""
    text, span = row['text'], row['span']
    try:
        start, end = find_aspect(text, span, row)
    except ValueError:
        if strict:
            raise
        start, end = -1, -1
    words = [t.text for t in doc]
    # Pre-tokenized sentence + aspect preserves sentence-level word_ids.
    encoded = tokenizer(words, [span], is_split_into_words=True, truncation=True,
                        max_length=MAX_LENGTH)
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
    aligned = any(aspect_mask)
    if strict and not aligned:
        raise ValueError(f'Aspect truncated or unaligned: {span!r} in {text!r}')
    return {
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask'],
        'token_type_ids': type_ids,
        'word_ids': mapped, 'adjacency': A.tolist(),
        'aspect_mask': aspect_mask,
        'labels': LABEL_TO_ID[row['label']],
    }, aligned


def round_up(n, multiple):
    return -(-n // multiple) * multiple


def collate(features):
    b = len(features)
    # Round padded sizes up to fixed buckets so MPS sees only a handful of tensor shapes.
    max_words = round_up(max(len(x['aspect_mask']) for x in features), PAD_MULTIPLE // 2)
    max_len = min(round_up(max(len(x['input_ids']) for x in features), PAD_MULTIPLE), MAX_LENGTH)
    A = torch.zeros(b, max_words, max_words, dtype=torch.float32)
    mask = torch.zeros(b, max_words, dtype=torch.float32)
    input_ids = torch.zeros(b, max_len, dtype=torch.long)  # [PAD] id is 0 for bert-base-uncased
    attention = torch.zeros(b, max_len, dtype=torch.long)
    type_ids = torch.zeros(b, max_len, dtype=torch.long)
    word_ids = torch.full((b, max_len), -1, dtype=torch.long)
    for i, x in enumerate(features):
        n, length = len(x['aspect_mask']), len(x['input_ids'])
        A[i, :n, :n] = torch.tensor(x['adjacency'], dtype=torch.float32)
        mask[i, :n] = torch.tensor(x['aspect_mask'], dtype=torch.float32)
        input_ids[i, :length] = torch.tensor(x['input_ids'])
        attention[i, :length] = torch.tensor(x['attention_mask'])
        type_ids[i, :length] = torch.tensor(x['token_type_ids'])
        word_ids[i, :length] = torch.tensor(x['word_ids'])
    return {
        'input_ids': input_ids, 'attention_mask': attention, 'token_type_ids': type_ids,
        'word_ids': word_ids, 'adjacency': A, 'aspect_mask': mask,
        'labels': torch.tensor([x['labels'] for x in features], dtype=torch.long),
    }


def main():
    parser = add_common_args(argparse.ArgumentParser())
    args = parser.parse_args()
    if not 0 < args.train_fraction <= 1:
        parser.error('--train-fraction must be in (0,1]')
    head_lr = args.head_lr if args.head_lr is not None else args.lr
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    tr, val = split_by_sentence(read_jsonl(f'data/raw/{args.domain}/train.jsonl'))
    tr = choose_train_fraction(tr, args.train_fraction)
    if args.smoke:
        tr, val = tr[:64], val[:32]
    print(f'Train: {len(tr)}, validation: {len(val)}')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    nlp = spacy.load('en_core_web_sm', disable=['ner'])
    suffix = experiment_suffix(args)

    def ds(rows, split, strict=True):
        """Encode rows. Returns (dataset, kept_rows, stats)."""
        encoded, kept, skipped, unaligned = [], [], [], 0
        docs = nlp.pipe([r['text'] for r in rows], batch_size=64)
        for row, doc in zip(rows, docs):
            try:
                features, aligned = encode_row(row, doc, tokenizer, strict=strict)
            except ValueError as exc:
                skipped.append({"text": row["text"], "span": row["span"], "reason": str(exc)})
                continue
            encoded.append(features)
            kept.append(row)
            unaligned += not aligned
        print(f"[{split}] encoded {len(encoded)}/{len(rows)} records; "
              f"skipped {len(skipped)}; kept with unaligned aspect {unaligned}")
        if skipped:
            audit_file = Path(args.results_dir) / "audit" / f"gcn_{suffix}_{split}_skipped.jsonl"
            audit_file.parent.mkdir(parents=True, exist_ok=True)
            with audit_file.open("w", encoding="utf-8") as f:
                for record in skipped:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print("Skipped-record audit:", audit_file)
        if not encoded:
            raise ValueError("No valid records remain after encoding")
        stats = {'n_input': len(rows), 'n_skipped': len(skipped), 'n_aspect_unaligned': unaligned}
        return Dataset.from_list(encoded), kept, stats

    train_ds, _, train_stats = ds(tr, 'train')
    val_ds, _, val_stats = ds(val, 'validation')
    model = BertDependencyGCN(model_name=MODEL)
    output_dir = f'checkpoints/gcn_{suffix}'
    opts = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1 if args.smoke else args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr, weight_decay=0.01,
        eval_strategy='epoch', save_strategy='epoch', save_only_model=True,
        load_best_model_at_end=True, metric_for_best_model='macro_f1',
        greater_is_better=True, save_total_limit=1,
        seed=args.seed, data_seed=DATA_SPLIT_SEED,
        report_to='none', dataloader_num_workers=0,
        dataloader_pin_memory=False, fp16=False, bf16=False,
        remove_unused_columns=False, logging_steps=10,
    )
    trainer = Trainer(model=model, args=opts, train_dataset=train_ds,
                      eval_dataset=val_ds, data_collator=collate,
                      compute_metrics=compute_metrics, processing_class=tokenizer,
                      optimizers=(build_optimizer(model, args.lr, head_lr), None))
    print('Trainer device:', trainer.args.device)
    trainer.train()
    result = {
        'experiment': suffix, 'model_type': 'BERT pair + 2-layer dependency GCN + aspect pooling',
        'device': str(trainer.args.device), 'seed': args.seed, 'domain': args.domain,
        'train_fraction': args.train_fraction,
        'lr': args.lr, 'head_lr': head_lr, 'epochs': args.epochs, 'batch_size': args.batch_size,
        'train_size': len(train_ds), 'validation_size': len(val_ds),
        'train_encoding': train_stats, 'validation_encoding': val_stats,
        'validation_metrics': trainer.evaluate(val_ds),
    }

    def encode_test(rows):
        dataset, kept, stats = ds(rows, 'test', strict=False)
        if len(kept) != len(rows):
            raise RuntimeError(f'{stats["n_skipped"]} test records could not be encoded; '
                               'test sets must be scored in full')
        return dataset, stats

    if not args.smoke:
        evaluate_test_sets(trainer, args, 'gcn', suffix, encode_test, read_jsonl, result)
    save_result(result, trainer, 'gcn', suffix, output_dir, args)
    print(json.dumps({k: v for k, v in result.items() if k != 'log_history'}, indent=2, default=float))


if __name__ == '__main__':
    main()
