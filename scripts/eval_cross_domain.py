"""Restaurant-trained BERT/GCN -> Laptop Gold Test, inference only.

Usage:
  python -m scripts.eval_cross_domain --model all --seeds 42 43 44

Uses saved Trainer BEST validation checkpoint; never trains on Laptop.
"""
import argparse
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import spacy
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from src.train_bert import MODEL, MODEL_REVISION, read_jsonl, make_dataset
from src.batch_padding import collate_bert
from src.evidence import INVALID_EVIDENCE, validate_result
from src.result_paths import best_restaurant_checkpoint, result_path, transfer_path, write_json
from src.predictions import prediction_path, write_predictions
from src.train_gcn import encode_row, collate
from src.model_gcn import BertDependencyGCN
from datasets import Dataset


def load_gcn_weights(model, checkpoint):
    safetensor = checkpoint / 'model.safetensors'
    binfile = checkpoint / 'pytorch_model.bin'
    if safetensor.is_file():
        from safetensors.torch import load_file
        state = load_file(str(safetensor), device='cpu')
    elif binfile.is_file():
        # Only load checkpoints created by your own training process.
        state = torch.load(binfile, map_location='cpu', weights_only=True)
    else:
        raise FileNotFoundError(f'No model.safetensors or pytorch_model.bin in {checkpoint}')
    model.load_state_dict(state, strict=True)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['bert', 'gcn', 'all'], default='all')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--gcn-run-tag', default='aspectfix', help='Corrected GCN training tag; aspectfix prefix required')
    args = parser.parse_args()
    if args.model in ('gcn', 'all') and not args.gcn_run_tag.startswith('aspectfix'):
        parser.error('GCN evaluation requires a corrected aspectfix run; legacy GCN results were retired')

    models = ['bert', 'gcn'] if args.model == 'all' else [args.model]
    pending = {}
    for model in models:
        graph = 'dependency' if model == 'gcn' else 'none'
        tag = args.gcn_run_tag if model == 'gcn' else ''
        for seed in args.seeds:
            target = transfer_path(model, seed, tag)
            if not args.overwrite and target.is_file():
                try:
                    validate_result(model, 'restaurant', 1.0, seed, graph, 'laptop', tag)
                except INVALID_EVIDENCE as exc:
                    print(f'Redoing incomplete {target}: {exc}')
                else:
                    continue
            pending.setdefault(model, []).append((seed, tag, target))
    if not pending:
        print('All requested transfer results and predictions validate.')
        return

    gold = Path('data/processed/laptop/test_gold.jsonl')
    if not gold.is_file():
        raise FileNotFoundError(f'{gold} missing. First run python scripts/prepare_gold.py')
    gold_rows = read_jsonl(gold)
    if len(gold_rows) != 638:
        print(f'WARNING: Laptop three-class Gold Test count is {len(gold_rows)}, expected 638')

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION, use_fast=True)
    output_dir = Path('results/cross_domain')
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_cache = {}

    for model_name, runs in pending.items():
        if model_name == 'bert':
            dataset_cache[model_name] = make_dataset(gold_rows, tokenizer)
        else:
            nlp = spacy.load('en_core_web_sm', disable=['ner'])
            encoded = [encode_row(r, tokenizer, nlp) for r in gold_rows]
            dataset_cache[model_name] = Dataset.from_list(encoded)

        for seed, tag, target in runs:
            graph = 'dependency' if model_name == 'gcn' else 'none'
            validate_result(model_name, 'restaurant', 1.0, seed, graph, tag=tag)
            source = result_path(model_name, 'restaurant', 1, seed, tag)
            source_hash = sha256(source.read_bytes()).hexdigest()
            checkpoint = best_restaurant_checkpoint(model_name, seed, tag)
            print(f'\nEvaluating {model_name} seed={seed}: Restaurant checkpoint {checkpoint} -> Laptop Gold Test')
            if model_name == 'bert':
                model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint))
            else:
                model = load_gcn_weights(BertDependencyGCN(model_name=MODEL, revision=MODEL_REVISION), checkpoint)

            opts = TrainingArguments(
                output_dir=str(output_dir / f'_eval_{model_name}_seed{seed}'),
                per_device_eval_batch_size=args.batch_size,
                report_to='none', remove_unused_columns=(model_name == 'bert'),
                dataloader_num_workers=0, dataloader_pin_memory=False,
                fp16=False, bf16=False,
            )
            trainer = Trainer(
                model=model, args=opts,
                data_collator=collate if model_name == 'gcn' else collate_bert,
                processing_class=tokenizer,
            )
            pred = trainer.predict(dataset_cache[model_name])
            truth = np.asarray(pred.label_ids)
            logits = pred.predictions
            if isinstance(logits, tuple):
                logits = logits[0]
            guessed = np.asarray(logits).argmax(axis=-1)
            if len(truth) != len(gold_rows):
                raise AssertionError(f'Prediction count mismatch {len(truth)} vs {len(gold_rows)}')
            result = {
                'experiment': 'restaurant_to_laptop_zero_shot',
                'source_training_domain': 'restaurant',
                'target_test_domain': 'laptop',
                'model': model_name,
                'run_tag': tag,
                'aspect_alignment': 'case_sensitive_ordinal_v1' if model_name == 'gcn' and args.gcn_run_tag else 'historical',
                'seed': seed,
                'checkpoint': str(checkpoint),
                'source_result_sha256': source_hash,
                'device': str(trainer.args.device),
                'test_size': len(truth),
                'test_accuracy': float(accuracy_score(truth, guessed)),
                'test_macro_f1': float(f1_score(truth, guessed, labels=[0, 1, 2], average='macro', zero_division=0)),
                'note': 'Zero-shot cross-domain evaluation only: no Laptop training or checkpoint selection.',
            }
            # Write the per-example predictions BEFORE the result record: the
            # result JSON is the resume marker, so the reverse order can leave a
            # finished record with no predictions and skip them forever.
            ppath = prediction_path(model_name, 'restaurant', 1, seed,
                                    tag, 'laptop')
            write_predictions(ppath, gold_rows, guessed, model_name, 'restaurant', 'laptop',
                              1, seed, tag,
                              graph=graph,
                              device=str(trainer.args.device))
            write_json(target, result)
            print('Saved predictions:', ppath)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            print('Saved:', target)


if __name__ == '__main__':
    main()
