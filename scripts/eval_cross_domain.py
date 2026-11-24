"""Restaurant-trained BERT/GCN -> Laptop Gold Test, inference only.

Usage:
  python -m scripts.eval_cross_domain --model all --seeds 42 43 44

Uses saved Trainer BEST validation checkpoint; never trains on Laptop.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import spacy
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from src.train_bert import MODEL, read_jsonl, make_dataset
from src.result_paths import checkpoint_dir
from src.train_gcn import encode_row, collate
from src.model_gcn import BertDependencyGCN
from datasets import Dataset


def find_best_checkpoint(model_name, seed, run_tag=""):
    root = checkpoint_dir(model_name, 'restaurant', 1, seed, run_tag)
    if not root.is_dir():
        raise FileNotFoundError(f'Missing {root}. Run original Restaurant training on this Mac first.')
    folders = sorted(root.glob('checkpoint-*'),
                     key=lambda p: int(p.name.split('-')[-1]) if p.name.split('-')[-1].isdigit() else -1)
    folders = [p for p in folders if p.is_dir()]
    if not folders:
        raise FileNotFoundError(f'No checkpoint-* folders under {root}. JSON metrics alone do not contain model weights.')
    # Hugging Face trainer_state.json records best_model_checkpoint. Do NOT assume highest step is best.
    for folder in reversed(folders):
        state_file = folder / 'trainer_state.json'
        if not state_file.is_file():
            continue
        info = json.loads(state_file.read_text(encoding='utf-8'))
        best = info.get('best_model_checkpoint')
        if best:
            p = Path(best)
            for candidate in (root / p.name,):
                if candidate.is_dir():
                    return candidate
    if len(folders) == 1:
        return folders[0]
    raise RuntimeError(f'Cannot identify best validation checkpoint in {root}; available: {folders}. '
                       'Please inspect trainer_state.json instead of selecting the last checkpoint arbitrarily.')


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

    def target_path(model, seed):
        tag = '_' + args.gcn_run_tag if model == 'gcn' and args.gcn_run_tag else ''
        return Path('results/cross_domain/restaurant_to_laptop') / model / f'{model}_restaurant_to_laptop_seed{seed}{tag}.json'

    models = ['bert', 'gcn'] if args.model == 'all' else [args.model]
    if not args.overwrite and all(
        target_path(model, seed).is_file()
        for model in models for seed in args.seeds
    ):
        print('All requested cross-domain results exist; skipping inference (--overwrite to rerun).')
        return

    gold = Path('data/processed/laptop/test_gold.jsonl')
    if not gold.is_file():
        raise FileNotFoundError(f'{gold} missing. First run python scripts/prepare_gold.py')
    gold_rows = read_jsonl(gold)
    if len(gold_rows) != 638:
        print(f'WARNING: Laptop three-class Gold Test count is {len(gold_rows)}, expected 638')

    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    models = ['bert', 'gcn'] if args.model == 'all' else [args.model]
    output_dir = Path('results/cross_domain')
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_cache = {}

    for model_name in models:
        if model_name == 'bert':
            dataset_cache[model_name] = make_dataset(gold_rows, tokenizer)
        else:
            nlp = spacy.load('en_core_web_sm', disable=['ner'])
            encoded = [encode_row(r, tokenizer, nlp) for r in gold_rows]
            dataset_cache[model_name] = Dataset.from_list(encoded)

        for seed in args.seeds:
            target = target_path(model_name, seed)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and not args.overwrite:
                print(f'Skipping existing {target} (--overwrite to rerun)')
                continue
            checkpoint = find_best_checkpoint(model_name, seed, args.gcn_run_tag if model_name == 'gcn' else '')
            print(f'\nEvaluating {model_name} seed={seed}: Restaurant checkpoint {checkpoint} -> Laptop Gold Test')
            if model_name == 'bert':
                model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint))
            else:
                model = load_gcn_weights(BertDependencyGCN(model_name=MODEL), checkpoint)

            opts = TrainingArguments(
                output_dir=str(output_dir / f'_eval_{model_name}_seed{seed}'),
                per_device_eval_batch_size=args.batch_size,
                report_to='none', remove_unused_columns=(model_name == 'bert'),
                dataloader_num_workers=0, dataloader_pin_memory=False,
                fp16=False, bf16=False,
            )
            trainer = Trainer(
                model=model, args=opts,
                data_collator=collate if model_name == 'gcn' else None,
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
                'run_tag': args.gcn_run_tag if model_name == 'gcn' else '',
                'aspect_alignment': 'case_sensitive_ordinal_v1' if model_name == 'gcn' and args.gcn_run_tag else 'historical',
                'seed': seed,
                'checkpoint': str(checkpoint),
                'device': str(trainer.args.device),
                'test_size': len(truth),
                'test_accuracy': float(accuracy_score(truth, guessed)),
                'test_macro_f1': float(f1_score(truth, guessed, labels=[0, 1, 2], average='macro', zero_division=0)),
                'note': 'Zero-shot cross-domain evaluation only: no Laptop training or checkpoint selection.',
            }
            target.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
            print(json.dumps(result, indent=2, ensure_ascii=False))
            print('Saved:', target)


if __name__ == '__main__':
    main()
