"""Build phrase-clique adjacencies from frozen Stanza constituency parses.

Install stanza==1.10.1 and download the English processors listed in PARSER
plus tokenize=combined with stanza.download. Run python -m src.constituency_graph
to cache the raw training corpus before starting the CUDA constituency shard.
Mount the downloaded stanza_resources directory read-only and point
STANZA_RESOURCES_DIR at it for final-evaluation parsing in the container.
"""
import json
import os
from fcntl import LOCK_EX, flock
from pathlib import Path

import numpy as np

from src.result_paths import DOMAINS, write_json

PARSER = {
    'name': 'stanza', 'version': '1.10.1', 'resources_version': '1.10.0',
    'constituency': 'ptb3-revised_charlm', 'pos': 'combined_charlm',
    'graph_rule': 'nonclausal_phrase_cliques_2_to_5_v1',
}
CACHE = Path('data/processed/constituency_stanza_1.10.1')


def phrase_spans(tree):
    """Index yields in leaf order and keep short nonclausal constituents."""
    spans = set()

    def visit(node, start):
        if not node.children:
            return start + 1
        end = start
        for child in node.children:
            end = visit(child, end)
        if node.label not in {'ROOT', 'TOP', 'S', 'SBAR', 'SBARQ', 'SINV', 'SQ', 'FRAG'}:
            if 2 <= end - start <= 5:
                spans.add((start, end))
        return end

    visit(tree, 0)
    return sorted(spans)


def adjacency(spans, lookup):
    matrix = np.eye(len(lookup), dtype=np.float32)
    for start, end in spans:
        indices = [lookup[index] for index in range(start, end) if index in lookup]
        matrix[np.ix_(indices, indices)] = 1
    return matrix


def parse_texts(texts, nlp):
    import stanza
    import torch
    from stanza.models.common.doc import Document

    if stanza.__version__ != PARSER['version']:
        raise ValueError(f"Expected Stanza {PARSER['version']}")
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '2')))
    pipeline = stanza.Pipeline(
        'en', processors={'tokenize': 'combined', 'pos': PARSER['pos'],
                          'constituency': PARSER['constituency']},
        package=None, tokenize_pretokenized=True, download_method=None,
        dir=os.environ.get('STANZA_RESOURCES_DIR', str(Path.home() / 'stanza_resources')),
        use_gpu=False, verbose=False,
    )
    texts = list(dict.fromkeys(texts))
    parsed = {}
    for offset in range(0, len(texts), 64):
        batch = texts[offset:offset + 64]
        tokens = [[token.text for token in nlp.tokenizer(text)] for text in batch]
        docs = pipeline.bulk_process([Document([], text=[words]) for words in tokens])
        for text, words, doc in zip(batch, tokens, docs, strict=True):
            sentence, = doc.sentences
            tree = sentence.constituency
            if [word.text for word in sentence.words] != words or len(tree.leaf_labels()) != len(words):
                raise ValueError(f'Stanza token alignment failed for {text!r}')
            parsed[text] = phrase_spans(tree)
    return parsed


def read_cache(path):
    contents = json.loads(path.read_text(encoding='utf-8'))
    if contents['parser'] != PARSER:
        raise ValueError(f'Unexpected parser metadata in {path}')
    return contents['spans']


def load_cache(domain):
    return read_cache(CACHE / f'{domain}.json')


def final_evaluation_spans(rows, domain, nlp):
    """Cache unlabeled test parses on first use, during final evaluation."""
    path = Path('checkpoints/constituency_cache') / f'{domain}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        flock(lock, LOCK_EX)
        if not path.exists():
            write_json(path, {'parser': PARSER,
                              'spans': parse_texts((row['text'] for row in rows), nlp)})
        return read_cache(path)


def main():
    import spacy
    from src.train_bert import read_jsonl

    nlp = spacy.load('en_core_web_sm', disable=['ner'])
    for domain in DOMAINS:
        rows = read_jsonl(f'data/raw/{domain}/train.jsonl')
        spans = parse_texts((row['text'] for row in rows), nlp)
        write_json(CACHE / f'{domain}.json', {'parser': PARSER, 'spans': spans})
        print(f'Saved {len(spans)} training-corpus parses for {domain}', flush=True)


if __name__ == '__main__':
    main()
