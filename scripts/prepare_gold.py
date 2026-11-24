"""Convert SemEval 2014 Task 4 gold test XML into aspect-sentiment JSONL.

Usage: python scripts/prepare_gold.py
Reads the two XML files in data/gold and writes data/processed/<domain>/test_gold.jsonl.
"""
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

FILES = {
    'restaurant': 'Restaurants_Test_Gold.xml',
    'laptop': 'Laptops_Test_Gold.xml',
}


def convert(xml_path, out_path):
    root = ET.parse(xml_path).getroot()
    rows = []
    for sentence in root.findall('.//sentence'):
        text = sentence.findtext('text')
        if not text:
            continue
        for aspect in sentence.findall('./aspectTerms/aspectTerm'):
            term = aspect.get('term', '')
            label = aspect.get('polarity', '')
            if not term or not label:
                continue
            char_start = int(aspect.get('from', '-1'))
            char_end = int(aspect.get('to', '-1'))
            # Ordinal is useful for repeated aspect mentions in the same sentence.
            candidates = [m.start() for m in re.finditer(re.escape(term), text, flags=re.IGNORECASE)]
            ordinal = sum(x < char_start for x in candidates) if candidates and char_start >= 0 else 0
            rows.append({
                'text': text,
                'span': term,
                'label': label,
                'ordinal': ordinal,
                'start': char_start,
                'end': char_end,
            })
    if len(rows) < 100:
        raise RuntimeError(f'Too few aspect-level examples ({len(rows)}); check {xml_path}')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'{xml_path} -> {out_path}: {len(rows)} aspect samples, labels={dict(Counter(x["label"] for x in rows))}')


if __name__ == '__main__':
    for domain, filename in FILES.items():
        convert(Path('data/gold') / filename, Path('data/processed') / domain / 'test_gold.jsonl')
