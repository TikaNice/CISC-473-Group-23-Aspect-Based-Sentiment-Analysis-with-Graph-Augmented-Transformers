import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset

datasets_to_download = {
    "restaurant": "tomaarsen/setfit-absa-semeval-restaurants",
    "laptop": "tomaarsen/setfit-absa-semeval-laptops",
}

for domain, dataset_id in datasets_to_download.items():
    print(f"\nDownloading {domain}...")
    dataset = load_dataset(dataset_id)

    output_dir = Path("data/raw") / domain
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "test"):
        rows = dataset[split]
        output_path = output_dir / f"{split}.jsonl"

        with output_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        counts = Counter(rows["label"])
        print(f"{domain}/{split}: {len(rows)} samples")
        print(f"Labels: {dict(counts)}")
        print(f"Saved to: {output_path}")

print("\nDownload complete.")
