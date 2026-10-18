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

# The SetFit test splits above are unlabelled, so the official gold test XML is
# downloaded separately; run scripts/prepare_gold.py afterwards to convert it.
from huggingface_hub import hf_hub_download

gold_dir = Path("data/gold")
gold_dir.mkdir(parents=True, exist_ok=True)
for filename in ("Restaurants_Test_Gold.xml", "Laptops_Test_Gold.xml"):
    cached = hf_hub_download(
        "alexcadillon/SemEval2014Task4",
        f"ABSA_Gold_TestData/{filename}",
        repo_type="dataset",
    )
    (gold_dir / filename).write_bytes(Path(cached).read_bytes())
    print(f"Saved gold test XML to: {gold_dir / filename}")
