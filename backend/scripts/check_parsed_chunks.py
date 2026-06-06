import os, json, sys
from pathlib import Path

data_dir = Path(__file__).parent.parent / "data" / "parsed"
files = list(data_dir.glob("*.json"))
print(f"Total parsed JSON files: {len(files)}")

valid = 0
invalid = 0
empty_chunks = 0
total_chunks = 0
errors = []

for f in files[:100]:
    try:
        with open(f, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            invalid += 1
            errors.append(f"{f.name}: not a list")
            continue
        if len(data) == 0:
            empty_chunks += 1
        total_chunks += len(data)
        for chunk in data:
            if not isinstance(chunk, dict):
                invalid += 1
                errors.append(f"{f.name}: chunk not dict")
                break
            if 'content' not in chunk or 'doc_id' not in chunk:
                invalid += 1
                errors.append(f"{f.name}: missing content/doc_id")
                break
        else:
            valid += 1
    except Exception as e:
        invalid += 1
        errors.append(f"{f.name}: {e}")

print(f"Sampled: {valid} valid, {invalid} invalid, {empty_chunks} empty")
print(f"Total chunks in sample: {total_chunks}")
if errors:
    print("Errors:")
    for e in errors[:10]:
        print(f"  - {e}")
