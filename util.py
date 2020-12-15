import json
import random

def load_strings(strings_path, split="full"):
    with open(strings_path, "r") as f:
        strings = f.read().splitlines()

    if split == "full":
        return strings

    elif split == "random":
        random.shuffle(strings)
        split_idx = int(0.9*len(strings))
        train_strings = strings[:split_idx]
        vali_strings = strings[split_idx:]

        return train_strings, vali_strings

def save_strings(strings_path, strings):
    with open(strings_path, "w") as f:
        f.write("\n".join(strings))

def load_json(record_path):
    with open(record_path, "r") as f:
        record = json.load(f)

    return record

def save_json(record, record_path):
    with open(record_path, "w") as f:
        json.dump(record, f)

