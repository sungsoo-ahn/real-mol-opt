import random

def load_strings(strings_path, split="random"):
    with open(strings_path, "r") as f:
        strings = f.read().splitlines()

    if split == "random":
        random.shuffle(strings)
        split_idx = int(0.9*len(strings))
        train_strings = strings[:split_idx]
        vali_strings = strings[split_idx:]

        return train_strings, vali_strings