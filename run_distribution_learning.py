import argparse
import os
import random

import torch

from scheme.distribution_learning import DistributionLearningScheme
from model import RecurrentNetwork
from util import load_strings
from vocab import load_vocab

import neptune

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="distribution_learning", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--vocab_name", type=str, default="zinc")
    parser.add_argument("--strings_path", type=str, default="./resource/data/zinc/smis.txt")
    parser.add_argument("--max_smiles_length", type=int, default=80)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--lstm_dropout", type=float, default=0.2)

    # Training parameters
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    # Evaluation parameters
    parser.add_argument("--eval_freq", type=int, default=1000)
    parser.add_argument("--eval_batch_size", type=int, default=256)

    # Directory to save the pretrained model
    parser.add_argument("--save_dir", default="./resource/checkpoint/zinc/")

    # Directory to save the pretrained model
    parser.add_argument("--disable_neptune", action="store_true")

    args = parser.parse_args()

    device = torch.device(0)
    random.seed(0)

    vocab = load_vocab(name=args.vocab_name)
    train_strings, vali_strings = load_strings(strings_path=args.strings_path, split="random")

    model = RecurrentNetwork(vocab_size=len(vocab), hidden_size=args.hidden_size, num_layers=args.num_layers,)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheme = DistributionLearningScheme()

    if not args.disable_neptune:
        neptune.init(project_qualified_name="sungsoo.ahn/real-mol-opt")
        neptune.create_experiment(name="distribution_learning", params=vars(args))

    scheme.run(
        model=model,
        optimizer=optimizer,
        vocab=vocab,
        train_strings=train_strings,
        vali_strings=vali_strings,
        num_steps=args.num_steps,
        train_batch_size=args.train_batch_size,
        eval_freq=args.eval_freq,
        eval_batch_size=args.eval_batch_size,
        device=device,
        save_dir=args.save_dir,
        disable_neptune=args.disable_neptune,
    )
