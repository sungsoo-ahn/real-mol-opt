import argparse
import os
import random

import torch

from scheme.hill_climb import HillClimbScheme
from model import RecurrentNetwork
from util import load_strings
from vocab import load_vocab
from chemistry.score import load_score_func

import neptune

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="distribution_learning", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--vocab_name", type=str, default="zinc")
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=3)

    parser.add_argument("--objective", type=str, default="logp")

    # Training parameters
    parser.add_argument("--pretrained_model_path", type=str, default="result/distribution_learning/checkpoint_best.pt")
    parser.add_argument("--num_stages", type=int, default=20)
    parser.add_argument("--queue_size", type=int, default=1024)
    parser.add_argument("--num_steps_per_stage", type=int, default=8)
    parser.add_argument("--optimize_batch_size", type=int, default=256)
    parser.add_argument("--generate_size", type=int, default=8196)
    parser.add_argument("--generate_batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--kl_div_coef", type=float, default=0.0)

    parser.add_argument("--disable_neptune", action="store_true")

    args = parser.parse_args()

    #args.disable_neptune = True

    device = torch.device(0)
    random.seed(0)

    vocab = load_vocab(name=args.vocab_name)
    obj_func = load_score_func(name=args.objective)

    model = RecurrentNetwork(vocab_size=len(vocab), hidden_size=args.hidden_size, num_layers=args.num_layers)
    model.load_state_dict(torch.load(args.pretrained_model_path))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheme = HillClimbScheme()

    if not args.disable_neptune:
        neptune.init(project_qualified_name="sungsoo.ahn/real-mol-opt")
        experiment = neptune.create_experiment(name="hill_climb", params=vars(args))
        result_dir = f"./result/tmp/{experiment.id}"
    else:
        result_dir = "./result/tmp/disable_neptune"

    scheme.run(
        model=model,
        optimizer=optimizer,
        vocab=vocab,
        obj_func=obj_func,
        num_stages=args.num_stages,
        queue_size=args.queue_size,
        num_steps_per_stage=args.num_steps_per_stage,
        optimize_batch_size=args.optimize_batch_size,
        kl_div_coef=args.kl_div_coef,
        generate_size=args.generate_size,
        generate_batch_size=args.generate_batch_size,
        device=device,
        result_dir=result_dir,
        disable_neptune=args.disable_neptune,
    )
