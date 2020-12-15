import argparse
import torch
from model import RecurrentNetwork, sample_smis
from vocab import load_vocab
from util import save_strings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="sample_model", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--vocab_name", type=str, default="zinc")
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=3)

    parser.add_argument("--model_path", type=str, default="./result/tmp/RMO-76/checkpoint_000.pt")
    parser.add_argument("--smis_path", type=str, default="./result/tmp/RMO-76/generated_smis_000.txt")
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device(0)

    print(args.model_path)

    vocab = load_vocab(name=args.vocab_name)

    model = RecurrentNetwork(vocab_size=len(vocab), hidden_size=args.hidden_size, num_layers=args.num_layers)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)

    all_smis = []
    while len(all_smis) < args.num_samples:
        smis = sample_smis(model, vocab, device, args.batch_size)
        all_smis += [smi for smi in smis if smi is not None]
        all_smis = list(set(all_smis))

    save_strings(args.smis_path, all_smis)

"""
python run_sample_model.py --model_path ./result/hill_climb/checkpoint_003.pt --smis_path ./result/hill_climb/generated_smis_003.txt;
python run_sample_model.py --model_path ./result/hill_climb/checkpoint_007.pt --smis_path ./result/hill_climb/generated_smis_007.txt;
python run_sample_model.py --model_path ./result/hill_climb/checkpoint_011.pt --smis_path ./result/hill_climb/generated_smis_011.txt;
python run_sample_model.py --model_path ./result/hill_climb/checkpoint_015.pt --smis_path ./result/hill_climb/generated_smis_015.txt;
python run_sample_model.py --model_path ./result/hill_climb/checkpoint_019.pt --smis_path ./result/hill_climb/generated_smis_019.txt
"""



