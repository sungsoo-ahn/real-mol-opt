import random
import argparse
import json
from util import load_strings
from chemistry.score import load_score_func

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="eval_quality", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--smiles_path", type=str, default="./resource/data/zinc/smis.txt")
    parser.add_argument("--result_path", type=str, default="./result/quality.json")
    parser.add_argument("--metrics", type=str, nargs="+", default=["sa", "qed", "rd_filter", "logp"])
    parser.add_argument("--max_num_samples", type=int, default=1000)
    args = parser.parse_args()

    strings = load_strings(args.smiles_path, split="full")
    if len(strings) > args.max_num_samples:
        strings = random.sample(strings, k=args.max_num_samples)

    result = {"smiles": strings}
    for metric in args.metrics:
        score_func = load_score_func(metric)
        result[metric] = [score_func(string) for string in strings]

    with open(args.result_path, "w") as f:
        json.dump(result, f)

"""
python run_quality_check.py --smiles_path ./result/hill_climb/generated_smis_003.txt --result_path ./result/hill_climb/result_003.json;
python run_quality_check.py --smiles_path ./result/hill_climb/generated_smis_007.txt --result_path ./result/hill_climb/result_007.json;
python run_quality_check.py --smiles_path ./result/hill_climb/generated_smis_011.txt --result_path ./result/hill_climb/result_011.json;
python run_quality_check.py --smiles_path ./result/hill_climb/generated_smis_015.txt --result_path ./result/hill_climb/result_015.json;
python run_quality_check.py --smiles_path ./result/hill_climb/generated_smis_019.txt --result_path ./result/hill_climb/result_019.json
"""