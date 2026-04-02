"""
main.py — Entry point for LA-TQD.

Usage:
    python main.py --model Qwen/Qwen2.5-7B-Instruct --rounds 30 --runs 5
    python main.py --smoke-test
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from src.config import DATA_DIR, RESULTS_DIR, N_ROUNDS, N_RUNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LA-TQD: Quality-Diversity for RGD Peptide Design")
    parser.add_argument("--model", default="mock",
                        help="HuggingFace model name or 'mock' for smoke test.")
    parser.add_argument("--rounds", type=int, default=N_ROUNDS)
    parser.add_argument("--runs", type=int, default=N_RUNS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds-file", default=os.path.join(DATA_DIR, "seed_sequences.json"))
    parser.add_argument("--output-dir", default=RESULTS_DIR)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--no-esm", action="store_true", help="Skip ESM-2; use charge-only fitness.")
    parser.add_argument("--smoke-test", action="store_true", help="3-round mock run, no GPU.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.smoke_test:
        print("[LA-TQD] Smoke test mode")
        args.model = "mock"
        args.rounds = 3
        args.runs = 1
        args.no_esm = True

    # Load seeds
    if not os.path.exists(args.seeds_file):
        print(f"[ERROR] Seeds file not found: {args.seeds_file}", file=sys.stderr)
        sys.exit(1)
    with open(args.seeds_file) as f:
        data = json.load(f)
    seeds = data if isinstance(data, list) else data.get("sequences", [])
    print(f"[main] {len(seeds)} seed sequences loaded")

    # Load mutator
    if args.model.lower() == "mock":
        from src.llm_mutator import MockMutator
        mutator = MockMutator()
    else:
        from src.llm_mutator import LLMMutator
        mutator = LLMMutator(
            args.model, device=args.device,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
        )

    # Load ESM-2
    esm_model, esm_tokenizer = None, None
    if not args.no_esm:
        from src.esm_scorer import load_esm2
        esm_model, esm_tokenizer = load_esm2(device=args.device)

    # Run
    experiment_name = args.experiment_name or f"latqd_{args.model.replace('/', '_').lower()}"

    from src.search import run_experiment
    results = run_experiment(
        mutator=mutator, seeds=seeds,
        n_runs=args.runs, n_rounds=args.rounds,
        esm_model=esm_model, esm_tokenizer=esm_tokenizer,
        device=args.device, base_seed=args.seed,
        save_dir=args.output_dir, experiment_name=experiment_name,
    )

    print(f"\n[Done] Coverage: {results['coverage_mean']:.2f}% +/- {results['coverage_std']:.2f}%")
    print(f"       Results: {args.output_dir}")


if __name__ == "__main__":
    main()
