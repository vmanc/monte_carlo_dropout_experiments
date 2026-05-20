import argparse
import os
from experimental_config import ExperimentalConfig
from experiment_runner import ExperimentRunner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=False, default=None)
    parser.add_argument("--no-dropout-inference", type=float, default=None,
                        help="Train with this dropout rate and run standard inference (no MC Dropout)")

    args = parser.parse_args()
    experimental_config = ExperimentalConfig()
    experiment_runner = ExperimentRunner(experimental_config)
    experiment_runner.load_data()

    if args.no_dropout_inference is not None:
        experiment_runner.run_standard_inference(args.no_dropout_inference)
    elif args.task_id is not None:
        experiment_runner.run_experiment(args.task_id)
    else:
        raise RuntimeError("Provide --task-id or --no-dropout-inference")

if __name__ == "__main__":
    main()

