import os
import sys
import argparse
from pathlib import Path
import torch
from mmengine.config import Config
from copy import deepcopy

ROOT = str(Path(__file__).resolve().parents[1])
sys.path.append(ROOT)

from pm.registry import ENVIRONMENT, AGENT, DATASET
from pm.utils import update_data_root, load_checkpoint
from pm.utils.export import export_allocation_history


def parse_args():
    parser = argparse.ArgumentParser(description="Export allocation history")
    parser.add_argument("--config", required=True, help="Config file path")
    parser.add_argument(
        "--checkpoint", default=None, help="Checkpoint file path (default: best.pth)"
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (default: same as checkpoint dir)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    cfg = Config.fromfile(args.config)
    update_data_root(cfg, root=ROOT)

    # Set up paths
    exp_path = os.path.join(cfg.root, cfg.workdir, cfg.tag)
    checkpoint_path = (
        args.checkpoint if args.checkpoint else os.path.join(exp_path, "best.pth")
    )
    output_dir = args.output_dir if args.output_dir else exp_path

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build dataset
    print("Building dataset...")
    dataset = DATASET.build(cfg.dataset)

    # Build environments
    print("Building environments...")

    # Train environment
    cfg.environment.update(
        dict(
            mode="train",
            if_norm=True,
            dataset=dataset,
            start_date=cfg.train_start_date,
            end_date=cfg.val_start_date,
        )
    )
    train_environment = ENVIRONMENT.build(cfg.environment)

    # Validation environment
    cfg.environment.update(
        dict(
            mode="val",
            if_norm=True,
            dataset=dataset,
            scaler=train_environment.scaler,
            start_date=cfg.val_start_date,
            end_date=cfg.test_start_date,
        )
    )
    val_environment = ENVIRONMENT.build(cfg.environment)

    # Test environment
    cfg.environment.update(
        dict(
            mode="test",
            if_norm=True,
            dataset=dataset,
            scaler=train_environment.scaler,
            start_date=cfg.test_start_date,
            end_date=getattr(cfg, "test_end_date", None),
        )
    )
    test_environment = ENVIRONMENT.build(cfg.environment)

    # Build agent
    print("Building agent...")
    cfg.agent.update(dict(device=device))
    agent = AGENT.build(cfg.agent)

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    episode = load_checkpoint(agent, checkpoint_path)
    print(f"Loaded checkpoint from episode {episode}")

    # Export allocation histories
    print("Exporting allocation histories...")

    # Train set
    train_output_path = os.path.join(output_dir, "train_allocation_history.csv")
    print(f"Exporting training allocation history to {train_output_path}...")
    export_allocation_history(agent, train_environment, train_output_path)

    # Validation set
    val_output_path = os.path.join(output_dir, "val_allocation_history.csv")
    print(f"Exporting validation allocation history to {val_output_path}...")
    export_allocation_history(agent, val_environment, val_output_path)

    # Test set
    test_output_path = os.path.join(output_dir, "test_allocation_history.csv")
    print(f"Exporting test allocation history to {test_output_path}...")
    export_allocation_history(agent, test_environment, test_output_path)

    print("All allocation histories exported successfully")


if __name__ == "__main__":
    main()
