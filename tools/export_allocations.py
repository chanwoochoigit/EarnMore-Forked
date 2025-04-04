import os
import sys
import argparse
from pathlib import Path
import torch
from mmengine.config import Config
import pandas as pd

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Building dataset...")
    dataset = DATASET.build(cfg.dataset)

    print("Building environments...")
    # create a training environment just to get the scaler
    cfg.environment.update(
        dict(
            mode="train",  # Must be in train mode to create the scaler
            if_norm=True,
            dataset=dataset,
            start_date=cfg.train_start_date,
            end_date=cfg.val_start_date,
        )
    )
    scaler_env = ENVIRONMENT.build(cfg.environment)
    scaler = scaler_env.scaler

    # Now create backtesting environment in val mode with fixed starting points
    cfg.environment.update(
        dict(
            mode="val",  # Val mode for deterministic behavior
            if_norm=True,
            dataset=dataset,
            scaler=scaler,  # Use scaler from first environment
            start_date=cfg.train_start_date,
            end_date=None,
        )
    )
    backtest_environment = ENVIRONMENT.build(cfg.environment)

    # Manually set day to start at the beginning of the dataset
    backtest_environment.day = backtest_environment.days - 1  # Start at beginning

    print("Building agent...")
    cfg.agent.update(dict(device=device))
    agent = AGENT.build(cfg.agent)

    print(f"Loading checkpoint from {checkpoint_path}...")
    episode = load_checkpoint(agent, checkpoint_path)
    print(f"Loaded checkpoint from episode {episode}")

    # Set agent to evaluation mode if it has that method
    if hasattr(agent, "eval"):
        agent.eval()

    # Export allocation histories
    print("Exporting allocation histories...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "allocation_history.csv")
    alloc_h_df = export_allocation_history(agent, backtest_environment, output_path)

    alloc_h_df = alloc_h_df.sort_values(by="date")
    stock_names = ["cash"] + backtest_environment.stocks  # assign assets

    # Rename the columns to match stock names
    # Column 0 is the date, columns 1+ are allocations
    column_names = ["date"] + stock_names

    # Verify we have the right number of columns
    if len(column_names) == alloc_h_df.shape[1]:
        alloc_h_df.columns = column_names
    else:
        print(
            f"Warning: Column count mismatch - found {alloc_h_df.shape[1]} columns but expected {len(column_names)}"
        )
        # Rename only what we can and leave the rest
        alloc_h_df.columns = ["date"] + [
            f"allocation_{i}" for i in range(alloc_h_df.shape[1] - 1)
        ]

    # Save the combined history
    alloc_h_df.to_csv(output_path, index=False)
    print(f"Allocation history saved to {output_path}")

    print("All allocation histories exported successfully")


if __name__ == "__main__":
    main()
