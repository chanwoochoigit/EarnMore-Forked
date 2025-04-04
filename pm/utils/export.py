import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F


def export_allocation_history(agent, env, output_path="allocation_history.csv"):
    """
    This util function is to export the allocation history
    to a CSV, which will be further analyzed and integrated
    into the codebase for comparison and reporting
    """
    print(f"Exporting allocation history to {output_path}...")

    # Reset the environment
    state = env.reset()
    done = False

    # Initialize lists to store data
    dates = []
    allocations = []

    # Get allocations for each step
    while not done:
        # Get current date
        date = (
            env.get_current_date() if hasattr(env, "get_current_date") else len(dates)
        )
        dates.append(date)

        # Reshape state to match agent's expected input format
        # Add batch and channel dimensions: (N, D, F) -> (1, 1, N, D, F)
        if isinstance(state, np.ndarray):
            state_tensor = (
                torch.tensor(state, dtype=torch.float32, device=agent.device)
                .unsqueeze(0)
                .unsqueeze(0)
            )  # Add batch and channel dims
        else:
            # If it's already a tensor, just add dimensions
            state_tensor = state.unsqueeze(0).unsqueeze(0)

        # Get action from agent
        with torch.no_grad():  # Ensure we're in evaluation mode
            action = agent.act(state_tensor)

            # Apply softmax to normalize allocations if needed
            if isinstance(action, torch.Tensor):
                if not torch.isclose(
                    action.sum(), torch.tensor(1.0, device=action.device)
                ):
                    action = F.softmax(action, dim=-1)
                action_np = action.detach().cpu().numpy()
            else:
                # If it's already numpy, apply softmax using numpy if needed
                if not np.isclose(np.sum(action), 1.0):
                    action_np = np.exp(action) / np.sum(
                        np.exp(action), axis=-1, keepdims=True
                    )
                else:
                    action_np = action

        # Record allocation
        allocations.append(action_np.flatten())

        # Step environment
        state, reward, done, info = env.step(action_np)

        # Handle vectorized environments
        if isinstance(done, np.ndarray) and len(done) > 1:
            done = done.any()

    # Create DataFrame
    allocation_df = pd.DataFrame(allocations)
    allocation_df.insert(0, "date", dates)

    return allocation_df
