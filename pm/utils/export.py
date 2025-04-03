import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F


def export_allocation_history(agent, env, output_path="allocation_history.csv"):
    """
    This util function is to export the allocation history
    to a CSV, which will be further analsed and integrated
    into my codebase for comparison and reporting
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
        # Convert state to tensor if it's a numpy array
        if isinstance(state, np.ndarray):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=agent.device)
        else:
            state_tensor = state

        # Get action from agent
        action = agent.act(state_tensor)

        # Apply softmax to normalize allocations
        if isinstance(action, torch.Tensor):
            action = F.softmax(action, dim=-1)
            action_np = action.detach().cpu().numpy()
        else:
            # If it's already numpy, apply softmax using numpy
            action_np = np.exp(action) / np.sum(np.exp(action), axis=-1, keepdims=True)
        # Step environment
        state, reward, done, info = env.step(action_np)

        if isinstance(done, np.ndarray) and len(done) > 1:
            done = done.any()
            breakpoint()

        # Extract date
        date = (
            env.get_current_date() if hasattr(env, "get_current_date") else len(dates)
        )

        dates.append(date)
        allocations.append(action_np.flatten())

    # Create DataFrame
    allocation_df = pd.DataFrame(allocations)
    allocation_df.insert(0, "date", dates)

    # Save to CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    allocation_df.to_csv(output_path, index=False)
    print(f"Allocation history exported to {output_path}")

    return allocation_df
