import os
import pandas as pd
import numpy as np
import torch
from einops import rearrange
import torch.nn.functional as F
import inspect


def mask_action(agent, state_tensor):
    state_tensor = state_tensor.unsqueeze(0)  # (B, E, N, D, F)
    # Reshape for the representation network
    reshaped_state = rearrange(state_tensor, "b e n d f -> (b e) n d f")
    agent_type = type(agent).__name__
    rep_state, _, _ = agent.rep.forward_state(reshaped_state)

    if "mask" in agent_type.lower() and hasattr(agent, "aux_stocks"):
        # Extract masks from aux_stocks - but only use the first one to match batch size
        # This is because during inference we're only processing one state at a time
        masks = []
        for i in range(rep_state.shape[0]):
            mask = agent.aux_stocks[i]["mask"]
            masks.append(mask)
        masks = np.array(masks)
        return agent.forward_action(x=rep_state, mask=masks)

    return agent.forward_action(rep_state)


def export_allocation_history(agent, envs, output_path="allocation_history.csv"):
    print(f"Exporting allocation history to {output_path}...")

    # Reset the environment
    state = envs.reset()
    done = False

    # Initialize lists to store data
    dates = []
    allocations = []

    # Check if this is a masked model with a representation network
    agent_type = type(agent).__name__
    is_masked = "mask" in agent_type.lower() and hasattr(agent, "rep")

    # Process each step
    while not done:
        # Get current date
        date = (
            envs.envs[0].get_current_date()
            if hasattr(envs.envs[0], "get_current_date")
            else len(dates)
        )
        dates.append(date)
        # Convert state to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32, device=agent.device)

        # Get action based on model type
        with torch.no_grad():
            if is_masked:
                action = mask_action(agent, state_tensor)
            else:
                action = agent.forward_action(state_tensor)

        # Convert to numpy and store
        action_np = action.detach().cpu().numpy()

        if len(action_np.shape) > 1:
            allocations.append(action_np[0].flatten())
        else:
            allocations.append(action_np.flatten())

        # Step environment with full action array for vectorized environments
        state, reward, done, info = envs.step(action_np)

        # Handle vectorized environments
        if isinstance(done, np.ndarray) and len(done) > 1:
            done = done.any()

    # Create DataFrame
    allocation_df = pd.DataFrame(allocations)
    allocation_df.insert(0, "date", dates)

    return allocation_df
