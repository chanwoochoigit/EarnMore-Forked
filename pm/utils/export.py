import os
import pandas as pd
import numpy as np
import torch
from einops import rearrange
import torch.nn.functional as F


def map_action_3d_to_5d(state_tensor):
    # For masked models, use the representation network 3-d to 5-d
    state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)  # (B, E, N, D, F)

    return state_tensor


def mask_action(agent, state_tensor):
    state_tensor = map_action_3d_to_5d(state_tensor)
    # Reshape for the representation network
    reshaped_state = rearrange(state_tensor, "b e n d f -> (b e) n d f")
    # Get masked representation
    rep_state, _, _ = agent.rep.forward_state(reshaped_state)
    # Get action using representation
    action = agent.act(rep_state)
    return action


def export_allocation_history(agent, env, output_path="allocation_history.csv"):
    print(f"Exporting allocation history to {output_path}...")

    # Reset the environment
    state = env.reset()
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
            env.current_date
            if hasattr(env, "current_date")
            else (
                env.get_current_date()
                if hasattr(env, "get_current_date")
                else len(dates)
            )
        )
        dates.append(date)
        # Convert state to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32, device=agent.device)

        # Get action based on model type
        with torch.no_grad():
            if is_masked:
                action = mask_action(agent, state_tensor)
            else:
                # DQN and SAC require 5-d action space
                if any([x in agent_type.lower() for x in ["dqn", "sac", "ppo"]]):
                    action = agent.act(map_action_3d_to_5d(state_tensor))
                else:
                    action = agent.act(state_tensor)
            action = F.softmax(action, dim=-1)

        # Convert to numpy and store
        action_np = action.detach().cpu().numpy()
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
