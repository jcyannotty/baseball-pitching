
'''
Work in progress:
This is just a script - need to test some ideas
Need to build out into an executible framework 

TODO: Add KL Divergence / Energy Stats for comparisons between pitchers
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import json

from pybaseball import statcast
from PitchLocations.Methods.normalizing_flows import *
from Data.pitcher import *

if len(sys.args) < 2:
    raise Exception("Missing run_config argument path.")

with open(sys.args[1]) as f:
    config = json.load(f)

with open(r"C:\Users\johny\Documents\Repos\baseball-pitching\PitchLocations\Estimation\run_config.json") as f:
    config = json.load(f)


# Load the data
# start_date = "2024-03-28"
# end_date = "2024-09-27"
start_date = config["start_date"]
end_date = config["end_date"]

# Pull all Statcast data for the specified pitcher and date range
df = statcast(start_dt=start_date, end_dt=end_date)

pitcher_subset = df[
    df["player_name"].isin(config["pitchers"]) &
    df["pitch_type"].isin(config["pitch_types"]) 
]

player_ids = pitcher_subset["pitcher"].unique().tolist()

dim = 2
learning_rate = config["model_properties"]["normalizing_flows"]["learning_rate"]
hidden_dim = config["model_properties"]["normalizing_flows"]["hidden_nodes"]
n_layers = config["model_properties"]["normalizing_flows"]["hidden_layers"]
n_epochs = config["model_properties"]["normalizing_flows"]["max_iterations"]
eps = config["model_properties"]["normalizing_flows"]["stop_criteria"]

database = {}
for player_id in player_ids:
    df_pitcher = df[df['pitcher'] == player_id].copy()

    # Drop pitches where location data is missing
    df_pitcher = df_pitcher.dropna(subset=['plate_x', 'plate_z'])

    flow = NormalizingFlow(dim, hidden_dim, n_layers = n_layers)

    optimizer = optim.Adam(flow.parameters(), lr=learning_rate)

    # Training loop
    current_loss = 0
    pitch_xy = torch.tensor(df_pitcher[['plate_x','plate_z']].values.tolist())
    for step in range(n_epochs):
        log_prob = flow.log_prob(pitch_xy)
        loss = -log_prob.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Stopping criteria for early termination
        if np.abs(current_loss - loss.item()) < eps:
            print(f"Step {step}, Loss {loss.item():.4f}")
            break
        
        current_loss = loss.item()
        
        if step % 1000 == 0:
            print(f"Step {step}, Loss {loss.item():.4f}")

    # Sample from trained flow
    samples = flow.sample(5000).detach().numpy()

    database[player_id] = flow



# Create a figure and axes for the plot
fig, ax = plt.subplots(figsize=(10, 8))

# Define the strike zone coordinates (approximate dimensions)
strike_zone_top = 3.5  # Top of the strike zone
strike_zone_bottom = 1.6  # Bottom of the strike zone
strike_zone_left = -0.95  # Left side of the strike zone
strike_zone_right = 0.95  # Right side of the strike zone

# Plot the strike zone as a rectangle
ax.add_patch(plt.Rectangle([strike_zone_left, strike_zone_bottom], 
                            strike_zone_right - strike_zone_left, 
                            strike_zone_top - strike_zone_bottom,
                            fill=False, edgecolor='black', linewidth=2))

# Plot the pitch locations, colored by pitch type
sns.scatterplot(data=df_pitcher, x='plate_x', y='plate_z', hue='pitch_name', ax=ax, alpha=0.6)

# Set plot titles and labels
plt.title(f'{df_pitcher["player_name"].iloc[0]} Pitch Locations (2024)', fontsize=16)
plt.xlabel("Horizontal Location (feet)", fontsize=12)
plt.ylabel("Vertical Location (feet)", fontsize=12)

# Set axis limits to make the strike zone centered
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(0, 5)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

plt.figure(figsize=(6,6))
plt.scatter(samples[:,0], samples[:,1], alpha=0.5)
plt.scatter(pitch_xy[:,0], pitch_xy[:,1], alpha=0.5)

plt.title("Samples from Normalizing Flow")
plt.show()



