import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

# Step 1: Authenticate
wandb.login()

# Step 2: Initialize Wandb and query your project
project_path = "math_dl/pattern_jumping_projection"
api = wandb.Api()
runs = api.runs(project_path)

# List to collect data rows
data_rows = []

for run in runs:

    # Extract Validation Accuracy Data
    history = run.scan_history()  # Adjust the sample size as needed
    
    # Find the maximum validation accuracy
    max_val_acc = max([row['val/acc'] for row in history if 'val/acc' in row and row['val/acc'] is not None])
    
    # Collect data in a list
    data_rows.append({
        'gating': run.config['model']['gradient_gating_p'],  # Accessing gating value from run config
        'k_hop': run.config['model']['k_hop'],              # Accessing k_hop value from run config
        'max_val_acc': 100 * max_val_acc                     # Multiplying by 100 for percentage
    })

# Convert the list to a DataFrame
df = pd.DataFrame(data_rows)

# Plotting
g = sns.catplot(
    data=df, kind="bar",
    x="k_hop", y="max_val_acc", hue="gating", errorbar=None,
    palette="dark", alpha=.6, height=6
)
g.despine(left=True)
g.set_axis_labels("", "Maximum accuracy reached [%]")
g.legend.set_title("")

plt.show()
