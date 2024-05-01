from models import InFusionRegression
from utils.utils import get_outputs
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

ROOT = './sklearn_models'
DATASET = 'HTS'

if not os.path.exists(f'./results/'): os.mkdir(f'./results/')
if not os.path.exists(f'./results/{ROOT}'): os.mkdir(f'./results/{ROOT}')
if not os.path.exists(f'./results/{ROOT}/{DATASET}'): os.mkdir(f'./results/{ROOT}/{DATASET}')
OUTPATH = f'./results/{ROOT}/{DATASET}'

score_data, ground_truth = get_outputs(f"{ROOT}/{DATASET}")

T = score_data
G = list(ground_truth['0'])

infuse_regression = InFusionRegression(T, weighting_schemes=['AC', 'WCDS'])
score_combinations, rank_combinations = infuse_regression.predict()
print(score_combinations)
data = []

# Collect data from score combinations
base_models = list(T.keys())
for M in score_combinations:
    C = 'SC' if M not in base_models else 'Base'
    r2, rmse = infuse_regression.get_r2_rmse(score_combinations[M], G)
    data.append({'Model': f'{C}_{M}', 'R2': r2, 'RMSE': rmse, 'Type': 'SC'})

# Collect data from rank combinations
# for M in rank_combinations:
    # C = 'RC' if M not in base_models else 'Base'
#     r2, rmse = infuse_regression.get_r2_rmse(rank_combinations[M], G)
#     data.append({'Model': f'{C}_{M}', 'R2': r2, 'RMSE': rmse, 'Type': 'RC'})

# Create a DataFrame
df = pd.DataFrame(data)

# Set the aesthetics for the plots
sns.set(style="whitegrid")

# Create a figure to hold the subplots
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot R2
sns.barplot(x='Model', y='R2', data=df, ax=ax[0], color='blue')
ax[0].set_title('R² for SC Models')
ax[0].set_ylabel('R² Value')
ax[0].set_xlabel('Model')
ax[0].tick_params(axis='x', rotation=90)

# Plot RMSE
sns.barplot(x='Model', y='RMSE', data=df, ax=ax[1], color='green')
ax[1].set_title('RMSE for SC Models')
ax[1].set_ylabel('RMSE Value')
ax[1].set_xlabel('Model')
ax[1].tick_params(axis='x', rotation=90)

# Improve layout and show the plot
plt.tight_layout()
plt.savefig(f'{OUTPATH}/regression_score_combination.PNG')
# plt.show()
# plt.close()

# Plot RSC
import torch

def normalize_and_sort_tensor(tensor):
    """ Normalizes a tensor to range [0, 1] and sorts it in descending order. """
    normalized_tensor = tensor / torch.max(tensor)
    sorted_tensor, _ = torch.sort(normalized_tensor, descending=True)
    return sorted_tensor

# Normalize and plot each tensor
plt.figure(figsize=(12, 6))
for model_name, tensor in score_combinations.items():
    normalized_tensor = normalize_and_sort_tensor(tensor)
    plt.plot(normalized_tensor.numpy(), label=model_name)  # Convert tensor to numpy array for plotting

# Customize the plot
plt.title('Rank-Score Characteristic')
plt.xlabel('Ranks')
plt.ylabel('Normalized Scores')
plt.legend(title='Models', loc='upper center', bbox_to_anchor=(1.15, 1))
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{OUTPATH}/rank_score_characteristics.PNG')
plt.show()
