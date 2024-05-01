from rank_score_characteristic import Weighting_Scheme
from utils.utils import get_outputs
from sklearn.metrics import mean_squared_error, r2_score
import torch
from itertools import combinations

class InFusionRegression:
    def __init__(self, score_data, OUTPATH = './', weighting_schemes = ['AC', 'WCDS']) -> None:
        self.score_tensors = {s: torch.tensor(score_data[s].iloc[:, 0].values, dtype=torch.float32) for s in score_data}
        # Use the function
        self.rank_tensors = self.get_tensor_ranks(self.score_tensors)

        # Weight_Schemes = Weighting_Scheme(self.score_tensors, self.rank_tensors)
        self.weighting_schemes = weighting_schemes
        self.weighting_schemes = self.weighting_scheme(self.score_tensors, self.rank_tensors)

    def get_combinations(self, models):
        lengths = [x for x in range(len(models)+1)]
        combs = []
        for x in lengths:
            comb = combinations(models, x)
            for i in comb:
                if len(i) > 1: 
                    combs.append(i)
        return combs

    def model_fusion(self, data, weights, label, fusion_type, sc=True):
        model_combination = data.copy()
        original_models = list(model_combination.keys())

        sum_numerator = torch.zeros_like(next(iter(model_combination.values())))  # Initialize to zeros of same shape as any tensor
        sum_denominator = 0

        # Iterate over the tensors and their corresponding weights
        for model, tensor in model_combination.items():
            if weights[model] == 0: continue
            w = weights[model] if sc else (1 / weights[model])
            sum_numerator += tensor * w
            sum_denominator += w
        scores = sum_numerator / sum_denominator if sum_denominator != 0 else torch.zeros_like(sum_numerator)

        model_combination[f"{label}_{fusion_type}"] = scores
        return model_combination

    def get_r2_rmse(self, tensor, G):
        rmse = mean_squared_error(G, tensor.tolist(), squared=False)
        r2 = r2_score(G, tensor.tolist())
        return r2, rmse

    def get_tensor_ranks(self, score_tensor):
        rank_tensors = {}
        for key, tensor in score_tensor.items():
            sorted_indices = torch.argsort(tensor)
            ranks = torch.argsort(sorted_indices) + 1  # Start ranks from 1 instead of 0
            rank_tensors[key] = ranks.float()
        return rank_tensors

    def weighting_scheme(self, scores_batch, ranks_batch):
        dic = {}
        Weight_Schemes = Weighting_Scheme(scores_batch, ranks_batch, norm_regression=True)
        for scheme in self.weighting_schemes:
            dic[scheme] = Weight_Schemes[scheme]
        return dic

    def combinatorial_fusion_analysis(self, score_tensors, weighting_schemes):
        sc, rc = {}, {}
        for C_TYPE in weighting_schemes:
            combs = self.get_combinations(list(score_tensors.keys()))

            for comb in combs:
                label = ''.join(comb)
                # Select tensors and weights for the current combination
                scores_batch = {key: score_tensors[key] for key in comb}
                ranks_batch = {key: self.rank_tensors[key] for key in comb}

                comb_weights = {key: weighting_schemes[C_TYPE][key] for key in comb if key in weighting_schemes[C_TYPE]}

                # Calculate weighted sum of tensors for the current combination
                sc.update(self.model_fusion(scores_batch, comb_weights, label, C_TYPE, sc=True))
                rc.update(self.model_fusion(ranks_batch, comb_weights, label, C_TYPE, sc=False))
        return sc, rc

    def predict(self):
        sc, rc = self.combinatorial_fusion_analysis(self.score_tensors, self.weighting_schemes)
        return sc, rc

ROOT = './sklearn_models'
DATASET = 'HTS'

score_data, ground_truth = get_outputs(f"{ROOT}/{DATASET}")

T = score_data
G = list(ground_truth['0'])

infuse_regression = InFusionRegression(T, OUTPATH=f'./results/sklearn_models/HTS')
sc, rc = infuse_regression.predict()


import pandas as pd

# Assuming infuse_regression.predict() returns two dictionaries `sc` and `rc`
# Assuming G is defined somewhere in your code as the ground truth values
data = []

# Collect data from sc
for M in sc:
    r2, rmse = infuse_regression.get_r2_rmse(sc[M], G)
    data.append({'Model': f'SC_{M}', 'R2': r2, 'RMSE': rmse, 'Type': 'SC'})

# Collect data from rc
# for M in rc:
#     r2, rmse = infuse_regression.get_r2_rmse(rc[M], G)
#     data.append({'Model': f'RC_{M}', 'R2': r2, 'RMSE': rmse, 'Type': 'RC'})

# Create a DataFrame
df = pd.DataFrame(data)

import seaborn as sns
import matplotlib.pyplot as plt

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
plt.show()

plt.close()
