import torch

class RankScoreCharacteristic:
    def __init__(self, regression=False) -> None:
        self.norm_sort_func = self.rank_score_regression if regression else self.rank_score_classification

    def rank_score_classification(self, x):
        """For Classification"""
        # Normalize the tensor row-wise between 0 and 1
        min_val = x.min(dim=1, keepdim=True).values
        max_val = x.max(dim=1, keepdim=True).values
        normalized_tensor = (x - min_val) / (max_val - min_val)

        # Sort the rows in descending order
        sorted_tensor, _ = torch.sort(normalized_tensor, dim=1, descending=True)
        return sorted_tensor

    def rank_score_regression(self, x):
        """For Regression"""
        # Normalize the tensor between 0 and 1
        min_val = x.min()
        max_val = x.max()
        normalized_tensor = (x - min_val) / (max_val - min_val)

        # Sort the tensor in descending order
        sorted_tensor, _ = torch.sort(normalized_tensor, descending=True)
        return sorted_tensor

    def cognitive_diversity(self, f_A, f_B):
        N = (len(f_A))
        return torch.sqrt(torch.sum((f_A - f_B)**2)/N)

    def diversity_strength(self, data):
        # Assuming data is a dictionary of tensors: {'T1': tensor1, 'T2': tensor2, ...}
        T = list(data.keys())
        num_items = len(T)
        
        # Initialize an empty tensor for results; shape [num_items, num_items]
        pairwise_matrix = torch.empty(num_items, num_items, dtype=torch.float)
        for i, T_i in enumerate(T):
            for j, T_j in enumerate(T):
                f_Ti = self.norm_sort_func(data[T_i])
                f_Tj = self.norm_sort_func(data[T_j])
                CD = self.cognitive_diversity(f_Ti, f_Tj)
                pairwise_matrix[i, j] = CD
        torch.diagonal(pairwise_matrix)[:] = float('nan')
        ds = torch.nanmean(pairwise_matrix, dim=0)

        # Setting up dictionary of model weights
        DS_dict = {T[i]: ds[i].item() for i in range(num_items)}

        return DS_dict

class Weighting_Scheme:
    def __init__(self, scores_batch, ranks_batch, model_accuracies:dict = {}, norm_regression=False):
        self.scores_batch = scores_batch
        self.ranks_batch = ranks_batch

        # Save model names as a list for clarity
        # (scores_batch and ranks_batch model names are the same)
        self.models = scores_batch.keys()
        self.model_accuracies = model_accuracies.copy()

        self.RSC = RankScoreCharacteristic(norm_regression)

        # List of Weight Functions
        self.init_functions = {
            'AC': self.average_combination,
            'WCDS': self.weighted_combination_diversity_strength,
            'WCP': self.weighted_combination_by_performance
        }
        self.values = {}

    def __getitem__(self, key):
        if key not in self.values:
            if key in self.init_functions:
                self.values[key] = self.init_functions[key]()
            else:
                raise KeyError(f"No function provided for key: {key}")
        return self.values[key]

    # Weight Functions for creating Model Dictionary of Weights
    def average_combination(self):
        return {m: 1 for m in self.models}

    def weighted_combination_diversity_strength(self):
        rank_score_functions = {m: self.scores_batch[m] * self.ranks_batch[m]**(-1) for m in self.models}
        return self.RSC.diversity_strength(rank_score_functions)
    
    def weighted_combination_by_performance(self):
        return {m: self.model_accuracies[m] * .01 for m in self.model_accuracies}