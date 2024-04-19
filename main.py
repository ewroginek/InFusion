import time
import torch
import pandas as pd
import os
from itertools import combinations
from utils import plot_average_rows

class RankScoreCharacteristic:
    def rank_score_function(self, x):
        # Normalize the tensor row-wise between 0 and 1
        min_val = x.min(dim=1, keepdim=True).values
        max_val = x.max(dim=1, keepdim=True).values
        normalized_tensor = (x - min_val) / (max_val - min_val)

        # Sort the rows in descending order
        sorted_tensor, _ = torch.sort(normalized_tensor, dim=1, descending=True)
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
                f_Ti = self.rank_score_function(data[T_i])
                f_Tj = self.rank_score_function(data[T_j])
                CD = self.cognitive_diversity(f_Ti, f_Tj)
                pairwise_matrix[i, j] = CD
        ds = pairwise_matrix.mean(dim=1) # Mean across columns

        # If you need to return a dictionary mapping each T to its ds value:
        DS_dict = {T[i]: ds[i].item() for i in range(num_items)}

        return DS_dict

class InFusionLayer:
    def __init__(self, ROOT = './sklearn_models/', DATASET = 'lidar_trees_classification_d2', BATCH_SIZE = 2048) -> None:
        self.ROOT = ROOT
        self.DATASET = DATASET

        if not os.path.exists(f'./results/'): os.mkdir(f'./results/')
        if not os.path.exists(f'./results/{ROOT}'): os.mkdir(f'./results/{ROOT}')
        if not os.path.exists(f'./results/{ROOT}/{DATASET}'): os.mkdir(f'./results/{ROOT}/{DATASET}')
        self.OUTPATH = f'./results/{ROOT}/{DATASET}'

        self.BATCH_SIZE = BATCH_SIZE
        self.K = 5
        self.PLOT_AVG_RSC = True

        score_data, ground_truth = self.get_outputs(f"{ROOT}/{DATASET}")
        self.score_data = score_data
        self.ground_truth = ground_truth
        rank_data = {i: score_data[i].rank(axis=1, ascending=False) for i in score_data}
        self.rank_data = rank_data

        self.base_models = list(score_data.keys())
        self.DATASET_LEN = DATASET_LEN = len(ground_truth['0'])
        print(f"Data Items: {DATASET_LEN}")

        self.RSC = RankScoreCharacteristic()

    def get_combinations(self, models):
        lengths = [x for x in range(len(models)+1)]
        combs = []
        for x in lengths:
            comb = combinations(models, x)
            for i in comb:
                if len(i) > 1: 
                    combs.append(i)
        return combs

    def model_fusion(self, data, weight, fusion_type, combs, sc=True):
        # Assuming data is a dictionary of tensors
        model_combination = data.copy()
        original_models = list(model_combination.keys())

        for comb in combs:
            label = ''.join(comb)
            # Using torch.zeros_like to initialize a tensor of zeros with the same shape as any of the input tensors
            sum_numerator = torch.zeros_like(next(iter(data.values())))
            sum_denominator = torch.zeros_like(next(iter(data.values())))

            # Apply Weighted Combination
            for model in comb:
                w = weight[model] if sc else (1/weight[model])
                sum_numerator += model_combination[model] * w
                sum_denominator += w
            scores = sum_numerator / sum_denominator
            model_combination[f"{label}_{fusion_type}"] = self.normalize(scores) if sc else scores

        if sc:
            for model in original_models:
                model_combination[model] = self.normalize(model_combination[model])  
        return model_combination

    def normalize(self, x):
        return (x - x.min()) / (x.max() - x.min())

    def get_outputs(self, ROOT):
        score_data = {}
        ground_truth = None

        paths = os.listdir(ROOT)

        for path in paths:
            model = path.split("_")[0]
            if path.endswith('_scores.csv'):
                score_data[model] = pd.read_csv(f"{ROOT}/{path}").iloc[:,1:]
            else:
                ground_truth = pd.read_csv(f"{ROOT}/{path}").iloc[:,1:]
        return score_data, ground_truth

    def combinatorial_fusion_analysis(self, scores_batch, ranks_batch, combination_types, combs):
        sc, rc = {}, {}
        for C_TYPE in combination_types:
            sc.update(self.model_fusion(scores_batch, combination_types[C_TYPE], C_TYPE, combs, sc=True))
            rc.update(self.model_fusion(ranks_batch, combination_types[C_TYPE], C_TYPE, combs, sc=False))
        return sc, rc

    def batch_combination(self, score_data, rank_data, batch_size=64):
        sc_batches = []
        rc_batches = []

        # Determine the size of batches based on your dataset and memory capacity
        N = len(next(iter(score_data.values())))  # Total number of rows

        combs = self.get_combinations(list(score_data.keys()))
        for n in range(0, N, batch_size):
            batch_indices = slice(n, n+batch_size)
            
            # Initialize tensors for scores and ranks for the batch
            scores_batch = {d: scores[batch_indices] for d, scores in score_data.items()}
            ranks_batch = {d: ranks[batch_indices] for d, ranks in rank_data.items()}
            
            # Example operations on the batch (assuming these functions are adapted to handle tensors)
            rank_score_functions = {m: scores_batch[m] * ranks_batch[m]**(-1) for m in scores_batch}
            ds_vector = self.RSC.diversity_strength(rank_score_functions)
            ac_vector = {m:1 for m in ds_vector}
            combination_types = {"AC": ac_vector, "WCDS": ds_vector}

            # CFA
            sc, rc = self.combinatorial_fusion_analysis(scores_batch, ranks_batch, combination_types, combs)
            sc_batches.append(sc)
            rc_batches.append(rc)

        concatenated_sc = {}
        concatenated_rc = {}
        # Assuming all dictionaries in sc_batches have the same keys
        keys = sc_batches[0].keys()

        for key in keys:
            concatenated_sc[key] = torch.cat([batch[key] for batch in sc_batches], dim=0).cpu()
            concatenated_rc[key] = torch.cat([batch[key] for batch in rc_batches], dim=0).cpu()
        
        return concatenated_sc, concatenated_rc

    def get_accuracies(self, models, ground_truth, sc=True):
        results = {}
        gt_tensor = torch.tensor(ground_truth['0'].values)

        for m in models:
            _, indices = torch.max(models[m], dim=1) if sc else torch.min(models[m], dim=1)
            matches = indices == gt_tensor
            percent_match = (matches.sum().item() / len(matches)) * 100
            results[m] = percent_match
        return results

    def check_tie_ranks(self, tensor):
        unique_elements = torch.unique(tensor)
        return len(tensor) != len(unique_elements)

    def rank_norm(self, rank_tensor):
        # Normalize the ranks tensor from 0 to 1, inverting the ranks
        return 1 - (rank_tensor - 1) / (rank_tensor.max() - 1)

    def update_max(self, top_m, highest_value_pair, max_val):
        try:
            highest_value_pair = max(top_m.items(), key=lambda item: item[1])
            max_val = highest_value_pair[1]
            return highest_value_pair, max_val
        except:
            return highest_value_pair, max_val

    def predict(self, matrices=False):
        start_time = time.time()
        score_tensors = {d: torch.tensor(df.values, dtype=torch.float32) for d, df in self.score_data.items()}

        base_model_accuracies = self.get_accuracies(score_tensors.copy(), self.ground_truth.copy(), True)
        for i in base_model_accuracies:
            print(f"{i}: {base_model_accuracies[i]}")
        self.highest_score_value_pair = max(base_model_accuracies.items(), key=lambda item: item[1])
        self.highest_start = self.highest_score_value_pair[1]
        print("Start:", self.highest_score_value_pair)

        # Initialize Rank Variables using max Score info as base
        rank_tensors = {d: torch.tensor(df.values, dtype=torch.float32) for d, df in self.rank_data.items()}
        self.highest_rank_value_pair = self.highest_score_value_pair

        for i in rank_tensors:
            print(f"Tie Ranks {i}: {self.check_tie_ranks(rank_tensors[i])}")

        if self.PLOT_AVG_RSC: plot_average_rows(score_tensors, rank_tensors, self.OUTPATH, self.DATASET_LEN, 0)

        fusion_models_sc, fusion_models_rc = self.batch_combination(score_tensors, rank_tensors, self.BATCH_SIZE)

        if matrices:
            return fusion_models_sc, fusion_models_rc
        else:
            max_s_val = self.highest_score_value_pair[1]
            max_r_val = max_s_val
            self.highest_score_value_pair, max_s_val = self.update_max(fusion_models_sc, self.highest_score_value_pair, max_s_val)
            self.highest_rank_value_pair, max_r_val = self.update_max(fusion_models_rc, self.highest_rank_value_pair, max_r_val)
            print("Algorithm time:", time.time() - start_time)
            top_2 = [self.highest_score_value_pair, self.highest_rank_value_pair]
            print(f"End: {max(top_2, key=lambda x: x[1])}")
            print(f"{max(top_2, key=lambda x: x[1])[1] - self.highest_start}% improvement from base models.")
            print("Done!")

class InFusionNet(InFusionLayer):
    def __init__(self, ROOT, DATASET, BATCH_SIZE) -> None:
        super(InFusionNet, self).__init__()

        IFL = InFusionLayer(ROOT = ROOT, DATASET = DATASET, BATCH_SIZE = BATCH_SIZE)

        self.fusion_models_sc, self.fusion_models_rc = IFL.predict(matrices=True)
        self.ground_truth = IFL.ground_truth
        self.highest_start = IFL.highest_start
        self.highest_score_value_pair = IFL.highest_score_value_pair 
        self.highest_rank_value_pair = IFL.highest_rank_value_pair

    def top_k(self, fusion_models, ground_truth, max_val, k=5, scores=True):
        results = self.get_accuracies(fusion_models.copy(), ground_truth.copy(), scores)
        top_performers = {key: value for key, value in results.items() if value > max_val}
        top_k_models = dict(sorted(top_performers.items(), key=lambda x: x[1], reverse=True)[:k])
        return top_k_models
    
    # Expansion-Reduction Algorithm
    def expansion_reduction_1(self, fusion_models_sc, fusion_models_rc, highest_score_value_pair, highest_rank_value_pair, OUTPATH, plot_avg_rsc=False):
        max_s_val = highest_score_value_pair[1]
        max_r_val = max_s_val
        for i in range(15): # turn to while (len(top_5) > 0)
            print(f"Iteration: {i+1}")
            
            if plot_avg_rsc: plot_average_rows(fusion_models_sc, fusion_models_rc, OUTPATH, self.DATASET_LEN, i+1)

            # Scores
            top_5_sc_models = self.top_k(fusion_models_sc, self.ground_truth, max_s_val, self.K, True)

            # Ranks
            top_5_rc_models = self.top_k(fusion_models_rc, self.ground_truth, max_r_val, self.K, False)

            # Pool SC & RC Models
            combined = {f'SC_{key}': value for key, value in top_5_sc_models.items()}
            combined.update({f'RC_{key}': value for key, value in top_5_rc_models.items()})
            top_5 = sorted(combined.items(), key=lambda item: item[1], reverse=True)[:self.K]
            
            # Termination condition
            if len(top_5) == 0: break

            # Update the base models with the newly fused higher performing models
            self.base_models = [m[0] for m in top_5]

            # Update max/highest model values
            highest_score_value_pair, max_s_val = self.update_max(top_5_sc_models, highest_score_value_pair, max_s_val)
            highest_rank_value_pair, max_r_val = self.update_max(top_5_rc_models, highest_rank_value_pair, max_r_val)

            # Create new score-rank tensors
            score_tensors = {}
            rank_tensors = {}
            for model in self.base_models:
                M = model[3:] # normalize the ranks for scores if using rank-scores
                score_tensors[model] = fusion_models_sc[M] if model[:2] == "SC" else self.rank_norm(fusion_models_rc[M])
                sorted_indices = score_tensors[model].argsort(dim=1, descending=True)
                ranks = sorted_indices.argsort(dim=1).float()
                rank_tensors[model] = ranks + 1
            
            # Batch combination
            if len(score_tensors) > 0:
                fusion_models_sc, fusion_models_rc = self.batch_combination(score_tensors, rank_tensors, self.BATCH_SIZE)

                # Filter out any tensors with NaN values
                fusion_models_sc = {k: v for k, v in fusion_models_sc.items() if not torch.isnan(v).any()}
                fusion_models_rc = {k: v for k, v in fusion_models_rc.items() if not torch.isnan(v).any()}
            else:
                break
        return highest_score_value_pair, highest_rank_value_pair

    def predict(self):
        highest_score_value_pair, highest_rank_value_pair = self.expansion_reduction_1(self.fusion_models_sc, self.fusion_models_rc, self.highest_score_value_pair, self.highest_rank_value_pair, self.OUTPATH, self.PLOT_AVG_RSC)
        top_2 = [highest_score_value_pair, highest_rank_value_pair]
        print(f"End: {max(top_2, key=lambda x: x[1])}")
        print(f"{max(top_2, key=lambda x: x[1])[1] - self.highest_start}% improvement from base models.")
        print("Done!")

infusionlayer = InFusionLayer()
infusionlayer.predict()

infusionnet = InFusionNet(ROOT = './sklearn_models/', DATASET='mnist', BATCH_SIZE = 2048)
infusionnet.predict()