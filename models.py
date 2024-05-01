import torch
import numpy as np
from itertools import combinations
from utils.utils import plot_average_rows, tensor_indices
from rank_score_characteristic import Weighting_Scheme
from majority_vote import JudgeMajorityVote
from sklearn.metrics import mean_squared_error, r2_score

class InFusionLayer:
    def __init__(self, score_data, ground_truth, OUTPATH = './results/sklearn_models/lidar_trees_classification_d2', weighting_schemes = ['AC', 'WCDS'], BATCH_SIZE = 2048) -> None:
        self.OUTPATH = OUTPATH

        self.weighting_schemes = weighting_schemes
        self.BATCH_SIZE = BATCH_SIZE
        self.K = 5
        self.PLOT_AVG_RSC = True

        self.score_data = score_data
        self.ground_truth = ground_truth
        rank_data = {i: score_data[i].rank(axis=1, ascending=False) for i in score_data}
        self.rank_data = rank_data

        self.base_models = list(score_data.keys())
        self.DATASET_LEN = len(ground_truth)
    
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
                if weight[model] == 0: continue
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

    # This function obtains weights only for the weighting schemes selected at layer initialization
    def weighting_scheme(self, scores_batch, ranks_batch, model_accuracies):
        dic = {}
        Weight_Schemes = Weighting_Scheme(scores_batch, ranks_batch, model_accuracies)
        for scheme in self.weighting_schemes:
            dic[scheme] = Weight_Schemes[scheme]
        return dic

    def combinatorial_fusion_analysis(self, scores_batch, ranks_batch, weighting_schemes, combs):
        sc, rc = {}, {}
        for C_TYPE in weighting_schemes:
            sc.update(self.model_fusion(scores_batch, weighting_schemes[C_TYPE], C_TYPE, combs, sc=True))
            rc.update(self.model_fusion(ranks_batch, weighting_schemes[C_TYPE], C_TYPE, combs, sc=False))
        return sc, rc

    def batch_combination(self, score_data, rank_data, model_accuracies, batch_size=64):
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
            
            # Obtain weight vectors based on a selection of weighting schemes
            weighting_schemes = self.weighting_scheme(scores_batch, ranks_batch, model_accuracies)

            # Use CFA to obtain score and rank combinations
            score_combinations, rank_combinations = self.combinatorial_fusion_analysis(scores_batch, ranks_batch, weighting_schemes, combs)
            sc_batches.append(score_combinations)
            rc_batches.append(rank_combinations)

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
        gt_tensor = torch.tensor(ground_truth.values)

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
        print(f"Data Items: {self.DATASET_LEN}\n")
        score_tensors = {d: torch.tensor(df.values, dtype=torch.float32) for d, df in self.score_data.items()}

        base_model_accuracies = self.get_accuracies(score_tensors.copy(), self.ground_truth.copy(), True)

        # Print Accuracies of the Base Models
        print("Base Models:")
        for i in base_model_accuracies:
            print(f"\t{i}: \t{base_model_accuracies[i]}")
        print()

        self.highest_score_value_pair = max(base_model_accuracies.items(), key=lambda item: item[1])
        self.highest_start = self.highest_score_value_pair[1]
        print("Start Top Model:", self.highest_score_value_pair)

        # Initialize Rank Variables using max Score info as base
        rank_tensors = {d: torch.tensor(df.values, dtype=torch.float32) for d, df in self.rank_data.items()}
        self.highest_rank_value_pair = self.highest_score_value_pair

        # Check for Tie Ranks
        # for i in rank_tensors:
        #     print(f"Tie Ranks {i}: {self.check_tie_ranks(rank_tensors[i])}")

        if self.PLOT_AVG_RSC: plot_average_rows(score_tensors, rank_tensors, self.OUTPATH, self.DATASET_LEN, 0)

        fusion_models_sc, fusion_models_rc = self.batch_combination(score_tensors, rank_tensors, base_model_accuracies, self.BATCH_SIZE)

        if matrices:
            return fusion_models_sc, fusion_models_rc
        else:
            max_s_val = self.highest_score_value_pair[1]
            max_r_val = max_s_val
            self.highest_score_value_pair, max_s_val = self.update_max(fusion_models_sc, self.highest_score_value_pair, max_s_val)
            self.highest_rank_value_pair, max_r_val = self.update_max(fusion_models_rc, self.highest_rank_value_pair, max_r_val)
            top_2 = [self.highest_score_value_pair, self.highest_rank_value_pair]
            print(f"End Top Model: {max(top_2, key=lambda x: x[1])}\n")
            print(f"{max(top_2, key=lambda x: x[1])[1] - self.highest_start}% improvement from base models.")
            print("Done!")

class InFusionNet(InFusionLayer):
    def __init__(self, score_data, ground_truth, OUTPATH, weighting_schemes, BATCH_SIZE) -> None:
        super().__init__(score_data, ground_truth, OUTPATH, weighting_schemes, BATCH_SIZE)

        IFL = InFusionLayer(score_data, ground_truth, OUTPATH, weighting_schemes = weighting_schemes, BATCH_SIZE = BATCH_SIZE)

        self.fusion_models_sc, self.fusion_models_rc = IFL.predict(matrices=True)
        self.ground_truth = IFL.ground_truth
        self.highest_start = IFL.highest_start
        self.highest_score_value_pair = IFL.highest_score_value_pair 
        self.highest_rank_value_pair = IFL.highest_rank_value_pair
        self.OUTPATH = IFL.OUTPATH

    def majority_vote(self):
        fusion_scores = tensor_indices(self.fusion_models_sc)
        fusion_ranks = tensor_indices(self.fusion_models_rc, use_argmin=True)
        fusion_scores = {"SC_" + key: np.array(value, dtype=np.int64) for key, value in fusion_scores.items()}
        fusion_ranks = {"RC_" + key: np.array(value, dtype=np.int64) for key, value in fusion_ranks.items()}
        fusion_dict = {**fusion_scores, **fusion_ranks}
        ensemble_with_ground_truth = JudgeMajorityVote(fusion_dict, self.ground_truth, use_ground_truth=True)
        ensemble_with_ground_truth.recursive_combinations(all=True)

    def top_k(self, fusion_models, ground_truth, max_val, k=5, scores=True):
        model_accuracies = self.get_accuracies(fusion_models.copy(), ground_truth.copy(), scores)
        top_performers = {key: value for key, value in model_accuracies.items() if value > max_val}
        top_k_models = dict(sorted(top_performers.items(), key=lambda x: x[1], reverse=True)[:k])
        return top_k_models
    
    # Expansion-Reduction Algorithm
    def expansion_reduction_1(self, plot_avg_rsc=False):
        fusion_models_sc, fusion_models_rc, highest_score_value_pair, highest_rank_value_pair, OUTPATH = self.fusion_models_sc, self.fusion_models_rc, self.highest_score_value_pair, self.highest_rank_value_pair, self.OUTPATH
        max_s_val = highest_score_value_pair[1]
        max_r_val = max_s_val
        for i in range(15): # turn to while (len(top_5) > 0)
            print(f"Optimization: {i+1}")
            
            if plot_avg_rsc: plot_average_rows(fusion_models_sc, fusion_models_rc, OUTPATH, self.DATASET_LEN, i+1)

            # Scores
            top_5_sc_models = self.top_k(fusion_models_sc, self.ground_truth, max_s_val, self.K, True)

            # Ranks
            top_5_rc_models = self.top_k(fusion_models_rc, self.ground_truth, max_r_val, self.K, False)

            # Pool SC & RC Models
            combined = {f'SC_{key}': value for key, value in top_5_sc_models.items()}
            combined.update({f'RC_{key}': value for key, value in top_5_rc_models.items()})
            top_k = sorted(combined.items(), key=lambda item: item[1], reverse=True)[:self.K]
            
            # Termination condition
            if len(top_k) == 0: break

            # Update the base models with the newly fused higher performing models
            self.base_models = [m[0] for m in top_k]
            model_accuracies = {model: accuracy for model, accuracy in top_k}

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
                fusion_models_sc, fusion_models_rc = self.batch_combination(score_tensors, rank_tensors, model_accuracies, self.BATCH_SIZE)

                # Filter out any invalid model combination tensors that contain NaN values
                fusion_models_sc = {k: v for k, v in fusion_models_sc.items() if not torch.isnan(v).any()}
                fusion_models_rc = {k: v for k, v in fusion_models_rc.items() if not torch.isnan(v).any()}
            else:
                break
        return highest_score_value_pair, highest_rank_value_pair

    def predict(self, model_selection="ER-Algorithm"):
        if model_selection == "ER-Algorithm":
            highest_score_value_pair, highest_rank_value_pair = self.expansion_reduction_1(self.PLOT_AVG_RSC)
            top_2 = [highest_score_value_pair, highest_rank_value_pair]
            print(f"End: {max(top_2, key=lambda x: x[1])}")
            print(f"{max(top_2, key=lambda x: x[1])[1] - self.highest_start}% improvement from base models.")
            print("Done!")
        else:
            self.majority_vote()

class InFusionRegression:
    def __init__(self, score_data, weighting_schemes = ['AC', 'WCDS']) -> None:
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