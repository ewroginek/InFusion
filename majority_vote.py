import numpy as np
from itertools import combinations

class MajorityRankVoter:
    def __init__(self, prediction_vectors, ground_truth=None, use_ground_truth=True):
        """
        Initialize the MajorityRankVoter with a dictionary of prediction vectors and optionally the ground truth.
        
        :param prediction_vectors: dict, where each key is a model identifier and each value is a list of predicted classes
        :param ground_truth: list or None, the actual labels for each instance, optional if use_ground_truth is False
        :param use_ground_truth: bool, determines whether to use ground truth for evaluating combinations
        """
        self.prediction_vectors = prediction_vectors
        self.ground_truth = ground_truth
        self.use_ground_truth = use_ground_truth
        self.best_accuracy = 0
        self.best_model = []

    def majority_vote(self, predictions):
        """
        Compute the majority vote across selected models for each instance.
        
        :param predictions: list of lists, where each sublist is model predictions
        :return: A list representing the combined output vector where each element is the majority voted class.
        """
        predictions_array = np.array(predictions).T
        majority_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=predictions_array)
        return majority_vote.tolist()

    def calculate_accuracy(self, combined_output):
        """
        Calculate the accuracy of a combined model output by comparing it to the ground truth.
        
        :return: float, accuracy of the ensemble
        """
        if not self.use_ground_truth:
            return None
        correct_predictions = sum(1 for gt, pred in zip(self.ground_truth, combined_output) if gt == pred)
        return correct_predictions / len(self.ground_truth)

    def evaluate_combinations(self):
        """
        Evaluate all combinations of models starting from combinations of 2.
        
        :return: dict, keys are tuples of model combinations and values are accuracies or prediction vectors
        """
        combinations_results = {}
        keys = list(self.prediction_vectors.keys())
        num_models = len(keys)

        for r in range(2, num_models + 1):
            for combo in combinations(keys, r):
                predictions = [self.prediction_vectors[key] for key in combo]
                combined_output = self.majority_vote(predictions)
                if self.use_ground_truth:
                    accuracy = self.calculate_accuracy(combined_output)
                    combinations_results[combo] = accuracy
                else:
                    combinations_results[combo] = combined_output

        return combinations_results

    def recursive_combinations(self):
        current_vectors = self.prediction_vectors.copy()
        iteration = 0
        while True:
            iteration += 1
            print(f"Starting iteration {iteration}")
            self.prediction_vectors = current_vectors
            results = self.evaluate_combinations()
            sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True) if self.use_ground_truth else results.items()

            if not results:
                print(f"No further combinations generated at iteration {iteration}. Stopping.")
                break

            if self.use_ground_truth:
                if iteration > 1 and sorted_results[0][1] <= self.best_accuracy:
                    print(f"No further improvement at iteration {iteration}. Stopping.")
                    break

                # Update best model if using ground truth
                self.best_accuracy = sorted_results[0][1]
                self.best_model = sorted_results[0][0]
                print(f"Top model at iteration {iteration}: {self.best_model}, Accuracy: {self.best_accuracy:.2f}")

                # Prepare top 5 new models for the next iteration
                top_combinations = sorted_results[:5] if self.use_ground_truth else list(results.items())[:5]
                current_vectors = {f"C{iteration}({''.join(combo[0])})": self.majority_vote([self.prediction_vectors[key] for key in combo[0]]) for i, combo in enumerate(top_combinations, 1)}
            else:
                top_combinations = sorted_results if self.use_ground_truth else list(results.items())
                current_vectors = {f"C{iteration}({''.join(combo[0])})": self.majority_vote([self.prediction_vectors[key] for key in combo[0]]) for i, combo in enumerate(top_combinations, 1)}

            # If not using ground truth, execute loop only once
            if not self.use_ground_truth:
                print(f"Generated model combinations without using ground truth at iteration {iteration}.")
                break

        if self.use_ground_truth:
            print(f"Final best model: C{iteration-1}{self.best_model}, Best accuracy: {self.best_accuracy:.2f}")
        else:
            return current_vectors  # Return dictionary of all model combinations without evaluation if not using ground truth


# Example usage (assuming we have prediction_vectors and optionally ground_truth defined elsewhere)
import torch
# Example usage
prediction_vectors = {
    'm1': torch.tensor([0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1]),
    'm2': torch.tensor([1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1]),
    'm3': torch.tensor([0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1]),
    'm4': torch.tensor([1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0]),
    'm5': torch.tensor([1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0])
}
# Correcting the ground truth definition and running the model ensemble

ground_truth = [0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0]  # Complete ground truth vector

# Create an instance of MajorityRankVoter with ground truth for accuracy evaluation
ensemble_with_ground_truth = MajorityRankVoter(prediction_vectors, ground_truth, use_ground_truth=True)
ensemble_with_ground_truth.recursive_combinations()

# Create an instance of MajorityRankVoter without ground truth to create combinations only
# ensemble_without_ground_truth = MajorityRankVoter(prediction_vectors_multiclass, use_ground_truth=False)
# combinations_without_ground_truth = ensemble_without_ground_truth.recursive_combinations()

# Print the output from the non-evaluative combinations
# print("Combinations without using ground truth:")
# for model_name, predictions in combinations_without_ground_truth.items():
#     print(f"{model_name}: {predictions}")