import matplotlib.pyplot as plt
import pandas as pd
import os
import math
import numpy as np

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def get_outputs(ROOT):
    score_data = {}
    ground_truth = None

    paths = os.listdir(ROOT)

    for path in paths:
        model = path.split("_")[0]
        if path.endswith('_scores.csv'):
            score_data[model] = pd.read_csv(f"{ROOT}/{path}").iloc[:,1:]
        elif path.endswith('.csv'):
            ground_truth = pd.read_csv(f"{ROOT}/{path}").iloc[:,1:]
        else:
            continue

    return score_data, ground_truth

def plot_rsc(data_1, data_2, OUTPATH, DATASET_LEN, iteration):
    # Perform the processing for both datasets
    datasets = [data_1.copy(), data_2.copy()]
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))  # Create 1 row and 2 columns of subplots
    
    for idx, scores in enumerate(datasets):
        ranks = scores.rank(ascending=False)
        labels = list(scores.keys())
        
        models = [f"M{i}" for i in range(len(labels))]
        for x, i in enumerate(labels):
            s_i_score = sorted(normalize(scores[i]), reverse=True)  # Assuming normalize is defined elsewhere
            s_i_rank = sorted(ranks[i], reverse=False)
            axes[idx].plot(s_i_rank, s_i_score, label=models[x])
        axes[idx].grid(True)
        # axes[idx].legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        C_TYPE = "Score Combination" if idx == 0 else "Rank Combination"
        
        axes[idx].set_title(f'{C_TYPE}: {len(axes[idx].lines)} models')
        axes[idx].set_xlabel('Classes/Ranks')
        axes[idx].set_ylabel(f"Normalized Scores")
    
    num_lines = len(axes[idx].lines)
    plural = 's ' if num_lines > 1 else ' '
    plt.suptitle(f'Averaged Rank-Score Characteristics\nD = {DATASET_LEN}, F = {math.floor(axes[0].get_xlim()[1])}', fontsize=20)
    plt.figtext(0.5, 0.01, f"Iteration {iteration}", ha="center", fontsize=16)
    plt.tight_layout()

    n = len(os.listdir(f'{OUTPATH}'))
    plt.savefig(f'{OUTPATH}/Avg RSC Iteration-{iteration}.png')
    plt.close()

def plot_fusion_model_accuracies(results, max_original, title):
    models = [f"M{i}" for i in range(len(results))]
    plt.bar(models, list(results.values()))
    plt.axhline(y=max_original, color='r', linestyle='--')
    plt.title(title)
    plt.xlabel("Models")
    plt.ylabel('Accuracy')
    plt.xticks(rotation=90) 
    plt.tight_layout()
    plt.show()
    plt.close()

def plot_average_rows(data_1, data_2, OUTPATH, DATASET_LEN, iteration):
    averages_1 = {}
    for key, tensor in data_1.items():
        # Compute the average across rows (dimension 0) for the tensor
        row_average = tensor.mean(dim=0).numpy()  # Assuming the tensor is 2D and on CPU
        averages_1[key] = row_average  # Store the average directly
    average_rows_df_1 = pd.DataFrame(averages_1)
    
    averages_2 = {}
    for key, tensor in data_2.items():
        # Compute the average across rows (dimension 0) for the tensor
        row_average = tensor.mean(dim=0).numpy()  # Assuming the tensor is 2D and on CPU
        averages_2[key] = row_average  # Store the average directly

    # Convert the averages to a pandas DataFrame
    average_rows_df_2 = pd.DataFrame(averages_2)
    plot_rsc(average_rows_df_1, average_rows_df_2, OUTPATH, DATASET_LEN, iteration)

def tensor_indices(tensor_dict, use_argmin=False):
    """
    Converts a dictionary of tensors (2D lists) into a dictionary where each tensor
    is represented as a list of indices of either the maximum or minimum element in each row,
    based on the use_argmin parameter.

    :param tensor_dict: dict, where keys are identifiers and values are lists of lists (tensors)
    :param use_argmin: bool, if True, finds the index of the minimum element; if False, finds the index of the maximum element
    :return: dict, same structure but each list in the tensor is replaced by the index of the min/max element in each row
    """
    indices_dict = {}
    for key, tensor in tensor_dict.items():
        # Convert list of lists to a numpy array for efficient processing
        np_tensor = np.array(tensor)
        # Find the indices of the max or min element in each row based on the use_argmin flag
        if use_argmin:
            indices = np.argmin(np_tensor, axis=1)
        else:
            indices = np.argmax(np_tensor, axis=1)
        # Store the result in the dictionary
        indices_dict[key] = indices.tolist()  # Convert numpy array to list
    return indices_dict