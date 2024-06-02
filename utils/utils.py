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

def plot_average_rsc(data_1, data_2, OUTPATH, DATASET_LEN, iteration):
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

def plot_rsc(data_1, data_2, OUTPATH, DATASET_LEN, iteration):
    # Perform the processing for both datasets
    datasets = [data_1.copy(), data_2.copy()]
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))  # Create 1 row and 2 columns of subplots
    
    for idx, scores in enumerate(datasets):
        ranks = scores.rank(ascending=False)
        labels = list(scores.keys())
        
        models = labels if iteration == 0 else [f"M{i}" for i in range(len(labels))]

        for x, i in enumerate(labels):
            s_i_score = sorted(normalize(scores[i]), reverse=True)  # Assuming normalize is defined elsewhere
            s_i_rank = sorted(ranks[i], reverse=False)
            axes[idx].plot(s_i_rank, s_i_score, label=models[x])
        axes[idx].grid(True)
        # axes[idx].legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        if iteration == 0:
            C_TYPE = "Scores" if idx == 0 else "Ranks"
        else:
            C_TYPE = "Score Combination" if idx == 0 else "Rank Combination"
        
        axes[idx].set_title(f'{C_TYPE}: {len(axes[idx].lines)} models')
        axes[idx].set_xlabel('Class Ranks')
        axes[idx].set_ylabel(f"Normalized Scores")
    
    # Add legend only for iteration 0
    if iteration == 0:
        for ax in axes:
            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    num_lines = len(axes[idx].lines)
    plural = 's ' if num_lines > 1 else ' '
    plt.suptitle(f'Rank-Score Characteristics', fontsize=20)
    plt.figtext(0.5, 0.01, f"Iteration {iteration}", ha="center", fontsize=16)
    plt.tight_layout()

    n = len(os.listdir(f'{OUTPATH}'))
    plt.savefig(f'{OUTPATH}/RSC Iteration-{iteration}.png')
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

# def plot_average_rows(data_1, data_2, OUTPATH, DATASET_LEN, iteration):
#     averages_1 = {}
#     for key, tensor in data_1.items():
#         # Compute the average across rows (dimension 0) for the tensor
#         row_average = tensor.mean(dim=0).numpy()  # Assuming the tensor is 2D and on CPU
#         averages_1[key] = row_average  # Store the average directly
#     average_rows_df_1 = pd.DataFrame(averages_1)
    
#     averages_2 = {}
#     for key, tensor in data_2.items():
#         # Compute the average across rows (dimension 0) for the tensor
#         row_average = tensor.mean(dim=0).numpy()  # Assuming the tensor is 2D and on CPU
#         averages_2[key] = row_average  # Store the average directly

#     # Convert the averages to a pandas DataFrame
#     average_rows_df_2 = pd.DataFrame(averages_2)
#     plot_rsc(average_rows_df_1, average_rows_df_2, OUTPATH, DATASET_LEN, iteration)

def plot_average_rows(data_1, data_2, OUTPATH, DATASET_LEN, iteration):
    selected_rows_1 = {}
    for key, tensor in data_1.items():
        # Select the first row (or any specific row) from the tensor
        selected_row = tensor[0].numpy()  # Assuming the tensor is 2D and on CPU
        selected_rows_1[key] = selected_row  # Store the selected row directly
    selected_rows_df_1 = pd.DataFrame(selected_rows_1)
    
    selected_rows_2 = {}
    for key, tensor in data_2.items():
        # Select the first row (or any specific row) from the tensor
        selected_row = tensor[0].numpy()  # Assuming the tensor is 2D and on CPU
        selected_rows_2[key] = selected_row  # Store the selected row directly

    # Convert the selected rows to a pandas DataFrame
    selected_rows_df_2 = pd.DataFrame(selected_rows_2)
    plot_rsc(selected_rows_df_1, selected_rows_df_2, OUTPATH, DATASET_LEN, iteration)

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

def plot_model_performance(OUTPATH, csv_file_path, A , B, C, D, E):
    df = pd.read_csv(csv_file_path)

    # Mapping old names to new simplified names
    def simplify_name(name):
        return name.replace(A, "A")\
                   .replace(B, "B")\
                   .replace(C, "C")\
                   .replace(D, "D")\
                   .replace(E, "E")

    # Simplify model names
    df["Model"] = df["Model"].apply(simplify_name)
    
    # Set rank values to 0 if the model name is just "A", "B", "C", "D", or "E"
    important_models = ["A", "B", "C", "D", "E"]
    df.loc[df["Model"].isin(important_models), "RC"] = df["SC"]
    
    top_performing_value = df[df["Model"].isin(important_models)][["SC"]].max().max()

    # Determine the lowest value in the data and set the bottom range of the y-axis
    lowest_value = df[["SC"]].min().min()
    bottom_y_value = lowest_value - 10

    # Dictionary to map categories to titles
    category_titles = {
        "_AC": "Average Combination",
        "_WCP": "Weighted Combination by Performance",
        "_WC-CDS": "Weighted Combination by Cognitive Diversity Strength",
        "_WC-KDS": "Weighted Combination by Ksi Diversity Strength"
    }

    # Function to plot the graph for each category
    def plot_category(category):
        filtered_df = df[df["Model"].str.endswith(category) | df["Model"].isin(important_models)]
        
        plt.figure(figsize=(18, 10))

        bar_width = 0.35
        index = range(len(filtered_df))

        bars1 = plt.bar(index, filtered_df["SC"], width=bar_width, color='blue', label='Score Combination')
        bars2 = plt.bar([i + bar_width for i in index], filtered_df["RC"], width=bar_width, color='red', label='Rank Combination')

        # Change the color of important models to green
        for i, model in enumerate(filtered_df["Model"]):
            if model in important_models:
                bars1[i].set_color('green')
                bars2[i].set_color('green')

        # Simplify the x-axis labels by removing everything after and including "_"
        x_labels = [label.split("_")[0] for label in filtered_df["Model"]]

        # Add labels
        plt.xlabel('Models', fontweight='bold')
        plt.ylabel('Accuracies', fontweight='bold')
        DATASET = OUTPATH.split("/")[-1]
        plt.title(f'{DATASET}\n{category_titles[category]}', fontweight='bold')

        # Adjust xticks to be at the center of grouped bars
        plt.xticks([i + bar_width / 2 for i in index], x_labels, rotation=90, ha='center')

        # Set y-axis limits
        plt.ylim(bottom=bottom_y_value, top=100)

        # Bold important models
        for i, label in enumerate(plt.gca().get_xticklabels()):
            if filtered_df.iloc[i]["Model"] in important_models:
                label.set_fontweight('bold')
                label.set_color('green')  # Change color to indicate importance

        # Create legend
        legend_elements = [
            plt.Line2D([0], [0], color='blue', lw=4, label='Score Combination'),
            plt.Line2D([0], [0], color='red', lw=4, label='Rank Combination'),
            plt.Line2D([0], [0], color='green', lw=4, label='Base Model')
        ]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        # Add a black dashed horizontal line at the top performing value among important models
        plt.axhline(y=top_performing_value, color='black', linestyle='--')

        # Adjust layout to increase distance between x-axis ticks
        plt.tight_layout()

        # Display the graph
        plt.savefig(f"{OUTPATH}/{category_titles[category]}.png")
    
    # Plot each category
    categories = ["_AC", "_WC-KDS", "_WCP", "_WC-CDS"]
    for category in categories:
        plot_category(category)