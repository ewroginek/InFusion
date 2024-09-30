import matplotlib.pyplot as plt
import pandas as pd
import os
import math
import numpy as np
from itertools import combinations
import torch
import csv

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
        
        models = [chr(65 + i) for i in range(len(labels))]
        for x, i in enumerate(labels):
            s_i_score = sorted(normalize(scores[i]), reverse=True)  # Assuming normalize is defined elsewhere
            s_i_rank = sorted(ranks[i], reverse=False)
            axes[idx].plot(s_i_rank, s_i_score, label=models[x])
        axes[idx].grid(True)
        # axes[idx].legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        if iteration == 0:
            C_TYPE = "Rank-Score Characteristics" if idx == 0 else "Rank-RankScore Characteristics"
        else:
            C_TYPE = "Score Combination" if idx == 0 else "Rank Combination"
        
        # Add legend only for iteration 0
        if iteration == 0:
            for ax in axes:
                ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

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

def plot_rsc(data_1, data_2, OUTPATH, DATA_ITEM, iteration):
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
            C_TYPE = "Rank-Score Characteristics" if idx == 0 else "Rank-RankScore Characteristics"
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
    plt.savefig(f'{OUTPATH}/RSC Data Item {DATA_ITEM}; Iteration-{iteration}.png')
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
    plot_average_rsc(average_rows_df_1, average_rows_df_2, OUTPATH, DATASET_LEN, iteration)

from rank_score_characteristic import RankScoreCharacteristic
def plot_single_row(data_1, data_2, OUTPATH, DATA_ITEM, iteration):
    selected_rows_1 = {}
    # print(data_1)
    # RSC = RankScoreCharacteristic(False)
    # print(RSC.diversity_strength(data_1))
    for key, tensor in data_1.items():
        # Select the first row (or any specific row) from the tensor
        selected_row = tensor[DATA_ITEM].numpy()  # Assuming the tensor is 2D and on CPU
        selected_rows_1[key] = selected_row  # Store the selected row directly
    selected_rows_df_1 = pd.DataFrame(selected_rows_1)
    
    selected_rows_2 = {}
    for key, tensor in data_2.items():
        # Select the first row (or any specific row) from the tensor
        selected_row = tensor[DATA_ITEM].numpy()  # Assuming the tensor is 2D and on CPU
        selected_rows_2[key] = selected_row  # Store the selected row directly

    # Convert the selected rows to a pandas DataFrame
    selected_rows_df_2 = pd.DataFrame(selected_rows_2)
    plot_rsc(selected_rows_df_1, selected_rows_df_2, OUTPATH, DATA_ITEM, iteration)

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

def plot_model_performance_histogram(OUTPATH, csv_file_path, A , B, C, D, E):
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
        "_WC-CDS": "Weighted Combination by Diversity Strength",
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


def plot_model_performance(OUTPATH, csv_file_path, A, B, C, D, E):
    """
    Plot model combination results in the same manner as shown in Yang et al. (2005)
    """
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

    # Set rank values to be equal to "SC" if the model name is just "A", "B", "C", "D", or "E"
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
        "_WC-CDS": "Weighted Combination by Diversity Strength",
        "_WC-KDS": "Weighted Combination by Ksi Diversity Strength"
    }

    # Function to generate combinations and sort them by RC values
    def get_sorted_combinations(group_size):
        comb_data = []
        combs = combinations(important_models, group_size)
        for comb in combs:
            comb_str = ''.join(comb)
            model_name = f"{comb_str}{category}"
            if model_name in filtered_df["Model"].values:
                rc_value = filtered_df[filtered_df["Model"] == model_name]["RC"].values[0]
                sc_value = filtered_df[filtered_df["Model"] == model_name]["SC"].values[0]
                comb_data.append((comb_str, sc_value, rc_value))
            elif comb_str in important_models:  # Check if it's an important model without suffix
                if comb_str in filtered_df["Model"].values:
                    rc_value = filtered_df[filtered_df["Model"] == comb_str]["RC"].values[0]
                    sc_value = filtered_df[filtered_df["Model"] == comb_str]["SC"].values[0]
                    comb_data.append((comb_str, sc_value, rc_value))
        comb_data.sort(key=lambda x: x[2])  # Sort by RC values
        return comb_data

    # Function to plot the graph for each category
    def plot_category(category):
        global filtered_df
        filtered_df = df[df["Model"].str.endswith(category) | df["Model"].isin(important_models)]

        plt.figure(figsize=(18, 10))

        all_x_labels = []
        all_sc_values = []
        all_rc_values = []
        group_boundaries = []

        # Generate and sort combinations for each group size
        for group_size in range(1, len(important_models) + 1):
            sorted_combinations = get_sorted_combinations(group_size)
            if sorted_combinations:
                all_x_labels.extend([item[0] for item in sorted_combinations])
                all_sc_values.extend([item[1] for item in sorted_combinations])
                all_rc_values.extend([item[2] for item in sorted_combinations])
                group_boundaries.append(len(all_x_labels) - 0.5)

        # Plot the scores and ranks as line graphs
        plt.plot(all_x_labels, all_sc_values, marker='o', linestyle='-', color='blue', label='Score Combination')
        plt.plot(all_x_labels, all_rc_values, marker='x', linestyle='-', color='red', label='Rank Combination')

        # Add vertical dashed lines at group boundaries
        for boundary in group_boundaries[:-1]:  # Exclude the last boundary as it is at the end of the plot
            plt.axvline(x=boundary, color='black', linestyle='--')

        # Add labels
        plt.xlabel('Model Combinations', fontweight='bold', fontsize=20)
        plt.ylabel('Accuracies', fontweight='bold', fontsize=22)
        DATASET = OUTPATH.split("/")[-1]
        plt.title(f'{DATASET}\n{category_titles[category]}', fontweight='bold', fontsize=28)

        # Set y-axis limits
        plt.ylim(bottom=bottom_y_value, top=100)

        # Set x-axis limits to ensure the first/last tick is the same distance as between two ticks
        plt.xlim(-0.5, len(all_x_labels) - 0.5)

        # Rotate x-axis labels and set font size
        plt.xticks(rotation=90, fontsize=16)
        plt.yticks(fontsize=16)

        # Bold important models
        for tick_label in plt.gca().get_xticklabels():
            if tick_label.get_text() in important_models:
                tick_label.set_fontweight('bold')
                tick_label.set_color('green')  # Change color to indicate importance

        # Create legend with larger font size
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., fontsize=18)

        # Add a black dashed horizontal line at the top performing value among important models
        plt.axhline(y=top_performing_value, color='black', linestyle='--')

        # Adjust layout to increase distance between x-axis ticks
        plt.tight_layout()
        plt.grid(True)

        # Save the graph
        plt.savefig(f"{OUTPATH}/{category_titles[category]}.png")
        plt.close()

    # Plot each category
    categories = ["_AC", "_WC-KDS", "_WCP", "_WC-CDS"]
    for category in categories:
        plot_category(category)


def average_multiclassification_accuracy(predictions1, predictions2_tuple, ground_truth, output_dir):
    # Convert ground_truth to a PyTorch tensor if it's not already
    if isinstance(ground_truth, (pd.Series, pd.DataFrame)):
        ground_truth = torch.tensor(ground_truth.values)
    elif not isinstance(ground_truth, torch.Tensor):
        raise ValueError("Ground truth should be a pandas DataFrame, Series, or a PyTorch tensor.")
    
    # Ensure ground_truth is a 1D tensor
    if ground_truth.ndim > 1:
        ground_truth = ground_truth.squeeze()

    # Function to calculate multi-class accuracy for each class using max or min column value
    def calculate_class_accuracy(predictions, ground_truth, scores=True):
        n_classes = predictions.shape[1]
        class_correct = torch.zeros(n_classes)
        class_counts = torch.zeros(n_classes)
        class_predictions = torch.argmax(predictions, dim=1) if scores else torch.argmin(predictions, dim=1)
        for class_idx in range(n_classes):
            mask = ground_truth == class_idx
            class_correct[class_idx] = (class_predictions[mask] == ground_truth[mask]).sum().float()
            class_counts[class_idx] = mask.sum().float()
        class_accuracies = class_correct / class_counts
        return class_accuracies

    # Verify the predictions dictionary structure
    def verify_predictions(predictions):
        if isinstance(predictions, dict) and all(isinstance(v, torch.Tensor) for v in predictions.values()):
            return True
        return False

    if verify_predictions(predictions1) and all(verify_predictions(pred) for pred in predictions2_tuple):
        # Calculate accuracy for each model in predictions1
        def get_model_accuracies(predictions, accuracy_function, scores=True):
            model_accuracies = []
            for model_name, prediction_tensor in predictions.items():
                accuracies = accuracy_function(prediction_tensor, ground_truth, scores)
                model_accuracies.append(accuracies)
            return torch.stack(model_accuracies)

        model_accuracies1 = get_model_accuracies(predictions1, calculate_class_accuracy, scores=True)
        model_accuracies2_max = get_model_accuracies(predictions2_tuple[0], calculate_class_accuracy, scores=True)
        model_accuracies2_min = get_model_accuracies(predictions2_tuple[1], calculate_class_accuracy, scores=False)

        # Calculate the average accuracy for each class across all models
        average_accuracies1 = model_accuracies1.mean(dim=0)
        average_accuracies2_max = model_accuracies2_max.mean(dim=0)
        average_accuracies2_min = model_accuracies2_min.mean(dim=0)
        
        # Average the accuracies from predictions2
        average_accuracies2 = (average_accuracies2_max + average_accuracies2_min) / 2

        # Plotting the accuracies
        classes = list(range(len(average_accuracies1)))
        plt.scatter(classes, average_accuracies1, label='Base Models')
        plt.scatter(classes, average_accuracies2, label='Comb Models')
        plt.xlabel('Class')
        plt.ylabel('Accuracy')
        plt.title('Average Accuracy per Class Across Models')
        plt.ylim([0, 1])
        plt.xticks(classes)  # Set x-axis ticks to be integers only
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
        plt.tight_layout()

        # Save the plot to the specified output directory
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'average_multiclassification_accuracy.png')
        plt.savefig(output_path)
        plt.close()

        return average_accuracies1, average_accuracies2
    else:
        raise ValueError("Both predictions should be dictionaries with tensor values.")

def write_dicts_to_csv(dict_list, filename):
    if not dict_list:
        return  # If the list is empty, there's nothing to write
    
    # Get the keys from the first dictionary as the column headers
    keys = dict_list[0].keys()
    
    with open(filename, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(dict_list)

from PIL import Image, ImageDraw, ImageFont
import pandas as pd

def annotate_image_with_csv(image_path, csv_path, DATA_ITEM):
    # Open the image file
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Load a font
    font = ImageFont.truetype("arial.ttf", 14)  # Larger font size

    # Load csv data
    csv_data = pd.read_csv(csv_path)
    print(csv_data)
    data_item = csv_data.iloc[DATA_ITEM]

    # Prepare the annotation text
    annotation_text = 'Diversity Strength\n\n' + '\n'.join([f"{key}: {round(value, 2)}" for key, value in data_item.items()])

    # Define position for the text (top-left corner of the image)
    position = (837, 200)

    # # Calculate text size for a background rectangle using textbbox (bounding box)
    bbox = draw.textbbox(position, annotation_text, font=font)
    background_rect = [(bbox[0] - 5, bbox[1] - 5), (bbox[2] + 5, bbox[3] + 5)]  # Add padding around the text

    # Draw a white rectangle with a black border for better contrast
    draw.rectangle(background_rect, fill="white", outline="black", width=3)  # White fill, black border

    # Annotate the image with the text
    draw.text(position, annotation_text, font=font, fill="black")

    # Save the annotated image
    image.save(image_path)