import math

import matplotlib.pyplot as plt
import numpy as np

from .data import SCVIC_CIDS_CLASSES, SCVIC_CIDS_CLASSES_INV


def plot_confusion_matrixies(confusion_matrices, labels=list(SCVIC_CIDS_CLASSES.keys())):
    rows = math.ceil(len(confusion_matrices) / 3)
    fig, axes = plt.subplots(rows, 3, figsize=(15, rows * 5))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i >= len(confusion_matrices):
            ax.axis('off')
            continue
        key = list(confusion_matrices.keys())[i]
        cax = ax.matshow(confusion_matrices[key], cmap='YlOrRd')
        fig.colorbar(cax, ax=ax)
        ax.set_title(f'Trial {key}')
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    plt.tight_layout()
    plt.show()

def plot_confusion_matrixies_binary(confusion_matrices, labels=list(SCVIC_CIDS_CLASSES.keys())):
    rows = math.ceil(len(confusion_matrices) / 3)
    fig, axes = plt.subplots(rows, 3, figsize=(9, rows * 5))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i >= len(confusion_matrices):
            ax.axis('off')
            continue
        key = list(confusion_matrices.keys())[i]
        cax = ax.matshow(confusion_matrices[key], cmap='YlOrRd')
        fig.colorbar(cax, ax=ax)
        ax.set_title(f'Trial {key}')
        ax.set_xticks(np.arange(2))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(["Benign", "Malicious"], rotation=90)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    plt.tight_layout()
    plt.show()

def plot_boxplot_accuracies_leaf_out(confusion_matrices, configs, benign_class=0):
    accuracies = []
    labels = []
    for (k, v) in confusion_matrices.items():
        config = configs[k]
        exclude_classes = [SCVIC_CIDS_CLASSES[cls] for cls in config['dataset']['exclude']]
        accuracy = calculate_accuracy_benign(v, exclude_classes, benign_class)
        accuracies.append(accuracy)
        labels.append(f"Exclude {', '.join(map(str, exclude_classes))}")

    plt.figure(figsize=(10, 6))
    plt.boxplot(accuracies, tick_labels=labels)
    plt.xlabel('Left-out Classes')
    plt.ylabel('Accuracy')
    plt.title(f'Classification of left-out-classes as malicious for {config["experiment"]["model_type"]}')
    plt.xticks(rotation=45)
    # Add a textbox on the right of the plot that shows the SCVIC dict
    textstr = '\n'.join([f'{k}: {v}' for k, v in SCVIC_CIDS_CLASSES_INV.items()])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gcf().text(0.95, 0.5, textstr, fontsize=12, verticalalignment='center', bbox=props)
    plt.show()

def boxplot_metric(confusion_matrices: dict[str: np.ndarray], metric=calculate_accuracy, labels=list(SCVIC_CIDS_CLASSES.keys())[1:]):
    total_metric = {}
    classwise_metric = {key: {} for key in confusion_matrices.keys()}

    for key, matrix in confusion_matrices.items():

        total_matrix = np.zeros((matrix.shape[0], 2, 2))
        total_matrix[:, 0, :] = matrix[:, 0, :]
        total_matrix[:, 1, :] = np.sum(matrix[:, 1:, :], axis=1)

        total_metric[key] = np.squeeze(metric(confusion=total_matrix))
        class_wise = metric(confusion=matrix)
        num_classes = len(labels)
        for cls in range(num_classes):
            
            classwise_metric[key][labels[cls]] = class_wise[:, cls]
    avg_scores = {cls: np.mean([classwise_metric[key][cls] for key in confusion_matrices.keys()]) for cls in labels}
    textstr = '\n'.join([f'{cls}: {avg_scores[cls]:.2f}' for cls in labels])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gcf().text(0.95, 0.5, textstr, fontsize=12, verticalalignment='center', bbox=props)
    plt.figure(figsize=(12, 6))
    plt.boxplot([total_metric[key] for key in confusion_matrices.keys()], tick_labels=confusion_matrices.keys())
    plt.xlabel('Metric')
    plt.title('Total Metric')
    plt.show()

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    for i, key in enumerate(confusion_matrices.keys()):
        ax = axes[i]
        ax.boxplot([classwise_metric[key][cls] for cls in labels], tick_labels=labels)
        ax.set_xlabel('Metric')
        ax.set_title(f'Classwise Metric for {key}')
    
    for ax in axes:
        for label in ax.get_xticklabels():
            label.set_rotation(90)
    plt.tight_layout()
    plt.show()
