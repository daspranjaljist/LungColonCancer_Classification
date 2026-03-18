import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import os

def plot_confusion_matrix(y_true, y_pred, classes, save_path='results/confusion_matrix.png'):
    """
    Generates and saves a Confusion Matrix heatmap.
    
    Args:
        y_true (list/array): True labels.
        y_pred (list/array): Predicted labels.
        classes (list): List of class names.
        save_path (str): Path to save the figure.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path)
    plt.show()
    print(f"Confusion matrix saved to {save_path}")

def plot_roc_curve(y_true_indices, y_score, classes, save_path='results/roc_curve.png'):
    """
    Generates and saves a multi-class ROC Curve.
    
    Args:
        y_true_indices (list/array): True labels (integer indices).
        y_score (list/array): Predicted probabilities for each class (shape: n_samples, n_classes).
        classes (list): List of class names.
        save_path (str): Path to save the figure.
    """
    # Binarize the output (One-hot encoding) for multi-class ROC
    y_true_bin = label_binarize(y_true_indices, classes=range(len(classes)))
    n_classes = len(classes)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    
    # Plot Micro-average
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-average ROC curve (area = {roc_auc["micro"]:.2f})',
             color='deeppink', linestyle=':', linewidth=4)

    # Plot individual class curves
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {classes[i]} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path)
    plt.show()
    print(f"ROC curve saved to {save_path}")