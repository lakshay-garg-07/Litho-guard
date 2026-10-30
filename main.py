"""Lithography Defect Prediction from Focus and Dose.

This script implements a machine learning pipeline to predict lithography defects
based on scanner Focus (nm) and Dose (mJ). It merges exposure logs with wafer 
inspection data, trains three distinct classifiers (MLP, Gaussian Process, SVM), 
evaluates their performance, and visualizes the decision boundaries.

Outputs (saved to the specified output directory):
    - raw_data.png: Scatter plot of the raw merged dataset.
    - decision_boundaries.png: Contour plots showing the learned boundaries for each model.
    - metrics.csv: Classification metrics (Accuracy, Precision, Recall, F1) for the defect class.

Usage:
    python litho_defect_prediction.py \
        --exposure data/exposure_data.csv \
        --inspection data/inspection_data.csv \
        --out results
"""

import argparse
from pathlib import Path

import matplotlib
# Use 'Agg' backend for headless environments (e.g., CI/CD pipelines, servers)
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Define columns used to join exposure and inspection datasets
MERGE_KEYS = ["Lot", "Wafer", "FieldX", "FieldY"]
SEED = 42


def find_col(df, keyword, fallback_idx):
    """
    Locates a column in a DataFrame by keyword, falling back to a positional index.

    Args:
        df (pd.DataFrame): The DataFrame to search.
        keyword (str): The substring to look for in the column names.
        fallback_idx (int): The positional index to use if the keyword is not found.

    Returns:
        str: The name of the matched or fallback column.
    """
    for col in df.columns:
        if keyword in str(col).lower():
            return col
    return df.columns[fallback_idx]


def load_data(exposure_path, inspection_path):
    """
    Loads, merges, and preprocesses the exposure and inspection datasets.

    Args:
        exposure_path (str): File path to the exposure CSV data.
        inspection_path (str): File path to the inspection CSV data.

    Returns:
        tuple: (X, y) where X is a numpy array of features (Focus, Dose) 
               and y is a numpy array of binary labels (0 = Good, 1 = Defect).
    """
    # Load datasets and remove rows with missing values
    exposure = pd.read_csv(exposure_path).dropna()
    inspection = pd.read_csv(inspection_path).dropna()
    
    # Merge on lot, wafer, and field coordinates
    merged = pd.merge(exposure, inspection, how="inner", on=MERGE_KEYS)

    # Dynamically locate feature and target columns based on keywords
    focus = find_col(merged, "focus", 9)
    dose = find_col(merged, "dose", 10)
    label = find_col(merged, "label", 12)
    
    print(f"Using columns -> focus: {focus!r}, dose: {dose!r}, label: {label!r}")

    # Extract features (X) and target variable (y)
    X = merged[[focus, dose]].to_numpy(dtype=float)
    y = merged[label].to_numpy().astype(int)
    
    print(f"Dataset summary | Total samples: {len(y)} | Defects: {int(y.sum())} | Good: {int((y == 0).sum())}")
    
    return X, y


def build_models():
    """
    Initializes the machine learning classifiers used for defect prediction.

    Returns:
        dict: A dictionary mapping model names to instantiated scikit-learn models.
    """
    return {
        "MLP": MLPClassifier(hidden_layer_sizes=(40, 40, 40), max_iter=1000, random_state=SEED),
        "GP": GaussianProcessClassifier(kernel=1.0 * RBF(1.0), random_state=SEED),
        "SVM": SVC(gamma=2, C=1, random_state=SEED),
    }


def scatter_points(ax, X, y):
    """
    Plots a 2D scatter of the dataset, color-coded by class.

    Args:
        ax (matplotlib.axes.Axes): The axis to plot on.
        X (np.ndarray): Feature matrix (Focus, Dose).
        y (np.ndarray): Label vector.
    """
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=20, c="C0", edgecolor="k", linewidths=0.5, label="Good")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=20, c="C3", edgecolor="k", linewidths=0.5, label="Defect")
    ax.set_xlabel("Focus [nm]")
    ax.set_ylabel("Dose [mJ]")


def plot_boundary(ax, model, scaler, X, y, title):
    """
    Generates a contour plot representing the model's decision boundary.

    Args:
        ax (matplotlib.axes.Axes): The axis to plot on.
        model (sklearn.base.BaseEstimator): A trained classifier.
        scaler (sklearn.preprocessing.StandardScaler): The fitted feature scaler.
        X (np.ndarray): Original feature matrix (used to determine plot limits).
        y (np.ndarray): Original label vector.
        title (str): The title for the subplot (usually the model name).
    """
    # Scale the original points to determine the grid boundaries in scaled space
    Xs = scaler.transform(X)
    
    # Create a mesh grid over the scaled feature space
    xx, yy = np.meshgrid(
        np.linspace(Xs[:, 0].min() - 0.5, Xs[:, 0].max() + 0.5, 150),
        np.linspace(Xs[:, 1].min() - 0.5, Xs[:, 1].max() + 0.5, 150),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Predict over the entire grid
    z = model.predict(grid).reshape(xx.shape)
    
    # Inverse transform the grid back to real physical units (nm, mJ) for plotting
    grid_orig = scaler.inverse_transform(grid)
    
    # Plot the decision contour and overlay the actual data points
    ax.contourf(
        grid_orig[:, 0].reshape(xx.shape),
        grid_orig[:, 1].reshape(xx.shape),
        z, cmap="RdYlBu_r", alpha=0.6,
    )
    scatter_points(ax, X, y)
    
    ax.set_title(title)
    ax.legend(loc=4)


def main():
    """
    Main execution pipeline: parses arguments, loads data, trains models, and saves results.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exposure", default="data/exposure_data.csv", help="Path to exposure logs.")
    parser.add_argument("--inspection", default="data/inspection_data.csv", help="Path to inspection results.")
    parser.add_argument("--out", default="results", help="Directory to save output plots and metrics.")
    args = parser.parse_args()

    # Ensure the output directory exists
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    X, y = load_data(args.exposure, args.inspection)

    # 2. Visualize and save raw data distribution
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    scatter_points(ax, X, y)
    ax.set_title("Raw data")
    ax.legend(loc=4)
    fig.savefig(out / "raw_data.png", bbox_inches="tight")
    plt.close(fig)

    # 3. Train-Test Split & Scaling
    # We split first to prevent data leakage, ensuring the scaler only sees the training set.
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # 4. Train Models and Visualize Boundaries
    models = build_models()
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5.5), dpi=100)
    rows = []
    
    for ax, (name, model) in zip(np.atleast_1d(axes), models.items()):
        print(f"\n=== Training {name} ===")
        model.fit(X_tr_s, y_tr)
        pred = model.predict(X_te_s)
        
        # Log textual evaluation metrics
        print("Confusion Matrix:")
        print(confusion_matrix(y_te, pred))
        print("\nClassification Report:")
        print(classification_report(y_te, pred))
        
        # Extract specific metrics for the defect class ("1")
        rep = classification_report(y_te, pred, output_dict=True)
        rows.append({
            "model": name,
            "accuracy": rep["accuracy"],
            "defect_precision": rep["1"]["precision"],
            "defect_recall": rep["1"]["recall"],
            "defect_f1": rep["1"]["f1-score"],
        })
        
        # Plot decision boundary on the respective subplot
        plot_boundary(ax, model, scaler, X, y, name)

    # Save the aggregated boundary plots
    fig.tight_layout()
    fig.savefig(out / "decision_boundaries.png", bbox_inches="tight")
    plt.close(fig)

    # 5. Export Metrics
    metrics = pd.DataFrame(rows).round(3)
    metrics.to_csv(out / "metrics.csv", index=False)
    
    print("\nFinal Performance Summary:\n", metrics.to_string(index=False))
    print(f"\nSuccess! Saved plots and metrics to: {out.resolve()}")


if __name__ == "__main__":
    main()