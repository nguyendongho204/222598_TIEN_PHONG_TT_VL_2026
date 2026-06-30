"""
Generate non-linear synthetic datasets where EBM+SVM can outperform SVM.
These datasets have complex decision boundaries that RBF SVM alone
struggles with, but EBM embeddings help capture.

Usage:
    python tools/generate_nonlinear_data.py
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split


def generate_xor_data(n_samples=2000, noise=0.15, random_state=42):
    """XOR-like non-linear decision boundary"""
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, 2) * 2
    y = np.where((X[:, 0] * X[:, 1]) > 0, 1, 0)
    X += rng.randn(n_samples, 2) * noise
    return X, y


def generate_spiral_data(n_samples=2000, noise=0.3, random_state=42):
    """Spiral pattern - very non-linear"""
    rng = np.random.RandomState(random_state)
    n = n_samples // 2
    theta = np.sqrt(rng.rand(n)) * 2 * np.pi
    r = 2 * theta + np.pi
    x1a = r * np.cos(theta) + rng.randn(n) * noise
    y1a = r * np.sin(theta) + rng.randn(n) * noise
    x1b = r * np.cos(theta + np.pi) + rng.randn(n) * noise
    y1b = r * np.sin(theta + np.pi) + rng.randn(n) * noise
    X = np.vstack([np.column_stack([x1a, y1a]), np.column_stack([x1b, y1b])])
    y = np.hstack([np.zeros(n), np.ones(n)])
    return X, y


def generate_ring_data(n_samples=2000, noise=0.1, random_state=42):
    """Concentric rings - inner vs outer"""
    rng = np.random.RandomState(random_state)
    n = n_samples // 2
    angles = rng.rand(n) * 2 * np.pi
    inner_r = 1.0
    outer_r = 3.0
    X_inner = np.column_stack([
        inner_r * np.cos(angles) + rng.randn(n) * noise,
        inner_r * np.sin(angles) + rng.randn(n) * noise
    ])
    angles = rng.rand(n) * 2 * np.pi
    X_outer = np.column_stack([
        outer_r * np.cos(angles) + rng.randn(n) * noise,
        outer_r * np.sin(angles) + rng.randn(n) * noise
    ])
    X = np.vstack([X_inner, X_outer])
    y = np.hstack([np.zeros(n), np.ones(n)])
    return X, y


def generate_high_dim_nonlinear(n_samples=2000, n_features=20, random_state=42):
    """High-dimensional with non-linear interactions"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=8,
        n_redundant=4,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        flip_y=0.05,
        random_state=random_state
    )
    return X, y


if __name__ == "__main__":
    datasets = {
        'xor': generate_xor_data,
        'moons': lambda n, rs: make_moons(n_samples=n, noise=0.15, random_state=rs),
        'circles': lambda n, rs: make_circles(n_samples=n, noise=0.08, factor=0.5, random_state=rs),
        'spiral': generate_spiral_data,
        'ring': generate_ring_data,
        'high_dim': generate_high_dim_nonlinear,
    }

    for name, gen_func in datasets.items():
        X, y = gen_func(2000)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        df_train = pd.DataFrame(X_train)
        df_train['target'] = y_train
        df_test = pd.DataFrame(X_test)
        df_test['target'] = y_test

        train_path = f'data/synthetic_{name}_train.csv'
        test_path = f'data/synthetic_{name}_test.csv'
        df_train.to_csv(train_path, index=False)
        df_test.to_csv(test_path, index=False)

        print(f"✓ {name}: {X.shape[0]} samples, {X.shape[1]} features, "
              f"train={X_train.shape[0]}, test={X_test.shape[0]} -> {train_path}")
