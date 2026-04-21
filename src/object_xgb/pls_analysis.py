import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MAPPING = {1: 'Epithelial', 2: 'Hyphal', 3: 'Invaded'}


def load_pls_data(csv_path: str):
    """
    Extracts features and labels from the object-xgb feature table.

    Parameters
    ----------
    csv_path : str
        Path to the exported CSV feature table.

    Returns
    -------
    X : pd.DataFrame
        The feature matrix.
    y : pd.Series
        The target labels (true_label).
    feature_cols : list[str]
        List of feature names.
    """
    df = pd.read_csv(csv_path)

    # Metadata and target columns to exclude from X
    meta_cols = ['label', 'slice_id', 'true_label']
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feature_cols].copy()
    y = df['true_label'].fillna(0).astype(int)

    return X, y, feature_cols


def run_plsda(X: pd.DataFrame, y: pd.Series, n_components: int = 3):
    """
    Fits PLS-DA on labeled objects only.

    Parameters
    ----------
    X : pd.DataFrame
        Full feature matrix.
    y : pd.Series
        Full label series.
    n_components : int
        Number of latent components (default 3).

    Returns
    -------
    pls : PLSRegression
        The fitted PLS model.
    mask : pd.Series
        Boolean mask indicating which objects have true labels (y > 0).
    classes : np.ndarray
        The unique classes found in the labeled data.
    """
    # Filter for labeled objects (y > 0)
    mask = y > 0
    X_labeled = X[mask]
    y_labeled = y[mask]

    # One-hot encode the labels for PLS-DA
    encoder = OneHotEncoder(sparse_output=False)
    Y_encoded = encoder.fit_transform(y_labeled.values.reshape(-1, 1))

    # Fit PLS (Scaling is enabled by default)
    pls = PLSRegression(n_components=n_components, scale=True)
    pls.fit(X_labeled, Y_encoded)

    return pls, mask, encoder.categories_[0]


def calculate_variance_explained(pls: PLSRegression, X_labeled: pd.DataFrame):
    """
    Calculates R2X (variance explained in X) for each component.

    Parameters
    ----------
    pls : PLSRegression
        The fitted PLS model.
    X_labeled : pd.DataFrame
        The feature matrix of labeled objects.

    Returns
    -------
    var_explained : list[float]
        Percentage of variance explained per component.
    """
    # Scale X as PLSRegression does internally
    X_scaled = StandardScaler().fit_transform(X_labeled)

    total_var = np.sum(np.var(X_scaled, axis=0))
    # T are the scores, P are the loadings
    T = pls.x_scores_
    P = pls.x_loadings_

    var_explained = []
    for i in range(pls.n_components):
        # Variance of the projection for this component
        comp_var = np.sum(np.var(np.outer(T[:, i], P[:, i]), axis=0))
        var_explained.append(comp_var / total_var * 100)

    return var_explained


def calculate_vip(pls: PLSRegression, feature_names: list[str]):
    """
    Calculates non-cumulative VIP scores for each feature per component.

    Parameters
    ----------
    pls : PLSRegression
        The fitted PLS model.
    feature_names : list[str]
        Names of the features.

    Returns
    -------
    vip_df : pd.DataFrame
        DataFrame with feature names as index and components as columns,
        containing the individual VIP contribution of each component.
    """
    T = pls.x_scores_
    W = pls.x_weights_
    Q = pls.y_loadings_

    p, n_comp = W.shape
    vips = np.zeros((p, n_comp))
    # s is the sum of squares explained by each component in the Y space
    s = np.diag(np.matmul(np.matmul(np.matmul(T.T, T), Q.T), Q))
    total_s = np.sum(s)

    for i in range(n_comp):
        # Normalized weight for this component
        # W[:, i] is the weight vector for component i
        w_norm = W[:, i] / np.linalg.norm(W[:, i])
        # Non-cumulative VIP contribution for this specific component
        vips[:, i] = np.sqrt(p * (s[i] * (w_norm**2)) / total_s)

    columns = [f'Component {i + 1}' for i in range(n_comp)]
    return pd.DataFrame(vips, index=feature_names, columns=columns)


def plot_plsda_biplots(
    pls: PLSRegression,
    X: pd.DataFrame,
    y: pd.Series,
    mask: pd.Series,
    feature_names: list[str],
    var_explained: list[float],
    top_n_features: int = 10,
):
    """
    Plots 3 biplots (Dim 1-2, 2-3, 1-3) with colored/gray objects and feature arrows.

    Parameters
    ----------
    pls : PLSRegression
        The fitted PLS model.
    X : pd.DataFrame
        Full feature matrix.
    y : pd.Series
        Full label series.
    mask : pd.Series
        Boolean mask of labeled objects.
    feature_names : list[str]
        Names of the features.
    var_explained : list[float]
        Variance explained per dimension.
    top_n_features : int
        Number of highest VIP features to show as arrows.
    """
    # Project ALL data (labeled and unlabeled) into the PLS space
    scores = pls.transform(X)
    loadings = pls.x_loadings_

    pairs = [(0, 1), (1, 2), (0, 2)]
    _fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Standardize loading scale for arrows
    arrow_scale = np.max(np.abs(scores)) / np.max(np.abs(loadings)) * 0.7

    # Calculate VIPs and find top features based on total VIP
    vip_df = calculate_vip(pls, feature_names)
    # Total VIP is the square root of the sum of squared component-wise VIPs
    total_vips = np.sqrt(np.sum(vip_df.values**2, axis=1))
    top_indices = np.argsort(total_vips)[-top_n_features:]

    for i, (d1, d2) in enumerate(pairs):
        ax = axes[i]

        # Plot unlabeled objects (y == 0) first so they are in the background
        unlabeled_scatter = ax.scatter(
            scores[~mask, d1],
            scores[~mask, d2],
            c='gray',
            alpha=0.3,
            s=15,
            label='Unlabeled',
        )

        # Plot labeled objects (y > 0)
        labeled_scatter = ax.scatter(
            scores[mask, d1],
            scores[mask, d2],
            c=y[mask],
            cmap='viridis',
            s=30,
            label='Labeled',
            edgecolors='white',
            linewidth=0.5,
        )

        # Plot Feature Arrows
        for idx in top_indices:
            ax.arrow(
                0,
                0,
                loadings[idx, d1] * arrow_scale,
                loadings[idx, d2] * arrow_scale,
                color='black',
                alpha=0.8,
                head_width=arrow_scale * 0.025,
            )
            ax.text(
                loadings[idx, d1] * arrow_scale * 1.1,
                loadings[idx, d2] * arrow_scale * 1.1,
                feature_names[idx],
                color='black',
                fontsize=9,
            )

        # Add Legend
        import re

        handles, labels = labeled_scatter.legend_elements()
        new_labels = []
        for label_str in labels:
            match = re.search(r'\d+', label_str)
            if match:
                class_id = int(match.group())
                new_labels.append(MAPPING.get(class_id, f'Class {class_id}'))
            else:
                new_labels.append(label_str)

        handles.append(unlabeled_scatter)
        new_labels.append('Unlabeled')
        ax.legend(
            handles, new_labels, title='Classes', loc='best', fontsize='small'
        )

        ax.set_xlabel(f'Component {d1 + 1} ({var_explained[d1]:.2f}%)')
        ax.set_ylabel(f'Component {d2 + 1} ({var_explained[d2]:.2f}%)')
        ax.axhline(0, color='black', lw=1, ls='--')
        ax.axvline(0, color='black', lw=1, ls='--')
        ax.set_title(f'PLS-DA Biplot: Dim {d1 + 1} vs {d2 + 1}')

    plt.tight_layout()
    plt.show()


def plot_plsda_3d(
    pls: PLSRegression,
    X: pd.DataFrame,
    y: pd.Series,
    mask: pd.Series,
    feature_names: list[str],
    var_explained: list[float],
    save_path: str = 'output/plsda/plsda_3d.html',
    top_n_features: int = 10,
):
    """
    Creates an interactive 3D biplot using Plotly and saves it as HTML.

    Parameters
    ----------
    pls : PLSRegression
        The fitted PLS model.
    X : pd.DataFrame
        Full feature matrix.
    y : pd.Series
        Full label series.
    mask : pd.Series
        Boolean mask of labeled objects.
    feature_names : list[str]
        Names of the features.
    var_explained : list[float]
        Variance explained per dimension.
    save_path : str
        Path to save the HTML output.
    top_n_features : int
        Number of highest VIP features to show as arrows.
    """
    import plotly.graph_objects as go

    scores = pls.transform(X)
    loadings = pls.x_loadings_

    # Create the figure
    fig = go.Figure()

    # 1. Plot Unlabeled Objects
    fig.add_trace(
        go.Scatter3d(
            x=scores[~mask, 0],
            y=scores[~mask, 1],
            z=scores[~mask, 2],
            mode='markers',
            marker={'size': 3, 'color': 'gray', 'opacity': 0.3},
            name='Unlabeled',
        )
    )

    # 2. Plot Labeled Objects by Class
    unique_labels = sorted(y[mask].unique())
    for label_id in unique_labels:
        class_mask = (y == label_id) & mask
        class_name = MAPPING.get(label_id, f'Class {label_id}')
        fig.add_trace(
            go.Scatter3d(
                x=scores[class_mask, 0],
                y=scores[class_mask, 1],
                z=scores[class_mask, 2],
                mode='markers',
                marker={'size': 5, 'line': {'width': 1, 'color': 'white'}},
                name=class_name,
            )
        )

    # 3. Plot Feature Arrows
    arrow_scale = np.max(np.abs(scores)) / np.max(np.abs(loadings)) * 0.7
    vip_df = calculate_vip(pls, feature_names)
    total_vips = np.sqrt(np.sum(vip_df.values**2, axis=1))
    top_indices = np.argsort(total_vips)[-top_n_features:]

    for idx in top_indices:
        lx = loadings[idx, 0] * arrow_scale
        ly = loadings[idx, 1] * arrow_scale
        lz = loadings[idx, 2] * arrow_scale

        # Arrow line
        fig.add_trace(
            go.Scatter3d(
                x=[0, lx],
                y=[0, ly],
                z=[0, lz],
                mode='lines+text',
                line={'color': 'black', 'width': 4},
                text=['', feature_names[idx]],
                textposition='top center',
                name=f'Feature: {feature_names[idx]}',
                showlegend=False,
            )
        )

    # Update Layout
    fig.update_layout(
        title='3D PLS-DA Biplot',
        scene={
            'xaxis_title': f'PC1 ({var_explained[0]:.2f}%)',
            'yaxis_title': f'PC2 ({var_explained[1]:.2f}%)',
            'zaxis_title': f'PC3 ({var_explained[2]:.2f}%)',
        },
        margin={'l': 0, 'r': 0, 'b': 0, 't': 40},
        legend={'yanchor': 'top', 'y': 0.99, 'xanchor': 'left', 'x': 0.01},
    )

    # Ensure output directory exists
    import os

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig.write_html(save_path)
    print(f'[Object XGB] 3D Biplot saved to {save_path}')


def run_pairwise_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    class_a: int,
    class_b: int,
    feature_names: list[str],
):
    """
    Performs PLS-DA on a specific pair of classes to find discriminating features.

    Parameters
    ----------
    X : pd.DataFrame
        Full feature matrix.
    y : pd.Series
        Full label series.
    class_a, class_b : int
        The class IDs to compare.
    feature_names : list[str]
        Names of the features.

    Returns
    -------
    vips : pd.DataFrame
        A single-column DataFrame containing VIP scores for the class pair.
    """
    mask = (y == class_a) | (y == class_b)
    X_pair = X[mask]
    # Binary 1 for class_a, 0 for class_b
    y_pair = (y[mask] == class_a).astype(int)

    if len(X_pair) < 2:
        return pd.DataFrame(index=feature_names)

    # 1 component is enough to define a single linear boundary between two classes
    pls = PLSRegression(n_components=1, scale=True)
    pls.fit(X_pair, y_pair)

    # Calculate VIP for this pair
    vips = calculate_vip(pls, feature_names)

    # Label with class names
    name_a = MAPPING.get(class_a, f'Class {class_a}')
    name_b = MAPPING.get(class_b, f'Class {class_b}')
    vips.columns = [f'{name_a} vs {name_b}']

    return vips


def analyze_all_pairs(X: pd.DataFrame, y: pd.Series, feature_names: list[str]):
    """
    Runs pairwise PLS-DA for all class combinations to find unique discriminators.

    Returns
    -------
    pair_table : pd.DataFrame
        A table where each column is a pairwise VIP comparison.
    """
    import itertools

    unique_classes = sorted(y[y > 0].unique())
    results = []

    for a, b in itertools.combinations(unique_classes, 2):
        pair_vips = run_pairwise_analysis(X, y, a, b, feature_names)
        results.append(pair_vips)

    return pd.concat(results, axis=1)


def analyze_grouped_separation(
    X: pd.DataFrame,
    y: pd.Series,
    group_ids: list[int],
    other_id: int,
    feature_names: list[str],
):
    """
    Analyzes separation between a group of classes (e.g., [1, 3]) and another class (e.g., 2).

    Parameters
    ----------
    X, y : pd.DataFrame, pd.Series
        The data.
    group_ids : list[int]
        IDs of classes to group together (e.g., [1, 3]).
    other_id : int
        The ID of the single class to compare against (e.g., 2).
    feature_names : list[str]
        The feature names.

    Returns
    -------
    vips : pd.DataFrame
        VIP scores for the group vs other comparison.
    """
    mask = y.isin(group_ids + [other_id])
    X_sub = X[mask]
    y_binary = y[mask].isin(group_ids).astype(int)

    if len(X_sub) < 2:
        return pd.DataFrame(index=feature_names)

    pls = PLSRegression(n_components=1, scale=True)
    pls.fit(X_sub, y_binary)

    group_name = '+'.join([MAPPING.get(gid, str(gid)) for gid in group_ids])
    other_name = MAPPING.get(other_id, str(other_id))

    vips = calculate_vip(pls, feature_names)
    vips.columns = [f'{group_name} vs {other_name}']
    return vips


def calculate_total_vip(vip_df: pd.DataFrame):
    """
    Calculates the total VIP score from a component-wise VIP DataFrame.

    Total VIP is calculated as the root sum of squares (RSS) of the
    individual component contributions.

    Parameters
    ----------
    vip_df : pd.DataFrame
        DataFrame with features as index and components as columns.

    Returns
    -------
    sorted_vip_df : pd.DataFrame
        The input DataFrame with a 'Total VIP' column added,
        sorted by total importance in descending order.
    """
    res_df = vip_df.copy()

    # Calculate total VIP: root sum of squares across columns
    total_vips = np.sqrt((res_df**2).sum(axis=1))
    res_df['Total VIP'] = total_vips

    # Sort by the actual importance
    return res_df.sort_values('Total VIP', ascending=False)
