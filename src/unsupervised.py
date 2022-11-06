from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, silhouette_score
from sklearn.metrics import DistanceMetric


def pca_transformer(numerical_data, numerical_test_data= [], threshold = 0.95):
    pca = PCA(n_components=threshold)
    pca_train_components = pca.fit(numerical_data).transform(numerical_data)
    if len(numerical_test_data)>0:
        pca_test_components = pca.transform(numerical_test_data)
        return pca, pca_train_components, pca_test_components
    else:
        return pca, pca_train_components


def visualize_pca_cum_variance(pca):
    """Plot a line showing the cumulative variance explained by each component, alongside the variance threshold

    Args:
        pca (sklearn.decmposisition.PCA): A PCA object that has been fitted on trianing data
    Returns:
        None
    """

    plt.rcParams["figure.figsize"] = (12,6)

    fig, ax = plt.subplots()
    xi = np.arange(0, pca.n_components, step=1)
    y = np.cumsum(pca.explained_variance_ratio_*100)

    plt.ylim(0.0, 100.1)
    plt.plot(xi, y, marker='o', linestyle='--', color='b')

    plt.xlabel('PCA Components', fontsize=16, labelpad=10)
    plt.ylabel('Cumulative variance (%)', fontsize=16)

    plt.xticks(np.arange(0, 44, step=2))
    plt.yticks(np.arange(0, 100.1, step=20))
    plt.tick_params(axis="both",direction="in", pad=10)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')

    plt.axhline(y=95, color='r', linestyle='-')
    plt.text(0.5, 85, '95% cut-off threshold', color = 'red', fontsize=16)

    ax.grid(axis='both', alpha=0.5)
    plt.show()

    return

def visualize_pca_3D(pca_train_components, train_labels):
    """Create a 3D scatterplot of components and labels

    Args:
        pca_train_components (np.ndarray): (N,D) A component for each customer with a reduced number of dimensions
        train_labels (pd.DataFrame): A dataframe containing the customer_ids and targets
    Returns:
        None
    """
    # TODO: train labels is not the same size as the number of training examples, so the boolean won't work in this new implementation
    plt.rcParams["figure.figsize"] = (12,6)
    target_names = ["compliance", "default"]
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    colors = ["turquoise", "darkorange"]

    for color, i, target_name in zip(colors, [0, 1], target_names):
        ax.scatter(
            pca_train_components[train_labels["target"]==i, 0], pca_train_components[train_labels["target"]==i, 1], pca_train_components[train_labels["target"]==i, 2], 
            s=9, linewidths=0.0, alpha=0.5 ,color=color, label=target_name
        )

    ax.set_xlabel("PCA Component 1", labelpad=10, rotation=0)
    ax.set_ylabel("PCA Component 2", labelpad=10, rotation=0)
    ax.set_zlabel("PCA Component 3", labelpad=10, rotation=0)

    ax.set_xticks(np.arange(-2, 3, 1))
    ax.set_yticks(np.arange(-2, 2, 1))
    ax.set_zticks(np.arange(-2, 2.5, 1))

    ax.legend(loc="best", shadow=False, scatterpoints=1)
    ax.view_init(elev=15, azim=-60)

    plt.show()

    return

def visualize_pca_2D(pca_train_components, train_labels):
    """Create a 2D scatterplot of components and labels

    Args:
        pca_train_components (np.ndarray): (N,D) A component for each customer with a reduced number of dimensions
        train_labels (pd.DataFrame): A dataframe containing the customer_ids and targets
    Returns:
        None
    """

    # TODO: train labels is not the same size as the number of training examples, so the boolean won't work in this new implementation
    target_names = ["compliance", "default"]
    fig = plt.figure()
    colors = ["turquoise", "darkorange"]
    marker_sizes = [9, 9]

    for color, i, ms, target_name in zip(colors, [0, 1], marker_sizes, target_names):
        plt.scatter(
            pca_train_components[train_labels["target"]==i, 1], pca_train_components[train_labels["target"]==i, 0], s=ms, alpha=0.5 ,color=color, label=target_name
        )

    plt.xlabel("PCA Component 2")
    plt.ylabel("PCA Component 1")
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("PCA of the AMEX dataset")
    plt.show()

    return

