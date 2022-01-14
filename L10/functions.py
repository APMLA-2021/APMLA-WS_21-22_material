import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import networkx as nx


def fl(x, dp=2):
    return round(x, dp)


def extract_overlapping_membership(i, cm, U, threshold=0.1):
    groups = np.where(U[i] > threshold)[0]
    wedge_sizes = U[i][groups]
    wedge_colors = [cm(c) for c in groups]
    return wedge_sizes, wedge_colors


def normalize_nonzero_membership(U):
    den1 = U.sum(axis=1, keepdims=True)
    nzz = den1 == 0.
    den1[nzz] = 1.
    return U / den1


def plot_L(values, indices=None, k_i=5, figsize=(7, 7), int_ticks=False, ylab='Log-likelihood', xlab='Iterations'):
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if indices is None:
        ax.plot(values[k_i:])
    else:
        ax.plot(indices[k_i:], values[k_i:])
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab+' values')
    if int_ticks:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid()

    plt.tight_layout()
    plt.show()


def plot_net_over(G, pos, u, plt, cm, wedgeprops=None):
    if wedgeprops is None:
        wedgeprops = {'edgecolor': 'lightgrey'}
    radius = 0.005
    ax = plt.gca()
    nx.draw_networkx_edges(G, pos, arrows=False, edge_color='lightgrey', alpha=0.5)
    for i,n in enumerate(list(G.nodes())):
        degree_n = 10
        wedge_sizes, wedge_colors = extract_overlapping_membership(i, cm, u, threshold=0.1)
        if len(wedge_sizes) > 0:
            wedge_sizes /= wedge_sizes.sum()
            _ = plt.pie(wedge_sizes, center=pos[n], colors=wedge_colors, radius=(min(10, degree_n)) * radius,
                             wedgeprops=wedgeprops, normalize=False)
    ax.axis("equal")


def plot_net_hard(G, pos, node_size, com, plt, cm):
    ax = plt.gca()
    nx.draw_networkx_edges(G, pos, arrows=False, edge_color='lightgrey', alpha=0.5)
    nx.draw_networkx_nodes(G, pos, node_size=node_size,
                           node_color=[cm(com[i]) for i, n in enumerate(G.nodes())])
    ax.axis("equal")
    ax.axis("off")
