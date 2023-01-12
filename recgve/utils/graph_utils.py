#!/usr/bin/env python
__author__ = "XXX"
__email__ = "XXX"

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sps


def nxgraph_from_user_item_interaction_df(df, user_col, item_col):
    """Create a networkx undirected graph from a pandas dataframe containing user item interactions

    Create a networkx undirected graph with two different kind of node, `user` and `item`, attribute stored inside `kind`
    item nodes start from last_user_id+1

    Args:
        df (pd.DataFrame): edges dataframe
        user_col (str): name of source node column
        item_col (str): name of dst node column

    Returns:
        nx.Graph: undirected graph
    """

    # let items node start from max(user_node) + 1
    max_user = max(df[user_col]) + 1
    # copy initial df
    copy_df = df.copy()
    copy_df[item_col] = copy_df[item_col] + max_user

    user_item_graph = nx.from_pandas_edgelist(copy_df, source=user_col, target=item_col)

    users_idxs = copy_df[user_col].unique()
    items_idxs = copy_df[item_col].unique()

    # set `kind` attribute on the nodes
    nx.set_node_attributes(
        user_item_graph, values={u: "user" for u in users_idxs}, name="kind"
    )
    nx.set_node_attributes(
        user_item_graph, values={i: "item" for i in items_idxs}, name="kind"
    )

    return user_item_graph


def symmetric_normalized_laplacian_matrix(graph, self_loop=True):
    """Symmetric normalized Laplacian Matrix

    Compute the symmetric normalized Laplacian matrix of a given networkx graph, if `self_loop` is True
    self loop is added to the initial graph before computing the Laplacian matrix

    .. math::
        S = D^{- \\frac{1}{2}} A D^{- \\frac{1}{2}}

    Args:
        graph (nx.Graph): networkx graph
        self_loop (bool): if add self loop to the initial graph

    Returns:
        sps.csr_matrix: sparse matrix containing symmetric normalized Laplacian
    """
    # note: we have to specify the nodelist in an ordered manner
    A = nx.adjacency_matrix(graph, nodelist=sorted(graph.nodes()))
    if self_loop:
        # convert to lil matrix for efficency of the method set diag
        A = A.tolil()
        # note: setdiag is an inplace operation
        A.setdiag(1)
        # bring back the matrix into csr format
        A = A.tocsr()

    # compute the degree matrix D
    degree = np.array(A.sum(axis=0)).squeeze()
    D = sps.diags(degree, format="csr")
    #D = D.power(-1 / 2)
    D = D.power(-1/2)

    S = D * A * D
    return S


def urm_from_nxgraph(train_graph):
    """Create the URM from a bipartite graph

    Note:
        It is expected a bipartite graph in which each node has an attribute `kind` assuming as value
        `user` or `item`

    Returns:
        sps.csr_matrix: user-rating matrix
    """
    users = sorted([x for x, y in train_graph.nodes(data=True) if y["kind"] == "user"])
    items = sorted([x for x, y in train_graph.nodes(data=True) if y["kind"] == "item"])
    urm = nx.bipartite.biadjacency_matrix(
        train_graph, row_order=users, column_order=items
    )
    return urm
