import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn import metrics

# linear_sum_assignment: Solve the linear sum assignment problem
# The details can be found here:
# https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html

# The "linear sum assignment" problem is also known as "minimum weight matching" in bipartite graphs.
# A problem instance is described by a matrix C, where each C[i,j] is the cost
# of matching vertex i of the first set (a “worker”) and vertex j of the second set (a “job”).
# The goal is to find a complete assignment of workers to jobs of minimal cost.
#
# Formally, let X be a boolean matrix where X[i,j]=1 iff row i is assigned to column j.
# Then the optimal assignment has cost
# min(sum_i sum_j C_ij X_ij)

# s.t. each row is assignment to at most one column, and each column to at most one row.
# This function can also solve a generalization of the classic assignment problem where the cost matrix
# is rectangular. If it has more rows than columns, then not every row needs to be assigned
# to a column, and vice versa.
#
# The method used is the Hungarian algorithm, also known as the Munkres or Kuhn-Munkres algorithm.


# This code is taken from "VaDE_pytorch_GuHongyang/model.py"
def clustering_accuracy(num_classes, y_pred, y_true):
    y_pred = np.asarray(y_pred, dtype=np.int32)
    y_true = np.asarray(y_true, dtype=np.int32)
    assert y_pred.shape == y_true.shape and y_pred.ndim == 1, \
        "y_pred and y_true must be 1D arrays of the same length. " \
        "Found y_pred.shape={y_pred.shape} and y_true.shape={y_true.shape}!"

    assign_mat = np.zeros((num_classes, num_classes), dtype=np.int32)
    for n in range(len(y_pred)):
        assign_mat[y_pred[n], y_true[n]] += 1

    # (num_classes,), (num_classes,)
    # In case of square matrix, 'row_ids' is equal to (0, 1, ..., num_classes - 1)
    # row_ids, col_ids = linear_assignment(np.max(assign_mat) - assign_mat)
    row_ids, col_ids = linear_assignment(-assign_mat)
    assert len(row_ids) == len(col_ids) == num_classes, \
        f"row_ids.shape={row_ids.shape}, col_ids.shape={col_ids.shape}, " \
        f"num_classes={num_classes}"

    # print(f"\nrow_ids:\n{row_ids}")
    # print(f"\ncol_ids:\n{col_ids}")
    # print(f"\nassign_mat:\n{assign_mat}")
    acc = assign_mat[row_ids, col_ids].sum() / len(y_pred)

    # Map predicted classes to real classes
    y_pred_mapped = col_ids[y_pred]

    return {
        'row_ids': row_ids,
        'col_ids': col_ids,
        'assign_mat': assign_mat,
        'acc': acc,

        'y_pred_mapped': y_pred_mapped,
        'pred_2_true_class_ids': col_ids,
    }


acc = clustering_accuracy


def nmi(y_pred, y_true):
    return metrics.normalized_mutual_info_score(y_true, y_pred)


def ari(y_pred, y_true):
    return metrics.adjusted_rand_score(y_true, y_pred)