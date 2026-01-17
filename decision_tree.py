import numpy as np
from treenode import TreeNode

class DecisionTree:
    """
    From-scratch decision tree (classification or regression).
    """

    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        min_gain=0.0,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.root = None

    # ======================
    # Public API
    # ======================

    def fit(self, X, y):
        """
        Build the tree.
        """
        self.root = self._build_tree(X, y, depth=0)

    def predict(self, X):
        """
        Predict for many samples.
        """
        return np.array([self._predict_one(x, self.root) for x in X])
        # ======================
    # Internal methods
    # ======================

    def _build_tree(self, X, y, depth):
        n_samples = len(y)

        # stopping conditions
        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or n_samples < self.min_samples_split
        ):
            return TreeNode(value=self._leaf_value(y), depth=depth)

        best_feature, best_threshold, best_gain = self._best_split(X, y)

        # no useful split
        if best_feature is None or best_gain <= self.min_gain:
            return TreeNode(value=self._leaf_value(y), depth=depth)

        # split using best feature
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = X[:, best_feature] > best_threshold

        left_child = self._build_tree(
            X[left_mask],
            y[left_mask],
            depth + 1,
        )

        right_child = self._build_tree(
            X[right_mask],
            y[right_mask],
            depth + 1,
        )

        return TreeNode(
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child,
            depth=depth,
        )
    def _best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_gain = -np.inf

        num_features = X.shape[1]

        for feature_idx in range(num_features):
            feature_values = np.unique(X[:, feature_idx])

            for threshold in feature_values:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = X[:, feature_idx] > threshold

                # skip invalid splits
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]

                gain = self._gain(y, y_left, y_right)

                if gain > best_gain:
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_gain = gain

        return best_feature, best_threshold, best_gain

    def _impurity(self, y):
        mean = np.mean(y)
        return np.mean((y - mean) ** 2)

    def _gain(self, y, y_left, y_right):
        parent_imp = self._impurity(y)

        n = len(y)
        n_left = len(y_left)
        n_right = len(y_right)

        child_imp = (
            (n_left / n) * self._impurity(y_left)
            + (n_right / n) * self._impurity(y_right)
        )

        return parent_imp - child_imp
            
    def _leaf_value(self, y):
        """
        Regression:
            return mean(y)
        """
        return np.mean(y)
    
    def _predict_one(self, x, node):
        """
        Traverse the tree.
        """
        if node.is_leaf():
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)
