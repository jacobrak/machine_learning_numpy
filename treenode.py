class TreeNode:
    """
    A single node in a decision tree.
    """

    def __init__(
        self,
        feature_index=None,
        threshold=None,
        left=None,
        right=None,
        value=None,
        depth=0,
    ):
        self.feature_index = feature_index   # int
        self.threshold = threshold           # float
        self.left = left                     # TreeNode
        self.right = right                   # TreeNode
        self.value = value                   # class label or float
        self.depth = depth                   # int

    def is_leaf(self):
        return self.value is not None
