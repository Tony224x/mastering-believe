"""
Solutions — Day 8 Trees & BST (easy exercises).
Run: python domains/algorithmie-python/03-exercises/solutions/08-trees-bst.py
"""

from collections import deque


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# =============================================================================
# Exercise 1: Max Depth of Binary Tree
# =============================================================================

def max_depth(root):
    """
    Recursive solution.
    Base case: empty tree has depth 0.
    Recursive case: 1 + max depth of left and right subtrees.

    Time : O(n) — we visit every node exactly once
    Space: O(h) — recursion stack, h is the tree height
    """
    if not root:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))


# Alternative iterative BFS version (useful for very deep trees)
def max_depth_bfs(root):
    if not root:
        return 0
    depth = 0
    queue = deque([root])
    while queue:
        depth += 1
        for _ in range(len(queue)):    # Process one full level
            node = queue.popleft()
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    return depth


# =============================================================================
# Exercise 2: Level Order Traversal
# =============================================================================

def level_order(root):
    """
    BFS with a deque.
    CRITICAL: capture level_size = len(queue) BEFORE the inner loop so that
    we only process the nodes that belong to the current level.
    Otherwise newly-added children would be mixed with current level nodes.

    Time : O(n)
    Space: O(w) — w = max width of the tree
    """
    if not root:
        return []
    result = []
    queue = deque([root])
    while queue:
        level_size = len(queue)        # Snapshot
        level = []
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
    return result


# =============================================================================
# Exercise 3: Validate Binary Search Tree
# =============================================================================

def is_valid_bst(root):
    """
    Validate BST using propagated bounds.
    Each recursive call carries (low, high) = the strict interval the subtree
    must live in. When we go left, the upper bound becomes root.val. When we
    go right, the lower bound becomes root.val.

    WHY the naive 'left.val < node.val' check fails:
    It only verifies the direct child, not deeper descendants. Example:
            5
           / \
          1   4      <- 4 is fine locally (4 > 5 is false... wait)
             / \
            3   6
    If root=5 and right child=4, a naive check would fail immediately, but
    the classic trap is when a deep descendant violates an ancestor's bound.

    Time : O(n)
    Space: O(h)
    """
    def validate(node, low, high):
        if not node:
            return True
        if not (low < node.val < high):
            return False
        return (validate(node.left, low, node.val) and
                validate(node.right, node.val, high))

    return validate(root, float('-inf'), float('inf'))


# Alternative: in-order traversal must be strictly increasing
def is_valid_bst_inorder(root):
    """
    Inorder of a BST yields sorted values. We check strict monotonicity.
    We keep a single 'prev' pointer instead of building a full list.
    """
    prev = [float('-inf')]

    def inorder(node):
        if not node:
            return True
        if not inorder(node.left):
            return False
        if node.val <= prev[0]:        # Strict: duplicates not allowed
            return False
        prev[0] = node.val
        return inorder(node.right)

    return inorder(root)


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    # -- Exercise 1 --
    assert max_depth(None) == 0
    assert max_depth(TreeNode(1)) == 1
    tree = TreeNode(3, TreeNode(9), TreeNode(20, TreeNode(15), TreeNode(7)))
    assert max_depth(tree) == 3
    chain = TreeNode(1, TreeNode(2, TreeNode(3, TreeNode(4))))
    assert max_depth(chain) == 4
    assert max_depth_bfs(tree) == 3
    print("Exercise 1 (max_depth): OK")

    # -- Exercise 2 --
    assert level_order(None) == []
    assert level_order(TreeNode(1)) == [[1]]
    assert level_order(tree) == [[3], [9, 20], [15, 7]]
    chain2 = TreeNode(1, TreeNode(2, TreeNode(3)))
    assert level_order(chain2) == [[1], [2], [3]]
    print("Exercise 2 (level_order): OK")

    # -- Exercise 3 --
    assert is_valid_bst(None) == True
    assert is_valid_bst(TreeNode(1)) == True
    assert is_valid_bst(TreeNode(2, TreeNode(1), TreeNode(3))) == True
    bad = TreeNode(5, TreeNode(1), TreeNode(4, TreeNode(3), TreeNode(6)))
    assert is_valid_bst(bad) == False
    assert is_valid_bst(TreeNode(1, TreeNode(1))) == False
    # Also verify with inorder variant
    assert is_valid_bst_inorder(tree) == False     # 3/9/20/15/7 is not a BST
    assert is_valid_bst_inorder(TreeNode(2, TreeNode(1), TreeNode(3))) == True
    print("Exercise 3 (is_valid_bst): OK")

    print("\nAll Day 8 solutions pass!")
