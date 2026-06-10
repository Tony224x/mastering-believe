"""
Solutions — Day 8 Trees & BST (easy, medium and hard exercises).
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
# Exercise 4 (Medium): Lowest Common Ancestor of a Binary Tree
# =============================================================================

def lowest_common_ancestor(root, p, q):
    """
    Post-order recursion: the answer bubbles up from the leaves.

    CONTRACT of the recursion:
    - Returns p or q if one of them is found in this subtree, None if
      neither is here, or the LCA if both are.

    THE THREE CASES at each node:
    - root is None, p or q → return root as is. If root is p we do NOT
      look below: either q is underneath (then p IS the LCA) or q is
      elsewhere (then an ancestor's other subtree will report it).
    - Both subtrees returned something → p and q split here: root is LCA.
    - One side is None → forward the non-None side upward.

    IDENTITY vs VALUE:
    - Nodes are compared with `is`; two distinct nodes can share a value.

    Time: O(n), Space: O(h) recursion stack
    """
    if root is None or root is p or root is q:
        return root

    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)

    if left and right:
        return root                 # p and q live in different subtrees
    return left if left else right  # Forward whichever side found something


# =============================================================================
# Exercise 5 (Medium): Binary Tree Right Side View
# =============================================================================

def right_side_view(root):
    """
    Level-order BFS, keeping only the LAST node of each level.

    WHY NOT "always go right":
    - A deep left node is visible when the right side is shorter. Only a
      full per-level scan (or a depth-tracking DFS) catches it.

    Time: O(n), Space: O(w) where w = max width
    """
    if not root:
        return []

    view = []
    queue = deque([root])
    while queue:
        level_size = len(queue)         # Freeze: only THIS level's nodes
        for i in range(level_size):
            node = queue.popleft()
            if i == level_size - 1:
                view.append(node.val)   # Rightmost node of the level
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    return view


# =============================================================================
# Exercise 6 (Medium): Kth Smallest Element in a BST
# =============================================================================

def kth_smallest(root, k):
    """
    Iterative in-order traversal with EARLY STOP.

    WHY IN-ORDER:
    - BST invariant: left < node < right at every node, recursively.
      In-order (left, node, right) therefore yields values in sorted
      order — the kth pop IS the kth smallest.

    WHY ITERATIVE:
    - A full recursive traversal visits all n nodes. The explicit stack
      lets us stop after exactly k pops: O(h + k) time.

    Time: O(h + k), Space: O(h)
    """
    stack = []
    node = root

    while stack or node:
        # Dive left as far as possible — smallest unvisited value on top
        while node:
            stack.append(node)
            node = node.left

        node = stack.pop()
        k -= 1
        if k == 0:
            return node.val             # Early stop: rest of tree untouched

        node = node.right               # Then explore the right subtree

    raise ValueError("k is larger than the tree size")


# =============================================================================
# Exercise 7 (Hard): Serialize and Deserialize Binary Tree
# =============================================================================

class Codec:
    """
    Pre-order serialization with explicit null markers.

    WHY NULL MARKERS ARE MANDATORY:
    - Without them, pre-order "1,2,3" could be a left chain OR a right
      chain. The "#" markers pin down exactly where subtrees end, making
      reconstruction deterministic from a single traversal.

    WHY AN ITERATOR FOR DESERIALIZE:
    - tokens.pop(0) is O(n) per call → O(n^2) total. iter() + next()
      consumes each token in O(1), preserving overall O(n).
    """

    def serialize(self, root) -> str:
        parts = []

        def preorder(node):
            if node is None:
                parts.append("#")       # Null marker — removes all ambiguity
                return
            parts.append(str(node.val))
            preorder(node.left)
            preorder(node.right)

        preorder(root)
        # "," separator keeps multi-digit and negative values intact
        return ",".join(parts)

    def deserialize(self, data: str):
        tokens = iter(data.split(","))  # O(1) consumption per token

        def build():
            token = next(tokens)
            if token == "#":
                return None
            node = TreeNode(int(token))
            # The recursion consumes tokens in the EXACT order serialize
            # produced them — no index bookkeeping needed
            node.left = build()
            node.right = build()
            return node

        return build()


# =============================================================================
# Exercise 8 (Hard): Binary Tree Maximum Path Sum
# =============================================================================

def max_path_sum(root):
    """
    Post-order DFS distinguishing TWO different quantities.

    1. GAIN RETURNED TO THE PARENT (path must stay simple):
       node.val + max(left_gain, right_gain) — only ONE branch can
       continue upward through the parent.
    2. GLOBAL CANDIDATE at this node (the "tent" path):
       node.val + left_gain + right_gain — uses BOTH branches but the
       path ends here; it can never extend to the parent.

    CLAMPING RULE:
    - Children's gains are clamped to 0 (a negative branch is better
      dropped), but node.val itself is NEVER clamped: the path must
      contain at least one node, so an all-negative tree returns the
      best single node, not 0.

    Time: O(n), Space: O(h)
    """
    best = float("-inf")                # Updated at EVERY node, not just root

    def gain(node):
        nonlocal best
        if not node:
            return 0
        # Negative subtree contribution → take 0 (drop that branch)
        left_gain = max(gain(node.left), 0)
        right_gain = max(gain(node.right), 0)

        # Tent-shaped path through this node: both branches, ends here
        best = max(best, node.val + left_gain + right_gain)

        # Only one branch may extend toward the parent
        return node.val + max(left_gain, right_gain)

    gain(root)
    return best


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

    # -- Exercise 4 --
    n7, n4 = TreeNode(7), TreeNode(4)
    n6, n2 = TreeNode(6), TreeNode(2, n7, n4)
    n0, n8 = TreeNode(0), TreeNode(8)
    n5, n1 = TreeNode(5, n6, n2), TreeNode(1, n0, n8)
    lca_root = TreeNode(3, n5, n1)
    assert lowest_common_ancestor(lca_root, n5, n1) is lca_root
    assert lowest_common_ancestor(lca_root, n5, n4) is n5   # p is q's ancestor
    assert lowest_common_ancestor(lca_root, n6, n4) is n5
    assert lowest_common_ancestor(lca_root, n7, n4) is n2
    assert lowest_common_ancestor(lca_root, n7, n8) is lca_root
    print("Exercise 4 (lowest_common_ancestor): OK")

    # -- Exercise 5 --
    rsv = TreeNode(1, TreeNode(2, None, TreeNode(5)), TreeNode(3, None, TreeNode(4)))
    assert right_side_view(rsv) == [1, 3, 4]
    # The deep node 4 is on the LEFT branch but still visible
    rsv2 = TreeNode(1, TreeNode(2, TreeNode(4)), TreeNode(3))
    assert right_side_view(rsv2) == [1, 3, 4]
    assert right_side_view(None) == []
    assert right_side_view(TreeNode(1)) == [1]
    assert right_side_view(TreeNode(1, TreeNode(2, TreeNode(3)))) == [1, 2, 3]
    print("Exercise 5 (right_side_view): OK")

    # -- Exercise 6 --
    bst = TreeNode(3, TreeNode(1, None, TreeNode(2)), TreeNode(4))
    assert kth_smallest(bst, 1) == 1
    assert kth_smallest(bst, 2) == 2
    assert kth_smallest(bst, 3) == 3
    assert kth_smallest(bst, 4) == 4
    bst2 = TreeNode(5, TreeNode(3, TreeNode(2, TreeNode(1)), TreeNode(4)), TreeNode(6))
    assert kth_smallest(bst2, 3) == 3
    assert kth_smallest(bst2, 6) == 6
    assert kth_smallest(TreeNode(1), 1) == 1
    print("Exercise 6 (kth_smallest): OK")

    # -- Exercise 7 --
    def same_tree(a, b):
        if not a and not b:
            return True
        if not a or not b or a.val != b.val:
            return False
        return same_tree(a.left, b.left) and same_tree(a.right, b.right)

    codec = Codec()
    t = TreeNode(1, TreeNode(2), TreeNode(3, TreeNode(4), TreeNode(5)))
    assert same_tree(codec.deserialize(codec.serialize(t)), t)
    assert codec.deserialize(codec.serialize(None)) is None
    t_neg = TreeNode(-42)
    assert same_tree(codec.deserialize(codec.serialize(t_neg)), t_neg)
    # Same values, different shapes — null markers must disambiguate
    left_chain = TreeNode(1, TreeNode(2, TreeNode(3)))
    right_chain = TreeNode(1, None, TreeNode(2, None, TreeNode(3)))
    assert same_tree(codec.deserialize(codec.serialize(left_chain)), left_chain)
    assert same_tree(codec.deserialize(codec.serialize(right_chain)), right_chain)
    assert not same_tree(codec.deserialize(codec.serialize(left_chain)), right_chain)
    t_dup = TreeNode(7, TreeNode(7, TreeNode(7)), TreeNode(7))
    assert same_tree(codec.deserialize(codec.serialize(t_dup)), t_dup)
    print("Exercise 7 (Codec serialize/deserialize): OK")

    # -- Exercise 8 --
    assert max_path_sum(TreeNode(1, TreeNode(2), TreeNode(3))) == 6
    mps = TreeNode(-10, TreeNode(9), TreeNode(20, TreeNode(15), TreeNode(7)))
    assert max_path_sum(mps) == 42      # Best path skips the root
    assert max_path_sum(TreeNode(-3)) == -3
    assert max_path_sum(TreeNode(-2, TreeNode(-1))) == -1   # All negative
    assert max_path_sum(TreeNode(2, TreeNode(-1))) == 2
    n11 = TreeNode(11, TreeNode(7), TreeNode(2))
    big = TreeNode(5, TreeNode(4, n11),
                   TreeNode(8, TreeNode(13), TreeNode(4, None, TreeNode(1))))
    assert max_path_sum(big) == 48
    print("Exercise 8 (max_path_sum): OK")

    print("\nAll Day 8 solutions pass!")
