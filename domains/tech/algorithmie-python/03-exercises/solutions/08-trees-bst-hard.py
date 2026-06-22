"""
Solutions — Day 8: Trees & BST (HARD)
Run: python domains/tech/algorithmie-python/03-exercises/solutions/08-trees-bst-hard.py

Each solution is numbered to match the exercise file (03-hard/08-trees-bst.md).
All solutions are verified with assertions at the end.
"""

from collections import deque


# =============================================================================
# SHARED HELPERS — TreeNode + build()/to_list() + traversals
# =============================================================================

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __repr__(self):
        return f"TreeNode({self.val})"


def build(values):
    """Build a tree from a BFS level-order list with None markers."""
    if not values:
        return None
    it = iter(values)
    root = TreeNode(next(it))
    queue = deque([root])
    while queue:
        node = queue.popleft()
        try:
            left_val = next(it)
        except StopIteration:
            break
        if left_val is not None:
            node.left = TreeNode(left_val)
            queue.append(node.left)
        try:
            right_val = next(it)
        except StopIteration:
            break
        if right_val is not None:
            node.right = TreeNode(right_val)
            queue.append(node.right)
    return root


def to_list(root):
    """
    Serialize a tree to a BFS level-order list with None markers, trimming
    trailing Nones — the canonical compact form for comparing tree shapes.
    """
    if not root:
        return []
    out = []
    queue = deque([root])
    while queue:
        node = queue.popleft()
        if node is None:
            out.append(None)
            continue
        out.append(node.val)
        queue.append(node.left)
        queue.append(node.right)
    while out and out[-1] is None:     # Trim trailing None markers
        out.pop()
    return out


def preorder_vals(root):
    out = []
    def dfs(node):
        if not node:
            return
        out.append(node.val)
        dfs(node.left)
        dfs(node.right)
    dfs(root)
    return out


def inorder_vals(root):
    out = []
    def dfs(node):
        if not node:
            return
        dfs(node.left)
        out.append(node.val)
        dfs(node.right)
    dfs(root)
    return out


# =============================================================================
# EXERCISE 7 (Hard): Serialize / Deserialize Binary Tree
# =============================================================================

class Codec:
    """
    Pre-order serialization with '#' markers for None nodes.
    Deserialization consumes tokens left-to-right via an iterator, rebuilding
    left then right (the order pre-order guarantees). Negative values are fine:
    we split on ',' not on '-'.

    Time: O(n) each way, Space: O(n).
    """

    def serialize(self, root):
        out = []
        def dfs(node):
            if not node:
                out.append("#")
                return
            out.append(str(node.val))
            dfs(node.left)
            dfs(node.right)
        dfs(root)
        return ",".join(out)

    def deserialize(self, data):
        tokens = iter(data.split(","))
        def build_node():
            val = next(tokens)
            if val == "#":
                return None
            node = TreeNode(int(val))
            node.left = build_node()
            node.right = build_node()
            return node
        return build_node()


def test_exercise_7():
    print("\nExercise 7: Serialize / Deserialize Binary Tree")

    codec = Codec()

    def roundtrip(values):
        tree = build(values)
        return to_list(codec.deserialize(codec.serialize(tree)))

    assert roundtrip([1, 2, 3, None, None, 4, 5]) == [1, 2, 3, None, None, 4, 5]
    assert roundtrip([]) == []
    assert roundtrip([1]) == [1]
    assert roundtrip([-1, -2, -3]) == [-1, -2, -3]
    assert roundtrip([1, 2, None, 3, None, 4]) == [1, 2, None, 3, None, 4]
    assert roundtrip([5, 3, 8, 1, 4, 7, 9]) == [5, 3, 8, 1, 4, 7, 9]

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 8 (Hard): Binary Tree Maximum Path Sum
# =============================================================================

def max_path_sum(root):
    """
    Recursion returns the max DOWNWARD gain (node + at most ONE side), while a
    global best is updated with the '^'-shaped path through each node
    (node + left_gain + right_gain). Negative child gains are clamped to 0.

    Time: O(n), Space: O(h)
    """
    best = float('-inf')

    def gain(node):
        nonlocal best
        if not node:
            return 0
        left_gain = max(gain(node.left), 0)    # Drop negative contributions
        right_gain = max(gain(node.right), 0)
        best = max(best, node.val + left_gain + right_gain)   # Path through node
        return node.val + max(left_gain, right_gain)          # One side for parent

    gain(root)
    return best


def test_exercise_8():
    print("\nExercise 8: Binary Tree Maximum Path Sum")

    assert max_path_sum(build([1, 2, 3])) == 6
    assert max_path_sum(build([-10, 9, 20, None, None, 15, 7])) == 42
    assert max_path_sum(build([-3])) == -3
    assert max_path_sum(build([2, -1])) == 2
    assert max_path_sum(build([-2, -1])) == -1
    assert max_path_sum(build([5, 4, 8, 11, None, 13, 4, 7, 2, None, None, None, 1])) == 48

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 9 (Hard): Construct Binary Tree from Preorder and Inorder
# =============================================================================

def build_tree(preorder, inorder):
    """
    preorder[0] is the root; its index in inorder splits left/right subtrees.
    We use a {value: inorder_index} map for O(1) lookup and a moving pointer on
    preorder so we never slice.

    Time: O(n), Space: O(n)
    """
    if not preorder:
        return None

    index = {val: i for i, val in enumerate(inorder)}   # value -> inorder pos
    pre_pos = 0

    def construct(lo, hi):
        nonlocal pre_pos
        if lo > hi:
            return None
        root_val = preorder[pre_pos]
        pre_pos += 1
        node = TreeNode(root_val)
        mid = index[root_val]          # Position of root in inorder
        node.left = construct(lo, mid - 1)    # Pre-order: build left first
        node.right = construct(mid + 1, hi)
        return node

    return construct(0, len(inorder) - 1)


def test_exercise_9():
    print("\nExercise 9: Construct Binary Tree from Preorder and Inorder")

    def roundtrip_pre_in(values):
        tree = build(values)
        pre, ino = preorder_vals(tree), inorder_vals(tree)
        return to_list(build_tree(pre, ino))

    assert to_list(build_tree([3, 9, 20, 15, 7], [9, 3, 15, 20, 7])) == [3, 9, 20, None, None, 15, 7]
    assert to_list(build_tree([-1], [-1])) == [-1]
    assert to_list(build_tree([], [])) == []
    assert roundtrip_pre_in([1, 2, 3, 4, 5, 6, 7]) == [1, 2, 3, 4, 5, 6, 7]
    assert roundtrip_pre_in([1, 2, None, 3]) == [1, 2, None, 3]
    assert roundtrip_pre_in([1, None, 2, None, 3]) == [1, None, 2, None, 3]

    print("  PASS — all test cases")


# =============================================================================
# MAIN — Run all tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SOLUTIONS — Day 8: Trees & BST (HARD)")
    print("=" * 70)

    test_exercise_7()
    test_exercise_8()
    test_exercise_9()

    print("\n" + "=" * 70)
    print("ALL HARD SOLUTIONS COMPLETE — All assertions passed")
    print("=" * 70)
