"""
Solutions — Day 8: Trees & BST (MEDIUM)
Run: python domains/algorithmie-python/03-exercises/solutions/08-trees-bst-medium.py

Each solution is numbered to match the exercise file (02-medium/08-trees-bst.md).
All solutions are verified with assertions at the end.
"""

from collections import deque


# =============================================================================
# SHARED HELPERS — TreeNode + build() from BFS level-order list
# =============================================================================

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __repr__(self):
        return f"TreeNode({self.val})"


def build(values):
    """
    Build a tree from a BFS level-order list with None markers (LeetCode style).
    e.g. [3, 9, 20, None, None, 15, 7] -> the classic example tree.
    """
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


# =============================================================================
# EXERCISE 4 (Medium): Zigzag Level Order Traversal
# =============================================================================

def zigzag_level_order(root):
    """
    Standard BFS level order, but flip the reading direction every level.
    Children are always enqueued left-to-right; only the per-level list is
    reversed on odd levels.

    Time: O(n), Space: O(n)
    """
    if not root:
        return []
    result = []
    queue = deque([root])
    left_to_right = True
    while queue:
        level_size = len(queue)        # Snapshot BEFORE adding children
        level = deque()
        for _ in range(level_size):
            node = queue.popleft()
            if left_to_right:
                level.append(node.val)
            else:
                level.appendleft(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(list(level))
        left_to_right = not left_to_right
    return result


def test_exercise_4():
    print("\nExercise 4: Zigzag Level Order Traversal")

    assert zigzag_level_order(build([3, 9, 20, None, None, 15, 7])) == [[3], [20, 9], [15, 7]]
    assert zigzag_level_order(build([1])) == [[1]]
    assert zigzag_level_order(build([])) == []
    assert zigzag_level_order(build([1, 2, 3, 4, 5, 6, 7])) == [[1], [3, 2], [4, 5, 6, 7]]

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 5 (Medium): Kth Smallest Element in a BST
# =============================================================================

def kth_smallest(root, k):
    """
    Iterative inorder traversal with early exit at the k-th popped node.
    Inorder of a BST yields sorted values, so the k-th popped is the answer.

    Time: O(h + k), Space: O(h)
    """
    stack = []
    node = root
    while node or stack:
        while node:                    # Go left as far as possible
            stack.append(node)
            node = node.left
        node = stack.pop()             # Smallest unvisited
        k -= 1
        if k == 0:
            return node.val            # Early exit at the k-th element
        node = node.right              # Pivot right
    raise ValueError("k larger than number of nodes")


def test_exercise_5():
    print("\nExercise 5: Kth Smallest Element in a BST")

    assert kth_smallest(build([3, 1, 4, None, 2]), 1) == 1
    assert kth_smallest(build([3, 1, 4, None, 2]), 2) == 2
    assert kth_smallest(build([5, 3, 6, 2, 4, None, None, 1]), 3) == 3
    assert kth_smallest(build([1]), 1) == 1
    assert kth_smallest(build([2, 1, 3]), 3) == 3

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 6 (Medium): Path Sum II
# =============================================================================

def path_sum(root, target_sum):
    """
    DFS with a mutable path accumulator + backtracking.
    Record a copy of the path only at LEAVES whose running sum hits the target.
    Values may be negative, so we never prune on sign.

    Time: O(n) nodes visited (O(n^2) worst case including path copies)
    Space: O(h) recursion + path
    """
    result = []
    path = []

    def dfs(node, remaining):
        if not node:
            return
        path.append(node.val)          # Choose
        if not node.left and not node.right and remaining == node.val:
            result.append(path[:])     # Leaf hit: copy the path
        else:
            dfs(node.left, remaining - node.val)
            dfs(node.right, remaining - node.val)
        path.pop()                     # Unchoose (backtrack)

    dfs(root, target_sum)
    return result


def test_exercise_6():
    print("\nExercise 6: Path Sum II")

    tree = build([5, 4, 8, 11, None, 13, 4, 7, 2, None, None, 5, 1])
    assert path_sum(tree, 22) == [[5, 4, 11, 2], [5, 8, 4, 5]]
    assert path_sum(build([1, 2, 3]), 5) == []
    assert path_sum(build([]), 0) == []
    assert path_sum(build([1, 2]), 0) == []
    assert path_sum(build([-2, None, -3]), -5) == [[-2, -3]]

    print("  PASS — all test cases")


# =============================================================================
# MAIN — Run all tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SOLUTIONS — Day 8: Trees & BST (MEDIUM)")
    print("=" * 70)

    test_exercise_4()
    test_exercise_5()
    test_exercise_6()

    print("\n" + "=" * 70)
    print("ALL MEDIUM SOLUTIONS COMPLETE — All assertions passed")
    print("=" * 70)
