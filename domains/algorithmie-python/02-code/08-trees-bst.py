"""
Day 8 — Trees & BST: DFS, BFS, BST validation, LCA, Serialize/Deserialize
Run: python domains/algorithmie-python/02-code/08-trees-bst.py

Every function is runnable and heavily commented with WHY each step matters.
We build a small tree once and reuse it across demos.
"""

from collections import deque


# =============================================================================
# THE BUILDING BLOCK: TreeNode
# =============================================================================

class TreeNode:
    """
    Binary tree node.
    A tree is simply a reference to the root node — everything else is
    reached by following .left and .right pointers.
    """
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __repr__(self):
        return f"TreeNode({self.val})"


def build_sample_tree():
    """
    Builds this tree:
            4
           / \
          2   6
         / \ / \
        1  3 5  7

    This is a valid BST. We reuse it across examples.
    """
    return TreeNode(4,
        TreeNode(2, TreeNode(1), TreeNode(3)),
        TreeNode(6, TreeNode(5), TreeNode(7)))


# =============================================================================
# SECTION 1: DFS TRAVERSALS (recursive + iterative)
# =============================================================================

def preorder_recursive(node, result=None):
    """
    Pre-order: Node -> Left -> Right.
    Used for: copying a tree, serializing.
    """
    if result is None:
        result = []
    if not node:
        return result
    result.append(node.val)           # Visit BEFORE descending
    preorder_recursive(node.left, result)
    preorder_recursive(node.right, result)
    return result


def inorder_recursive(node, result=None):
    """
    In-order: Left -> Node -> Right.
    KEY: on a BST, this yields sorted values.
    """
    if result is None:
        result = []
    if not node:
        return result
    inorder_recursive(node.left, result)
    result.append(node.val)           # Visit AFTER left, BEFORE right
    inorder_recursive(node.right, result)
    return result


def postorder_recursive(node, result=None):
    """
    Post-order: Left -> Right -> Node.
    Used for: deleting a tree, evaluating expression trees.
    """
    if result is None:
        result = []
    if not node:
        return result
    postorder_recursive(node.left, result)
    postorder_recursive(node.right, result)
    result.append(node.val)           # Visit AFTER both children
    return result


def preorder_iterative(root):
    """
    Iterative pre-order using an explicit stack.
    WHY: avoids recursion depth limit for very deep trees.
    """
    if not root:
        return []
    stack = [root]
    result = []
    while stack:
        node = stack.pop()
        result.append(node.val)
        # Push RIGHT first so LEFT is processed next (LIFO).
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return result


def inorder_iterative(root):
    """
    Iterative in-order — the trickiest one, memorize this template.
    Strategy: go left as far as possible, pop, visit, then pivot right.
    """
    stack = []
    result = []
    node = root
    while node or stack:
        while node:                   # Go left to the bottom
            stack.append(node)
            node = node.left
        node = stack.pop()            # Deepest unvisited left
        result.append(node.val)       # Visit
        node = node.right             # Pivot right and repeat
    return result


# =============================================================================
# SECTION 2: BFS LEVEL ORDER
# =============================================================================

def level_order(root):
    """
    BFS traversal returning values grouped by level.
    KEY: snapshot `level_size` BEFORE adding children to keep levels separated.
    """
    if not root:
        return []
    result = []
    queue = deque([root])
    while queue:
        level_size = len(queue)       # How many nodes are in THIS level
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


def right_side_view(root):
    """
    Return the rightmost node at each level — what you'd see standing
    to the right of the tree.
    """
    if not root:
        return []
    result = []
    queue = deque([root])
    while queue:
        level_size = len(queue)
        for i in range(level_size):
            node = queue.popleft()
            if i == level_size - 1:   # Last node of this level = rightmost
                result.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    return result


# =============================================================================
# SECTION 3: RECURSIVE PROPERTIES (depth, diameter, balanced)
# =============================================================================

def max_depth(root):
    """
    Height of the tree.
    Base case: empty tree has depth 0.
    Recursive case: 1 + max of left/right subtree depth.
    """
    if not root:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))


def diameter(root):
    """
    Longest path between any two nodes (edges, not nodes).
    KEY INSIGHT: for each node, the longest path THROUGH that node
    is left_depth + right_depth. We track the max across all nodes.
    """
    best = [0]                        # Mutable container so inner fn can update

    def depth(node):
        if not node:
            return 0
        l = depth(node.left)
        r = depth(node.right)
        best[0] = max(best[0], l + r) # Path going through this node
        return 1 + max(l, r)          # Height for the caller

    depth(root)
    return best[0]


def is_balanced(root):
    """
    A tree is balanced if for every node, |left_height - right_height| <= 1.
    TRICK: return -1 to signal imbalance up the recursion stack early.
    """
    def check(node):
        if not node:
            return 0
        l = check(node.left)
        if l == -1:
            return -1                 # Early exit: left subtree already broken
        r = check(node.right)
        if r == -1:
            return -1
        if abs(l - r) > 1:
            return -1                 # This node breaks the rule
        return 1 + max(l, r)

    return check(root) != -1


# =============================================================================
# SECTION 4: BST OPERATIONS
# =============================================================================

def is_valid_bst(root, low=float('-inf'), high=float('inf')):
    """
    A BST requires left < node < right recursively and GLOBALLY.
    We pass down the allowed range [low, high] for each subtree.
    """
    if not root:
        return True
    if not (low < root.val < high):
        return False
    # Left subtree must be strictly less than root.val (new upper bound)
    # Right subtree must be strictly greater than root.val (new lower bound)
    return (is_valid_bst(root.left, low, root.val) and
            is_valid_bst(root.right, root.val, high))


def search_bst(root, target):
    """
    O(h) search: go left/right based on comparison, exit when found or None.
    """
    while root:
        if root.val == target:
            return root
        root = root.left if target < root.val else root.right
    return None


def insert_bst(root, val):
    """
    Insert a value into a BST, returning the (possibly new) root.
    """
    if not root:
        return TreeNode(val)
    if val < root.val:
        root.left = insert_bst(root.left, val)
    elif val > root.val:
        root.right = insert_bst(root.right, val)
    # val == root.val: skip (no duplicates)
    return root


# =============================================================================
# SECTION 5: LOWEST COMMON ANCESTOR
# =============================================================================

def lca_binary_tree(root, p, q):
    """
    LCA in a generic binary tree.
    Returns the root if found either node OR if p and q are on opposite sides.
    """
    if not root or root is p or root is q:
        return root
    left = lca_binary_tree(root.left, p, q)
    right = lca_binary_tree(root.right, p, q)
    if left and right:
        return root                   # Split point: p and q in different subtrees
    return left or right              # Whichever side actually found something


def lca_bst(root, p, q):
    """
    LCA in a BST: exploit ordering.
    - If both p and q are less than root -> LCA is in left subtree
    - If both are greater -> in right subtree
    - Otherwise, root is the split point = LCA
    """
    while root:
        if p.val < root.val and q.val < root.val:
            root = root.left
        elif p.val > root.val and q.val > root.val:
            root = root.right
        else:
            return root
    return None


# =============================================================================
# SECTION 6: PATH SUM
# =============================================================================

def has_path_sum(root, target):
    """
    True if there exists a root-to-leaf path whose sum equals target.
    KEY: we only check the sum at LEAVES (no children).
    """
    if not root:
        return False
    if not root.left and not root.right:
        return root.val == target
    remaining = target - root.val
    return (has_path_sum(root.left, remaining) or
            has_path_sum(root.right, remaining))


# =============================================================================
# SECTION 7: SERIALIZE / DESERIALIZE
# =============================================================================

def serialize(root):
    """
    Pre-order serialization with '#' markers for None.
    WHY markers: they remove ambiguity so we can rebuild the exact shape.
    """
    result = []
    def dfs(node):
        if not node:
            result.append("#")
            return
        result.append(str(node.val))
        dfs(node.left)
        dfs(node.right)
    dfs(root)
    return ",".join(result)


def deserialize(data):
    """
    Rebuild the tree from its serialized form.
    We consume tokens left-to-right using an iterator.
    """
    tokens = iter(data.split(","))

    def build():
        val = next(tokens)
        if val == "#":
            return None
        node = TreeNode(int(val))
        node.left = build()           # Pre-order: build left then right
        node.right = build()
        return node

    return build()


# =============================================================================
# MAIN DEMO
# =============================================================================

if __name__ == "__main__":
    root = build_sample_tree()
    print("Sample tree values (inorder sorted):")
    print("  preorder :", preorder_recursive(root))
    print("  inorder  :", inorder_recursive(root))
    print("  postorder:", postorder_recursive(root))
    print("  iterative preorder:", preorder_iterative(root))
    print("  iterative inorder :", inorder_iterative(root))

    print("\nBFS level order:", level_order(root))
    print("Right side view:", right_side_view(root))

    print("\nMax depth :", max_depth(root))
    print("Diameter  :", diameter(root))
    print("Balanced? :", is_balanced(root))

    print("\nValid BST?", is_valid_bst(root))
    print("Search 5 :", search_bst(root, 5))
    print("Search 42:", search_bst(root, 42))

    # LCA demo: find ancestor of nodes with values 1 and 3
    p = root.left.left                # value 1
    q = root.left.right               # value 3
    print("\nLCA(1, 3) generic:", lca_binary_tree(root, p, q).val)
    print("LCA(1, 3) BST    :", lca_bst(root, p, q).val)

    print("\nHas path sum 7 (4+2+1)?", has_path_sum(root, 7))
    print("Has path sum 100?", has_path_sum(root, 100))

    s = serialize(root)
    print("\nSerialized:", s)
    root2 = deserialize(s)
    print("Deserialized inorder:", inorder_recursive(root2))
    print("Round trip OK?", inorder_recursive(root) == inorder_recursive(root2))
