#Uses python3

import sys
# from typing import List, Set, Union
from collections import deque

sys.setrecursionlimit(200000)

clock = 0

class Node():
    def __init__(self, key=None, graph_node_id=None):
        self.parent = None
        self.key = key # post-order count
        self.height = 1
        # self.subtree_node_count = 1
        self.graph_node_id = graph_node_id
        self.left = None
        self.right = None


class SCC:
    def __init__(self):
        self.root = None
        self.postorder_to_graph_node_id = dict()
        self.graph_node_id_to_postorder = dict()


def adjust_height(N: Node):
    assert N is not None
    
    N.height = 1
    if N.left is not None and N.right is not None:
        N.height += max(N.left.height, N.right.height)
    elif N.left is not None:
        N.height += N.left.height
    elif N.right is not None:
        N.height += N.right.height


"""
def adjust_subtree_node_count(N: Node):
    assert N is not None

    N.subtree_node_count = 1
    if N.left is not None:
        N.subtree_node_count += N.left.subtree_node_count
    if N.right is not None:
        N.subtree_node_count += N.right.subtree_node_count
"""


def rebalance(N: Node):
    # assert N is not None
    P = N.parent
    root = N
    if (N.left is not None) and (((N.right is None) and (N.left.height > 1)) or (N.right and (N.left.height > N.right.height + 1))):
        root = rebalance_right(N)
    elif (N.right is not None) and (((N.left is None) and (N.right.height > 1)) or (N.left and (N.right.height > N.left.height + 1))):
        root = rebalance_left(N)
    else:
        adjust_height(N)
        # adjust_subtree_node_count(N)
    if P:
        return rebalance(P)
    return root


def avl_insert(scc: SCC, postorder_number: int, graph_node_id: int) -> Node:
    # print("avl_insert")
    new_node = insert_node(scc, postorder_number, graph_node_id)
    # print_node(new_node)
    # print(f"tree in avl_insert:")
    # print_tree(scc.root)
    return rebalance(new_node)


def avl_delete(root: Node, key: int):  # -> Union[Node, None]:
    # print(f"avl_delete key={key}")
    replacement_node = delete_node(root, key)
    # print(f"avl_delete replacement_node={replacement_node}")
    if replacement_node:
        return rebalance(replacement_node)
    else:
        return None


def rebalance_right(N: Node):
    assert N.left is not None
    M = N.left
    if (M.right is not None) and (M.left is None or M.right.height > M.left.height):
        rotate_left(M)
    return rotate_right(N)


def rebalance_left(N: Node):
    assert N.right is not None
    M = N.right
    if (M.left is not None) and (M.right is None or M.left.height > M.right.height):
        rotate_right(M)
    return rotate_left(N)


# X guaranteed to have left child Y.
# X not guaranteed to have parent P
# X not guaranteed to have right child
# Y not guaranteed to have right child
# def rotate_right(root: Node, X_key: int):
def rotate_right(X: Node):
    # X = find_node(root, X_key)
    assert X.left is not None

    P = X.parent
    Y = X.left
    B = Y.right

    Y.parent = P
    if P:
        if P.left == X:
            P.left = Y
        elif P.right == X:
            P.right = Y

    X.parent = Y
    Y.right = X

    if B:
        B.parent = X
    X.left = B

    adjust_height(X)
    # adjust_subtree_node_count(X)
    adjust_height(Y)
    # adjust_subtree_node_count(Y)
    return Y


# X guaranteed to have right child Y.
# X not guaranteed to have parent P
# X not guaranteed to have left child
# Y not guaranteed to have left child
# def rotate_left(root: Node, X_key: int):
def rotate_left(X: Node):
    # X = find_node(root, X_key)
    assert X.right is not None

    P = X.parent
    Y = X.right
    B = Y.left

    Y.parent = P
    if P:
        if P.left == X:
            P.left = Y
        elif P.right == X:
            P.right = Y

    X.parent = Y
    Y.left = X

    if B:
        B.parent = X
    X.right = B

    adjust_height(X)
    # adjust_subtree_node_count(X)
    adjust_height(Y)
    # adjust_subtree_node_count(Y)

    return Y

"""
def print_node(node: Node):
    print(f"Node_key={node.key}", end=' ')
    print(f"Node_height={node.height}", end=' ')
    print(f"Node_graph_node_id={node.graph_node_id}", end=' ')

    #print(f"Node_subtree_node_count={node.subtree_node_count}", end=' ')
    if node.parent:
        print(f"Parent_key={node.parent.key}", end=' ')
    else:
        print("Parent_key=None", end=' ')
    if node.left:
        print(f"Left_key={node.left.key}", end=' ')
    else:
        print("Left_key=None", end=' ')
    if node.right:
        print(f"Right_key={node.right.key}", end=' ')
    else:
        print("Right_key=None", end=' ')
    print()

# after find_node, check if returned node is None,
# case 1) returned node is None
# case 2) returned node.key == key
# case 3) returned node.key != key, and returned node should be parent of new node with node.key == key
def print_tree(root: Node):
    node_q = deque()
    node_q.append(root)
    print("Begin Whole Tree")
    while len(node_q) > 0:
        de_queued_node = node_q.popleft()
        if not de_queued_node:
            break
        print_node(de_queued_node)
        if de_queued_node.left:
            node_q.append(de_queued_node.left)
        if de_queued_node.right:
            node_q.append(de_queued_node.right)
    print("End Whole Tree")
    print()
"""

def find_node(root: Node, key: int):
    if root is None:
        return None

    elif root.key == key:
        return root

    elif key < root.key:
        if root.left is not None:
            return find_node(root.left, key)
        else:
            return root

    elif key > root.key:
        if root.right is not None:
            return find_node(root.right, key)
        else:
            return root

    assert False


def insert_node(scc: SCC, key: int, graph_node_id: int) -> Node: # Tuple[Node, Node]:
    if scc.root is None:
        new_root = Node(key, graph_node_id)
        scc.root = new_root
        return scc.root

    parent = find_node(scc.root, key)

    new_node = Node(key, graph_node_id)
    new_node.parent = parent

    if key < parent.key:
        parent.left = new_node
    elif key > parent.key:
        parent.right = new_node

    assert key != parent.key

    return new_node
    #return root, new_node
    #return root


def left_descendant(node: Node):  # -> Union[Node, None]:
    if node is None:
        return None
    elif node.left is None:
        return node
    else:
        return left_descendant(node.left)


def delete_node(root: Node, key: int):  # -> Union[Node, None]:
    node = find_node(root, key)
    assert node is not None
    if node.left is None and node.right is None:
        if node.parent is None:
            return None
        else:
            if node == node.parent.left:
                node.parent.left = None
            elif node == node.parent.right:
                node.parent.right = None
            parent = node.parent
            node.parent = None
            return parent


    if node.right is None:
        # remove node, promote node.left
        if node.parent is None:
            node.left.parent = None
        else:
            node.left.parent = node.parent

            if node == node.parent.left:
                node.parent.left = node.left
            elif node == node.parent.right:
                node.parent.right = node.left

            node.parent = None

        node_left_of_deleted_node = node.left
        node.left = None

        return node_left_of_deleted_node

    replacement = left_descendant(node.right)
    # replacement.left is None

    # node to delete is parent of replacement node
    if node == replacement.parent:

        replacement.parent = node.parent
        if replacement.parent:
            if replacement.parent.right == node:
                replacement.parent.right = replacement
            elif replacement.parent.left == node:
                replacement.parent.left = replacement
            else:
                assert False

        replacement.left = node.left
        if replacement.left:
            replacement.left.parent = replacement

        node.parent = None
        node.left = None
        node.right = None

        return replacement

    else:
        # replace node with replacement, detach and promote replacement.right
        if replacement.right:
            replacement.right.parent = replacement.parent

        if replacement == replacement.parent.left:
            replacement.parent.left = replacement.right
        elif replacement == replacement.parent.right:
            replacement.parent.right = replacement.right

        # need to store value to return, since the value will be overwritten
        # replacement_parent = replacement.parent

        replacement.parent = node.parent
        replacement.right = node.right
        replacement.left = node.left
        if replacement.right:
            replacement.right.parent = replacement
        if replacement.left:
            replacement.left.parent = replacement
        if replacement.parent:
            if replacement.parent.left == node:
                replacement.parent.left = replacement
            elif replacement.parent.right == node:
                replacement.parent.right = replacement

        node.parent = None
        node.left = None
        node.right = None

        return replacement
        #return replacement_parent


def reverse_adj(adj):  # List[Set[int]]) -> List[Set[int]]:
    adj_reverse = [set() for _ in range(len(adj))]
    for v, w_list in enumerate(adj):
        for w in w_list:
            adj_reverse[w].add(v)
    return adj_reverse


def count_non_empty_sets(adj_reverse):
    count = 0
    for v_set in adj_reverse:
        if len(v_set):
            count += 1
    return count

# def dfs_explore(adj_reverse: List[Set[int]], visited: Set[int], scc: Union[SCC, None], graph_node_id: int):
def dfs_explore(adj_reverse, visited, scc, graph_node_id: int):

    global clock
    visited.add(graph_node_id)
    for other_node in adj_reverse[graph_node_id]:
        # print(f"visited={visited} scc={scc} other_node={other_node}")
        if other_node not in visited:
            # print(f"inner loop through other_node other_node={other_node}")
            dfs_explore(adj_reverse, visited, scc, other_node)
    clock += 1
    # print(f"clock={clock}")
    # print(f"clock={clock} graph_node_id={graph_node_id}")
    # new_node = Node(clock, graph_node_id)
    # clock is postorder value
    new_root = avl_insert(scc, clock, graph_node_id)
    # print(f"tree after avl_insert")
    scc.root = new_root
    # print_tree(scc.root)
    scc.postorder_to_graph_node_id[clock] = graph_node_id
    scc.graph_node_id_to_postorder[graph_node_id] = clock

# def dfs(n: int, adj_reverse: List[Set[int]], scc: SCC):
def dfs(n: int, adj_reverse, scc: SCC):
    global clock
    clock = 0
    visited = set()
    for graph_node_id in range(n):
        if graph_node_id not in visited:
            # print(f"outer loop through graph_node_id: graph_node_id {graph_node_id}")
            dfs_explore(adj_reverse, visited, scc, graph_node_id)


# def set_adj(adj_sets: List[Set[int]]) -> List[Set[int]]:
def set_adj(adj_sets):  # ): List[Set[int]]) -> List[Set[int]]:
    adj = [set() for _ in range(len(adj_sets))]
    for v, w_set in enumerate(adj_sets):
        for w in w_set:
            adj[v].add(w)
    return adj

def get_tree_max(root: Node) -> Node:
    if root.right is None:
        return root
    else:
        return get_tree_max(root.right)

# def node_ids_in_single_scc_explore(adj_sets: List[Set[int]], visited: Set[int], graph_node_id: int,
#                                   node_ids_in_single_scc: List[int]):
def node_ids_in_single_scc_explore(adj_sets, visited, graph_node_id: int,
                                   node_ids_in_single_scc):
    visited.add(graph_node_id)
    node_ids_in_single_scc.append(graph_node_id)
    for other_node in adj_sets[graph_node_id]:
        # print(f"visited={visited} other_node={other_node}")
        if other_node not in visited:
            # print(f"inner loop through other_node other_node={other_node}")
            node_ids_in_single_scc_explore(adj_sets, visited, other_node, node_ids_in_single_scc)

    # print(f"graph_node_id={graph_node_id}")
    return node_ids_in_single_scc


# def get_all_node_ids_in_single_scc(adj_sets: List[Set[int]], node_id_with_largest_postorder: int):
def get_all_node_ids_in_single_scc(adj_sets, node_id_with_largest_postorder: int):
    visited = set()
    node_ids_in_single_scc = []
    return node_ids_in_single_scc_explore(adj_sets, visited, node_id_with_largest_postorder, node_ids_in_single_scc)

# def number_of_strongly_connected_components(adj_sets: List[Set[int]], n: int) -> int:
def number_of_strongly_connected_components(adj_sets, n: int) -> int:
    result = 0
    # write your code here

    # make set version of edge list
    #adj = set_adj(adj_sets)
    # print("adj_sets", adj_sets, sep='\n')

    adj_reverse = reverse_adj(adj_sets)
    # print("adj_reverse", adj_reverse, sep="\n")
    # for v_set in adj_reverse:
        # print(len(v_set), end=' ')
    # print()
    non_empty_set_count = count_non_empty_sets(adj_reverse)
    # print(f"non_empty_set_count: {non_empty_set_count}")

    scc = SCC()

    # make tree with dfs, using keys postorder numbering of nodes
    dfs(n, adj_reverse, scc)

    # print_tree(scc.root)
    # print(scc.postorder_to_graph_node_id)

    scc_count = 0

    while scc.root is not None:
        # print_node(scc.root)
        # get sink node_id from node tree with largest postorder
        node_id_with_largest_postorder = (get_tree_max(scc.root)).graph_node_id
        # print(f"node_id_with_largest_postorder={node_id_with_largest_postorder}")
        # on forward graph, explore all nodes from sink and make node list
        all_node_ids_in_single_scc = get_all_node_ids_in_single_scc(adj_sets, node_id_with_largest_postorder)
        # print(f"all_node_ids_in_single_scc={all_node_ids_in_single_scc}")
        for node_id in all_node_ids_in_single_scc:
            # print(f"delete for loop node_id={node_id}")
            scc.root = avl_delete(scc.root, scc.graph_node_id_to_postorder[node_id])
            # print("delete for loop after avl_delete")
            # print_tree(scc.root)
            # (f"before deletion scc.graph_node_id_to_postorder {scc.graph_node_id_to_postorder}")
            for reverse_node_id in adj_sets[node_id]:
                adj_reverse[reverse_node_id].discard(node_id)
            for forward_node_id in adj_reverse[node_id]:
                adj_sets[forward_node_id].discard(node_id)
            adj_sets[node_id] = set()
            adj_reverse[node_id] = set()

        # remove all explored nodes from node tree
        # remove all explored nodes from forward graph

        # increment scc count
        scc_count += 1
        # print(f"scc_count={scc_count}")
    return scc_count


if __name__ == '__main__':
    input = sys.stdin.read()
    data = list(map(int, input.split()))
    n, m = data[0:2]
    data = data[2:]
    edges = list(zip(data[0:(2 * m):2], data[1:(2 * m):2]))
    """
    adj_list = [[] for _ in range(n)]
    for (a, b) in edges:
        adj_list[a - 1].append(b - 1)
    """
    adj_sets = [set() for _ in range(n)]
    for (a, b) in edges:
        adj_sets[a - 1].add(b - 1)

    print(number_of_strongly_connected_components(adj_sets, n))
