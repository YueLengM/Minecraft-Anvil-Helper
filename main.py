#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
from typing import List, Optional, Tuple


# every node is an item
class Node:
    def __init__(self,
                 value: int = 0,
                 penalty: Optional[int] = None,
                 l_weight: Optional[int] = None,
                 r_weight: Optional[int] = None,
                 parent: Optional['Node'] = None,
                 is_right_node: bool = False,
                 is_main_path: bool = False,
                 is_leaf_node: bool = False) -> None:
        self.left: Optional['Node'] = None
        self.right: Optional['Node'] = None

        self.value = value
        self.penalty = penalty

        self.parent = parent
        self.is_right_node = is_right_node
        self.is_main_path = is_main_path
        self.is_leaf_node = is_leaf_node

        if l_weight:
            self.l_weight = l_weight
        elif parent:
            self.l_weight = parent.l_weight if is_right_node else parent.l_weight + 1
        else:
            self.l_weight = 0

        if r_weight:
            self.r_weight = r_weight
        elif parent:
            self.r_weight = parent.r_weight + 1 if is_right_node else parent.r_weight
        else:
            self.r_weight = 0

        if parent:
            self.layer = parent.layer + 1
        else:
            self.layer = 0

    def __str__(self) -> str:
        lines = _build_tree_string(self, '-')[0]
        return '\n' + '\n'.join((line.rstrip() for line in lines))

    def print(self, debug: bool = False):
        lines = _build_tree_string(self, '-', debug)[0]
        print('\n' + '\n'.join((line.rstrip() for line in lines)))


# modified from https://github.com/joowani/binarytree/blob/d53a4b22472934ca7e23ac35c9677c70bf139071/binarytree/__init__.py#L1858
def _build_tree_string(root: Optional[Node],
                       delimiter: str = '-',
                       debug: bool = False) -> Tuple[List[str], int, int, int]:
    if root is None:
        return [], 0, 0, 0

    line1 = []
    line2 = []

    if debug:
        node_repr = '|'.join([
            '{}{}{}|W{}L{}'.format(root.value, delimiter, root.penalty,
                                   root.r_weight, root.layer),
            '{}{}{}'.format('M' if root.is_main_path else '',
                            'R' if root.is_right_node else '',
                            'B' if root.is_leaf_node else '')
        ])
    else:
        node_repr = '{}{}{}'.format(root.value, delimiter, root.penalty)

    new_root_width = gap_size = len(node_repr)

    l_box, l_box_width, l_root_start, l_root_end = _build_tree_string(
        root.left, delimiter, debug)
    r_box, r_box_width, r_root_start, r_root_end = _build_tree_string(
        root.right, delimiter, debug)

    if l_box_width > 0:
        l_root = (l_root_start + l_root_end) // 2 + 1
        line1.append(' ' * (l_root + 1))
        line1.append('_' * (l_box_width - l_root))
        line2.append(' ' * l_root + '/')
        line2.append(' ' * (l_box_width - l_root))
        new_root_start = l_box_width + 1
        gap_size += 1
    else:
        new_root_start = 0

    line1.append(node_repr)
    line2.append(' ' * new_root_width)

    if r_box_width > 0:
        r_root = (r_root_start + r_root_end) // 2
        line1.append('_' * r_root)
        line1.append(' ' * (r_box_width - r_root + 1))
        line2.append(' ' * r_root + '\\')
        line2.append(' ' * (r_box_width - r_root))
        gap_size += 1
    new_root_end = new_root_start + new_root_width - 1

    gap = ' ' * gap_size
    new_box = [''.join(line1), ''.join(line2)]
    for i in range(max(len(l_box), len(r_box))):
        l_line = l_box[i] if i < len(l_box) else ' ' * l_box_width
        r_line = r_box[i] if i < len(r_box) else ' ' * r_box_width
        new_box.append(l_line + gap + r_line)

    return new_box, len(new_box[0]), new_root_start, new_root_end


def gen_tree(layer, node_count, book_value) -> Node:
    book_value = sorted(book_value)
    # possible positions for new nodes
    options: List[List[Node, str, int, int, int]] = []
    min_used = 0

    def add_option(parent: Node, dirciton):
        if parent.layer + 1 == layer:
            return

        new_layer_cost = 1
        if parent.penalty == 1:
            # if adding a new node will grow the subtree one layer
            # then assume a layer penalty to the option
            new_layer_cost = test_penalty(parent)

        if dirciton == 'left':
            options.append([
                parent, 'left', new_layer_cost, parent.l_weight + 1,
                parent.r_weight
            ])
        elif dirciton == 'right':
            options.append([
                parent, 'right', new_layer_cost, parent.l_weight,
                parent.r_weight + 1
            ])
        elif dirciton == 'both':
            options.append([
                parent, 'left', new_layer_cost, parent.l_weight + 1,
                parent.r_weight
            ])
            options.append([
                parent, 'right', new_layer_cost, parent.l_weight,
                parent.r_weight + 1
            ])

    # after adding a new node, needs to change penalties for each parent node
    def refresh_penalties(item: Node):
        item.penalty = 1
        while True:
            item = item.parent
            if item:
                if item.is_main_path:
                    break

                if item.left and item.right:
                    i = max(item.left.penalty, item.left.penalty)
                elif item.left:
                    i = item.left.penalty
                elif item.right:
                    i = item.right.penalty
                else:
                    i = 1

                new_penalty = 2 * i + 1
                if item.penalty == new_penalty:
                    # no change will happen exit early
                    break
                else:
                    item.penalty = new_penalty
            else:
                break
        # the assumed penalties in the options needs to refresh too
        refresh_option_penalties()

    def refresh_option_penalties():
        for opt in options:
            if opt[0].penalty != 1:
                # if parent penalty isn't 1 then there is another child node
                opt[2] = 1

    def test_penalty(item: Node):
        penalty = 0
        while True:
            if item.is_main_path:
                break
            penalty = item.penalty
            item = item.parent
        return 2 * penalty + 1

    root = Node(r_weight=0, penalty=pow(2, layer) - 1, is_main_path=True)
    add_option(root, 'right')

    # building main path (the far left branch)
    item = root
    for i in range(layer - 1):
        item.left = Node(penalty=pow(2, layer - 1 - i) - 1,
                         parent=item,
                         is_main_path=True)
        item = item.left
        add_option(item, 'right')

    remains = node_count - layer
    while remains:
        # sort option by costs
        options.sort(key=lambda x: (x[2] + x[4] * book_value[min_used], -x[3]))
        # print(options)

        # adding node to the tree
        opt = options.pop(0)
        p: Node = opt[0]
        if opt[1] == 'left':
            p.left = Node(parent=p)
            new = p.left
        else:
            p.right = Node(parent=p, is_right_node=True)
            new = p.right

        refresh_penalties(new)

        remains -= 1
        if remains == 0:
            break

        add_option(new, 'both')

        nxt = options[0]
        if (opt[2] + opt[3] * book_value[min_used]) == (
                nxt[2] + nxt[3] * book_value[min_used]):
            # if there is an equal node need to use a more expensive one
            min_used += 1
        else:
            min_used = 0

        # print(root)
    return root


# add leaf nodes (book nodes) to the tree
def add_leaf(tree: Node):
    queue = [tree]
    while queue:
        cuur = queue.pop(0)
        if cuur.left:
            queue.append(cuur.left)
        else:
            cuur.left = Node(penalty=0, parent=cuur, is_leaf_node=True)
        if cuur.right:
            queue.append(cuur.right)
        else:
            cuur.right = Node(penalty=0,
                              parent=cuur,
                              is_right_node=True,
                              is_leaf_node=True)


def get_leaf(tree: Node) -> List[Node]:
    queue = [tree]
    leaves = []

    while queue:
        cuur = queue.pop(0)
        if cuur.left:
            queue.insert(0, cuur.right)
            queue.insert(0, cuur.left)
        else:
            leaves.append(cuur)
    return leaves


def get_main_parent(node: Node) -> Node:
    curr = node
    while not curr.is_main_path:
        curr = curr.parent
    return curr


# total prior work penalties of a tree
def get_penalties(tree: Node) -> int:
    total = 0
    queue = [tree]
    while queue:
        cuur = queue.pop(0)
        if cuur.left:
            total += cuur.left.penalty
            queue.append(cuur.left)
        if cuur.right:
            total += cuur.right.penalty
            queue.append(cuur.right)
    return total


# all the leaf node weights
def get_weights(tree: Node) -> List[int]:
    weights = []
    queue = [tree]
    while queue:
        cuur = queue.pop(0)
        if cuur.left:
            queue.insert(0, cuur.right)
            queue.insert(0, cuur.left)
        else:
            weights.append(cuur.r_weight)
    # print(weights)
    return weights


def calc_book_order(weights: List[int], leaf: List[Node],
                    books: List[int]) -> Tuple[List[int], int]:
    length = len(books)
    reamin = books.copy()
    # add leaf index to weights list and sort with weights
    indexed = [(i, x, leaf[i + 1]) for i, x in enumerate(weights[1:])]
    ordered = []

    while indexed:
        indexed.sort(key=lambda x: (
            x[1], get_main_parent(x[2]).left.penalty + get_main_parent(x[
                2]).right.penalty + get_main_parent(x[2]).right.value +
            (0 if x[2].parent.is_main_path else books[-1])))
        curr_book = reamin.pop(0)
        curr_node = indexed.pop(0)
        get_main_parent(curr_node[2]).right.value += curr_book
        # add sorted leaf index to book values
        ordered.append((curr_book, curr_node[0]))

    # sort back to leaf order
    ordered.sort(key=lambda x: x[1])

    total = 0
    for i in range(length):
        total += ordered[i][0] * weights[i + 1]

    return ([x[0] for x in ordered], total)


def fill_in_value(tree: Node, values: List[int]):
    leaf = get_leaf(tree)

    leaf[0].value = 0
    for i, v in enumerate(values):
        leaf[i + 1].value = v
        if leaf[i + 1].is_right_node:
            bubble = leaf[i + 1]
            while bubble:
                bubble = bubble.parent
                bubble.value = bubble.left.value + bubble.right.value
                if not bubble.is_right_node:
                    break
    # print([x.value for x in get_leaf(tree)])


def get_highest_cost(tree: Node) -> int:
    highest = 0
    queue = [tree]
    while queue:
        cuur = queue.pop(0)
        if cuur.left:
            queue.insert(0, cuur.right)
            queue.insert(0, cuur.left)
            highest = max(
                highest,
                cuur.left.penalty + cuur.right.value + cuur.right.penalty)
    return highest


def main():
    ipt = input('List of book values: ')
    # ipt = '12 12 6 4 4 3 2'
    ipt = ipt.split()
    ipt = [int(x) for x in ipt]

    if not ipt:
        return
    ipt.sort(reverse=True)

    cnt = len(ipt)

    trees: List[Node] = []
    for i in range(math.ceil(math.log2(cnt + 1)), cnt + 1):
        # generating diferent layers of trees
        trees.append(gen_tree(layer=i, node_count=cnt, book_value=ipt))

    result_length = len(trees)

    base_costs = []  # total value cost of each anvil combine of each trees
    penalties = []  # total prior work penalties of each trees
    for i in range(result_length):
        add_leaf(trees[i])
        base_costs.append(
            calc_book_order(get_weights(trees[i]), get_leaf(trees[i]), ipt))
        penalties.append(get_penalties(trees[i]))
        fill_in_value(trees[i], base_costs[i][0])

    exp_costs = [base_costs[i][1] + penalties[i] for i in range(result_length)]
    max_single_cost = [get_highest_cost(tree) for tree in trees]

    results_title = ('Minimum Prior Work Penalties',
                     'Minimum total Exp required',
                     'Minimum single step Exp required')
    results_idx = [
        trees.index(min(trees, key=lambda x: x.penalty)),
        exp_costs.index(min(exp_costs)),
        max_single_cost.index(min(max_single_cost))
    ]

    for i in range(len(results_title)):
        print(results_title[i] + ':')
        trees[results_idx[i]].print()

        exp = exp_costs[results_idx[i]]
        print('Total Exp required: {} level{}'.format(exp, 's' [:exp ^ 1]))
        highest = max_single_cost[results_idx[i]]
        print('The highest Exp cost step: {} level{}'.format(
            highest, 's' [:highest ^ 1]))

        print('-' * 40)


if __name__ == '__main__':
    while True:
        main()
